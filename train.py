import argparse
import random
import numpy as np
import torch
import os
import gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from env import make_env
from env.observation_wrapper import TemporalObservationWrapper
from config import OBSERVATION_HISTORY, NUM_LIDAR

# ------------------------------------------------------------
# MODIFIED GymWrapper with seed support
# ------------------------------------------------------------
class GymWrapper(gym.Env):
    def __init__(self, hybrid=False, seed=None):
        super().__init__()
        self.env = make_env(hybrid)
        self._seed = seed
        self.hybrid = hybrid
        if hybrid:
            low = np.array([0.0] * NUM_LIDAR + [-1.0, -1.0, -1.0])
            high = np.array([1.0] * NUM_LIDAR + [1.0, 1.0, 1.0])
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_LIDAR,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        return obs, {}
    
    """
    def reset(self, *, seed=None, options=None):
        actual_seed = seed if seed is not None else self._seed
        if actual_seed is not None:
            obs = self.env.reset(seed=actual_seed)
        else:
            obs = self.env.reset()
        return obs, {}
        """

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def get_robot_position(self):
        return self.env.get_robot_position()

    def get_dynamic_obstacle_distances(self):
        return self.env.get_dynamic_obstacle_distances()

# ------------------------------------------------------------
# Environment factory for SubprocVecEnv
# ------------------------------------------------------------

def make_env_fn(hybrid, seed, rank, algorithm, history_len):
    def _init():
        # Set seeds for this subprocess (for numpy, random, etc.)
        import random
        import numpy as np
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        env = GymWrapper(hybrid=hybrid)  # no seed argument
        if algorithm in ['lstm-sac', 'tx-sac']:
            env = TemporalObservationWrapper(env, history_len=history_len)
        return env
    return _init

"""
def make_env_fn(hybrid, seed, rank, algorithm, history_len):
    """ 
"""Returns a function that creates a single environment with temporal wrapper if needed."""
"""
    def _init():
        # Create base env with unique seed
        env = GymWrapper(hybrid=hybrid, seed=seed + rank)
        # Apply temporal wrapper only for LSTM/Transformer policies
        if algorithm in ['lstm-sac', 'tx-sac']:
            env = TemporalObservationWrapper(env, history_len=history_len)
        return env
    return _init
    
    """

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_type', type=str, default='mixed', choices=['static','mixed','dense'])
    parser.add_argument('--algorithm', type=str, default='sac', choices=['sac','ppo','lstm-sac','tx-sac'])
    parser.add_argument('--hybrid', action='store_true')
    parser.add_argument('--total_timesteps', type=int, default=500000)
    parser.add_argument('--eval_freq', type=int, default=10000)
    parser.add_argument('--gui', action='store_true', help='Only for visualisation, NEVER during training')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_envs', type=int, default=1,
                        help='Number of parallel environments (1 = DummyVecEnv, >1 = SubprocVecEnv)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to a pretrained model zip file')
    parser.add_argument('--exp_suffix', type=str, default='',
                        help='Suffix to add to model name (e.g., independent, curriculum)')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.gui:
        print("WARNING: --gui enabled. This will slow training dramatically. Use only for debugging/visualisation.")
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Seed: {args.seed} | Parallel envs: {args.num_envs}")
    set_seed(args.seed)

    import config
    config.ENV_TYPE = args.env_type
    config.GUI = args.gui

    # --------------------------------------------------------
    # CREATE TRAINING ENVIRONMENT(S)
    # --------------------------------------------------------
    if args.num_envs > 1:
        # Parallel environments using SubprocVecEnv
        env_fns = [make_env_fn(args.hybrid, args.seed, i, args.algorithm, OBSERVATION_HISTORY)
                   for i in range(args.num_envs)]
        env = SubprocVecEnv(env_fns)
    else:
        # Single environment (original behaviour)
        env = GymWrapper(hybrid=args.hybrid, seed=args.seed)
        if args.algorithm in ['lstm-sac', 'tx-sac']:
            env = TemporalObservationWrapper(env, history_len=OBSERVATION_HISTORY)
        env = DummyVecEnv([lambda: env])

    # Normalise observations and rewards (works with both DummyVecEnv and SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # --------------------------------------------------------
    # CREATE EVALUATION ENVIRONMENT (always single process)
    # --------------------------------------------------------
    config.GUI = False  # headless for evaluation
    eval_env = GymWrapper(hybrid=args.hybrid, seed=args.seed + 1000)  # separate seed
    if args.algorithm in ['lstm-sac', 'tx-sac']:
        eval_env = TemporalObservationWrapper(eval_env, history_len=OBSERVATION_HISTORY)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    config.GUI = args.gui  # restore

    # --------------------------------------------------------
    # POLICY KWARGS FOR TEMPORAL MODELS
    # --------------------------------------------------------
    actual_obs_dim = NUM_LIDAR + 3 if args.hybrid else NUM_LIDAR
    policy_kwargs = None
    suffix = f"_{args.exp_suffix}" if args.exp_suffix else ""
    model_name = f'{args.env_type}_{args.algorithm}_{"hybrid" if args.hybrid else "end2end"}_seed{args.seed}{suffix}'
    if args.algorithm == 'lstm-sac':
        from policies.lstm_policy import LSTMFeatureExtractor
        policy_kwargs = dict(
            features_extractor_class=LSTMFeatureExtractor,
            features_extractor_kwargs=dict(history_len=OBSERVATION_HISTORY, input_dim=actual_obs_dim,
                                            hidden_dim=128, features_dim=256)
        )
    elif args.algorithm == 'tx-sac':
        from policies.transformer_policy import TemporalTransformerExtractor
        policy_kwargs = dict(
            features_extractor_class=TemporalTransformerExtractor,
            features_extractor_kwargs=dict(history_len=OBSERVATION_HISTORY, input_dim=actual_obs_dim,
                                            features_dim=256)
        )

    # --------------------------------------------------------
    # CREATE OR LOAD MODEL
    # --------------------------------------------------------
    if args.algorithm == 'sac':
        if args.pretrained:
            model = SAC.load(args.pretrained, env=env, device=device)
            norm_path = os.path.join(os.path.dirname(args.pretrained), "vecnormalize.pkl")
            if os.path.exists(norm_path):
                env = VecNormalize.load(norm_path, env)
                env.training = True
        else:
            model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=100000,
                        batch_size=256, tau=0.005, gamma=0.99, train_freq=1,
                        gradient_steps=1, ent_coef="auto_0.1", verbose=1,
                        tensorboard_log=f'./tb_logs/{model_name}/', device=device)
    elif args.algorithm == 'ppo':
        if args.pretrained:
            model = PPO.load(args.pretrained, env=env, device=device)
            norm_path = os.path.join(os.path.dirname(args.pretrained), "vecnormalize.pkl")
            if os.path.exists(norm_path):
                env = VecNormalize.load(norm_path, env)
                env.training = True
        else:
            model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                        gae_lambda=0.95, gamma=0.99, ent_coef=0.01, verbose=1,
                        tensorboard_log=f'./tb_logs/{model_name}/', device=device)
    elif args.algorithm in ['lstm-sac', 'tx-sac']:
        if args.pretrained:
            model = SAC.load(args.pretrained, env=env, device=device)
            norm_path = os.path.join(os.path.dirname(args.pretrained), "vecnormalize.pkl")
            if os.path.exists(norm_path):
                env = VecNormalize.load(norm_path, env)
                env.training = True
        else:
            model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=300000,
                        batch_size=256, tau=0.005, gamma=0.99, train_freq=1,
                        gradient_steps=1, ent_coef="auto_0.1", verbose=1,
                        tensorboard_log=f'./tb_logs/{model_name}/', device=device, policy_kwargs=policy_kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

   
    eval_callback = EvalCallback(eval_env, best_model_save_path=f'./models/{model_name}',
                                 log_path=f'./logs/{model_name}', eval_freq=args.eval_freq,
                                 deterministic=True, render=False)

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

    # Save final model and normalisation stats
    model_dir = f"./models/{model_name}_final"
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "model.zip"))
    env.save(os.path.join(model_dir, "vecnormalize.pkl"))

if __name__ == "__main__":
    main()