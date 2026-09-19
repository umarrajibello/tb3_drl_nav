# FULL CORRECTED TRAIN.PY
# Works with end‑to‑end (TB3Env) and hybrid (TB3HybridEnv)
# Supports SAC, PPO, and temporal models (LSTM, Transformer, CNN‑GRU)

import argparse
import random
import numpy as np
import torch
import os

import gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from env.tb3_env import TB3Env
from env.tb3_env_hybrid1 import TB3HybridEnv
from env.observation_wrapper import TemporalObservationWrapper
from config import OBSERVATION_HISTORY

# ------------------------------------------------------------
# Seeding helper
# ------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ------------------------------------------------------------
# Base env factory
# ------------------------------------------------------------
def make_base_env(hybrid: bool, env_type: str, gui: bool):
    if hybrid:
        return TB3HybridEnv(env_type=env_type, gui=gui)
    else:
        return TB3Env(env_type=env_type, gui=gui)

# ------------------------------------------------------------
# VecEnv factory for SubprocVecEnv
# ------------------------------------------------------------
def make_env_fn(hybrid, seed, rank, algorithm, env_type, gui):
    def _init():
        env_seed = seed + rank
        random.seed(env_seed)
        np.random.seed(env_seed)
        env = make_base_env(hybrid=hybrid, env_type=env_type, gui=gui)
        env.reset(seed=env_seed)
        if algorithm in ['lstm-sac', 'tx-sac', 'cnn-gru-sac']:
            env = TemporalObservationWrapper(env)
        return env
    return _init

# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_type', type=str, default='mixed')
    parser.add_argument('--algorithm', type=str, default='cnn-gru-sac',
                        choices=['sac', 'ppo', 'lstm-sac', 'tx-sac', 'cnn-gru-sac'])
    parser.add_argument('--hybrid', action='store_true')
    parser.add_argument('--total_timesteps', type=int, default=500000)
    parser.add_argument('--eval_freq', type=int, default=10000)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--exp_suffix', type=str, default='')
    return parser.parse_args()

# ------------------------------------------------------------
# Main training
# ------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    import config
    config.ENV_TYPE = args.env_type
    config.GUI = args.gui

    # ---------------- ENV CREATION ----------------
    if args.num_envs > 1:
        env_fns = [make_env_fn(args.hybrid, args.seed, i, args.algorithm, args.env_type, args.gui)
                   for i in range(args.num_envs)]
        env = SubprocVecEnv(env_fns)
    else:
        def _single_env():
            base = make_base_env(args.hybrid, args.env_type, args.gui)
            base.reset(seed=args.seed)
            if args.algorithm in ['lstm-sac', 'tx-sac', 'cnn-gru-sac']:
                base = TemporalObservationWrapper(base)
            return base
        env = DummyVecEnv([_single_env])

    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    

    # ---------------- EVAL ENV ----------------
    def _eval_env():
        base = make_base_env(args.hybrid, args.env_type, args.gui)
        if args.algorithm in ['lstm-sac', 'tx-sac', 'cnn-gru-sac']:
            base = TemporalObservationWrapper(base)
        return base
    eval_env = DummyVecEnv([_eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    eval_env.norm_reward = False
    eval_env.obs_rms = env.obs_rms  
    eval_env.training = False

    # ---------------- MODEL NAME ----------------
    mode = "hybrid" if args.hybrid else "end2end"
    model_name = f'{args.env_type}_{args.algorithm}_{mode}_seed{args.seed}_{args.exp_suffix}'

    # ---------------- POLICY KWARGS ----------------
    policy_kwargs = None
    if args.algorithm == 'cnn-gru-sac':
        try:
            from policies.cnn_gru_extractor import SB3_CNNGRU_Extractor
            policy_kwargs = dict(
                features_extractor_class=SB3_CNNGRU_Extractor,
                features_extractor_kwargs=dict(features_dim=256),
                net_arch=dict(pi=[256], qf=[256])
            )
        except ImportError:
            print("WARNING: CNN-GRU extractor not found. Falling back to MLP.")
            args.algorithm = 'sac'
    elif args.algorithm == 'lstm-sac':
        try:
            from policies.lstm_extractor import SB3_LSTM_Extractor
            policy_kwargs = dict(
                features_extractor_class=SB3_LSTM_Extractor,
                features_extractor_kwargs=dict(features_dim=256),
                net_arch=dict(pi=[256], qf=[256])
            )
        except ImportError:
            print("WARNING: LSTM extractor not found. Falling back to MLP.")
            args.algorithm = 'sac'
    elif args.algorithm == 'tx-sac':
        try:
            from policies.transformer_extractor import SB3_Transformer_Extractor
            policy_kwargs = dict(
                features_extractor_class=SB3_Transformer_Extractor,
                features_extractor_kwargs=dict(features_dim=256),
                net_arch=dict(pi=[256], qf=[256])
            )
        except ImportError:
            print("WARNING: Transformer extractor not found. Falling back to MLP.")
            args.algorithm = 'sac'

    # ---------------- MODEL CREATION ----------------
    if args.algorithm == 'sac':
        model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=300000,
                    batch_size=256, tau=0.005, gamma=0.99, ent_coef="auto_0.5",
                    verbose=1, tensorboard_log=f"./tb_logs/{model_name}", device=device)
    elif args.algorithm == 'ppo':
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=128,
                    n_epochs=10, gamma=0.995, gae_lambda=0.95, clip_range=0.2,
                    vf_coef=0.5, max_grad_norm=0.5, ent_coef=0.01,
                    policy_kwargs=dict(net_arch=[256,256]), verbose=1,
                    tensorboard_log=f"./tb_logs/{model_name}", device=device)
    else:  # temporal models
        model = SAC("MlpPolicy", env, learning_rate=1e-4, buffer_size=800000,
                    batch_size=512, tau=0.005, gamma=0.99, ent_coef="auto_0.5",
                    policy_kwargs=policy_kwargs, verbose=1,
                    tensorboard_log=f"./tb_logs/{model_name}", device=device)

    if args.pretrained and os.path.isfile(args.pretrained):
        model.set_parameters(args.pretrained)

    # ---------------- TRAIN ----------------
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"./models/{model_name}",
                                 log_path=f"./logs/{model_name}", eval_freq=args.eval_freq,
                                 deterministic=True)
    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

    # ---------------- SAVE ----------------
    model_dir = f"./models/{model_name}_final"
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "model.zip"))
    env.save(os.path.join(model_dir, "vecnormalize.pkl"))

if __name__ == "__main__":
    main()