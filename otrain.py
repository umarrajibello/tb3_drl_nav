# train.py – Complete W&B integration with normalised reward and sweep support

import argparse
import random
import numpy as np
import torch
import os
import sys
import subprocess
from datetime import datetime

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from env.tb3_env import TB3Env
from env.tb3_env_hybrid import TB3HybridEnv
from env.observation_wrapper import TemporalObservationWrapper
from config import OBSERVATION_HISTORY


# ============================================================
# ARGUMENT PARSING
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    
    # Environment
    parser.add_argument('--env_type', type=str, default='mixed',
                        choices=['empty','static', 'mixed', 'dense'])
    parser.add_argument('--hybrid', type=lambda x: x.lower() == 'true', default=False,
                        help='Use hybrid environment (true/false)')
    parser.add_argument('--gui', action='store_true')
    
    # Algorithm
    parser.add_argument('--algorithm', type=str, default='sac',
                        choices=['sac', 'ppo', 'lstm-sac', 'tx-sac', 'cnn-gru-sac'])
    
    # Training
    parser.add_argument('--total_timesteps', type=int, default=500000)
    parser.add_argument('--eval_freq', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--gpu', type=lambda x: x.lower() == 'true', default=True,
                        help='Use GPU (true/false)')
    parser.add_argument('--exp_suffix', type=str, default='')
    
    # Reward parameters (fully tunable for sweeps)
    parser.add_argument('--goal_reward', type=float, default=20.0,
                        help='Reward for reaching the goal')
    parser.add_argument('--collision_penalty', type=float, default=10.0,
                        help='Penalty for collision')
    parser.add_argument('--progress_weight', type=float, default=3.0,
                        help='Weight for progress toward waypoint/goal')
    parser.add_argument('--ttc_weight', type=float, default=1.0,
                        help='Weight for Time‑to‑Collision penalty')
    parser.add_argument('--safety_dist_weight', type=float, default=0.5,
                        help='Weight for distance‑based safety penalty')
    parser.add_argument('--angular_penalty', type=float, default=0.05,
                        help='Penalty for angular velocity')
    parser.add_argument('--angular_jerk_penalty', type=float, default=0.1,
                        help='Penalty for angular jerk')
    parser.add_argument('--step_penalty', type=float, default=0.02,
                        help='Constant step penalty')
    parser.add_argument('--idle_penalty', type=float, default=0.05,
                        help='Penalty for low linear velocity')
    
    # Hyperparameter override
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Override learning rate (default depends on algorithm)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size (default 256 for SAC/Tx-SAC)')
    
    # W&B
    parser.add_argument('--wandb_project', type=str, default='tb3-nav-research',
                        help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name (auto-generated if not provided)')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[],
                        help='Tags for W&B run')
    
    return parser.parse_args()


# ============================================================
# SEEDING
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# BASE ENV FACTORY
# ============================================================
def make_base_env(hybrid: bool, env_type: str, gui: bool):
    if hybrid:
        return TB3HybridEnv(env_type=env_type, gui=gui)
    else:
        return TB3Env(env_type=env_type, gui=gui)


# ============================================================
# VEC ENV FACTORY FOR SUBPROC
# ============================================================
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


# ============================================================
# POLICY KWARGS FOR TEMPORAL MODELS
# ============================================================
def get_policy_kwargs(algorithm, feature_dim=256):
    policy_kwargs = None
    if algorithm == 'cnn-gru-sac':
        try:
            from policies.cnn_gru_extractor import SB3_CNNGRU_Extractor
            policy_kwargs = dict(
                features_extractor_class=SB3_CNNGRU_Extractor,
                features_extractor_kwargs=dict(features_dim=feature_dim),
                net_arch=dict(pi=[feature_dim], qf=[feature_dim])
            )
            print(f"[Policy] Using CNN-GRU extractor with features_dim={feature_dim}")
        except ImportError:
            print("WARNING: CNN-GRU extractor not found. Falling back to MLP.")
            return None
    elif algorithm == 'lstm-sac':
        try:
            from policies.lstm_extractor import SB3_LSTM_Extractor
            policy_kwargs = dict(
                features_extractor_class=SB3_LSTM_Extractor,
                features_extractor_kwargs=dict(features_dim=feature_dim),
                net_arch=dict(pi=[feature_dim], qf=[feature_dim])
            )
            print(f"[Policy] Using LSTM extractor with features_dim={feature_dim}")
        except ImportError:
            print("WARNING: LSTM extractor not found. Falling back to MLP.")
            return None
    elif algorithm == 'tx-sac':
        try:
            from policies.transformer_extractor import SB3_Transformer_Extractor
            policy_kwargs = dict(
                features_extractor_class=SB3_Transformer_Extractor,
                features_extractor_kwargs=dict(features_dim=feature_dim),
                net_arch=dict(pi=[feature_dim], qf=[feature_dim])
            )
            print(f"[Policy] Using Transformer extractor with features_dim={feature_dim}")
        except ImportError:
            print("WARNING: Transformer extractor not found. Falling back to MLP.")
            return None
    return policy_kwargs


# ============================================================
# GET GIT HASH
# ============================================================
def get_git_hash():
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except:
        pass
    return "unknown"


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using: {device}")
    set_seed(args.seed)
    
    import config
    config.ENV_TYPE = args.env_type
    config.GUI = args.gui
    
    # Push all reward parameters to environment variables for reward.py
    os.environ['GOAL_REWARD'] = str(args.goal_reward)
    os.environ['COLLISION_PENALTY'] = str(args.collision_penalty)
    os.environ['PROGRESS_WEIGHT'] = str(args.progress_weight)
    os.environ['TTC_WEIGHT'] = str(args.ttc_weight)
    os.environ['SAFETY_DIST_WEIGHT'] = str(args.safety_dist_weight)
    os.environ['ANGULAR_PENALTY'] = str(args.angular_penalty)
    os.environ['ANGULAR_JERK_PENALTY'] = str(args.angular_jerk_penalty)
    os.environ['STEP_PENALTY'] = str(args.step_penalty)
    os.environ['IDLE_PENALTY'] = str(args.idle_penalty)
    
    print(f"[Reward] goal={args.goal_reward}, collision={args.collision_penalty}, "
          f"progress={args.progress_weight}, ttc={args.ttc_weight}, "
          f"safety_dist={args.safety_dist_weight}, angular={args.angular_penalty}, "
          f"step={args.step_penalty}, idle={args.idle_penalty}")
    
    # ----------------------------------------------------------
    # W&B INIT
    # ----------------------------------------------------------
    run_name = args.wandb_run_name or f"{args.env_type}_{args.algorithm}_{'hybrid' if args.hybrid else 'e2e'}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    git_hash = get_git_hash()
    
    # Default hyperparameters per algorithm
    if args.algorithm == 'sac':
        learning_rate = 3e-4
        buffer_size = 300000
        ent_coef = "auto_0.5"
    elif args.algorithm == 'ppo':
        learning_rate = 3e-4
        buffer_size = None
        ent_coef = 0.01
    else:  # temporal models
        learning_rate = 5e-5
        buffer_size = 800000
        ent_coef = 0.1
    
    batch_size = 256
    if args.learning_rate is not None:
        learning_rate = args.learning_rate
    if args.batch_size is not None:
        batch_size = args.batch_size
    
    wandb_run = wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode,
        tags=args.wandb_tags,
        config={
            # Environment & algorithm
            "env_type": args.env_type,
            "algorithm": args.algorithm,
            "hybrid": args.hybrid,
            "total_timesteps": args.total_timesteps,
            "eval_freq": args.eval_freq,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "pretrained": args.pretrained is not None,
            "use_gpu": args.gpu,
            "exp_suffix": args.exp_suffix,
            # Reward parameters
            "goal_reward": args.goal_reward,
            "collision_penalty": args.collision_penalty,
            "progress_weight": args.progress_weight,
            "ttc_weight": args.ttc_weight,
            "safety_dist_weight": args.safety_dist_weight,
            "angular_penalty": args.angular_penalty,
            "angular_jerk_penalty": args.angular_jerk_penalty,
            "step_penalty": args.step_penalty,
            "idle_penalty": args.idle_penalty,
            # Hyperparameters
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "gamma": 0.99,
            "tau": 0.005,
            "ent_coef": ent_coef,
            # System
            "device": str(device),
            "git_commit": git_hash,
            "python_version": sys.version[:50],
            "torch_version": torch.__version__,
        },
        name=run_name,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )
    
    print(f"\n{'='*60}")
    print(f"W&B Run: {wandb_run.name}")
    print(f"W&B URL: {wandb_run.get_url()}")
    print(f"{'='*60}\n")
    
    # ----------------------------------------------------------
    # ENVIRONMENT CREATION
    # ----------------------------------------------------------
    print("[Env] Creating training environments...")
    if args.num_envs > 1:
        env_fns = [make_env_fn(args.hybrid, args.seed, i, args.algorithm, args.env_type, args.gui)
                   for i in range(args.num_envs)]
        env = SubprocVecEnv(env_fns)
        print(f"[Env] Created {args.num_envs} parallel environments (SubprocVecEnv)")
    else:
        def _single_env():
            base = make_base_env(args.hybrid, args.env_type, args.gui)
            base.reset(seed=args.seed)
            if args.algorithm in ['lstm-sac', 'tx-sac', 'cnn-gru-sac']:
                base = TemporalObservationWrapper(base)
            return base
        env = DummyVecEnv([_single_env])
        print("[Env] Created single environment (DummyVecEnv)")
    
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    print("[Env] VecNormalize added (norm_obs=True, norm_reward=True)")
    
    # ----------------------------------------------------------
    # EVALUATION ENVIRONMENT
    # ----------------------------------------------------------
    print("[Env] Creating evaluation environment...")
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
    
    # ----------------------------------------------------------
    # MODEL NAME & POLICY KWARGS
    # ----------------------------------------------------------
    mode = "hybrid" if args.hybrid else "end2end"
    model_name = f'{args.env_type}_{args.algorithm}_{mode}_seed{args.seed}_{args.exp_suffix}'
    print(f"[Model] Name: {model_name}")
    
    policy_kwargs = None
    if args.algorithm in ['lstm-sac', 'tx-sac', 'cnn-gru-sac']:
        policy_kwargs = get_policy_kwargs(args.algorithm)
        if policy_kwargs is None:
            print("[Policy] Falling back to SAC with MLP policy")
            args.algorithm = 'sac'
    
    # ----------------------------------------------------------
    # MODEL CREATION
    # ----------------------------------------------------------
    print(f"[Model] Creating {args.algorithm} model...")
    if args.algorithm == 'sac':
        model = SAC("MlpPolicy", env,
                    learning_rate=learning_rate,
                    buffer_size=300000,
                    batch_size=batch_size,
                    tau=0.005,
                    gamma=0.99,
                    ent_coef="auto_0.5",
                    verbose=1,
                    tensorboard_log=f"./tb_logs/{model_name}",
                    device=device)
    elif args.algorithm == 'ppo':
        model = PPO("MlpPolicy", env,
                    learning_rate=learning_rate,
                    n_steps=2048,
                    batch_size=batch_size if batch_size else 128,
                    n_epochs=10,
                    gamma=0.995,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    vf_coef=0.5,
                    ent_coef=0.01,
                    policy_kwargs=dict(net_arch=[256,256]),
                    verbose=1,
                    tensorboard_log=f"./tb_logs/{model_name}",
                    device=device)
    else:  # temporal models
        model = SAC("MlpPolicy", env,
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,
                    batch_size=batch_size,
                    tau=0.005,
                    gamma=0.99,
                    ent_coef=ent_coef,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=f"./tb_logs/{model_name}",
                    device=device)
    
    if args.pretrained and os.path.isfile(args.pretrained):
        model.set_parameters(args.pretrained)
        wandb_run.config.update({"pretrained_model": args.pretrained}, allow_val_change=True)
        print(f"[Model] Loaded pretrained weights from {args.pretrained}")
    
    # ----------------------------------------------------------
    # CALLBACKS
    # ----------------------------------------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{model_name}",
        log_path=f"./logs/{model_name}",
        eval_freq=args.eval_freq,
        deterministic=True,
        verbose=1
    )
    wandb_callback = WandbCallback(
        verbose=2,
        model_save_path=f"./models/{model_name}",
        model_save_freq=args.eval_freq,
        gradient_save_freq=1000
    )
    
    # ----------------------------------------------------------
    # TRAINING
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Starting training: {model_name}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Evaluation frequency: {args.eval_freq:,} steps")
    print(f"{'='*60}\n")
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, wandb_callback]
    )
    
    print(f"\n✅ Training finished. Best model saved in ./models/{model_name}")
    
    # ----------------------------------------------------------
    # SAVE FINAL MODEL
    # ----------------------------------------------------------
    model_dir = f"./models/{model_name}_final"
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "model.zip"))
    env.save(os.path.join(model_dir, "vecnormalize.pkl"))
    print(f"[Model] Final model saved to {model_dir}")
    
    model_artifact = wandb.Artifact('trained_model', type='model')
    model_artifact.add_file(os.path.join(model_dir, "model.zip"))
    wandb_run.log_artifact(model_artifact)
    print("[W&B] Model artifact uploaded")
    
    wandb_run.summary["final_model_path"] = model_dir
    wandb_run.summary["total_timesteps"] = args.total_timesteps
    wandb_run.finish()
    print(f"\n✅ W&B run finished. View results at: {wandb_run.get_url()}")


if __name__ == "__main__":
    main()
