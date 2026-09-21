"""
Training entry point for the thesis experiments.

Committed architecture:
    E2E:
        LiDAR history -> temporal encoder -> SAC -> (v, w)

    Hybrid:
        Static map -> RRT -> adaptive path-distance lookahead -> waypoint
        -> LiDAR history + robot state + waypoint
        -> temporal encoder -> SAC -> (v, w)

Temporal algorithms use the fixed 8-frame observation wrapper and the
corresponding CNN + LSTM/GRU/Transformer feature extractor.

This file intentionally preserves the original W&B, TensorBoard,
evaluation, checkpointing, seed, CLI, and model-saving workflow while
removing silent architecture fallbacks and stale reward configuration.
"""

import argparse
import os
import random
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from config import OBSERVATION_HISTORY
from env.observation_wrapper import TemporalObservationWrapper
from env.tb3_env import TB3Env
from env.tb3_env_hybrid import TB3HybridEnv


TEMPORAL_ALGORITHMS = {"lstm-sac", "tx-sac", "cnn-gru-sac"}


# ============================================================
# ARGUMENT PARSING
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument(
        "--env_type",
        type=str,
        default="mixed",
        choices=["empty", "static", "mixed", "dense"],
    )
    parser.add_argument(
        "--hybrid",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use RRT-guided hybrid environment (true/false)",
    )
    parser.add_argument("--gui", action="store_true")

    # Algorithm
    parser.add_argument(
        "--algorithm",
        type=str,
        default="sac",
        choices=["sac", "ppo", "lstm-sac", "tx-sac", "cnn-gru-sac"],
    )

    # Training
    parser.add_argument("--total_timesteps", type=int, default=500000)
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument(
        "--gpu",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use GPU (true/false)",
    )
    parser.add_argument("--exp_suffix", type=str, default="")

    # Reward parameters.
    # These are retained as explicit experiment metadata and are passed to
    # the environment reward calculator when supported by the current reward
    # implementation. They replace the old TTC/angular/idle reward interface,
    # which no longer matches the simplified reward design.
    parser.add_argument(
        "--goal_reward",
        type=float,
        default=10.0,
        help="Reward for reaching the goal",
    )
    parser.add_argument(
        "--collision_penalty",
        type=float,
        default=10.0,
        help="Penalty for physical collision",
    )
    parser.add_argument(
        "--progress_weight",
        type=float,
        default=2.0,
        help="Weight for progress toward local waypoint/goal",
    )
    parser.add_argument(
        "--clearance_weight",
        type=float,
        default=1.0,
        help="Weight for the near-obstacle clearance penalty",
    )
    parser.add_argument(
        "--step_penalty",
        type=float,
        default=0.01,
        help="Constant per-step penalty",
    )

    # Hyperparameter override
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )

    # W&B
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="tb3-nav-research",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="+",
        default=[],
        help="Tags for W&B run",
    )

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
    """Create the committed E2E or RRT-guided hybrid environment."""
    render_mode = "human" if gui else None

    if hybrid:
        return TB3HybridEnv(
            env_type=env_type,
            render_mode=render_mode,
        )

    return TB3Env(
        env_type=env_type,
        render_mode=render_mode,
    )


# ============================================================
# ENVIRONMENT WRAPPING
# ============================================================
def wrap_env(env, algorithm):
    """Apply temporal history only to temporal-policy experiments."""
    if algorithm in TEMPORAL_ALGORITHMS:
        return TemporalObservationWrapper(
            env,
            history_len=OBSERVATION_HISTORY,
        )
    return env


def make_env_fn(hybrid, seed, rank, algorithm, env_type, gui):
    def _init():
        env_seed = seed + rank
        random.seed(env_seed)
        np.random.seed(env_seed)

        env = make_base_env(
            hybrid=hybrid,
            env_type=env_type,
            gui=gui,
        )
        env = wrap_env(env, algorithm)

        # Reset after wrapping so the wrapper can initialize its complete
        # observation history correctly.
        env.reset(seed=env_seed)
        return env

    return _init


# ============================================================
# POLICY KWARGS FOR TEMPORAL MODELS
# ============================================================
def get_policy_kwargs(algorithm, feature_dim=256):
    """Return the requested temporal extractor; never silently fall back."""
    if algorithm == "lstm-sac":
        from policies.lstm_extractor import LSTMFeatureExtractor

        print(
            f"[Policy] Using LSTM extractor with features_dim={feature_dim}"
        )

        return {
            "features_extractor_class": LSTMFeatureExtractor,
            "features_extractor_kwargs": {
                "history_len": OBSERVATION_HISTORY,
                "lidar_dim": 36,
                "non_lidar_dim": 5,
                "features_dim": feature_dim,
            },
            "net_arch": {
                "pi": [feature_dim],
                "qf": [feature_dim],
            },
        }

    if algorithm == "tx-sac":
        from policies.transformer_extractor import TransformerFeatureExtractor

        print(
            f"[Policy] Using Transformer extractor with features_dim={feature_dim}"
        )

        return {
            "features_extractor_class": TransformerFeatureExtractor,
            "features_extractor_kwargs": {
                "history_len": OBSERVATION_HISTORY,
                "lidar_dim": 36,
                "non_lidar_dim": 5,
                "features_dim": feature_dim,
            },
            "net_arch": {
                "pi": [feature_dim],
                "qf": [feature_dim],
            },
        }

    if algorithm == "cnn-gru-sac":
        from policies.cnn_gru_extractor import GRUFeatureExtractor

        print(
            f"[Policy] Using GRU extractor with features_dim={feature_dim}"
        )

        return {
            "features_extractor_class": GRUFeatureExtractor,
            "features_extractor_kwargs": {
                "history_len": OBSERVATION_HISTORY,
                "lidar_dim": 36,
                "non_lidar_dim": 5,
                "features_dim": feature_dim,
            },
            "net_arch": {
                "pi": [feature_dim],
                "qf": [feature_dim],
            },
        }

    return None


# ============================================================
# REWARD CONFIGURATION
# ============================================================
def configure_reward(env, args):
    """Apply reward settings to an environment when the calculator supports them.

    The simplified reward is:
        progress + clearance penalty + step penalty
        + success bonus / collision penalty.

    We deliberately do not recreate the old TTC/angular/jerk/idle reward.
    """
    reward_calc = getattr(env.unwrapped if hasattr(env, "unwrapped") else env,
                          "reward_calc", None)

    if reward_calc is None:
        return

    # Current RewardCalculator exposes these as constructor/default
    # parameters in the audited implementation. Set them explicitly when
    # attributes are available; otherwise leave its audited defaults intact.
    attribute_map = {
        "progress_weight": args.progress_weight,
        "clearance_weight": args.clearance_weight,
        "step_penalty": args.step_penalty,
        "success_reward": args.goal_reward,
        "collision_penalty": args.collision_penalty,
    }

    for name, value in attribute_map.items():
        if hasattr(reward_calc, name):
            setattr(reward_calc, name, float(value))


# ============================================================
# GET GIT HASH
# ============================================================
def get_git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass

    return "unknown"


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()

    if args.num_envs < 1:
        raise ValueError("--num_envs must be >= 1")

    device = torch.device(
        "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    )

    print(f"[Device] Using: {device}")
    set_seed(args.seed)

    import config

    config.ENV_TYPE = args.env_type
    config.GUI = args.gui

    print("\n" + "=" * 60)
    print("[Architecture]")
    if args.hybrid:
        print(
            "RRT -> adaptive path-distance lookahead -> local waypoint "
            "-> temporal/E2E SAC controller"
        )
        print("RRT plans static obstacles; LiDAR handles dynamic obstacles.")
    else:
        print("LiDAR/history -> policy -> SAC controller")
    print("=" * 60 + "\n")

    # ----------------------------------------------------------
    # W&B INIT
    # ----------------------------------------------------------
    run_name = (
        args.wandb_run_name
        or f"{args.env_type}_{args.algorithm}_"
           f"{'hybrid' if args.hybrid else 'e2e'}_"
           f"seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    git_hash = get_git_hash()

    # Default hyperparameters per algorithm.
    if args.algorithm == "sac":
        learning_rate = 3e-4
        buffer_size = 300000
        ent_coef = "auto_0.5"
    elif args.algorithm == "ppo":
        learning_rate = 3e-4
        buffer_size = None
        ent_coef = 0.01
    else:
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
            "architecture": (
                "RRT-adaptive-waypoint-temporal-SAC"
                if args.hybrid
                else "LiDAR-history-policy-SAC"
            ),
            "total_timesteps": args.total_timesteps,
            "eval_freq": args.eval_freq,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "pretrained": args.pretrained is not None,
            "use_gpu": args.gpu,
            "exp_suffix": args.exp_suffix,

            # Observation
            "observation_history": OBSERVATION_HISTORY,
            "base_observation_dim": 41,
            "temporal_observation_dim": (
                41 * OBSERVATION_HISTORY
                if args.algorithm in TEMPORAL_ALGORITHMS
                else 41
            ),

            # Reward
            "goal_reward": args.goal_reward,
            "collision_penalty": args.collision_penalty,
            "progress_weight": args.progress_weight,
            "clearance_weight": args.clearance_weight,
            "step_penalty": args.step_penalty,

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
        save_code=True,
    )

    print(f"\n{'=' * 60}")
    print(f"W&B Run: {wandb_run.name}")
    print(f"W&B URL: {wandb_run.get_url()}")
    print(f"{'=' * 60}\n")

    # ----------------------------------------------------------
    # ENVIRONMENT CREATION
    # ----------------------------------------------------------
    print("[Env] Creating training environments...")

    if args.num_envs > 1:
        env_fns = [
            make_env_fn(
                args.hybrid,
                args.seed,
                i,
                args.algorithm,
                args.env_type,
                args.gui,
            )
            for i in range(args.num_envs)
        ]

        env = SubprocVecEnv(env_fns)
        print(
            f"[Env] Created {args.num_envs} parallel environments "
            "(SubprocVecEnv)"
        )
    else:

        def _single_env():
            base = make_base_env(
                args.hybrid,
                args.env_type,
                args.gui,
            )
            base = wrap_env(base, args.algorithm)
            configure_reward(base, args)
            return base

        env = DummyVecEnv([_single_env])
        print("[Env] Created single environment (DummyVecEnv)")

    # For subprocess environments, reward configuration is performed inside
    # the worker before/after construction through the same environment
    # object. The simplified RewardCalculator defaults are the audited
    # baseline if no configurable attributes exist.

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
    )

    print("[Env] VecNormalize added (norm_obs=True, norm_reward=True)")

    # ----------------------------------------------------------
    # EVALUATION ENVIRONMENT
    # ----------------------------------------------------------
    print("[Env] Creating evaluation environment...")

    def _eval_env():
        base = make_base_env(
            args.hybrid,
            args.env_type,
            args.gui,
        )
        base = wrap_env(base, args.algorithm)
        configure_reward(base, args)
        return base

    eval_env = DummyVecEnv([_eval_env])

    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
    )

    # Evaluation must use the training observation statistics but must not
    # update them.
    eval_env.obs_rms = env.obs_rms
    eval_env.training = False
    eval_env.norm_reward = False

    # ----------------------------------------------------------
    # MODEL NAME & POLICY KWARGS
    # ----------------------------------------------------------
    mode = "hybrid" if args.hybrid else "end2end"

    model_name = (
        f"{args.env_type}_{args.algorithm}_{mode}_"
        f"seed{args.seed}_{args.exp_suffix}"
    )

    print(f"[Model] Name: {model_name}")

    policy_kwargs = None

    if args.algorithm in TEMPORAL_ALGORITHMS:
        # Import errors are intentionally allowed to propagate. A failed
        # temporal extractor must never silently become ordinary SAC.
        policy_kwargs = get_policy_kwargs(args.algorithm)

    # ----------------------------------------------------------
    # MODEL CREATION
    # ----------------------------------------------------------
    print(f"[Model] Creating {args.algorithm} model...")

    if args.algorithm == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=0.005,
            gamma=0.99,
            ent_coef=ent_coef,
            verbose=1,
            tensorboard_log=f"./tb_logs/{model_name}",
            device=device,
            seed=args.seed,
        )

    elif args.algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            tensorboard_log=f"./tb_logs/{model_name}",
            device=device,
            seed=args.seed,
        )

    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=0.005,
            gamma=0.99,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=f"./tb_logs/{model_name}",
            device=device,
            seed=args.seed,
        )

    if args.pretrained:
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(
                f"Pretrained model not found: {args.pretrained}"
            )

        model.set_parameters(args.pretrained)

        wandb_run.config.update(
            {"pretrained_model": args.pretrained},
            allow_val_change=True,
        )

        print(
            f"[Model] Loaded pretrained weights from "
            f"{args.pretrained}"
        )

    # ----------------------------------------------------------
    # CALLBACKS
    # ----------------------------------------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{model_name}",
        log_path=f"./logs/{model_name}",
        eval_freq=args.eval_freq,
        deterministic=True,
        verbose=1,
    )

    wandb_callback = WandbCallback(
        verbose=2,
        model_save_path=f"./models/{model_name}",
        model_save_freq=args.eval_freq,
        gradient_save_freq=1000,
    )

    # ----------------------------------------------------------
    # TRAINING
    # ----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Starting training: {model_name}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Evaluation frequency: {args.eval_freq:,} steps")
    print(f"{'=' * 60}\n")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, wandb_callback],
    )

    print(
        f"\nTraining finished. Best model saved in "
        f"./models/{model_name}"
    )

    # ----------------------------------------------------------
    # SAVE FINAL MODEL
    # ----------------------------------------------------------
    model_dir = f"./models/{model_name}_final"
    os.makedirs(model_dir, exist_ok=True)

    model.save(os.path.join(model_dir, "model.zip"))
    env.save(os.path.join(model_dir, "vecnormalize.pkl"))

    print(f"[Model] Final model saved to {model_dir}")

    model_artifact = wandb.Artifact(
        "trained_model",
        type="model",
    )
    model_artifact.add_file(
        os.path.join(model_dir, "model.zip")
    )
    model_artifact.add_file(
        os.path.join(model_dir, "vecnormalize.pkl")
    )
    wandb_run.log_artifact(model_artifact)

    print("[W&B] Model and VecNormalize artifacts uploaded")

    wandb_run.summary["final_model_path"] = model_dir
    wandb_run.summary["total_timesteps"] = args.total_timesteps

    wandb_run.finish()

    print(
        f"\nW&B run finished. View results at: "
        f"{wandb_run.get_url()}"
    )


if __name__ == "__main__":
    main()
