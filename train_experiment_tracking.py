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

This file preserves the thesis experiment workflow:
    - W&B
    - TensorBoard
    - evaluation
    - checkpointing
    - seeding
    - model saving
    - E2E / Hybrid architectures
    - SAC / PPO / temporal SAC variants

Important:
    num_envs controls how many environments run in parallel.
    Callback frequencies are automatically adjusted so that evaluation
    and checkpointing remain approximately consistent in terms of
    actual environment transitions.
"""

import argparse
import json
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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

from config import OBSERVATION_HISTORY
from env.observation_wrapper import TemporalObservationWrapper
from env.tb3_env import TB3Env
from env.tb3_env_hybrid import TB3HybridEnv


TEMPORAL_ALGORITHMS = {
    "lstm-sac",
    "tx-sac",
    "cnn-gru-sac",
}


# ============================================================
# ARGUMENT PARSING
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------
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

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable PyBullet GUI",
    )

    # --------------------------------------------------------
    # Algorithm
    # --------------------------------------------------------
    parser.add_argument(
        "--algorithm",
        type=str,
        default="sac",
        choices=[
            "sac",
            "ppo",
            "lstm-sac",
            "tx-sac",
            "cnn-gru-sac",
        ],
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=500000,
    )

    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10000,
        help=(
            "Evaluation interval measured in approximate environment "
            "transitions. Automatically adjusted for num_envs."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help=(
            "Number of parallel training environments. "
            "Use 1 initially; increase only after benchmarking."
        ),
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--gpu",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use GPU if CUDA is available (true/false)",
    )

    parser.add_argument(
        "--exp_suffix",
        type=str,
        default="",
    )

    # --------------------------------------------------------
    # Off-policy training control
    # --------------------------------------------------------
    parser.add_argument(
        "--gradient_steps",
        type=int,
        default=1,
        help=(
            "Number of SAC gradient updates after each rollout step. "
            "Use -1 to perform one gradient update per collected "
            "environment transition."
        ),
    )

    # --------------------------------------------------------
    # Reward parameters
    # --------------------------------------------------------
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
        help="Weight for near-obstacle clearance penalty",
    )

    parser.add_argument(
        "--step_penalty",
        type=float,
        default=0.01,
        help="Constant per-step penalty",
    )

    # --------------------------------------------------------
    # Hyperparameter overrides
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # W&B
    # --------------------------------------------------------
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
    """Set the main Python/NumPy/PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        # Reproducibility is useful for thesis experiments.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# BASE ENVIRONMENT FACTORY
# ============================================================
def make_base_env(hybrid: bool, env_type: str, gui: bool):
    """
    Create the committed E2E or hybrid environment.

    GUI is controlled through Gymnasium's render_mode rather than
    the global config.GUI value so training can safely run headless.
    """
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
    """
    Apply temporal observation history only to temporal experiments.

    SAC and PPO use the normal 41-dimensional observation unless
    a temporal algorithm explicitly requests the 8-frame history.
    """
    if algorithm in TEMPORAL_ALGORITHMS:
        env = TemporalObservationWrapper(
            env,
            history_len=OBSERVATION_HISTORY,
        )

    return env


# ============================================================
# REWARD CONFIGURATION
# ============================================================
def configure_reward(env, args):
    """
    Apply the experiment's reward parameters.

    The current reward contains:
        progress
        clearance penalty
        step penalty
        success reward
        collision penalty

    This function is intentionally simple and does not recreate
    the previous TTC/angular/jerk/idle reward.
    """
    reward_calc = getattr(
        env.unwrapped if hasattr(env, "unwrapped") else env,
        "reward_calc",
        None,
    )

    if reward_calc is None:
        return

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
# TRAINING ENVIRONMENT FACTORY
# ============================================================
def make_env_fn(
    hybrid,
    seed,
    rank,
    algorithm,
    env_type,
    gui,
    args,
):
    """
    Factory used by SubprocVecEnv.

    IMPORTANT:
        This function runs inside each subprocess.

    Therefore reward configuration must happen here, not only
    in the parent process.
    """

    def _init():
        env_seed = seed + rank

        # Seed Python and NumPy inside this worker.
        random.seed(env_seed)
        np.random.seed(env_seed)

        env = make_base_env(
            hybrid=hybrid,
            env_type=env_type,
            gui=gui,
        )

        env = wrap_env(
            env,
            algorithm,
        )

        # IMPORTANT:
        # Configure the reward inside the worker process.
        configure_reward(
            env,
            args,
        )

        # Reset after wrapping so the temporal wrapper can correctly
        # initialize its history.
        env.reset(
            seed=env_seed,
        )

        # Monitor must be inside the vectorized environment so SB3 can
        # correctly record episode reward and episode length.
        env = Monitor(env)

        return env

    return _init


# ============================================================
# POLICY KWARGS
# ============================================================
def get_policy_kwargs(
    algorithm,
    feature_dim=256,
):
    """
    Return the requested temporal feature extractor.

    There is intentionally no silent fallback to ordinary MLP SAC.
    """
    if algorithm == "lstm-sac":
        from policies.lstm_extractor import LSTMFeatureExtractor

        print(
            f"[Policy] Using LSTM extractor "
            f"with features_dim={feature_dim}"
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
        from policies.transformer_extractor import (
            TransformerFeatureExtractor,
        )

        print(
            f"[Policy] Using Transformer extractor "
            f"with features_dim={feature_dim}"
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
            f"[Policy] Using GRU extractor "
            f"with features_dim={feature_dim}"
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
# GIT HASH
# ============================================================
def get_git_hash():
    """Return a short git commit hash for experiment tracking."""
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
# LOCAL EXPERIMENT METADATA
# ============================================================
def build_experiment_metadata(
    args,
    device,
    learning_rate,
    batch_size,
    buffer_size,
    ent_coef,
    callback_eval_freq,
    git_hash,
):
    """Build a self-contained local record of the training run."""
    import config

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_hash,

        "environment": {
            "env_type": args.env_type,
            "hybrid": args.hybrid,
            "gui": args.gui,
            "num_envs": args.num_envs,
            "max_steps": config.MAX_STEPS,
            "goal_threshold": config.GOAL_THRESHOLD,
            "collision_distance": config.COLLISION_DIST,
            "lidar_rays": config.NUM_LIDAR,
            "lidar_range": config.LIDAR_RANGE,
            "observation_history": OBSERVATION_HISTORY,
            "planner_update_freq": config.PLANNER_UPDATE_FREQ,
            "waypoint_distance": config.WAYPOINT_DISTANCE,
        },

        "training": {
            "algorithm": args.algorithm,
            "total_timesteps": args.total_timesteps,
            "eval_freq_transitions": args.eval_freq,
            "callback_eval_freq": callback_eval_freq,
            "seed": args.seed,
            "gradient_steps": args.gradient_steps,
            "pretrained": args.pretrained,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "gamma": 0.99,
            "tau": 0.005,
            "ent_coef": ent_coef,
        },

        "reward": {
            "goal_reward": args.goal_reward,
            "collision_penalty": args.collision_penalty,
            "progress_weight": args.progress_weight,
            "clearance_weight": args.clearance_weight,
            "step_penalty": args.step_penalty,
        },

        "observation": {
            "base_dim": 41,
            "history_length": OBSERVATION_HISTORY,
            "temporal_dim": (
                41 * OBSERVATION_HISTORY
                if args.algorithm in TEMPORAL_ALGORITHMS
                else 41
            ),
        },

        "system": {
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": (
                int(torch.cuda.device_count())
                if torch.cuda.is_available()
                else 0
            ),
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },

        "tracking": {
            "wandb_project": args.wandb_project,
            "wandb_mode": args.wandb_mode,
            "wandb_tags": args.wandb_tags,
            "exp_suffix": args.exp_suffix,
        },
    }


def save_experiment_metadata(path, metadata):
    """Save experiment metadata locally as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)



# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------
    if args.num_envs < 1:
        raise ValueError(
            "--num_envs must be >= 1"
        )

    if args.gradient_steps == 0:
        raise ValueError(
            "--gradient_steps cannot be 0. "
            "Use a positive integer or -1."
        )

    if args.eval_freq < 1:
        raise ValueError(
            "--eval_freq must be >= 1"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    device = torch.device(
        "cuda"
        if args.gpu and torch.cuda.is_available()
        else "cpu"
    )

    print(f"[Device] Using: {device}")

    if args.gpu and not torch.cuda.is_available():
        print(
            "[Device] GPU requested, but CUDA is unavailable. "
            "Training will run on CPU."
        )

    set_seed(args.seed)

    import config

    config.ENV_TYPE = args.env_type
    config.GUI = args.gui

    # --------------------------------------------------------
    # Architecture summary
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Architecture]")

    if args.hybrid:
        print(
            "RRT -> adaptive path-distance lookahead -> "
            "local waypoint -> temporal/E2E SAC controller"
        )
        print(
            "RRT plans static obstacles; "
            "LiDAR handles dynamic obstacles."
        )
    else:
        print(
            "LiDAR/history -> policy -> SAC/PPO controller"
        )

    print("=" * 60)

    print("\n[Training configuration]")
    print(f"  Algorithm:       {args.algorithm}")
    print(f"  Environment:     {args.env_type}")
    print(f"  Hybrid:          {args.hybrid}")
    print(f"  Num envs:        {args.num_envs}")
    print(f"  Timesteps:       {args.total_timesteps:,}")
    print(f"  Seed:             {args.seed}")
    print(f"  Gradient steps:  {args.gradient_steps}")

    if args.num_envs > 1:
        print(
            "\n[Parallel training note]"
            "\n  Each vectorized step collects "
            f"{args.num_envs} transitions."
        )

    print("=" * 60 + "\n")

    # --------------------------------------------------------
    # Algorithm hyperparameters
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # W&B
    # --------------------------------------------------------
    run_name = (
        args.wandb_run_name
        or f"{args.env_type}_{args.algorithm}_"
        f"{'hybrid' if args.hybrid else 'e2e'}_"
        f"seed{args.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    git_hash = get_git_hash()

    metadata = build_experiment_metadata(
        args=args,
        device=device,
        learning_rate=learning_rate,
        batch_size=batch_size,
        buffer_size=buffer_size,
        ent_coef=ent_coef,
        callback_eval_freq=max(
            args.eval_freq // args.num_envs,
            1,
        ),
        git_hash=git_hash,
    )

    wandb_run = None

    if args.wandb_mode != "disabled":
        wandb_run = wandb.init(
            project=args.wandb_project,
            mode=args.wandb_mode,
            tags=args.wandb_tags,
            config=metadata,
            name=run_name,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

        print("\n" + "=" * 60)
        print(f"W&B Run: {wandb_run.name}")
        print(f"W&B URL: {wandb_run.get_url()}")
        print("=" * 60 + "\n")
    else:
        print("\n" + "=" * 60)
        print("[W&B] Disabled: no W&B run will be created.")
        print("[W&B] Local experiment metadata will still be saved.")
        print("=" * 60 + "\n")

    # --------------------------------------------------------
    # TRAINING ENVIRONMENT
    # --------------------------------------------------------
    print("[Env] Creating training environments...")

    if args.num_envs > 1:

        env_fns = [
            make_env_fn(
                hybrid=args.hybrid,
                seed=args.seed,
                rank=i,
                algorithm=args.algorithm,
                env_type=args.env_type,
                gui=args.gui,
                args=args,
            )
            for i in range(args.num_envs)
        ]

        env = SubprocVecEnv(
            env_fns
        )

        print(
            f"[Env] Created {args.num_envs} parallel "
            "environments using SubprocVecEnv."
        )

    else:

        def _single_env():
            base = make_base_env(
                hybrid=args.hybrid,
                env_type=args.env_type,
                gui=args.gui,
            )

            base = wrap_env(
                base,
                args.algorithm,
            )

            configure_reward(
                base,
                args,
            )

            base = Monitor(
                base
            )

            return base

        env = DummyVecEnv(
            [_single_env]
        )

        print(
            "[Env] Created single environment "
            "using DummyVecEnv."
        )

    # --------------------------------------------------------
    # VecNormalize
    # --------------------------------------------------------
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
    )

    print(
        "[Env] VecNormalize added "
        "(norm_obs=True, norm_reward=True)"
    )

    # --------------------------------------------------------
    # EVALUATION ENVIRONMENT
    # --------------------------------------------------------
    print("[Env] Creating evaluation environment...")

    def _eval_env():
        base = make_base_env(
            hybrid=args.hybrid,
            env_type=args.env_type,
            gui=args.gui,
        )

        base = wrap_env(
            base,
            args.algorithm,
        )

        configure_reward(
            base,
            args,
        )

        # IMPORTANT:
        # Monitor is applied before VecNormalize so evaluation can
        # recover true episode rewards and episode lengths.
        base = Monitor(
            base
        )

        return base

    eval_env = DummyVecEnv(
        [_eval_env]
    )

    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
    )

    # Evaluation uses training observation statistics.
    eval_env.obs_rms = env.obs_rms

    # Do not update normalization statistics during evaluation.
    eval_env.training = False

    # We want the original reward scale during evaluation.
    eval_env.norm_reward = False

    print(
        "[Eval] Using training observation normalization statistics."
    )
    print(
        "[Eval] Reward normalization disabled."
    )

    # --------------------------------------------------------
    # Callback frequency
    # --------------------------------------------------------
    #
    # SB3 callback counters count vectorized env.step() calls.
    #
    # Example:
    #     num_envs = 4
    #     eval_freq = 10000
    #
    # Without adjustment:
    #     10000 callback calls
    #     ≈ 40000 environment transitions
    #
    # We therefore divide by num_envs.
    #
    callback_eval_freq = max(
        args.eval_freq // args.num_envs,
        1,
    )

    print(
        f"[Eval] Requested evaluation interval: "
        f"{args.eval_freq:,} environment transitions"
    )

    print(
        f"[Eval] Actual callback frequency: "
        f"{callback_eval_freq:,} vectorized steps"
    )

    # --------------------------------------------------------
    # MODEL NAME
    # --------------------------------------------------------
    mode = (
        "hybrid"
        if args.hybrid
        else "end2end"
    )

    model_name = (
        f"{args.env_type}_{args.algorithm}_{mode}_"
        f"seed{args.seed}_{args.exp_suffix}"
    )

    print(
        f"[Model] Name: {model_name}"
    )

    # --------------------------------------------------------
    # POLICY KWARGS
    # --------------------------------------------------------
    policy_kwargs = None

    if args.algorithm in TEMPORAL_ALGORITHMS:
        policy_kwargs = get_policy_kwargs(
            args.algorithm
        )

    # --------------------------------------------------------
    # MODEL CREATION
    # --------------------------------------------------------
    print(
        f"[Model] Creating {args.algorithm} model..."
    )

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
            train_freq=1,
            gradient_steps=args.gradient_steps,
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
            policy_kwargs=dict(
                net_arch=[256, 256]
            ),
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
            train_freq=1,
            gradient_steps=args.gradient_steps,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=f"./tb_logs/{model_name}",
            device=device,
            seed=args.seed,
        )

    # --------------------------------------------------------
    # PRETRAINED PARAMETERS
    # --------------------------------------------------------
    if args.pretrained:

        if not os.path.isfile(
            args.pretrained
        ):
            raise FileNotFoundError(
                f"Pretrained model not found: "
                f"{args.pretrained}"
            )

        model.set_parameters(
            args.pretrained
        )

        wandb_run.config.update(
            {
                "pretrained_model": args.pretrained
            },
            allow_val_change=True,
        )

        print(
            "[Model] Loaded pretrained neural-network "
            "parameters from:"
        )
        print(
            f"        {args.pretrained}"
        )

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{model_name}",
        log_path=f"./logs/{model_name}",
        eval_freq=callback_eval_freq,
        deterministic=True,
        verbose=1,
    )

    callbacks = [eval_callback]

    if wandb_run is not None:
        wandb_callback = WandbCallback(
            verbose=2,
            model_save_path=f"./models/{model_name}",
            model_save_freq=callback_eval_freq,
            gradient_save_freq=1000,
        )
        callbacks.append(wandb_callback)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print(
        f"Starting training: {model_name}"
    )
    print(
        f"Total timesteps: {args.total_timesteps:,}"
    )
    print(
        f"Evaluation every approximately "
        f"{args.eval_freq:,} environment transitions"
    )
    print(
        f"Parallel environments: {args.num_envs}"
    )
    print(
        f"Gradient steps per rollout: "
        f"{args.gradient_steps}"
    )
    print("=" * 60 + "\n")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
    )

    print(
        f"\nTraining finished. "
        f"Best model saved in "
        f"./models/{model_name}"
    )

    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------
    model_dir = (
        f"./models/{model_name}_final"
    )

    os.makedirs(
        model_dir,
        exist_ok=True,
    )

    metadata["paths"] = {
        "final_model_dir": model_dir,
        "best_model_dir": f"./models/{model_name}",
        "tensorboard_dir": f"./tb_logs/{model_name}",
        "evaluation_log_dir": f"./logs/{model_name}",
    }

    save_experiment_metadata(
        os.path.join(
            model_dir,
            "experiment_metadata.json",
        ),
        metadata,
    )

    save_experiment_metadata(
        os.path.join(
            f"./models/{model_name}",
            "experiment_metadata.json",
        ),
        metadata,
    )

    model.save(
        os.path.join(
            model_dir,
            "model.zip",
        )
    )

    env.save(
        os.path.join(
            model_dir,
            "vecnormalize.pkl",
        )
    )

    print(
        f"[Model] Final model saved to "
        f"{model_dir}"
    )

    # --------------------------------------------------------
    # --------------------------------------------------------
    # W&B ARTIFACT
    # --------------------------------------------------------
    if wandb_run is not None:
        model_artifact = wandb.Artifact(
            f"{model_name}_model",
            type="model",
            metadata={
                "git_commit": git_hash,
                "seed": args.seed,
                "algorithm": args.algorithm,
                "env_type": args.env_type,
                "hybrid": args.hybrid,
            },
        )

        for artifact_file in (
            "model.zip",
            "vecnormalize.pkl",
            "experiment_metadata.json",
        ):
            model_artifact.add_file(
                os.path.join(
                    model_dir,
                    artifact_file,
                )
            )

        wandb_run.log_artifact(
            model_artifact
        )

        print(
            "[W&B] Model, VecNormalize, and metadata "
            "artifacts uploaded."
        )

        wandb_run.summary[
            "final_model_path"
        ] = model_dir

        wandb_run.summary[
            "total_timesteps"
        ] = args.total_timesteps

    else:
        print(
            "[W&B] Disabled; no artifact uploaded."
        )

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------
    eval_env.close()
    env.close()

    if wandb_run is not None:
        wandb_run.finish()

        print(
            "\nW&B run finished. "
            f"View results at: "
            f"{wandb_run.get_url()}"
        )
    else:
        print(
            "\nTraining run finished. "
            "W&B was disabled; local metadata was saved."
        )


if __name__ == "__main__":
    main()