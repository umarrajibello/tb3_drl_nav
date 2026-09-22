#!/usr/bin/env python3

"""
Diagnostic script for inspecting the actions produced by a trained policy.

This script does NOT modify the training code, environment, reward, or model.

It loads a trained model and reports:
- normalized linear actions
- normalized angular actions
- physical linear velocity
- physical angular velocity
- action statistics
- near-zero action frequency
- robot displacement
- total displacement
- final distance to goal
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import ANGULAR_VEL, LINEAR_VEL
from env.observation_wrapper import TemporalObservationWrapper
from env.tb3_env import TB3Env
from env.tb3_env_hybrid import TB3HybridEnv


TEMPORAL_ALGORITHMS = {
    "cnn-gru-sac",
    "lstm-sac",
    "tx-sac",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--env_type",
        type=str,
        default="static",
        choices=["empty", "static", "mixed", "dense"],
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=[
            "sac",
            "ppo",
            "cnn-gru-sac",
            "lstm-sac",
            "tx-sac",
        ],
    )

    parser.add_argument(
        "--num_episodes",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--hybrid",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--vecnormalize_path",
        type=str,
        default=None,
    )

    return parser.parse_args()


def find_vecnormalize_path(model_path, explicit_path=None):
    if explicit_path is not None:
        path = Path(explicit_path)

        if not path.exists():
            raise FileNotFoundError(
                f"VecNormalize file not found: {path}"
            )

        return path

    model_path = Path(model_path)

    candidates = [
        model_path.parent / "vecnormalize.pkl",
        (
            model_path.parent.parent
            / f"{model_path.parent.name}_final"
            / "vecnormalize.pkl"
        ),
    ]

    if model_path.is_dir():
        candidates.append(
            model_path / "vecnormalize.pkl"
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find VecNormalize statistics. "
        "Use --vecnormalize_path explicitly."
    )


def make_env(args):
    render_mode = None

    if args.hybrid:
        env = TB3HybridEnv(
            env_type=args.env_type,
            render_mode=render_mode,
        )
    else:
        env = TB3Env(
            env_type=args.env_type,
            render_mode=render_mode,
        )

    if args.algorithm in TEMPORAL_ALGORITHMS:
        env = TemporalObservationWrapper(env)

    return env


def unwrap_env(env):
    current = env

    while hasattr(current, "venv"):
        current = current.venv

    if isinstance(current, DummyVecEnv):
        current = current.envs[0]

    while hasattr(current, "env"):
        current = current.env

    return current


def main():
    args = parse_args()

    set_seed(args.seed)

    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    norm_path = find_vecnormalize_path(
        model_path,
        args.vecnormalize_path,
    )

    print("=" * 70)
    print("POLICY ACTION DIAGNOSTIC")
    print("=" * 70)
    print(f"Model:        {model_path}")
    print(f"Environment:  {args.env_type}")
    print(f"Algorithm:    {args.algorithm}")
    print(f"Hybrid:       {args.hybrid}")
    print(f"Episodes:     {args.num_episodes}")
    print(f"VecNormalize: {norm_path}")
    print("=" * 70)

    base_env = make_env(args)

    env = DummyVecEnv(
        [lambda: base_env]
    )

    env = VecNormalize.load(
        str(norm_path),
        env,
    )

    env.training = False
    env.norm_reward = False

    if args.algorithm in {
        "sac",
        "cnn-gru-sac",
        "lstm-sac",
        "tx-sac",
    }:
        model = SAC.load(
            str(model_path),
            env=env,
        )
    else:
        model = PPO.load(
            str(model_path),
            env=env,
        )

    original_env = unwrap_env(env)

    all_linear_actions = []
    all_angular_actions = []
    all_linear_velocities = []
    all_angular_velocities = []
    all_step_displacements = []

    episode_summaries = []

    for episode in range(args.num_episodes):
        obs = env.reset()

        original_env = unwrap_env(env)

        start_pos = np.array(
            original_env.get_robot_pose()[:2],
            dtype=np.float32,
        )

        goal_pos = np.array(
            original_env.goal_pos[:2],
            dtype=np.float32,
        )

        previous_pos = start_pos.copy()

        done = False
        steps = 0
        episode_linear = []
        episode_angular = []
        episode_displacement = []

        while not done:
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            action = np.asarray(
                action,
                dtype=np.float32,
            )

            if action.ndim == 1:
                action = action.reshape(1, -1)

            linear_action = float(action[0][0])
            angular_action = float(action[0][1])

            linear_velocity = (
                linear_action * LINEAR_VEL
            )

            angular_velocity = (
                angular_action * ANGULAR_VEL
            )

            all_linear_actions.append(
                linear_action
            )
            all_angular_actions.append(
                angular_action
            )
            all_linear_velocities.append(
                linear_velocity
            )
            all_angular_velocities.append(
                angular_velocity
            )

            episode_linear.append(
                linear_action
            )
            episode_angular.append(
                angular_action
            )

            obs, rewards, dones, infos = env.step(
                action
            )

            done = bool(
                np.asarray(dones).reshape(-1)[0]
            )

            current_pos = np.array(
                original_env.get_robot_pose()[:2],
                dtype=np.float32,
            )

            displacement = float(
                np.linalg.norm(
                    current_pos - previous_pos
                )
            )

            all_step_displacements.append(
                displacement
            )
            episode_displacement.append(
                displacement
            )

            previous_pos = current_pos
            steps += 1

        final_pos = np.array(
            original_env.get_robot_pose()[:2],
            dtype=np.float32,
        )

        final_distance = float(
            np.linalg.norm(
                final_pos - goal_pos
            )
        )

        episode_summaries.append(
            {
                "episode": episode,
                "steps": steps,
                "total_displacement": float(
                    np.sum(episode_displacement)
                ),
                "final_distance": final_distance,
                "linear_mean": float(
                    np.mean(episode_linear)
                ),
                "linear_std": float(
                    np.std(episode_linear)
                ),
                "angular_mean": float(
                    np.mean(episode_angular)
                ),
                "angular_std": float(
                    np.std(episode_angular)
                ),
            }
        )

    linear_actions = np.asarray(
        all_linear_actions
    )
    angular_actions = np.asarray(
        all_angular_actions
    )
    linear_velocities = np.asarray(
        all_linear_velocities
    )
    angular_velocities = np.asarray(
        all_angular_velocities
    )
    displacements = np.asarray(
        all_step_displacements
    )

    near_zero_linear = (
        np.abs(linear_actions) < 0.05
    )

    near_zero_angular = (
        np.abs(angular_actions) < 0.05
    )

    print("\n" + "=" * 70)
    print("ACTION STATISTICS")
    print("=" * 70)

    print("\nNormalized linear action [-1, 1]:")
    print(
        f"  mean:       {np.mean(linear_actions): .6f}"
    )
    print(
        f"  std:        {np.std(linear_actions): .6f}"
    )
    print(
        f"  min:        {np.min(linear_actions): .6f}"
    )
    print(
        f"  max:        {np.max(linear_actions): .6f}"
    )

    print("\nNormalized angular action [-1, 1]:")
    print(
        f"  mean:       {np.mean(angular_actions): .6f}"
    )
    print(
        f"  std:        {np.std(angular_actions): .6f}"
    )
    print(
        f"  min:        {np.min(angular_actions): .6f}"
    )
    print(
        f"  max:        {np.max(angular_actions): .6f}"
    )

    print("\nPhysical linear velocity [m/s]:")
    print(
        f"  mean:       {np.mean(linear_velocities): .6f}"
    )
    print(
        f"  std:        {np.std(linear_velocities): .6f}"
    )
    print(
        f"  min:        {np.min(linear_velocities): .6f}"
    )
    print(
        f"  max:        {np.max(linear_velocities): .6f}"
    )

    print("\nPhysical angular velocity [rad/s]:")
    print(
        f"  mean:       {np.mean(angular_velocities): .6f}"
    )
    print(
        f"  std:        {np.std(angular_velocities): .6f}"
    )
    print(
        f"  min:        {np.min(angular_velocities): .6f}"
    )
    print(
        f"  max:        {np.max(angular_velocities): .6f}"
    )

    print("\nNear-zero actions:")
    print(
        f"  linear  |a| < 0.05: "
        f"{np.mean(near_zero_linear) * 100:.2f}%"
    )
    print(
        f"  angular |a| < 0.05: "
        f"{np.mean(near_zero_angular) * 100:.2f}%"
    )

    print("\nRobot displacement per control step:")
    print(
        f"  mean:       {np.mean(displacements): .6f} m"
    )
    print(
        f"  std:        {np.std(displacements): .6f} m"
    )
    print(
        f"  min:        {np.min(displacements): .6f} m"
    )
    print(
        f"  max:        {np.max(displacements): .6f} m"
    )

    print("\n" + "=" * 70)
    print("EPISODE SUMMARY")
    print("=" * 70)

    for summary in episode_summaries:
        print(
            f"Episode {summary['episode']:2d}: "
            f"steps={summary['steps']:4d}, "
            f"displacement={summary['total_displacement']:.4f} m, "
            f"final_dist={summary['final_distance']:.4f} m, "
            f"linear_mean={summary['linear_mean']:.4f}, "
            f"linear_std={summary['linear_std']:.4f}, "
            f"angular_mean={summary['angular_mean']:.4f}, "
            f"angular_std={summary['angular_std']:.4f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
