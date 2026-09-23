#!/usr/bin/env python3
"""
Evaluation entry point for the thesis experiments.

Evaluation metrics:
- success rate
- collision rate
- timeout rate
- dynamic-obstacle collision diagnostic
- episode reward
- episode steps
- simulation/control duration
- time to goal for successful episodes
- path length
- path efficiency
- final distance to goal
- minimum LiDAR clearance
- angular-velocity smoothness
- stuck events
"""

import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import (
    ANGULAR_VEL,
    COLLISION_DIST,
    GOAL_THRESHOLD,
    LIDAR_RANGE,
    TIME_STEP,
    LINEAR_VEL,
)
from env.observation_wrapper import TemporalObservationWrapper
from env.tb3_env import TB3Env
from env.tb3_env_hybrid import TB3HybridEnv


TEMPORAL_ALGORITHMS = {
    "gru-sac",
    "lstm-sac",
    "tx-sac",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_base_env(hybrid: bool, env_type: str, gui: bool):
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


def unwrap_original_env(env):
    current = env

    while hasattr(current, "venv"):
        current = current.venv

    if isinstance(current, DummyVecEnv):
        current = current.envs[0]

    while hasattr(current, "env"):
        current = current.env

    return current


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model .zip file.",
    )

    parser.add_argument(
        "--env_type",
        type=str,
        default="mixed",
        choices=["empty", "static", "mixed", "dense"],
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=[
            "sac",
            "ppo",
            "gru-sac",
            "lstm-sac",
            "tx-sac",
        ],
    )

    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Evaluate the RRT-guided hybrid environment.",
    )

    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--gui",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.csv",
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
        help=(
            "Optional explicit path to vecnormalize.pkl. "
            "If omitted, the evaluator searches automatically."
        ),
    )

    return parser.parse_args()


def resolve_model_path(model_path):
    """
    Resolve a Stable-Baselines3 model path.

    Accepted inputs:
      1. Direct .zip model path
      2. Directory containing model.zip
      3. Directory containing exactly one .zip model
    """
    path = Path(model_path)

    # Case 1: explicitly supplied .zip file
    if path.is_file():
        return path

    # Case 2: supplied model directory
    if path.is_dir():
        model_zip = path / "model.zip"

        if model_zip.is_file():
            return model_zip

        # Fallback: look for exactly one .zip file
        zip_files = sorted(path.glob("*.zip"))

        if len(zip_files) == 1:
            return zip_files[0]

        if len(zip_files) > 1:
            raise FileNotFoundError(
                f"Multiple .zip model files found in directory: {path}\n"
                f"Please specify the exact model .zip file."
            )

        raise FileNotFoundError(
            f"No model .zip file found in directory: {path}"
        )

    # Case 3: path does not exist
    raise FileNotFoundError(
        f"Model path does not exist: {path}"
    )



def find_vecnormalize_path(model_path, explicit_path=None):
    if explicit_path is not None:
        path = Path(explicit_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Specified VecNormalize file does not exist: {path}"
            )

        return path

    model_path = Path(model_path)
    candidates = []

    candidates.append(
        model_path.parent / "vecnormalize.pkl"
    )

    parent_name = model_path.parent.name

    candidates.append(
        model_path.parent.parent
        / f"{parent_name}_final"
        / "vecnormalize.pkl"
    )

    if model_path.is_dir():
        candidates.append(
            model_path / "vecnormalize.pkl"
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def main():
    args = parse_args()

    if args.num_episodes < 1:
        raise ValueError(
            "--num_episodes must be >= 1"
        )

    set_seed(args.seed)

    print("\n" + "=" * 60)
    print(
        f"Evaluating {args.algorithm} "
        f"on {args.env_type} environment"
    )
    print(f"Hybrid mode: {args.hybrid}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Seed: {args.seed}")
    print("=" * 60 + "\n")

    model_path = resolve_model_path(
        args.model_path
    )

    print(
        f"[Eval] Using model: {model_path}"
    )

    base_env = make_base_env(
        hybrid=args.hybrid,
        env_type=args.env_type,
        gui=args.gui,
    )

    if args.algorithm in TEMPORAL_ALGORITHMS:
        base_env = TemporalObservationWrapper(
            base_env
        )

        print(
            "[Eval] TemporalObservationWrapper applied "
            "(history=8)"
        )

    env = DummyVecEnv(
        [lambda: base_env]
    )

    norm_path = find_vecnormalize_path(
        model_path,
        args.vecnormalize_path,
    )

    if norm_path is None:
        raise FileNotFoundError(
            "\n[Eval] VecNormalize statistics were not found.\n"
            "This model was trained with VecNormalize, so evaluation "
            "should not continue without the training observation "
            "statistics.\n"
            "\nProvide the file explicitly with:\n"
            "  --vecnormalize_path /path/to/vecnormalize.pkl\n"
        )

    print(
        "[Eval] Loading VecNormalize statistics from:\n"
        f"       {norm_path}"
    )

    env = VecNormalize.load(
        str(norm_path),
        env,
    )

    env.training = False
    env.norm_reward = False

    print(
        "[Eval] VecNormalize configured:"
        "\n       training=False"
        "\n       norm_reward=False"
    )

    if args.algorithm in {
        "sac",
        "gru-sac",
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

    print(
        "[Eval] Model loaded successfully.\n"
    )

    metrics = []

    physics_steps_per_action = 8
    control_dt = (
        physics_steps_per_action * TIME_STEP
    )

    print(
        f"[Eval] Control timestep: "
        f"{control_dt:.6f} s"
    )

    for ep in range(args.num_episodes):
        obs = env.reset()

        obs = np.asarray(
            obs,
            dtype=np.float32,
        )

        original_env = unwrap_original_env(
            env
        )

        start_pos = np.array(
            original_env.get_robot_pose()[:2],
            dtype=np.float32,
        )

        goal_pos = np.array(
            original_env.goal_pos[:2],
            dtype=np.float32,
        )

        straight_line_dist = float(
            np.linalg.norm(
                goal_pos - start_pos
            )
        )

        done = False
        steps = 0
        episode_reward = 0.0
        path_length = 0.0

        prev_pos = start_pos.copy()

        collision = False
        dynamic_collision = False
        success = False

        commanded_linear_velocities = []
        commanded_angular_velocities = []

        realized_linear_velocities = []
        realized_angular_velocities = []

        min_distances = []

        stuck_counter = 0
        stuck_events = 0

        goal_reached_info = False

        episode_start_time = time.time()

        while not done:
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            normalized_action = np.asarray(action).reshape(-1)

            if ep == 0 and steps < 20:
                print(
                    f"[Action diagnostic] step={steps:3d}, "
                    f"a_v={normalized_action[0]: .4f}, "
                    f"a_w={normalized_action[1]: .4f}"
                )

            action = np.asarray(
                action,
                dtype=np.float32,
            )

            if action.ndim == 1:
                action = action.reshape(
                    1,
                    -1,
                )



            linear_vel = float(
                action[0][0] * LINEAR_VEL
            )

            ang_vel = float(
                action[0][1] * ANGULAR_VEL
            )

            real_linear_vel, real_angular_vel = (
                original_env.get_robot_velocity()
            )

            commanded_linear_velocities.append(
                linear_vel
            )

            commanded_angular_velocities.append(
                ang_vel
            )

            realized_linear_velocities.append(
                float(real_linear_vel)
            )

            realized_angular_velocities.append(
                float(real_angular_vel)
            )

            

            obs, reward, dones, infos = env.step(
                action
            )

            obs = np.asarray(
                obs,
                dtype=np.float32,
            )

            done = bool(
                np.asarray(dones).reshape(-1)[0]
            )

            info = infos[0]

            reward_value = float(
                np.asarray(reward).reshape(-1)[0]
            )

            episode_reward += reward_value
            steps += 1

            curr_pos = np.array(
                original_env.get_robot_pose()[:2],
                dtype=np.float32,
            )

            step_dist = float(
                np.linalg.norm(
                    curr_pos - prev_pos
                )
            )

            path_length += step_dist
            prev_pos = curr_pos

            if step_dist < 0.01:
                stuck_counter += 1
            else:
                stuck_counter = 0

            if stuck_counter >= 50:
                stuck_events += 1
                stuck_counter = 0

            raw_lidar = np.asarray(
                original_env.get_lidar(),
                dtype=np.float32,
            )

            raw_lidar = np.clip(
                raw_lidar,
                0.0,
                1.0,
            )

            min_dist = float(
                np.min(raw_lidar)
                * LIDAR_RANGE
            )

            min_distances.append(
                min_dist
            )

            if isinstance(info, dict):
                if info.get(
                    "collision",
                    False,
                ):
                    collision = True

                    try:
                        dyn_dists = (
                            original_env
                            .get_dynamic_obstacle_distances()
                        )

                        if dyn_dists:
                            if (
                                min(dyn_dists)
                                < COLLISION_DIST
                            ):
                                dynamic_collision = True

                    except AttributeError:
                        pass

                if info.get(
                    "reached_goal",
                    False,
                ):
                    goal_reached_info = True

        episode_wall_time = (
            time.time()
            - episode_start_time
        )

        final_pos = np.array(
            original_env.get_robot_pose()[:2],
            dtype=np.float32,
        )

        final_distance_to_goal = float(
            np.linalg.norm(
                final_pos - goal_pos
            )
        )

        if not collision:
            if (
                goal_reached_info
                or final_distance_to_goal < GOAL_THRESHOLD
            ):
                success = True

        if success:
            termination_reason = "success"
        elif collision:
            termination_reason = "collision"
        else:
            termination_reason = "timeout"

        simulation_duration = (
            steps * control_dt
        )

        if success:
            time_to_goal_s = simulation_duration
        else:
            time_to_goal_s = float("nan")

        if len(commanded_angular_velocities) > 1:
            angular_jerk = float(
                np.mean(
                    np.abs(
                        np.diff(
                            commanded_angular_velocities
                        )   
                        
                    )
                )
            )
        else:
            angular_jerk = 0.0

        if min_distances:
            min_clearance = float(
                np.min(min_distances)
            )
        else:
            min_clearance = float("nan")


        mean_commanded_linear_velocity = float(
            np.mean(commanded_linear_velocities)
        ) if commanded_linear_velocities else float("nan")

        mean_abs_commanded_angular_velocity = float(
            np.mean(
                np.abs(commanded_angular_velocities)
            )
        ) if commanded_angular_velocities else float("nan")

        mean_realized_linear_velocity = float(
            np.mean(realized_linear_velocities)
        ) if realized_linear_velocities else float("nan")

        mean_abs_realized_angular_velocity = float(
            np.mean(
                np.abs(realized_angular_velocities)
            )
        ) if realized_angular_velocities else float("nan")

        low_motion_fraction = float(
            np.mean(
                np.asarray(
                    realized_linear_velocities
                ) < 0.01
            )
        ) if realized_linear_velocities else float("nan")

        net_displacement = float(
            np.linalg.norm(
                final_pos - start_pos
            )
        )

        motion_efficiency = (
            net_displacement / path_length
            if path_length > 1e-8
            else float("nan")
        )



        

        if success and path_length > 1e-8:
            path_efficiency = (
            straight_line_dist
            / path_length
            )
        else:
            path_efficiency = float("nan")


        metrics.append(
            {
                "episode": ep,
                "success": success,
                "termination_reason": termination_reason,
                "collision": collision,
                "dynamic_collision": dynamic_collision,
                "steps": steps,
                "episode_reward": episode_reward,
                "simulation_duration_s": simulation_duration,
                "episode_wall_time_s": episode_wall_time,
                "time_to_goal_s": time_to_goal_s,
                "path_length": path_length,
                "path_efficiency": path_efficiency,
                "final_distance_to_goal": final_distance_to_goal,
                "smoothness_jerk": angular_jerk,
                "min_clearance": min_clearance,
                "stuck_events": stuck_events,
                "mean_commanded_linear_velocity":
                    mean_commanded_linear_velocity,
                "mean_abs_commanded_angular_velocity":
                    mean_abs_commanded_angular_velocity,
                "mean_realized_linear_velocity":
                    mean_realized_linear_velocity,
                "mean_abs_realized_angular_velocity":
                    mean_abs_realized_angular_velocity,
                "low_motion_fraction":
                    low_motion_fraction,
                "net_displacement":
                    net_displacement,
                "motion_efficiency":
                    motion_efficiency,
            }
        )

        print(
            f"Episode {ep:3d}: "
            f"result={termination_reason:8s}, "
            f"steps={steps:4d}, "
            f"reward={episode_reward:8.3f}, "
            f"final_dist={final_distance_to_goal:.3f} m, "
            f"eff={path_efficiency:.3f}, "
            f"jerk={angular_jerk:.3f}, "
            f"clearance={min_clearance:.3f} m, "
            f"stuck={stuck_events}"
        )

    if metrics:
        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=metrics[0].keys(),
            )

            writer.writeheader()
            writer.writerows(metrics)

    success_rate = (
        np.mean(
            [m["success"] for m in metrics]
        )
        * 100.0
    )

    collision_rate = (
        np.mean(
            [m["collision"] for m in metrics]
        )
        * 100.0
    )

    timeout_rate = (
        np.mean(
            [
                m["termination_reason"] == "timeout"
                for m in metrics
            ]
        )
        * 100.0
    )

    dynamic_collision_rate = (
        np.mean(
            [
                m["dynamic_collision"]
                for m in metrics
            ]
        )
        * 100.0
    )

    avg_steps = np.mean(
        [m["steps"] for m in metrics]
    )

    avg_reward = np.mean(
        [m["episode_reward"] for m in metrics]
    )

    avg_path_length = np.mean(
        [m["path_length"] for m in metrics]
    )

    efficiency_values = [
        m["path_efficiency"]
        for m in metrics
        if np.isfinite(m["path_efficiency"])
    ]

    avg_efficiency = (
        float(np.mean(efficiency_values))
        if efficiency_values
        else float("nan")
    )

    

    avg_final_distance = np.mean(
        [
            m["final_distance_to_goal"]
            for m in metrics
        ]
    )

    avg_clearance = np.mean(
        [m["min_clearance"] for m in metrics]
    )

    avg_stuck = np.mean(
        [m["stuck_events"] for m in metrics]
    )

    successful_times = [
        m["time_to_goal_s"]
        for m in metrics
        if m["success"]
        and math.isfinite(
            m["time_to_goal_s"]
        )
    ]

    if successful_times:
        avg_time_to_goal = float(
            np.mean(successful_times)
        )
    else:
        avg_time_to_goal = float("nan")

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Success rate:                 "
        f"{success_rate:.1f}%"
    )

    print(
        f"Collision rate:               "
        f"{collision_rate:.1f}%"
    )

    print(
        f"Timeout rate:                 "
        f"{timeout_rate:.1f}%"
    )

    print(
        f"Dynamic-collision diagnostic: "
        f"{dynamic_collision_rate:.1f}%"
    )

    print(
        f"Average episode reward:       "
        f"{avg_reward:.3f}"
    )

    print(
        f"Average steps:                "
        f"{avg_steps:.1f}"
    )

    print(
        f"Average path length:          "
        f"{avg_path_length:.3f} m"
    )

    print(
        f"Average path efficiency:      "
        f"{avg_efficiency:.3f}"
    )

    print(
        f"Average final goal distance:  "
        f"{avg_final_distance:.3f} m"
    )

    print(
        f"Average min clearance:        "
        f"{avg_clearance:.3f} m"
    )

    if math.isfinite(avg_time_to_goal):
        print(
            f"Average time to goal:         "
            f"{avg_time_to_goal:.3f} s"
        )
    else:
        print(
            "Average time to goal:         N/A"
        )

    print(
        f"Average stuck events:         "
        f"{avg_stuck:.2f}"
    )

    print(
        f"Results saved to:             "
        f"{args.output}"
    )

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
