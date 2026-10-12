#!/usr/bin/env python3
"""
Motion-audit evaluator for the thesis experiments.

Purpose
-------
Validate physical-motion measurements independently of the reward audit.

All transition-level robot state used after env.step() is read from
the environment's transition info dictionary:

    transition_linear_velocity
    transition_angular_velocity
    transition_robot_x
    transition_robot_y
    transition_robot_yaw

This avoids reading the underlying environment after SB3/DummyVecEnv
has automatically reset a terminated episode.

Metrics
-------
Primary evaluation:
- success rate
- collision rate
- timeout rate
- out-of-bounds rate
- episode reward
- episode steps
- simulation/control duration
- time to goal
- path length
- path efficiency
- final distance to goal
- minimum LiDAR clearance

Motion diagnostics:
- mean commanded linear velocity
- mean absolute commanded angular velocity
- mean realized linear velocity
- mean absolute realized angular velocity
- commanded-realized linear velocity error
- low-motion fraction
- low-motion/stuck events
- net displacement
- motion efficiency
- angular-velocity smoothness

This file deliberately does NOT require reward-audit instrumentation.
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
    "cnn-gru-sac",
    "lstm-sac",
    "tx-sac",
}

PHYSICS_STEPS_PER_ACTION = 8
CONTROL_DT = PHYSICS_STEPS_PER_ACTION * TIME_STEP

# Diagnostic only. 0.01 m/s is used as the near-zero-motion threshold.
LOW_MOTION_THRESHOLD = 0.01

# 50 control steps at the current control period.
LOW_MOTION_EVENT_STEPS = 50


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
    parser = argparse.ArgumentParser(
        description="Evaluate a trained policy with transition-level motion diagnostics."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model .zip file or model directory.",
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
            "cnn-gru-sac",
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
        default="motion_audit_results.csv",
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
        help="Optional explicit path to vecnormalize.pkl.",
    )

    return parser.parse_args()


def resolve_model_path(model_path):
    path = Path(model_path)

    if path.exists():
        if path.is_dir():
            candidate = path / "model.zip"
            if candidate.exists():
                return candidate

            zips = sorted(path.glob("*.zip"))
            if len(zips) == 1:
                return zips[0]

            raise FileNotFoundError(
                f"Model directory does not contain a unique model zip: {path}"
            )

        return path

    if path.suffix != ".zip":
        zip_path = path.with_suffix(".zip")
        if zip_path.exists():
            return zip_path

    raise FileNotFoundError(f"Model file not found: {path}")


def find_vecnormalize_path(model_path, explicit_path=None):
    if explicit_path is not None:
        path = Path(explicit_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Specified VecNormalize file does not exist: {path}"
            )

        return path

    model_path = Path(model_path)
    candidates = [
        model_path.parent / "vecnormalize.pkl",
    ]

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


def get_transition_value(info, key, default=None):
    """Read a scalar transition value from SB3 info."""
    if not isinstance(info, dict):
        return default

    value = info.get(key, default)

    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main():
    args = parse_args()

    if args.num_episodes < 1:
        raise ValueError("--num_episodes must be >= 1")

    set_seed(args.seed)

    print("\n" + "=" * 68)
    print(f"Motion audit: {args.algorithm}")
    print(f"Environment: {args.env_type}")
    print(f"Hybrid mode: {args.hybrid}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Seed: {args.seed}")
    print("=" * 68 + "\n")

    model_path = resolve_model_path(args.model_path)
    print(f"[MotionAudit] Using model: {model_path}")

    base_env = make_base_env(
        hybrid=args.hybrid,
        env_type=args.env_type,
        gui=args.gui,
    )

    if args.algorithm in TEMPORAL_ALGORITHMS:
        base_env = TemporalObservationWrapper(base_env)
        print("[MotionAudit] TemporalObservationWrapper applied (history=8)")

    env = DummyVecEnv([lambda: base_env])

    norm_path = find_vecnormalize_path(
        model_path,
        args.vecnormalize_path,
    )

    if norm_path is None:
        raise FileNotFoundError(
            "\n[MotionAudit] VecNormalize statistics were not found.\n"
            "Provide them explicitly with:\n"
            "  --vecnormalize_path /path/to/vecnormalize.pkl\n"
        )

    print(f"[MotionAudit] Loading VecNormalize: {norm_path}")

    env = VecNormalize.load(
        str(norm_path),
        env,
    )

    env.training = False
    env.norm_reward = False

    print(
        "[MotionAudit] VecNormalize configured: "
        "training=False, norm_reward=False"
    )

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

    print("[MotionAudit] Model loaded successfully.")
    print(f"[MotionAudit] Control timestep: {CONTROL_DT:.6f} s")
    print(
        "[MotionAudit] Transition state source: "
        "info['transition_*']"
    )

    metrics = []

    for ep in range(args.num_episodes):
        obs = env.reset()
        obs = np.asarray(obs, dtype=np.float32)

        original_env = unwrap_original_env(env)

        # Reset-state queries are safe here because this is intentionally
        # before the first transition of the episode.
        start_pose = original_env.get_robot_pose()
        start_pos = np.array(
            start_pose[:2],
            dtype=np.float32,
        )

        goal_pos = np.array(
            original_env.goal_pos[:2],
            dtype=np.float32,
        )

        straight_line_dist = float(
            np.linalg.norm(goal_pos - start_pos)
        )

        done = False
        steps = 0
        episode_reward = 0.0
        path_length = 0.0

        prev_pos = start_pos.copy()
        final_transition_pos = start_pos.copy()

        collision = False
        dynamic_collision = False
        success = False
        out_of_bounds = False
        goal_reached_info = False

        commanded_linear_velocities = []
        commanded_angular_velocities = []

        realized_linear_velocities = []
        realized_angular_velocities = []

        min_distances = []

        low_motion_steps = 0
        low_motion_counter = 0
        low_motion_events = 0

        episode_start_time = time.time()

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

            commanded_linear = float(
                action[0][0] * LINEAR_VEL)
            commanded_angular = float(
                action[0][1] * ANGULAR_VEL
            )

            commanded_linear_velocities.append(
                commanded_linear
            )
            commanded_angular_velocities.append(
                commanded_angular
            )

            obs, reward, dones, infos = env.step(action)

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

            # ----------------------------------------------------------
            # Critical motion-audit rule:
            # Never query original_env.get_robot_pose() or
            # original_env.get_robot_velocity() here.
            #
            # DummyVecEnv may already have reset the environment after
            # a terminal transition. The instrumented environment
            # captures the actual transition state before returning.
            # ----------------------------------------------------------
            transition_x = get_transition_value(
                info,
                "transition_robot_x",
            )
            transition_y = get_transition_value(
                info,
                "transition_robot_y",
            )
            transition_linear = get_transition_value(
                info,
                "transition_linear_velocity",
            )
            transition_angular = get_transition_value(
                info,
                "transition_angular_velocity",
            )

            if (
                transition_x is None
                or transition_y is None
                or transition_linear is None
                or transition_angular is None
            ):
                raise RuntimeError(
                    "Motion-audit transition fields are missing. "
                    "Run this evaluator only with the instrumented "
                    "motion-audit environment files."
                )

            curr_pos = np.array(
                [transition_x, transition_y],
                dtype=np.float32,
            )

            final_transition_pos = curr_pos.copy()

            step_dist = float(
                np.linalg.norm(curr_pos - prev_pos)
            )

            path_length += step_dist
            prev_pos = curr_pos

            realized_linear_velocities.append(
                transition_linear
            )
            realized_angular_velocities.append(
                transition_angular
            )

            if (
                abs(transition_linear)
                < LOW_MOTION_THRESHOLD
            ):
                low_motion_steps += 1
                low_motion_counter += 1
            else:
                low_motion_counter = 0

            if low_motion_counter >= LOW_MOTION_EVENT_STEPS:
                low_motion_events += 1
                low_motion_counter = 0

            # Use the transition-captured LiDAR minimum if available.
            # The motion-audit environment does not require reward-audit
            # instrumentation, so capture LiDAR through the info field
            # when present; otherwise use the post-step sensor only on
            # non-terminal transitions.
            transition_min_lidar = get_transition_value(
                info,
                "transition_min_lidar",
            )

            if transition_min_lidar is not None:
                min_distances.append(
                    transition_min_lidar
                )
            elif not done:
                raw_lidar = np.asarray(
                    original_env.get_lidar(),
                    dtype=np.float32,
                )
                raw_lidar = np.clip(
                    raw_lidar,
                    0.0,
                    1.0,
                )
                min_distances.append(
                    float(np.min(raw_lidar) * LIDAR_RANGE)
                )

            if isinstance(info, dict):
                if info.get("collision", False):
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

                if info.get("reached_goal", False):
                    goal_reached_info = True

                if info.get("out_of_bounds", False):
                    out_of_bounds = True

        episode_wall_time = (
            time.time() - episode_start_time
        )

        final_pos = final_transition_pos

        final_distance_to_goal = float(
            np.linalg.norm(final_pos - goal_pos)
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
        elif out_of_bounds:
            termination_reason = "out_of_bounds"
        else:
            termination_reason = "timeout"

        simulation_duration = steps * CONTROL_DT

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

        if path_length > 1e-8:
            path_efficiency = (
                straight_line_dist / path_length
            )
        else:
            path_efficiency = 0.0

        net_displacement = float(
            np.linalg.norm(
                final_pos - start_pos
            )
        )

        if path_length > 1e-8:
            motion_efficiency = (
                net_displacement / path_length
            )
        else:
            motion_efficiency = 0.0

        mean_commanded_linear = float(
            np.mean(commanded_linear_velocities)
        )

        mean_abs_commanded_angular = float(
            np.mean(
                np.abs(commanded_angular_velocities)
            )
        )

        mean_realized_linear = float(
            np.mean(realized_linear_velocities)
        )

        mean_abs_realized_angular = float(
            np.mean(
                np.abs(realized_angular_velocities)
            )
        )

        mean_abs_linear_tracking_error = float(
            np.mean(
                np.abs(
                    np.asarray(
                        commanded_linear_velocities
                    )
                    - np.asarray(
                        realized_linear_velocities
                    )
                )
            )
        )

        low_motion_fraction = (
            low_motion_steps / steps
            if steps > 0
            else 0.0
        )

        metrics.append(
            {
                "episode": ep,
                "success": success,
                "termination_reason": termination_reason,
                "collision": collision,
                "dynamic_collision": dynamic_collision,
                "out_of_bounds": out_of_bounds,
                "steps": steps,
                "episode_reward": episode_reward,
                "simulation_duration_s": simulation_duration,
                "episode_wall_time_s": episode_wall_time,
                "time_to_goal_s": time_to_goal_s,
                "path_length": path_length,
                "path_efficiency": path_efficiency,
                "final_distance_to_goal": final_distance_to_goal,
                "min_clearance": min_clearance,
                "smoothness_jerk": angular_jerk,
                "stuck_events": low_motion_events,
                "low_motion_fraction": low_motion_fraction,
                "mean_commanded_linear_velocity": mean_commanded_linear,
                "mean_abs_commanded_angular_velocity": mean_abs_commanded_angular,
                "mean_realized_linear_velocity": mean_realized_linear,
                "mean_abs_realized_angular_velocity": mean_abs_realized_angular,
                "mean_abs_linear_velocity_error": mean_abs_linear_tracking_error,
                "net_displacement": net_displacement,
                "motion_efficiency": motion_efficiency,
            }
        )

        print(
            f"Episode {ep:3d}: "
            f"result={termination_reason:12s}, "
            f"steps={steps:4d}, "
            f"reward={episode_reward:8.3f}, "
            f"final_dist={final_distance_to_goal:.3f} m, "
            f"path={path_length:.3f} m, "
            f"cmd_v={mean_commanded_linear:.3f}, "
            f"real_v={mean_realized_linear:.3f}, "
            f"low_motion={low_motion_fraction:.3f}, "
            f"stuck={low_motion_events}"
        )

    output_path = Path(args.output)
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
        np.mean([m["success"] for m in metrics])
        * 100.0
    )

    collision_rate = (
        np.mean([m["collision"] for m in metrics])
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

    out_of_bounds_rate = (
        np.mean([m["out_of_bounds"] for m in metrics])
        * 100.0
    )

    dynamic_collision_rate = (
        np.mean(
            [m["dynamic_collision"] for m in metrics]
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

    successful_efficiencies = [
        m["path_efficiency"]
        for m in metrics
        if m["success"]
    ]

    if successful_efficiencies:
        avg_efficiency = float(
            np.mean(successful_efficiencies)
        )
    else:
        avg_efficiency = float("nan")

    avg_final_distance = np.mean(
        [m["final_distance_to_goal"] for m in metrics]
    )

    avg_clearance = np.mean(
        [m["min_clearance"] for m in metrics]
    )

    avg_stuck = np.mean(
        [m["stuck_events"] for m in metrics]
    )

    avg_low_motion_fraction = np.mean(
        [m["low_motion_fraction"] for m in metrics]
    )

    avg_cmd_v = np.mean(
        [
            m["mean_commanded_linear_velocity"]
            for m in metrics
        ]
    )

    avg_real_v = np.mean(
        [
            m["mean_realized_linear_velocity"]
            for m in metrics
        ]
    )

    avg_linear_error = np.mean(
        [
            m["mean_abs_linear_velocity_error"]
            for m in metrics
        ]
    )

    avg_net_displacement = np.mean(
        [m["net_displacement"] for m in metrics]
    )

    avg_motion_efficiency = np.mean(
        [m["motion_efficiency"] for m in metrics]
    )

    successful_times = [
        m["time_to_goal_s"]
        for m in metrics
        if m["success"]
        and math.isfinite(m["time_to_goal_s"])
    ]

    if successful_times:
        avg_time_to_goal = float(
            np.mean(successful_times)
        )
    else:
        avg_time_to_goal = float("nan")

    print("\n" + "=" * 68)
    print("MOTION AUDIT SUMMARY")
    print("=" * 68)

    print(f"Success rate:                  {success_rate:.1f}%")
    print(f"Collision rate:                {collision_rate:.1f}%")
    print(f"Timeout rate:                  {timeout_rate:.1f}%")
    print(f"Out-of-bounds rate:            {out_of_bounds_rate:.1f}%")
    print(
        f"Dynamic-collision diagnostic:  "
        f"{dynamic_collision_rate:.1f}%"
    )
    print(f"Average episode reward:        {avg_reward:.3f}")
    print(f"Average steps:                 {avg_steps:.1f}")
    print(f"Average path length:           {avg_path_length:.3f} m")

    if math.isfinite(avg_efficiency):
        print(
            f"Average successful path efficiency: "
            f"{avg_efficiency:.3f}"
        )
    else:
        print(
            "Average successful path efficiency: N/A"
        )

    print(
        f"Average final goal distance:   "
        f"{avg_final_distance:.3f} m"
    )

    print(
        f"Average minimum clearance:     "
        f"{avg_clearance:.3f} m"
    )

    if math.isfinite(avg_time_to_goal):
        print(
            f"Average time to goal:          "
            f"{avg_time_to_goal:.3f} s"
        )
    else:
        print("Average time to goal:          N/A")

    print(
        f"Average commanded linear v:    "
        f"{avg_cmd_v:.4f} m/s"
    )

    print(
        f"Average realized linear v:     "
        f"{avg_real_v:.4f} m/s"
    )

    print(
        f"Mean |commanded-realized v|:   "
        f"{avg_linear_error:.4f} m/s"
    )

    print(
        f"Average low-motion fraction:   "
        f"{avg_low_motion_fraction:.3f}"
    )

    print(
        f"Average low-motion events:     "
        f"{avg_stuck:.2f}"
    )

    print(
        f"Average net displacement:      "
        f"{avg_net_displacement:.3f} m"
    )

    print(
        f"Average motion efficiency:     "
        f"{avg_motion_efficiency:.3f}"
    )

    print(f"Results saved to:              {args.output}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
