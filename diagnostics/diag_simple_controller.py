#!/usr/bin/env python3

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np

from config import (
    ANGULAR_VEL,
    GOAL_THRESHOLD,
    LIDAR_RANGE,
    LINEAR_VEL,
    MAX_STEPS,
)
from env.tb3_env import TB3Env


def wrap_angle(angle):
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def goal_controller(env, lidar):
    """
    Simple reactive goal-seeking controller.

    Returns a normalized [linear, angular] action in [-1, 1].
    LiDAR is expected to be normalized to [0, 1].
    """
    pose = env.get_robot_pose()
    goal = env.goal_pos

    dx = float(goal[0] - pose[0])
    dy = float(goal[1] - pose[1])

    desired_heading = math.atan2(dy, dx)
    heading_error = wrap_angle(desired_heading - float(pose[2]))

    # LiDAR indices:
    # 18 is approximately straight ahead.
    # Higher indices around 24 look to the left.
    # Lower indices around 12 look to the right.
    scan = np.asarray(lidar, dtype=np.float32).reshape(-1)

    front = float(np.min(scan[15:21])) * LIDAR_RANGE
    left_clearance = float(np.mean(scan[21:27]))
    right_clearance = float(np.mean(scan[9:15]))

    # If an obstacle is close ahead, turn toward the more open side.
    if front < 0.45:
        if left_clearance >= right_clearance:
            angular = 0.65
        else:
            angular = -0.65

        # Slow down while turning away from an obstacle.
        linear = 0.12

    else:
        # Proportional steering toward the goal.
        angular = np.clip(
            1.5 * heading_error / ANGULAR_VEL,
            -0.8,
            0.8,
        )

        # Move more slowly when the robot is facing away from the goal.
        if abs(heading_error) > 0.8:
            linear = 0.20
        else:
            linear = 0.65

        # Slow further when an obstacle is approaching.
        if front < 0.75:
            linear = min(linear, 0.25)

    return np.array(
        [
            np.clip(linear, -1.0, 1.0),
            np.clip(angular, -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a simple goal-seeking controller."
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env_type", default="mixed")
    parser.add_argument(
        "--output",
        default="diagnostics/simple_controller_results.csv",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = TB3Env(
        render_mode=None,
        env_type=args.env_type,
        goal_dist_min=1.0,
        goal_dist_max=2.0,
    )

    rows = []

    try:
        for episode in range(args.episodes):
            obs, reset_info = env.reset(seed=args.seed + episode)

            start_pose = env.get_robot_pose()
            start_xy = np.asarray(start_pose[:2], dtype=float)
            goal_xy = np.asarray(env.goal_pos[:2], dtype=float)

            initial_distance = float(
                np.linalg.norm(goal_xy - start_xy)
            )

            previous_xy = start_xy.copy()
            path_length = 0.0
            total_reward = 0.0
            collision = False
            out_of_bounds = False
            success = False
            steps = 0

            for step in range(MAX_STEPS):
                # Use the current environment LiDAR directly.
                lidar = env.get_lidar()
                action = goal_controller(env, lidar)

                obs, reward, terminated, truncated, info = env.step(
                    action
                )

                total_reward += float(reward)
                steps += 1

                # Plain Gymnasium environment: no vectorized auto-reset.
                pose = env.get_robot_pose()
                current_xy = np.asarray(pose[:2], dtype=float)

                path_length += float(
                    np.linalg.norm(current_xy - previous_xy)
                )
                previous_xy = current_xy.copy()

                collision = collision or bool(
                    info.get("collision", False)
                )
                out_of_bounds = out_of_bounds or bool(
                    info.get("out_of_bounds", False)
                )

                final_distance = float(
                    np.linalg.norm(goal_xy - current_xy)
                )

                success = bool(
                    info.get("reached_goal", False)
                    or final_distance < GOAL_THRESHOLD
                )

                if terminated or truncated:
                    break

            final_pose = env.get_robot_pose()
            final_xy = np.asarray(final_pose[:2], dtype=float)
            final_distance = float(
                np.linalg.norm(goal_xy - final_xy)
            )

            if success and not collision:
                terminal = "success"
            elif collision:
                terminal = "collision"
            elif out_of_bounds:
                terminal = "out_of_bounds"
            else:
                terminal = "timeout"

            row = {
                "episode": episode + 1,
                "seed": args.seed + episode,
                "env_type": args.env_type,
                "terminal": terminal,
                "success": int(terminal == "success"),
                "collision": int(collision),
                "timeout": int(terminal == "timeout"),
                "out_of_bounds": int(out_of_bounds),
                "steps": steps,
                "initial_goal_distance_m": initial_distance,
                "final_goal_distance_m": final_distance,
                "path_length_m": path_length,
                "path_efficiency": (
                    initial_distance / path_length
                    if path_length > 1e-9
                    else 0.0
                ),
                "episode_reward": total_reward,
            }
            rows.append(row)

            print(
                f"Episode {episode + 1:3d}/{args.episodes}: "
                f"{terminal:12s} "
                f"steps={steps:4d} "
                f"goal_dist={final_distance:.3f} m "
                f"path={path_length:.2f} m"
            )

    finally:
        env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        with output.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

        n = len(rows)
        successes = sum(r["success"] for r in rows)
        collisions = sum(r["collision"] for r in rows)
        timeouts = sum(r["timeout"] for r in rows)
        out_of_bounds = sum(r["out_of_bounds"] for r in rows)

        print("\n" + "=" * 60)
        print("SIMPLE CONTROLLER SUMMARY")
        print("=" * 60)
        print(f"Episodes:                 {n}")
        print(f"Success rate:             {100 * successes / n:.1f}%")
        print(f"Collision rate:           {100 * collisions / n:.1f}%")
        print(f"Timeout rate:             {100 * timeouts / n:.1f}%")
        print(f"Out-of-bounds rate:       {100 * out_of_bounds / n:.1f}%")
        print(
            "Mean final goal distance: "
            f"{np.mean([r['final_goal_distance_m'] for r in rows]):.3f} m"
        )
        print(
            "Mean path length:         "
            f"{np.mean([r['path_length_m'] for r in rows]):.3f} m"
        )
        print(f"CSV saved to:              {output}")
        print("=" * 60)


if __name__ == "__main__":
    main()
