#!/usr/bin/env python3
"""
Run one episode with the hybrid environment and log all reward components.
Usage: python debug_reward.py
"""

import numpy as np
from env.tb3_env_hybrid import TB3HybridEnv   # adjust name if needed
from config import MAX_STEPS

def main():
    # Create environment (no GUI for speed, or set gui=True to watch)
    env = TB3HybridEnv(env_type="static", gui=True)

    obs, _ = env.reset()
    total_reward = 0.0
    step = 0
    print("Step, GoalDist, WaypointDist, MinLidar, Reward, Cumulative, Action[0], Action[1], Collision, ReachedGoal")

    for step in range(MAX_STEPS):
        # Use a simple random action, or better: take zero action to see baseline
        # action = np.random.uniform(-1, 1, size=2)   # random
        action = np.array([0.5, 0.0])   # forward only (to see progress)

        obs, reward, terminated, truncated, info = env.step(action)

        goal_dist = info.get("distance_to_goal", np.inf)
        waypoint_dist = info.get("waypoint_distance", np.nan)  # not always in info; we can compute manually
        # Get min lidar distance (in meters)
        min_lidar_m = np.min(obs[:36]) * env.planner.lidar_range   # assuming lidar_range = config.LIDAR_RANGE
        # But easier: use environment's lidar directly:
        lidar_raw = env.get_lidar()
        min_lidar_m = np.min(lidar_raw) * env.LIDAR_RANGE

        print(f"{step}, {goal_dist:.3f}, {waypoint_dist:.3f}, {min_lidar_m:.3f}, "
              f"{reward:.2f}, {total_reward + reward:.2f}, {action[0]:.2f}, {action[1]:.2f}, "
              f"{info.get('collision', False)}, {info.get('reached_goal', False)}")

        total_reward += reward

        if terminated or truncated:
            print(f"\nEpisode ended after {step+1} steps. Total reward: {total_reward:.2f}")
            break

if __name__ == "__main__":
    main()
