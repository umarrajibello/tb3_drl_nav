#!/usr/bin/env python3
"""
Classical baseline: RRT global planner + DWA local controller.
Runs multiple episodes, logs metrics to CSV.
"""

import numpy as np
import time
import sys
import csv
import os
from env.tb3_env import TB3Env
from env.controllers import DWAPlanner
from env.planners import RRTPlanner
from config import STATIC_OBSTACLES, ENV_TYPE, GUI, GOAL_THRESHOLD, COLLISION_DIST, MAX_STEPS

def run_rrt_dwa(env_type, num_episodes=50, gui=False, output_csv="rrt_dwa_results.csv"):
    """
    Run RRT + DWA baseline.
    """
    # Set config
    import config
    config.ENV_TYPE = env_type
    config.GUI = gui

    env = TB3Env(env_type=env_type, gui=gui)
    rrt = RRTPlanner(static_obstacles=STATIC_OBSTACLES, bounds=env.world_bounds)
    dwa = DWAPlanner(max_lin=0.22, max_ang=2.84, lin_acc=0.5, ang_acc=3.2, dt=0.1, horizon=1.0)

    """
    dwa = DWAPlanner(
        max_lin=0.35,      # faster
        max_ang=3.5,       # sharper turns
        lin_acc=0.8,       # quicker acceleration
        ang_acc=5.0,       # quicker rotation
        dt=0.05,           # finer simulation
        horizon=1.5,       # look ahead further
        num_samples_v=7,
        num_samples_w=15,
        min_clearance=0.12 # less conservative
    )
    """

    metrics = []

    for ep in range(num_episodes):
        obs = env.reset()
        start = env.get_robot_pose()[:2]
        goal = env.goal_pos[:2]
        straight_line_dist = np.linalg.norm(goal - start)

        # Plan global path
        rrt.set_start(start)
        rrt.set_goal(goal)
        path = rrt.plan()
        if path is None or len(path) < 2:
            print(f"Episode {ep}: RRT failed to find path")
            # Log as failure
            metrics.append({
                'episode': ep, 'steps': MAX_STEPS, 'success': False, 'collision': False,
                'time_to_goal': 0, 'path_length': 0, 'path_efficiency': 0,
                'smoothness_jerk': 0, 'min_clearance': 0, 'stuck_events': 0
            })
            continue

        # Waypoint tracking
        wp_idx = 0
        waypoint = np.array(path[0])
        steps = 0
        done = False
        collision = False
        success = False
        path_length = 0.0
        prev_pos = start.copy()
        angular_velocities = []
        min_distances = []
        stuck_counter = 0
        stuck_events = 0
        start_time = time.time()

        while not done and steps < MAX_STEPS:
            robot_pose = env.get_robot_pose()
            robot_xy = robot_pose[:2]

            # Advance waypoint when close
            if np.linalg.norm(robot_xy - waypoint) < 0.2 and wp_idx < len(path)-1:
                wp_idx += 1
                waypoint = np.array(path[wp_idx])

            # DWA command
            action = dwa.compute_cmd(env, waypoint)
            angular_velocities.append(action[1] * config.ANGULAR_VEL)  # for jerk

            # Step environment (handles both Gym and Gymnasium)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            steps += 1
            curr_pos = np.array(env.get_robot_pose()[:2])

            # Path length
            step_dist = np.linalg.norm(curr_pos - prev_pos)
            path_length += step_dist
            prev_pos = curr_pos

            # Stuck detection
            if step_dist < 0.01:
                stuck_counter += 1
            else:
                stuck_counter = 0
            if stuck_counter >= 50:
                stuck_events += 1
                stuck_counter = 0

            # LiDAR clearance
            lidar = env.get_lidar()
            min_dist = np.min(lidar) * config.LIDAR_RANGE
            min_distances.append(min_dist)

            # Collision detection (from info or reward)
            if isinstance(info, dict) and info.get('collision', False):
                collision = True
                done = True
            elif reward < -100:  # heuristic
                collision = True
                done = True

            # Goal check
            dist_to_goal = np.linalg.norm(curr_pos - goal)
            if dist_to_goal < GOAL_THRESHOLD:
                success = True
                done = True

        # Episode finished
        time_to_goal = time.time() - start_time
        jerk = np.mean(np.abs(np.diff(angular_velocities))) if len(angular_velocities) > 1 else 0.0
        min_clearance = np.min(min_distances) if min_distances else 0.0
        path_efficiency = straight_line_dist / path_length if path_length > 0 else 0.0

        metrics.append({
            'episode': ep,
            'steps': steps,
            'time_to_goal': time_to_goal,
            'path_length': path_length,
            'path_efficiency': path_efficiency,
            'smoothness_jerk': jerk,
            'min_clearance': min_clearance,
            'collision': collision,
            'stuck_events': stuck_events,
            'success': success
        })

        print(f"Episode {ep}: success={success}, steps={steps}, collision={collision}, time={time_to_goal:.2f}s")

    # Save CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    # Summary
    success_rate = np.mean([m['success'] for m in metrics]) * 100
    collision_rate = np.mean([m['collision'] for m in metrics]) * 100
    avg_steps = np.mean([m['steps'] for m in metrics])
    avg_eff = np.mean([m['path_efficiency'] for m in metrics])
    print("\n===== RRT+DWA Summary =====")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Collision rate: {collision_rate:.1f}%")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Path efficiency: {avg_eff:.2f}")
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    env_type = sys.argv[1] if len(sys.argv) > 1 else "static"
    num_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    gui = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
    output = sys.argv[4] if len(sys.argv) > 4 else f"rrt_dwa_{env_type}.csv"
    run_rrt_dwa(env_type, num_episodes=num_ep, gui=gui, output_csv=output)