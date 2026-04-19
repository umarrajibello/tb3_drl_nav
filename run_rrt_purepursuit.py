#!/usr/bin/env python3
"""
Classical baseline: RRT global planner + Pure Pursuit local controller.
"""

import numpy as np
import time
import sys
import csv
import pybullet as p
from env.tb3_env import TB3Env
from env.planners import RRTPlanner
from config import STATIC_OBSTACLES, LINEAR_VEL, ANGULAR_VEL, GOAL_THRESHOLD, COLLISION_DIST, MAX_STEPS

class PurePursuitController:
    def __init__(self, env, lookahead=0.5):
        self.env = env
        self.lookahead = lookahead

    def compute_cmd(self, waypoint):
        """Return normalized action [linear, angular] in [-1,1]."""
        robot_pos = self.env.get_robot_position()
        robot_xy = np.array(robot_pos[:2])
        _, ori = p.getBasePositionAndOrientation(self.env.robot_id)
        _, _, robot_yaw = p.getEulerFromQuaternion(ori)

        dx = waypoint[0] - robot_xy[0]
        dy = waypoint[1] - robot_xy[1]
        angle_to_wp = np.arctan2(dy, dx)
        angle_error = angle_to_wp - robot_yaw
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

        # Angular control
        k_ang = 2.0
        angular = np.clip(k_ang * angle_error, -ANGULAR_VEL, ANGULAR_VEL)

        # Linear speed based on heading alignment
        heading_factor = max(0.0, 1.0 - abs(angle_error) / np.pi)
        linear = LINEAR_VEL * (0.2 + 0.8 * heading_factor)

        # Normalize to action space [-1,1]
        return np.array([linear / LINEAR_VEL, angular / ANGULAR_VEL])

def run_rrt_purepursuit(env_type, num_episodes=50, gui=False, output_csv="rrt_purepursuit_results.csv"):
    import config
    config.ENV_TYPE = env_type
    config.GUI = gui

    env = TB3Env(env_type=env_type, gui=gui)
    controller = PurePursuitController(env, lookahead=0.5)
    rrt = RRTPlanner(static_obstacles=STATIC_OBSTACLES, bounds=env.world_bounds)

    metrics = []

    for ep in range(num_episodes):
        obs = env.reset()
        start = env.get_robot_pose()[:2]
        goal = env.goal_pos[:2]
        straight_line_dist = np.linalg.norm(goal - start)

        rrt.set_start(start)
        rrt.set_goal(goal)
        path = rrt.plan()
        if path is None or len(path) < 2:
            print(f"Episode {ep}: RRT failed")
            metrics.append({
                'episode': ep, 'steps': MAX_STEPS, 'success': False, 'collision': False,
                'time_to_goal': 0, 'path_length': 0, 'path_efficiency': 0,
                'smoothness_jerk': 0, 'min_clearance': 0, 'stuck_events': 0
            })
            continue

        # Visualize path
        env.draw_path(path, color=[0,1,0])

        wp_idx = 1
        waypoint = np.array(path[wp_idx])
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

            # Advance waypoint
            if np.linalg.norm(robot_xy - waypoint) < 0.3 and wp_idx < len(path)-1:
                wp_idx += 1
                waypoint = np.array(path[wp_idx])

            # Pure pursuit action
            action = controller.compute_cmd(waypoint)
            angular_velocities.append(action[1] * ANGULAR_VEL)

            # Step
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

            # Stuck
            if step_dist < 0.01:
                stuck_counter += 1
            else:
                stuck_counter = 0
            if stuck_counter >= 50:
                stuck_events += 1
                stuck_counter = 0

            # Clearance
            lidar = env.get_lidar()
            min_dist = np.min(lidar) * config.LIDAR_RANGE
            min_distances.append(min_dist)

            # Collision
            if isinstance(info, dict) and info.get('collision', False):
                collision = True
                done = True
            elif reward < -100:
                collision = True
                done = True

            # Goal
            dist_to_goal = np.linalg.norm(curr_pos - goal)
            if dist_to_goal < GOAL_THRESHOLD:
                success = True
                done = True

        # Metrics
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

        print(f"Episode {ep}: success={success}, steps={steps}, collision={collision}")

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
    print("\n===== RRT+PurePursuit Summary =====")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Collision rate: {collision_rate:.1f}%")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Path efficiency: {avg_eff:.2f}")
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    env_type = sys.argv[1] if len(sys.argv) > 1 else "static"
    num_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    gui = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
    output = sys.argv[4] if len(sys.argv) > 4 else f"rrt_purepursuit_{env_type}.csv"
    run_rrt_purepursuit(env_type, num_episodes=num_ep, gui=gui, output_csv=output)