#!/usr/bin/env python3
# debug_env.py – Test hybrid environment waypoint logic and basic navigation
# Uses a simple proportional controller (not a learned policy) to follow waypoints.
# This will reveal if the environment and waypoint system are functioning.

import numpy as np
import time
from env.tb3_env_hybrid import TB3HybridEnv

def simple_controller(robot_pose, waypoint, max_linear=0.22, max_angular=2.84, kp=1.0):
    """
    Simple proportional controller that steers the robot toward the waypoint.
    robot_pose: [x, y, theta]
    waypoint: [x_wp, y_wp]
    Returns action (linear, angular) in normalized range [-1, 1].
    """
    dx = waypoint[0] - robot_pose[0]
    dy = waypoint[1] - robot_pose[1]
    angle_to_wp = np.arctan2(dy, dx)
    heading_error = angle_to_wp - robot_pose[2]
    # Normalise angle to [-pi, pi]
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    
    # Linear velocity: reduce when heading error is large (turn in place)
    linear = max_linear * (1.0 - min(1.0, abs(heading_error) / np.pi))
    angular = max_angular * np.clip(kp * heading_error, -1.0, 1.0)
    
    # Normalise to action space [-1, 1]
    action = np.array([linear / max_linear, angular / max_angular])
    return action

def run_debug(env_type='static', episodes=3, max_steps=1000, gui=True):
    env = TB3HybridEnv(env_type=env_type, gui=gui)
    
    for ep in range(episodes):
        obs, info = env.reset()
        print(f"\n{'='*60}")
        print(f"Episode {ep+1} started. Goal position: {env.goal_pos[:2]}")
        
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            robot_pose = env.get_robot_pose()   # [x, y, theta]
            waypoint = env.current_waypoint
            
            if waypoint is None:
                print("WARNING: waypoint is None! Using goal as fallback.")
                waypoint = env.goal_pos[:2]
            
            # Use simple controller to get action
            action = simple_controller(robot_pose, waypoint)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Print debug info every 50 steps
            if steps % 50 == 0 or done:
                dist_to_wp = np.linalg.norm(robot_pose[:2] - waypoint)
                dist_to_goal = np.linalg.norm(robot_pose[:2] - env.goal_pos[:2])
                print(f"Step {steps:4d}: pos=({robot_pose[0]:5.2f},{robot_pose[1]:5.2f}), "
                      f"theta={robot_pose[2]:5.2f}, wp_dist={dist_to_wp:6.3f}, "
                      f"goal_dist={dist_to_goal:6.3f}, action=({action[0]:5.2f},{action[1]:5.2f}), "
                      f"reward={reward:6.2f}, done={done}")
            
            # Optional: slow down for visual inspection
            if gui:
                time.sleep(0.02)
        
        print(f"Episode {ep+1} finished after {steps} steps. Total reward: {total_reward:.2f}. "
              f"Reached goal: {info.get('reached_goal', False)}")
    
    env.close()

if __name__ == "__main__":
    # Run on static environment with GUI
    run_debug(env_type='static', episodes=3, max_steps=1000, gui=True)
