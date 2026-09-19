# env/reward.py – COMPLETE CORRECTED VERSION
# Includes reset_step_count() method for episode reset

import os
import numpy as np
from config import LIDAR_RANGE


class RewardCalculator:
    """
    Reward calculator for mobile robot navigation.
    
    Reward components:
    - Progress toward goal (+)
    - Moving away from goal (-)
    - Step penalty (-)
    - Idle penalty (-)
    - Smoothness penalty (-)
    - Safety penalty (when too close to obstacles)
    - Goal reached (large positive, terminal)
    - Collision (large negative, terminal)
    
    Parameters can be overridden via environment variables for sweeps:
    - GOAL_REWARD
    - COLLISION_PENALTY
    - PROGRESS_WEIGHT
    - AWAY_PENALTY
    """
    
    def __init__(self):
        # ========== READ FROM ENVIRONMENT VARIABLES (for sweeps) ==========
        self.goal_reward = float(os.environ.get('GOAL_REWARD', 100.0))
        self.collision_penalty = float(os.environ.get('COLLISION_PENALTY', 50.0))
        self.lambda_goal_progress = float(os.environ.get('PROGRESS_WEIGHT', 2.0))
        self.lambda_away_penalty = float(os.environ.get('AWAY_PENALTY', 3.0))
        
        # ========== FIXED PARAMETERS (not swept) ==========
        self.step_penalty = 0.01          # Per step penalty (time efficiency)
        self.idle_penalty = 0.1           # Penalty when linear velocity < 0.05
        self.lambda_smooth = 0.02         # Penalty for high angular velocity
        
        # Safety parameters
        self.safe_dist = 0.30             # Meters – safe distance threshold
        self.near_collision_dist = 0.12   # Meters – near collision threshold
        self.lambda_safety = 2.0          # Weight for safety penalty
        
        # ========== INTERNAL STATE ==========
        self.prev_goal_dist = None
        self.step_count = 0
        
        # Print configuration for verification
        print(f"[RewardCalculator] Initialized with:")
        print(f"  goal_reward={self.goal_reward}")
        print(f"  collision_penalty={self.collision_penalty}")
        print(f"  progress_weight={self.lambda_goal_progress}")
        print(f"  away_penalty={self.lambda_away_penalty}")
        print(f"  step_penalty={self.step_penalty}")
        print(f"  idle_penalty={self.idle_penalty}")
        print(f"  smoothness_weight={self.lambda_smooth}")
        print(f"  safe_dist={self.safe_dist}")
        print(f"  safety_weight={self.lambda_safety}")
    
    def compute(self, prev_wp_dist, curr_wp_dist, robot_pose, path, 
                collision, reached_goal, action, lidar_scan, goal_xy):
        """
        Compute reward and termination signal.
        
        Args:
            prev_wp_dist: Previous distance to waypoint (unused in reward for end-to-end)
            curr_wp_dist: Current distance to waypoint (unused in reward for end-to-end)
            robot_pose: [x, y, theta] of robot
            path: RRT global path (unused in reward, only for hybrid)
            collision: Boolean collision flag
            reached_goal: Boolean goal reached flag
            action: [linear, angular] normalized actions
            lidar_scan: Normalized LiDAR readings (0 to 1)
            goal_xy: (x, y) goal position
            
        Returns:
            reward: Float reward value
            done: Boolean termination flag
        """
        reward = 0.0
        done = False
        
        # Increment step counter
        self.step_count += 1
        
        # ========== GOAL DISTANCE TRACKING ==========
        curr_goal_dist = np.linalg.norm(robot_pose[:2] - goal_xy)
        
        if self.prev_goal_dist is not None:
            # Progress toward goal (positive reward)
            goal_progress = max(0.0, self.prev_goal_dist - curr_goal_dist)
            reward += self.lambda_goal_progress * goal_progress
            
            # Moving away from goal (negative reward) – CRITICAL to prevent oscillation
            away = max(0.0, curr_goal_dist - self.prev_goal_dist)
            reward -= self.lambda_away_penalty * away
            
            # Debug print for progress (every 100 steps)
            if self.step_count % 100 == 0 and (goal_progress > 0 or away > 0):
                print(f"[Step {self.step_count}] Progress: +{self.lambda_goal_progress * goal_progress:.2f}, "
                      f"Away: -{self.lambda_away_penalty * away:.2f}, "
                      f"Goal dist: {curr_goal_dist:.2f}m")
        
        # ========== STEP PENALTY ==========
        # Encourages time-efficient navigation
        reward -= self.step_penalty
        
        # ========== IDLE PENALTY ==========
        # Prevents standing still
        if abs(action[0]) < 0.05:
            reward -= self.idle_penalty
        
        # ========== SMOOTHNESS PENALTY ==========
        # Penalise sharp turns for passenger comfort and mechanical wear
        reward -= self.lambda_smooth * abs(action[1])
        
        # ========== SAFETY PENALTY (soft penalty near obstacles) ==========
        min_dist = np.min(lidar_scan) * LIDAR_RANGE
        
        if min_dist < self.safe_dist:
            safety_penalty = self.lambda_safety * (self.safe_dist - min_dist)
            reward -= safety_penalty
            
            if min_dist < self.near_collision_dist:
                reward -= 1.0  # Extra penalty for very close obstacles
                if self.step_count % 50 == 0:
                    print(f"[Step {self.step_count}] Near collision! min_dist={min_dist:.3f}m, penalty={safety_penalty + 1.0:.2f}")
        
        # ========== TERMINAL CONDITIONS ==========
        if collision:
            reward -= self.collision_penalty
            print(f"[!] Step {self.step_count}: COLLISION! reward: {reward:.2f}")
            done = True
            
        elif reached_goal:
            reward += self.goal_reward
            print(f"[✓] Step {self.step_count}: GOAL REACHED! Total reward: {reward:.2f}")
            done = True
        
        # Update stored distance for next step
        self.prev_goal_dist = curr_goal_dist
        
        return reward, done
    
    def reset_step_count(self):
        """
        Reset internal state for a new episode.
        
        This method MUST be called at the beginning of each episode
        (in the environment's reset() method) to ensure clean state.
        """
        self.step_count = 0
        self.prev_goal_dist = None
        print("[RewardCalculator] State reset for new episode")
