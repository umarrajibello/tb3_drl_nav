import numpy as np
from config import LIDAR_RANGE

class RewardCalculator:
    def __init__(self):
        # Scaled down rewards for stability
        self.lambda_goal_progress = 5.0
        self.lambda_wp_progress   = 2.0
        self.lambda_forward       = 0.1
        self.lambda_smooth        = 0.05
        self.lambda_safety        = 1.0
        self.lambda_time          = 0.01
        self.lambda_idle          = 0.05
        self.lambda_away_penalty  = 0.5

        self.goal_reward          = 15.0
        self.collision_penalty    = 8.0

        self.safe_dist            = 0.25
        self.near_collision_dist  = 0.12

        self.prev_goal_dist = None

    def compute(self, prev_wp_dist, curr_wp_dist,
                robot_pose, path, collision, reached_goal,
                action, lidar_scan, goal_xy):
        reward = 0.0
        done = False

        curr_goal_dist = np.linalg.norm(robot_pose[:2] - goal_xy)

        if self.prev_goal_dist is not None:
            goal_progress = max(0.0, self.prev_goal_dist - curr_goal_dist)
            away = max(0.0, curr_goal_dist - self.prev_goal_dist)
            reward += self.lambda_goal_progress * goal_progress
            reward -= self.lambda_away_penalty * away

            #if goal_progress > 0:
             #   print(f"[DEBUG] Goal progress: {goal_progress:.3f} m, reward: +{self.lambda_goal_progress * goal_progress:.2f}")

        # Waypoint progress (secondary)
        wp_progress = max(0.0, prev_wp_dist - curr_wp_dist)
        reward += self.lambda_wp_progress * wp_progress

        # Forward motion
        if action[0] > 0:
            reward += self.lambda_forward * action[0]

        reward -= self.lambda_smooth * abs(action[1])

        min_dist = np.min(lidar_scan) * LIDAR_RANGE
        if min_dist < self.safe_dist:
            reward -= self.lambda_safety * (self.safe_dist - min_dist)
        if min_dist < self.near_collision_dist:
            reward -= 1.0

        reward -= self.lambda_time
        if abs(action[0]) < 0.05:
            reward -= self.lambda_idle

        if collision:
            reward -= self.collision_penalty
            print(f"Collision! reward -{self.collision_penalty}")
            done = True
        elif reached_goal:
            reward += self.goal_reward
            print(f"Goal reached! +{self.goal_reward}")
            done = True

        self.prev_goal_dist = curr_goal_dist
        return reward, done