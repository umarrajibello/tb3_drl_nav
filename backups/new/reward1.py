import numpy as np
from config import LIDAR_RANGE

class RewardCalculator:
    """
    Corrected navigation reward that prevents reward hacking.
    - Clamped progress (only positive distance reduction)
    - No heading or global distance rewards (non-stationary)
    - Strong safety and idle penalties
    - Balanced terminal rewards
    """

    def __init__(self):
        # Core weights
        self.lambda_progress = 5.0
        self.lambda_forward = 0.3
        self.lambda_smooth = 0.05
        self.lambda_safety = 5.0          # increased from 2.0
        self.lambda_time = 0.05           # increased from 0.01
        self.lambda_idle = 0.2            # increased from 0.05

        # Disabled conflicting/non-stationary terms
        self.lambda_path = 0.0
        self.lambda_heading = 0.0
        self.lambda_global_dist = 0.0

        # Terminal rewards (balanced)
        self.goal_reward = 150.0          # slightly higher
        self.collision_penalty = 200.0    # much stronger

        self.safe_dist = 0.3

    def compute(self, prev_wp_dist, curr_wp_dist, robot_pose, path,
                collision, reached_goal, action, lidar_scan, goal_xy):

        reward = 0.0
        done = False

        # 1. Progress toward goal – CLAMPED (critical fix)
        progress = prev_wp_dist - curr_wp_dist
        progress = max(0.0, progress)
        reward += self.lambda_progress * progress

        # 2. Small forward motion bonus
        linear_vel = action[0]
        if linear_vel > 0:
            reward += self.lambda_forward * linear_vel

        # 3. Smoothness penalty
        angular_vel = action[1]
        reward -= self.lambda_smooth * abs(angular_vel)

        # 4. Safety shaping (strong penalty near obstacles)
        min_dist = np.min(lidar_scan) * LIDAR_RANGE
        if min_dist < self.safe_dist:
            reward -= self.lambda_safety * (self.safe_dist - min_dist)

        # 5. Time and idle penalties
        reward -= self.lambda_time
        if abs(linear_vel) < 0.05:
            reward -= self.lambda_idle

        # 6. Terminal conditions
        if collision:
            reward -= self.collision_penalty
            done = True
        elif reached_goal:
            reward += self.goal_reward
            done = True

        return reward, done

    # Geometry helpers (kept for compatibility – not used when lambda_path=0)
    def _distance_to_path(self, robot_xy, path):
        P = np.array(robot_xy)
        min_dist = float('inf')
        for i in range(len(path)-1):
            A = np.array(path[i])
            B = np.array(path[i+1])
            AB = B - A
            AP = P - A
            t = np.dot(AP, AB) / (np.dot(AB, AB) + 1e-8)
            t = np.clip(t, 0.0, 1.0)
            closest = A + t * AB
            dist = np.linalg.norm(P - closest)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _get_path_direction(self, robot_xy, path):
        P = np.array(robot_xy)
        best_theta = 0.0
        best_dist = float('inf')
        for i in range(len(path)-1):
            A = np.array(path[i])
            B = np.array(path[i+1])
            AB = B - A
            AP = P - A
            t = np.dot(AP, AB) / (np.dot(AB, AB) + 1e-8)
            t = np.clip(t, 0.0, 1.0)
            closest = A + t * AB
            dist = np.linalg.norm(P - closest)
            if dist < best_dist:
                best_dist = dist
                best_theta = np.arctan2(AB[1], AB[0])
        return best_theta