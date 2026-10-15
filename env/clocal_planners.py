# env/local_planners.py
import numpy as np
import pybullet as p

class LocalPlannerBase:
    def compute_cmd(self, env, goal_xy, path=None):
        raise NotImplementedError

class DWALocalPlanner(LocalPlannerBase):
    def __init__(self, max_lin=0.22, max_ang=2.84, lin_acc=0.5, ang_acc=3.2,
                 dt=0.1, horizon=1.0, num_samples_v=5, num_samples_w=11,
                 min_clearance=0.15):
        self.max_lin = max_lin
        self.max_ang = max_ang
        self.lin_acc = lin_acc
        self.ang_acc = ang_acc
        self.dt = dt
        self.horizon = horizon
        self.num_samples_v = num_samples_v
        self.num_samples_w = num_samples_w
        self.min_clearance = min_clearance

    def compute_cmd(self, env, goal_xy, path=None):
        pos = env.get_robot_position()
        x, y = pos[0], pos[1]
        _, quat = p.getBasePositionAndOrientation(env.robot)
        _, _, yaw = p.getEulerFromQuaternion(quat)

        v_curr, w_curr = 0.0, 0.0
        v_min = max(-self.max_lin, v_curr - self.lin_acc * self.dt)
        v_max = min(self.max_lin,  v_curr + self.lin_acc * self.dt)
        w_min = max(-self.max_ang, w_curr - self.ang_acc * self.dt)
        w_max = min(self.max_ang,  w_curr + self.ang_acc * self.dt)

        vs = np.linspace(v_min, v_max, self.num_samples_v)
        ws = np.linspace(w_min, w_max, self.num_samples_w)

        lidar = env.get_obs()
        lidar_max = env.LIDAR_RANGE if hasattr(env, "LIDAR_RANGE") else 3.5

        best_score = -1e9
        best_v, best_w = 0.0, 0.0

        for v in vs:
            for w in ws:
                x_sim, y_sim, yaw_sim, clearance = self._simulate(
                    x, y, yaw, v, w, lidar, lidar_max
                )

                if clearance < self.min_clearance:
                    continue

                dist_goal = np.linalg.norm(np.array(goal_xy) - np.array([x_sim, y_sim]))
                heading_score = -dist_goal
                clearance_score = clearance
                speed_score = v

                score = 1.0 * heading_score + 0.8 * clearance_score + 0.2 * speed_score

                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w

        lin_norm = np.clip(best_v / self.max_lin, -1, 1)
        ang_norm = np.clip(best_w / self.max_ang, -1, 1)
        return np.array([lin_norm, ang_norm])

    def _simulate(self, x, y, yaw, v, w, lidar_ranges, lidar_max_range):
        t = 0.0
        min_clearance = lidar_max_range
        while t < self.horizon:
            x += v * np.cos(yaw) * self.dt
            y += v * np.sin(yaw) * self.dt
            yaw += w * self.dt
            t += self.dt
            min_clearance = min(min_clearance, np.min(lidar_ranges) * lidar_max_range)
        return x, y, yaw, min_clearance
