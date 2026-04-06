import pybullet as p
import numpy as np
import csv
from env.tb3_env import TB3Env
from env.planners import RRTPlanner
from config import STATIC_OBSTACLES, LINEAR_VEL, ANGULAR_VEL, GOAL_THRESHOLD

class SimpleController:
    def __init__(self, env):
        self.env = env

    def compute_velocity(self, waypoint):
        robot_pos = self.env.get_robot_position()
        dx = waypoint[0] - robot_pos[0]
        dy = waypoint[1] - robot_pos[1]
        angle_to_waypoint = np.arctan2(dy, dx)
        _, _, robot_yaw = p.getEulerFromQuaternion(self.env.get_robot_orientation())
        angle_error = angle_to_waypoint - robot_yaw
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        linear = LINEAR_VEL * 0.5
        angular = np.clip(angle_error * 2.0, -ANGULAR_VEL, ANGULAR_VEL)
        return np.array([linear / LINEAR_VEL, angular / ANGULAR_VEL])

def run_rrt_baseline(env_type='mixed', num_episodes=20):
    env = TB3Env()
    import config
    config.ENV_TYPE = env_type
    controller = SimpleController(env)

    metrics = []
    for ep in range(num_episodes):
        obs = env.reset()
        start = env.get_robot_position()[:2]
        goal = env.goal
        planner = RRTPlanner(start=start, goal=goal, obstacles=STATIC_OBSTACLES)
        path = planner.plan()
        if path is None:
            print(f"Episode {ep}: RRT failed to find path")
            metrics.append({'episode': ep, 'steps': 0, 'collision': False, 'success': False})
            continue
        waypoint_idx = 1
        waypoint = path[waypoint_idx] if len(path) > 1 else goal
        done = False
        steps = 0
        collision = False
        while not done:
            robot_pos = env.get_robot_position()[:2]
            if np.linalg.norm(robot_pos - waypoint) < 0.2:
                waypoint_idx = min(waypoint_idx + 1, len(path)-1)
                waypoint = path[waypoint_idx]
            action = controller.compute_velocity(waypoint)
            obs, reward, done, _ = env.step(action)
            steps += 1
            if reward == -150.0:
                collision = True
                break
            if steps > 1000:
                break
        success = not collision and np.linalg.norm(env.get_robot_position()[:2] - goal) < GOAL_THRESHOLD
        metrics.append({'episode': ep, 'steps': steps, 'collision': collision, 'success': success})
        print(f"Episode {ep}: success={success}, steps={steps}, collision={collision}")

    with open('rrt_baseline_results.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['episode','steps','collision','success'])
        writer.writeheader()
        writer.writerows(metrics)

if __name__ == "__main__":
    run_rrt_baseline(env_type='mixed', num_episodes=20)