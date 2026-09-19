import pybullet as p
import numpy as np
import time
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
        linear = LINEAR_VEL
        angular = np.clip(angle_error * 2.0, -ANGULAR_VEL, ANGULAR_VEL)
        return np.array([linear / LINEAR_VEL, angular / ANGULAR_VEL])

def run_single_episode(gui=True):
    import config
    config.GUI = gui
    env = TB3Env()
    controller = SimpleController(env)   # <-- CRITICAL: instantiate controller
    
    obs = env.reset()
    start = np.array(env.get_robot_position()[:2])
    goal = env.goal
    print(f"Start: {start}, Goal: {goal}")
    
    planner = RRTPlanner(start=start, goal=goal, obstacles=STATIC_OBSTACLES,
                         bounds=[-3,3,-3,3], step_size=0.2)
    path = planner.plan()
    if path is None:
        print("RRT failed to find a path")
        return
    print(f"Path found with {len(path)} points")
    
    waypoint_idx = 1
    waypoint = path[waypoint_idx]
    done = False
    steps = 0

    

    while steps < 300:
        action = np.array([1.0, 0.0])  # full forward, no turn
        obs, reward, done, _ = env.step(action)
        robot_pos = np.array(env.get_robot_position()[:2])
        print(steps, robot_pos)
        steps += 1
        time.sleep(0.01)

    """
    while not done and steps < 2000:
        robot_pos = np.array(env.get_robot_position()[:2])
        if np.linalg.norm(robot_pos - waypoint) < 0.4:
            waypoint_idx = min(waypoint_idx + 1, len(path)-1)
            waypoint = path[waypoint_idx]
            print(f"New waypoint: {waypoint}")
        
        action = controller.compute_velocity(waypoint)
        print(f"Step {steps}: action={action}, robot={robot_pos}")
        obs, reward, done, _ = env.step(action)
        steps += 1
        if gui:
            time.sleep(0.01)
    """
    print(f"Episode finished after {steps} steps")

if __name__ == "__main__":
    run_single_episode(gui=True)