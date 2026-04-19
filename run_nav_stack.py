from env.tb3_env import TB3Env
from env.global_planners import RRTGlobalPlanner
from env.local_planners import DWALocalPlanner
import numpy as np

def run_nav_stack(num_episodes=1):
    env = TB3Env()

    for ep in range(num_episodes):
        env.reset()
        start = env.get_robot_position()[:2]
        goal = env.goal[:2]

        # GLOBAL PLANNER
        obstacles = env.static_obstacles
        global_planner = RRTGlobalPlanner(obstacles)
        path = global_planner.plan(start, goal)

        if path is None:
            print("Global planner failed")
            continue

        env.draw_path(path)

        # LOCAL PLANNER
        local_planner = DWALocalPlanner()

        waypoint_idx = 0
        done = False

        while not done:
            robot_pos = env.get_robot_position()[:2]

            if waypoint_idx < len(path) - 1:
                waypoint = path[waypoint_idx]
                if np.linalg.norm(robot_pos - waypoint) < 0.3:
                    waypoint_idx += 1
            else:
                waypoint = goal

            action = local_planner.compute_cmd(env, waypoint, path)
            obs, reward, done, info = env.step(action)

        print(f"Episode {ep} done.")

if __name__ == "__main__":
    run_nav_stack()
