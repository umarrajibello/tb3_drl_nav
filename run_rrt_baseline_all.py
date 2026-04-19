import pybullet as p
import numpy as np
import csv
import time
from env.tb3_env import TB3Env
from env.planners import RRTPlanner
from config import STATIC_OBSTACLES, LINEAR_VEL, ANGULAR_VEL, GOAL_THRESHOLD, COLLISION_DIST, LIDAR_RANGE

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
        # Simple P controller
        linear = LINEAR_VEL    # half speed for safety
        angular = np.clip(angle_error * 2.0, -ANGULAR_VEL, ANGULAR_VEL)
        # Normalise actions to [-1,1] as in RL
        return np.array([linear / LINEAR_VEL, angular / ANGULAR_VEL])

def run_rrt_baseline(env_type='mixed', num_episodes=100, output_file='rrt_baseline.csv'):
    env = TB3Env()
    controller = SimpleController(env)
    import config
    config.ENV_TYPE = env_type
    config.GUI = False   # headless for speed

    metrics = []
    for ep in range(num_episodes):
        obs = env.reset()
        start = np.array(env.get_robot_position()[:2])
        goal = env.goal
        planner = RRTPlanner(start=start, goal=goal, obstacles=STATIC_OBSTACLES,
                             bounds=[-3,3,-3,3], step_size=0.2)
        path = planner.plan()
        if path is None or len(path) < 2:
            print(f"Episode {ep}: RRT failed to find path")
            metrics.append({
                'episode': ep,
                'success': False,
                'collision': False,
                'steps': 0,
                'path_length': 0.0,
                'path_efficiency': 0.0,
                'min_clearance': 0.0,
                'stuck_events': 0
            })
            continue

        waypoint_idx = 1
        waypoint = path[waypoint_idx]
        done = False
        steps = 0
        collision = False
        path_length = 0.0
        prev_pos = start
        min_clearance = float('inf')
        stuck_counter = 0
        stuck_events = 0

        start_time = time.time()

        while not done:
            robot_pos = np.array(env.get_robot_position()[:2])
            # Update waypoint if close enough
            if np.linalg.norm(robot_pos - waypoint) < 0.2:
                waypoint_idx = min(waypoint_idx + 1, len(path)-1)
                waypoint = path[waypoint_idx]

            # Compute action
            action = controller.compute_velocity(waypoint)
            obs, reward, done, _ = env.step(action)
            steps += 1

            # Update metrics
            curr_pos = np.array(env.get_robot_position()[:2])
            path_length += np.linalg.norm(curr_pos - prev_pos)
            prev_pos = curr_pos

            # Lidar based min distance
            lidar = env.get_obs()
            min_dist = np.min(lidar) * LIDAR_RANGE
            min_clearance = min(min_clearance, min_dist)

            # Stuck detection
            if steps > 1:
                progress = np.linalg.norm(curr_pos - prev_pos)
                if progress < 0.01:
                    stuck_counter += 1
                else:
                    stuck_counter = 0
                if stuck_counter >= 50:
                    stuck_events += 1
                    stuck_counter = 0

            # Collision detection (use same as RL: min distance < COLLISION_DIST)
            if min_dist < COLLISION_DIST:
                collision = True
                done = True
                break

            if steps >= 5000:   # same max steps as RL
                break

        time_taken = time.time() - start_time
        success = (not collision) and (np.linalg.norm(curr_pos - goal) < GOAL_THRESHOLD)
        straight_line_dist = np.linalg.norm(goal - start)
        path_efficiency = straight_line_dist / path_length if path_length > 0 else 0.0

        metrics.append({
            'episode': ep,
            'success': success,
            'collision': collision,
            'steps': steps,
            'path_length': path_length,
            'path_efficiency': path_efficiency,
            'min_clearance': min_clearance,
            'stuck_events': stuck_events,
            'time_to_goal': time_taken if success else None
        })

        print(f"Episode {ep}: success={success}, steps={steps}, collision={collision}, efficiency={path_efficiency:.2f}")

    # Save CSV
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['episode','success','collision','steps','path_length',
                      'path_efficiency','min_clearance','stuck_events','time_to_goal']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    # Print summary
    success_rate = np.mean([m['success'] for m in metrics]) * 100
    collision_rate = np.mean([m['collision'] for m in metrics]) * 100
    avg_steps = np.mean([m['steps'] for m in metrics])
    print(f"\n===== RRT Baseline Summary ({env_type}) =====")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Collision rate: {collision_rate:.1f}%")
    print(f"Average steps: {avg_steps:.1f}")

if __name__ == "__main__":
    # Run for all environment types
    for env_type in ['static', 'mixed', 'dense']:
        run_rrt_baseline(env_type=env_type, num_episodes=100, output_file=f'rrt_baseline_{env_type}.csv')