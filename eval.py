import argparse
import os
import csv
import time
import random
import numpy as np
import torch
import gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import make_env
from env.observation_wrapper import TemporalObservationWrapper
from config import OBSERVATION_HISTORY, GOAL_THRESHOLD, COLLISION_DIST, LIDAR_RANGE, NUM_LIDAR, ANGULAR_VEL

class GymWrapper(gym.Env):
    def __init__(self, hybrid=False):
        super().__init__()
        self.env = make_env(hybrid)
        self.hybrid = hybrid
        if hybrid:
            low = np.array([0.0] * NUM_LIDAR + [-1.0, -1.0, -1.0])
            high = np.array([1.0] * NUM_LIDAR + [1.0, 1.0, 1.0])
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(NUM_LIDAR,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def get_robot_position(self):
        return self.env.get_robot_position()

    def get_dynamic_obstacle_distances(self):
        return self.env.get_dynamic_obstacle_distances()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--env_type', type=str, default='mixed', choices=['static','mixed','dense'])
    parser.add_argument('--algorithm', type=str, required=True, choices=['sac','ppo','lstm-sac','tx-sac'])
    parser.add_argument('--hybrid', action='store_true')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--output', type=str, default='eval_results.csv')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    import config
    config.ENV_TYPE = args.env_type
    config.GUI = args.gui

    env = GymWrapper(hybrid=args.hybrid)
    if args.algorithm in ['lstm-sac', 'tx-sac']:
        env = TemporalObservationWrapper(env, history_len=OBSERVATION_HISTORY)
    env = DummyVecEnv([lambda: env])

    # Load normalization stats if they exist
    norm_path = os.path.join(os.path.dirname(args.model_path), "vecnormalize.pkl")
    if os.path.exists(norm_path):
        try:
            env = VecNormalize.load(norm_path, env)
            env.training = False
            env.norm_reward = False
        except AssertionError as e:
            print(f"Warning: Could not load normalization stats due to shape mismatch: {e}. Proceeding without normalization.")
            env = VecNormalize(env, norm_obs=False, norm_reward=False, training=False)
    else:
        print("Warning: No normalization stats found. Proceeding without normalization.")
        env = VecNormalize(env, norm_obs=False, norm_reward=False, training=False)

    # Load model
    if args.algorithm in ['sac', 'lstm-sac', 'tx-sac']:
        model = SAC.load(args.model_path, env=env)
    elif args.algorithm == 'ppo':
        model = PPO.load(args.model_path, env=env)

    metrics = []
    for ep in range(args.num_episodes):
        obs = env.reset()
        # Access underlying environment for position and goal
        base_env = env.envs[0] if hasattr(env, 'envs') else env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        original_env = base_env.env if hasattr(base_env, 'env') else base_env

        start_pos = original_env.get_robot_position()[:2]
        goal_pos = original_env.goal[:2]
        straight_line_dist = np.linalg.norm(np.array(goal_pos) - np.array(start_pos))

        done = False
        steps = 0
        path_length = 0.0
        prev_pos = np.array(start_pos)
        collision = False
        dynamic_collision = False
        success = False
        angular_velocities = []
        min_distances = []
        stuck_counter = 0
        stuck_events = 0
        start_time = time.time()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            ang_vel = action[0][1] * ANGULAR_VEL
            angular_velocities.append(ang_vel)

            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
                terminated = done
                truncated = False

            steps += 1
            curr_pos = np.array(original_env.get_robot_position()[:2])
            path_length += np.linalg.norm(curr_pos - prev_pos)
            prev_pos = curr_pos

            raw_lidar = original_env.get_obs()
            min_dist = np.min(raw_lidar) * LIDAR_RANGE
            min_distances.append(min_dist)

            if steps > 1:
                progress = np.linalg.norm(curr_pos - prev_pos)
                if progress < 0.01:
                    stuck_counter += 1
                else:
                    stuck_counter = 0
                if stuck_counter >= 50:
                    stuck_events += 1
                    stuck_counter = 0

            # Collision detection based on reward -200 (from your reward function)
            if reward == -200.0 and not collision:
                collision = True
                try:
                    dyn_dists = original_env.get_dynamic_obstacle_distances()
                    if dyn_dists and min(dyn_dists) < COLLISION_DIST:
                        dynamic_collision = True
                except AttributeError:
                    pass

        time_to_goal = time.time() - start_time
        angular_velocities = np.array(angular_velocities)
        if len(angular_velocities) > 1:
            jerk = np.mean(np.abs(np.diff(angular_velocities)))
        else:
            jerk = 0.0
        min_clearance = np.min(min_distances) if min_distances else 0.0
        path_efficiency = straight_line_dist / path_length if path_length > 0 else 0.0

        if not collision:
            dist_to_goal = np.linalg.norm(curr_pos - np.array(goal_pos))
            if dist_to_goal < GOAL_THRESHOLD:
                success = True

        metrics.append({
            'episode': ep,
            'steps': steps,
            'time_to_goal': time_to_goal,
            'path_length': path_length,
            'path_efficiency': path_efficiency,
            'smoothness_jerk': jerk,
            'min_clearance': min_clearance,
            'collision': collision,
            'dynamic_collision': dynamic_collision,
            'stuck_events': stuck_events,
            'success': success
        })
        print(f"Episode {ep}: success={success}, steps={steps}, time={time_to_goal:.2f}s, "
              f"path_efficiency={path_efficiency:.2f}, jerk={jerk:.3f}, min_clearance={min_clearance:.2f}")

    # Write detailed results
    with open(args.output, 'w', newline='') as f:
        fieldnames = ['episode','steps','time_to_goal','path_length','path_efficiency',
                      'smoothness_jerk','min_clearance','collision','dynamic_collision',
                      'stuck_events','success']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    # Print summary statistics
    success_rate = np.mean([m['success'] for m in metrics]) * 100
    collision_rate = np.mean([m['collision'] for m in metrics]) * 100
    dynamic_collision_rate = np.mean([m['dynamic_collision'] for m in metrics]) * 100
    avg_steps = np.mean([m['steps'] for m in metrics])
    avg_path_efficiency = np.mean([m['path_efficiency'] for m in metrics])
    print("\n===== Evaluation Summary =====")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Collision rate: {collision_rate:.1f}% (dynamic: {dynamic_collision_rate:.1f}%)")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Path efficiency: {avg_path_efficiency:.2f}")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()