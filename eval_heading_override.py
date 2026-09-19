#!/usr/bin/env python3
# eval_heading_override.py – Override angular velocity to steer toward goal
# Uses policy for linear velocity only.

import argparse
import os
import csv
import time
import random
import numpy as np
import torch
from pathlib import Path

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.tb3_env_hybrid import TB3HybridEnv
from env.observation_wrapper import TemporalObservationWrapper
from config import GOAL_THRESHOLD, COLLISION_DIST, LIDAR_RANGE, ANGULAR_VEL, LINEAR_VEL

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_base_env(hybrid: bool, env_type: str, gui: bool):
    if hybrid:
        return TB3HybridEnv(env_type=env_type, gui=gui)
    else:
        raise NotImplementedError("Only hybrid supported for this test")

def unwrap_original_env(env):
    while hasattr(env, 'venv'):
        env = env.venv
    if isinstance(env, DummyVecEnv):
        env = env.envs[0]
    while hasattr(env, 'env'):
        env = env.env
    return env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--env_type', type=str, default='static')
    parser.add_argument('--algorithm', type=str, default='sac')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--output', type=str, default='eval_heading_override.csv')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    import config
    config.ENV_TYPE = args.env_type
    config.GUI = args.gui

    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        model_path_zip = model_path.with_suffix('.zip')
        if model_path_zip.exists():
            model_path = model_path_zip
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    base_env = make_base_env(True, args.env_type, args.gui)
    if args.algorithm in ['cnn-gru-sac', 'lstm-sac', 'tx-sac']:
        base_env = TemporalObservationWrapper(base_env)
    env = DummyVecEnv([lambda: base_env])
    norm_path = model_path.parent / "vecnormalize.pkl"
    if norm_path.exists():
        env = VecNormalize.load(str(norm_path), env)
        env.training = False
        env.norm_reward = False
    if args.algorithm in ['sac', 'cnn-gru-sac', 'lstm-sac', 'tx-sac']:
        model = SAC.load(str(model_path), env=env)
    else:
        model = PPO.load(str(model_path), env=env)

    metrics = []
    for ep in range(args.num_episodes):
        obs = env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        original_env = unwrap_original_env(env)
        start_pos = np.array(original_env.get_robot_pose()[:2])
        goal_pos = np.array(original_env.goal_pos[:2])
        straight_dist = np.linalg.norm(goal_pos - start_pos)

        done = False
        steps = 0
        path_length = 0.0
        prev_pos = start_pos.copy()
        collision = False
        success = False
        start_time = time.time()

        while not done:
            # Get policy action
            action, _ = model.predict(obs, deterministic=True)
            linear_norm = action[0][0]   # keep policy's linear (may be small)
            # Override angular: pure pursuit toward goal
            robot_pose = original_env.get_robot_pose()
            dx = goal_pos[0] - robot_pose[0]
            dy = goal_pos[1] - robot_pose[1]
            angle_to_goal = np.arctan2(dy, dx)
            heading_error = angle_to_goal - robot_pose[2]
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            # P-controller: angular velocity proportional to heading error, clip to [-1,1]
            angular_norm = np.clip(2.0 * heading_error / np.pi, -1.0, 1.0)
            # Use a constant linear if policy's linear is too small (optional)
            if abs(linear_norm) < 0.05:
                linear_norm = 0.3   # force some forward movement
            modified_action = np.array([[linear_norm, angular_norm]])

            step_result = env.step(modified_action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                obs = np.asarray(obs, dtype=np.float32)
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
                obs = np.asarray(obs, dtype=np.float32)

            steps += 1
            curr_pos = np.array(original_env.get_robot_pose()[:2])
            step_dist = np.linalg.norm(curr_pos - prev_pos)
            path_length += step_dist
            prev_pos = curr_pos
            if isinstance(info, dict) and info.get('collision', False):
                collision = True
                break
            if steps >= 1000:
                break

        time_to_goal = time.time() - start_time
        if not collision:
            dist_to_goal = np.linalg.norm(curr_pos - goal_pos)
            if dist_to_goal < GOAL_THRESHOLD:
                success = True
        metrics.append({'episode': ep, 'success': success, 'steps': steps,
                        'collision': collision, 'path_length': path_length,
                        'path_efficiency': straight_dist / path_length if path_length > 0 else 0})

        print(f"Episode {ep}: success={success}, steps={steps}, collision={collision}")

    success_rate = np.mean([m['success'] for m in metrics]) * 100
    print(f"\nSuccess rate with heading override: {success_rate:.1f}%")

if __name__ == "__main__":
    main()
