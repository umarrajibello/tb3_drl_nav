#!/usr/bin/env python3
# eval.py – Complete evaluation for TB3 navigation (end‑to‑end and hybrid)
# Works with all algorithms, handles missing VecNormalize, correct file path resolution

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

from env.tb3_env import TB3Env
from env.tb3_env_hybrid import TB3HybridEnv
from env.observation_wrapper import TemporalObservationWrapper

from config import GOAL_THRESHOLD, COLLISION_DIST, LIDAR_RANGE, ANGULAR_VEL


# ----------------------------------------------------------------------
# Helper: set random seeds
# ----------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------
# Base environment factory
# ----------------------------------------------------------------------
def make_base_env(hybrid: bool, env_type: str, gui: bool):
    if hybrid:
        return TB3HybridEnv(env_type=env_type, gui=gui)
    else:
        return TB3Env(env_type=env_type, gui=gui)


# ----------------------------------------------------------------------
# Recursively unwrap VecNormalize → DummyVecEnv → underlying env
# ----------------------------------------------------------------------
def unwrap_original_env(env):
    while hasattr(env, 'venv'):
        env = env.venv
    if isinstance(env, DummyVecEnv):
        env = env.envs[0]
    while hasattr(env, 'env'):
        env = env.env
    return env


# ----------------------------------------------------------------------
# Parse arguments
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model .zip file (or base name without .zip)')
    parser.add_argument('--env_type', type=str, default='mixed',
                        choices=['static', 'mixed', 'dense'])
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['sac', 'ppo', 'cnn-gru-sac', 'lstm-sac', 'tx-sac'])
    parser.add_argument('--hybrid', action='store_true')
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--output', type=str, default='eval_results.csv')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    import config
    config.ENV_TYPE = args.env_type
    config.GUI = args.gui

    print(f"\n{'='*60}")
    print(f"Evaluating {args.algorithm} on {args.env_type} environment")
    print(f"Hybrid mode: {args.hybrid}")
    print(f"Model path: {args.model_path}")
    print(f"Episodes: {args.num_episodes}")
    print(f"{'='*60}\n")

    # --------------------------------------------------------------
    # 1. Locate model file (handle both .zip and no extension)
    # --------------------------------------------------------------
    model_path = Path(args.model_path)
    if not model_path.exists():
        # Try adding .zip
        model_path_zip = model_path.with_suffix('.zip')
        if model_path_zip.exists():
            model_path = model_path_zip
        else:
            raise FileNotFoundError(f"Model file not found: {model_path} or {model_path_zip}")
    print(f"[Eval] Using model: {model_path}")

    # --------------------------------------------------------------
    # 2. Create environment
    # --------------------------------------------------------------
    base_env = make_base_env(args.hybrid, args.env_type, args.gui)
    if args.algorithm in ['cnn-gru-sac', 'lstm-sac', 'tx-sac']:
        base_env = TemporalObservationWrapper(base_env)
        print("[Eval] TemporalObservationWrapper applied (history=8)")

    env = DummyVecEnv([lambda: base_env])

    # --------------------------------------------------------------
    # 3. Load VecNormalize stats (if present)
    # --------------------------------------------------------------
    norm_path = model_path.parent / "vecnormalize.pkl"
    if norm_path.exists():
        try:
            env = VecNormalize.load(str(norm_path), env)
            env.training = False
            env.norm_reward = False
            print(f"[Eval] Loaded VecNormalize stats from {norm_path}")
        except Exception as e:
            print(f"[Eval] Warning: could not load VecNormalize ({e})")
    else:
        print("[Eval] No VecNormalize stats found. Running without normalisation.")

    # --------------------------------------------------------------
    # 4. Load trained model
    # --------------------------------------------------------------
    if args.algorithm in ['sac', 'cnn-gru-sac', 'lstm-sac', 'tx-sac']:
        model = SAC.load(str(model_path), env=env)
    else:
        model = PPO.load(str(model_path), env=env)
    print("[Eval] Model loaded successfully.\n")

    # --------------------------------------------------------------
    # 5. Evaluation loop
    # --------------------------------------------------------------
    metrics = []

    for ep in range(args.num_episodes):
        obs = env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        original_env = unwrap_original_env(env)

        start_pos = np.array(original_env.get_robot_pose()[:2])
        goal_pos = np.array(original_env.goal_pos[:2])
        straight_line_dist = np.linalg.norm(goal_pos - start_pos)

        done = False
        steps = 0
        path_length = 0.0
        prev_pos = start_pos.copy()
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

            step_result = env.step(action)
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

            if step_dist < 0.01:
                stuck_counter += 1
            else:
                stuck_counter = 0
            if stuck_counter >= 50:
                stuck_events += 1
                stuck_counter = 0

            raw_lidar = np.array(original_env.get_lidar(), dtype=np.float32)
            raw_lidar = np.clip(raw_lidar, 0.0, 1.0)
            min_dist = np.min(raw_lidar) * LIDAR_RANGE
            min_distances.append(min_dist)

            if isinstance(info, dict) and info.get('collision', False):
                collision = True
                try:
                    dyn_dists = original_env.get_dynamic_obstacle_distances()
                    if dyn_dists and min(dyn_dists) < COLLISION_DIST:
                        dynamic_collision = True
                except AttributeError:
                    pass

        # Episode finished – compute metrics
        time_to_goal = time.time() - start_time
        jerk = np.mean(np.abs(np.diff(angular_velocities))) if len(angular_velocities) > 1 else 0.0
        min_clearance = np.min(min_distances) if min_distances else 0.0
        path_efficiency = straight_line_dist / path_length if path_length > 0 else 0.0

        if not collision:
            dist_to_goal = np.linalg.norm(curr_pos - goal_pos)
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

        print(f"Episode {ep:3d}: success={success}, steps={steps:3d}, "
              f"time={time_to_goal:.2f}s, eff={path_efficiency:.2f}, "
              f"jerk={jerk:.3f}, clearance={min_clearance:.2f}, "
              f"collision={collision}, dyn_coll={dynamic_collision}")

    # --------------------------------------------------------------
    # 6. Save and summarise
    # --------------------------------------------------------------
    if metrics:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)

    success_rate = np.mean([m['success'] for m in metrics]) * 100
    collision_rate = np.mean([m['collision'] for m in metrics]) * 100
    dynamic_collision_rate = np.mean([m['dynamic_collision'] for m in metrics]) * 100
    avg_steps = np.mean([m['steps'] for m in metrics])
    avg_eff = np.mean([m['path_efficiency'] for m in metrics])
    avg_clearance = np.mean([m['min_clearance'] for m in metrics])
    avg_stuck = np.mean([m['stuck_events'] for m in metrics])

    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Success rate:            {success_rate:.1f}%")
    print(f"Collision rate:          {collision_rate:.1f}%")
    print(f"  - Dynamic collision:   {dynamic_collision_rate:.1f}%")
    print(f"Average steps:           {avg_steps:.1f}")
    print(f"Path efficiency:         {avg_eff:.3f}")
    print(f"Min clearance (avg):     {avg_clearance:.2f} m")
    print(f"Stuck events (avg):      {avg_stuck:.2f}")
    print(f"Results saved to:        {args.output}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
