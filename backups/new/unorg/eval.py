# Fully Corrected eval.py
# Works with both old Gym (4‑value step) and Gymnasium (5‑value step)
# Compatible with end‑to‑end and hybrid environments, with or without temporal wrappers.

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

from env.tb3_env import TB3Env
from env.tb3_env_hybrid import TB3HybridEnv
from env.observation_wrapper import TemporalObservationWrapper

from config import (
    OBSERVATION_HISTORY, GOAL_THRESHOLD, COLLISION_DIST,
    LIDAR_RANGE, ANGULAR_VEL
)

# ------------------------------------------------------------
# Seeding for reproducibility
# ------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ------------------------------------------------------------
# Base environment factory
# ------------------------------------------------------------
def make_base_env(hybrid: bool, env_type: str, gui: bool):
    """Create the appropriate base environment (TB3Env or TB3HybridEnv)."""
    if hybrid:
        return TB3HybridEnv(env_type=env_type, gui=gui)
    else:
        return TB3Env(env_type=env_type, gui=gui)

# ------------------------------------------------------------
# Unwrap VecNormalize → DummyVecEnv → underlying base env
# ------------------------------------------------------------
def unwrap_original_env(env):
    """
    Recursively unwrap VecNormalize and DummyVecEnv to access the original
    TB3Env or TB3HybridEnv instance. Needed to call get_robot_pose(), get_lidar(), etc.
    """
    if isinstance(env, VecNormalize):
        env = env.venv
    if isinstance(env, DummyVecEnv):
        env = env.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env

# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the trained model .zip file")
    parser.add_argument('--env_type', type=str, default='mixed',
                        choices=['static', 'mixed', 'dense'])
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['sac', 'ppo', 'cnn-gru-sac', 'lstm-sac', 'tx-sac'])
    parser.add_argument('--hybrid', action='store_true',
                        help="Use hybrid environment (RRT + waypoint)")
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--gui', action='store_true',
                        help="Show PyBullet GUI")
    parser.add_argument('--output', type=str, default='eval_results.csv')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    # Update global config (used by environments)
    import config
    config.ENV_TYPE = args.env_type
    config.GUI = args.gui

    # ---------------- CREATE BASE ENVIRONMENT ----------------
    base_env = make_base_env(args.hybrid, args.env_type, args.gui)
    if args.algorithm in ['cnn-gru-sac', 'lstm-sac', 'tx-sac']:
        # Wrap with TemporalObservationWrapper to stack history
        base_env = TemporalObservationWrapper(base_env)
    env = DummyVecEnv([lambda: base_env])

    # ---------------- LOAD NORMALIZATION STATS ----------------
    norm_path = os.path.join(os.path.dirname(args.model_path), "vecnormalize.pkl")
    if os.path.exists(norm_path):
        try:
            env = VecNormalize.load(norm_path, env)
            env.training = False      # disable updates
            env.norm_reward = False   # we want raw rewards for metrics
            print(f"Loaded VecNormalize stats from {norm_path}")
        except Exception as e:
            print(f"Warning: VecNormalize load failed ({e}). Running without normalization.")
    else:
        print("Warning: No VecNormalize stats found.")

    # ---------------- LOAD TRAINED MODEL ----------------
    if args.algorithm in ['sac', 'cnn-gru-sac', 'lstm-sac', 'tx-sac']:
        model = SAC.load(args.model_path, env=env)
    else:  # PPO
        model = PPO.load(args.model_path, env=env)

    metrics = []

    # ---------------- EVALUATION LOOP ----------------
    for ep in range(args.num_episodes):
        obs = env.reset()
        original_env = unwrap_original_env(env)

        # Record start and goal positions
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
            # Get action from policy (deterministic for evaluation)
            action, _ = model.predict(obs, deterministic=True)

            # Record angular velocity for smoothness metric
            ang_vel = action[0][1] * ANGULAR_VEL
            angular_velocities.append(ang_vel)

            # ---------- CRITICAL FIX: handle both 4-value and 5-value step ----------
            step_result = env.step(action)
            if len(step_result) == 5:          # Gymnasium style
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:                              # Old Gym style (4 values)
                obs, reward, done, info = step_result
            # ------------------------------------------------------------------------

            steps += 1

            # Get current robot position (x, y)
            curr_pos = np.array(original_env.get_robot_pose()[:2])

            # Path length (Euclidean distance travelled)
            step_dist = np.linalg.norm(curr_pos - prev_pos)
            path_length += step_dist
            prev_pos = curr_pos

            # Stuck detection: if robot moves less than 1 cm for 50 steps
            if step_dist < 0.01:
                stuck_counter += 1
            else:
                stuck_counter = 0
            if stuck_counter >= 50:
                stuck_events += 1
                stuck_counter = 0

            # LiDAR clearance (minimum distance to any obstacle)
            raw_lidar = np.array(original_env.get_lidar(), dtype=np.float32)
            raw_lidar = np.clip(raw_lidar, 0.0, 1.0)
            min_dist = np.min(raw_lidar) * LIDAR_RANGE
            min_distances.append(min_dist)

   
            # Collision detection – info is a list of dicts (VecEnv)
                       
            if isinstance(info, list) and len(info) > 0:
                # For DummyVecEnv, info[0] contains the dict for the single environment
                env_info = info[0]
                if env_info.get('collision', False):
                    collision = True
                    # Also check dynamic collision if needed
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

        # Success if reached goal without collision
        if not collision:
            dist_to_goal = np.linalg.norm(curr_pos - goal_pos)
            if dist_to_goal < GOAL_THRESHOLD:
                success = True

        # Store episode results
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

        print(f"Episode {ep}: success={success}, steps={steps}, "
              f"time={time_to_goal:.2f}s, eff={path_efficiency:.2f}, "
              f"jerk={jerk:.3f}, clearance={min_clearance:.2f}")

    # ---------------- SAVE RESULTS TO CSV ----------------
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    # ---------------- PRINT SUMMARY ----------------
    success_rate = np.mean([m['success'] for m in metrics]) * 100
    collision_rate = np.mean([m['collision'] for m in metrics]) * 100
    dynamic_collision_rate = np.mean([m['dynamic_collision'] for m in metrics]) * 100
    avg_steps = np.mean([m['steps'] for m in metrics])
    avg_eff = np.mean([m['path_efficiency'] for m in metrics])

    print("\n===== Evaluation Summary =====")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Collision rate: {collision_rate:.1f}% (dynamic: {dynamic_collision_rate:.1f}%)")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Path efficiency: {avg_eff:.2f}")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()