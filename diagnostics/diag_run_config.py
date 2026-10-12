"""Verify actual run configurations from saved metadata."""
import json
import os
from pathlib import Path

MODELS_DIR = Path("./models")

if not MODELS_DIR.exists():
    print("No models directory found.")
    exit(1)

print(f"Scanning {MODELS_DIR} for run metadata...\n")

for run_dir in sorted(MODELS_DIR.iterdir()):
    if not run_dir.is_dir():
        continue

    meta_path = run_dir / "experiment_metadata.json"
    if not meta_path.exists():
        continue

    with open(meta_path) as f:
        meta = json.load(f)

    print(f"=== {run_dir.name} ===")
    print(f"  Algorithm:          {meta.get('training', {}).get('algorithm')}")
    print(f"  Env type:           {meta.get('environment', {}).get('env_type')}")
    print(f"  Hybrid:             {meta.get('environment', {}).get('hybrid')}")
    print(f"  Lookahead mode:     {meta.get('environment', {}).get('lookahead_mode')}")
    print(f"  Total timesteps:    {meta.get('training', {}).get('total_timesteps')}")
    print(f"  Seed:               {meta.get('training', {}).get('seed')}")
    print(f"  Goal distance:      {meta.get('environment', {}).get('goal_dist_min')} - "
          f"{meta.get('environment', {}).get('goal_dist_max')}")
    print(f"  Learning rate:      {meta.get('training', {}).get('learning_rate')}")
    print(f"  Batch size:         {meta.get('training', {}).get('batch_size')}")
    print(f"  Buffer size:        {meta.get('training', {}).get('buffer_size')}")
    print(f"  Entropy coef:       {meta.get('training', {}).get('ent_coef')}")
    print(f"  Reward progress:    {meta.get('reward', {}).get('progress_weight')}")
    print(f"  Reward clearance:   {meta.get('reward', {}).get('clearance_weight')}")
    print(f"  Reward step:        {meta.get('reward', {}).get('step_penalty')}")
    print(f"  Goal reward:        {meta.get('reward', {}).get('goal_reward')}")
    print(f"  Collision penalty:  {meta.get('reward', {}).get('collision_penalty')}")
    print()