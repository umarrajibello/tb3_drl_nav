#!/usr/bin/env python3
# run_sweep.py – Helper to launch and manage sweeps

import subprocess
import sys
import os

def print_banner(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def create_sweep(sweep_file):
    """Create a new sweep and return the sweep ID"""
    print_banner(f"Creating sweep from {sweep_file}")
    
    if not os.path.exists(sweep_file):
        print(f"❌ File not found: {sweep_file}")
        return None
    
    result = subprocess.run(
        f"wandb sweep {sweep_file}",
        shell=True,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    # Extract sweep ID
    for line in result.stdout.split('\n'):
        if 'wandb agent' in line:
            sweep_id = line.split('wandb agent ')[-1].strip()
            print(f"\n✅ Sweep created! ID: {sweep_id}")
            return sweep_id
    
    print("❌ Could not extract sweep ID")
    return None

def main():
    print_banner("W&B Sweep Launcher")
    
    print("Select sweep type:")
    print("  1. Quick Test (2 runs, ~30 min)")
    print("  2. Reward Ablation (16 runs, ~9 hours)")
    print("  3. Hyperparameter (Bayesian, ~14 hours)")
    print("  4. Architecture Comparison (12 runs, ~5 hours)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    sweep_files = {
        '1': 'sweep_test.yaml',
        '2': 'sweep_reward.yaml',
        '3': 'sweep_hyperparam.yaml',
        '4': 'sweep_architecture.yaml',
    }
    
    sweep_file = sweep_files.get(choice)
    if not sweep_file:
        print("Invalid choice")
        return
    
    sweep_id = create_sweep(sweep_file)
    if not sweep_id:
        return
    
    count = input(f"\nNumber of runs to execute (default: all): ").strip()
    count_cmd = f" --count {count}" if count else ""
    
    print(f"\nRun this command to start the agent:")
    print(f"\n  wandb agent {sweep_id}{count_cmd}")
    print(f"\nOr run on multiple GPUs:")
    print(f"  CUDA_VISIBLE_DEVICES=0 wandb agent {sweep_id}{count_cmd}")
    print(f"  CUDA_VISIBLE_DEVICES=1 wandb agent {sweep_id}{count_cmd}")
    print(f"  CUDA_VISIBLE_DEVICES=2 wandb agent {sweep_id}{count_cmd}")
    print(f"  CUDA_VISIBLE_DEVICES=3 wandb agent {sweep_id}{count_cmd}")

if __name__ == "__main__":
    main()
