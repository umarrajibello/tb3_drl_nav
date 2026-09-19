#!/usr/bin/env python3
"""Simple sweep launcher - prints the command you need to run"""

import subprocess
import sys
import os

def main():
    print("\n" + "="*60)
    print("  W&B Sweep Launcher (Simple)")
    print("="*60)
    
    print("\nSelect sweep file:")
    print("  1. sweep_test.yaml (quick test)")
    print("  2. sweep_reward.yaml (reward ablation)")
    print("  3. sweep_hyperparam.yaml (hyperparameter)")
    print("  4. sweep_architecture.yaml (architecture)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    files = {
        '1': 'sweep_test.yaml',
        '2': 'sweep_reward.yaml',
        '3': 'sweep_hyperparam.yaml',
        '4': 'sweep_architecture.yaml',
    }
    sweep_file = files.get(choice)
    if not sweep_file:
        print("Invalid choice")
        return
    
    if not os.path.exists(sweep_file):
        print(f"File not found: {sweep_file}")
        return
    
    print(f"\nRunning: wandb sweep {sweep_file}")
    result = subprocess.run(f"wandb sweep {sweep_file}", shell=True, text=True)
    
    if result.returncode != 0:
        print("\n❌ Sweep creation failed. Check the error above.")
        return
    
    print("\n" + "="*60)
    print("✅ Sweep created!")
    print("="*60)
    print("\nNow run the agent using the Sweep ID shown above.")
    print("Example:")
    print("  CUDA_VISIBLE_DEVICES=0 wandb agent <sweep-id>")
    print("\nOr run on multiple GPUs:")
    print("  CUDA_VISIBLE_DEVICES=0 wandb agent <sweep-id> --count 2")
    print("  CUDA_VISIBLE_DEVICES=1 wandb agent <sweep-id> --count 2")
    print("  ...")

if __name__ == "__main__":
    main()
