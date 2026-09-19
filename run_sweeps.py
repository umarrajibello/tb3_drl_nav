#!/usr/bin/env python3
# run_sweep.py – Launch and manage W&B sweeps
# Usage: python run_sweep.py

import subprocess
import sys
import os
import time

def print_banner(text):
    """Print a banner for visual separation"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def create_sweep(sweep_file):
    """Create a new sweep and return the sweep ID"""
    print_banner(f"Creating sweep from {sweep_file}")
    
    # Check if file exists
    if not os.path.exists(sweep_file):
        print(f"❌ Sweep file not found: {sweep_file}")
        return None
    
    result = subprocess.run(
        f"wandb sweep {sweep_file}",
        shell=True,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Extract sweep ID from output
    for line in result.stdout.split('\n'):
        if 'wandb agent' in line:
            sweep_id = line.split('wandb agent ')[-1].strip()
            print(f"\n✅ Sweep created! ID: {sweep_id}")
            return sweep_id
    
    print("❌ Could not extract sweep ID from output")
    return None

def run_agent(sweep_id, count=10):
    """Run a sweep agent for a specified number of runs"""
    print_banner(f"Running agent for {count} runs")
    
    cmd = f"wandb agent {sweep_id} --count {count}"
    print(f"Command: {cmd}")
    print("\nMonitoring progress...\n")
    
    subprocess.run(cmd, shell=True)

def list_sweeps(project_name):
    """List all sweeps in a project"""
    print_banner(f"Listing sweeps in {project_name}")
    result = subprocess.run(
        f"wandb sweep list -p {project_name}",
        shell=True,
        capture_output=True,
        text=True
    )
    print(result.stdout)

def get_sweep_status(sweep_id):
    """Get status of a sweep"""
    print_banner(f"Status for sweep {sweep_id}")
    result = subprocess.run(
        f"wandb sweep status {sweep_id}",
        shell=True,
        capture_output=True,
        text=True
    )
    print(result.stdout)

def fetch_results(sweep_id, output_file="sweep_results.csv"):
    """Fetch results from a completed sweep using Python API"""
    print_banner(f"Fetching results for sweep {sweep_id}")
    
    try:
        import wandb
        api = wandb.Api()
        
        # Get all runs in the sweep
        runs = api.runs(f"your-username/tb3-nav-reward-ablation/sweeps/{sweep_id}")
        
        results = []
        for run in runs:
            config = run.config
            summary = run.summary
            
            results.append({
                'run_name': run.name,
                'state': run.state,
                'goal_reward': config.get('goal_reward', 'N/A'),
                'collision_penalty': config.get('collision_penalty', 'N/A'),
                'progress_weight': config.get('progress_weight', 'N/A'),
                'away_penalty': config.get('away_penalty', 'N/A'),
                'learning_rate': config.get('learning_rate', 'N/A'),
                'batch_size': config.get('batch_size', 'N/A'),
                'success_rate': summary.get('eval/success_rate', 0),
                'collision_rate': summary.get('eval/collision_rate', 0),
                'avg_steps': summary.get('eval/avg_steps', 0),
                'best_reward': summary.get('episode_reward_mean', 0),
            })
        
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"✅ Results saved to {output_file}")
        print(f"\nFound {len(results)} completed runs")
        
        # Print summary
        if not results.empty:
            print("\n" + "="*70)
            print("SUMMARY STATISTICS")
            print("="*70)
            print(df.groupby(['goal_reward', 'collision_penalty'])['success_rate'].agg(['mean', 'std', 'count']))
        
        return df
        
    except ImportError:
        print("❌ pandas not installed. Install with: pip install pandas")
        return None
    except Exception as e:
        print(f"❌ Error fetching results: {e}")
        return None

def main():
    print_banner("W&B Sweep Launcher for TB3 Navigation Research")
    
    print("Select sweep type:")
    print("  1. Reward Ablation Study (grid search)")
    print("  2. Hyperparameter Sensitivity (bayesian)")
    print("  3. Architecture Comparison (grid)")
    print("  4. List existing sweeps")
    print("  5. Check sweep status")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    sweep_files = {
        '1': 'sweep_reward.yaml',
        '2': 'sweep_hyperparams.yaml',
        '3': 'sweep_architecture.yaml',
    }
    
    if choice == '1':
        print("\n=== Reward Ablation Study ===")
        print("This will test combinations of:")
        print("  - goal_reward: 50, 100, 200")
        print("  - collision_penalty: 30, 50, 80")
        print("  - progress_weight: 1.0, 2.0, 4.0")
        print("  - away_penalty: 2.0, 3.0, 5.0")
        print(f"  - Total combinations: 3×3×3×3 = 81")
        print(f"  - With 3 seeds: 243 runs")
        print("  - Time estimate: ~36 hours with 8 parallel agents")
        
    elif choice == '2':
        print("\n=== Hyperparameter Sensitivity ===")
        print("This will use Bayesian optimization for:")
        print("  - learning_rate: 1e-5 to 1e-3")
        print("  - batch_size: 128, 256, 512")
        print("  - gamma: 0.95, 0.99, 0.995")
        print("  - tau: 0.001, 0.005, 0.01")
        print("  - Total runs: 50 (Bayesian sampling)")
        print("  - Time estimate: ~4 hours with 4 parallel agents")
        
    elif choice == '3':
        print("\n=== Architecture Comparison ===")
        print("This will compare:")
        print("  - algorithms: sac, lstm-sac, tx-sac, cnn-gru-sac")
        print("  - hybrid vs end-to-end")
        print(f"  - Total combinations: 4×2×3 = 24 runs")
        print("  - Time estimate: ~6 hours with 4 parallel agents")
        
    elif choice == '4':
        project = input("\nEnter project name (default: tb3-nav-research): ").strip()
        if not project:
            project = "tb3-nav-research"
        list_sweeps(project)
        return
        
    elif choice == '5':
        sweep_id = input("\nEnter sweep ID: ").strip()
        get_sweep_status(sweep_id)
        return
    
    else:
        print("Invalid choice")
        return
    
    sweep_file = sweep_files.get(choice)
    if not sweep_file:
        return
    
    # Create sweep
    sweep_id = create_sweep(sweep_file)
    if not sweep_id:
        return
    
    print(f"\n✅ Sweep created successfully!")
    print(f"   Sweep ID: {sweep_id}")
    print(f"   View online: https://wandb.ai/your-username/")
    
    # Ask for run configuration
    count = input("\nNumber of runs to execute (default: 10): ").strip()
    count = int(count) if count else 10
    
    if count > 0:
        print(f"\nStarting agent for {count} runs...")
        run_agent(sweep_id, count=count)
    
    # Ask if user wants to download results
    if input("\nDownload results as CSV? (y/n): ").lower() == 'y':
        project_name = input("Project name (default: tb3-nav-reward-ablation): ").strip()
        if not project_name:
            project_name = "tb3-nav-reward-ablation"
        fetch_results(sweep_id, "sweep_results.csv")
    
    print_banner("Sweep completed!")

if __name__ == "__main__":
    main()
