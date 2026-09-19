import pandas as pd
import glob
import re
import numpy as np

# Find all evaluation CSV files
csv_files = glob.glob("eval_results/*.csv")

results = []

for file in csv_files:
    # Extract info from filename
    name = file.split("/")[-1].replace(".csv", "")
    # Example: dense_sac_end2end_seed42_curriculum_final
    parts = name.split('_')
    # Expected: [dense, algorithm, representation, seedX, training, final]
    # But algorithm may be 'lstm-sac' or 'tx-sac' which have dashes, so careful
    # Simpler: use regex or manual parsing
    try:
        # Assume format: dense_{algo}_{rep}_seed{seed}_{training}_final
        # Algo can be sac, ppo, lstm-sac, tx-sac
        # rep: end2end or hybrid
        if 'lstm-sac' in name:
            algo = 'lstm-sac'
            rest = name.replace('dense_lstm-sac_', '')
        elif 'tx-sac' in name:
            algo = 'tx-sac'
            rest = name.replace('dense_tx-sac_', '')
        elif 'sac' in name:
            algo = 'sac'
            rest = name.replace('dense_sac_', '')
        elif 'ppo' in name:
            algo = 'ppo'
            rest = name.replace('dense_ppo_', '')
        else:
            continue
        
        # rest format: end2end_seed42_curriculum_final or hybrid_seed42_curriculum_final
        parts_rest = rest.split('_')
        rep = parts_rest[0]  # end2end or hybrid
        seed_part = parts_rest[1]  # seed42
        seed = int(seed_part.replace('seed', ''))
        training = parts_rest[2]  # curriculum or independent
        
    except Exception as e:
        print(f"Skipping {name}: {e}")
        continue
    
    # Read CSV
    df = pd.read_csv(file)
    
    # Compute metrics (mean over 100 episodes)
    success_rate = df['success'].mean() * 100
    collision_rate = df['collision'].mean() * 100
    avg_steps = df['steps'].mean()
    avg_path_length = df['path_length'].mean()
    avg_path_efficiency = df['path_efficiency'].mean()
    avg_jerk = df['smoothness_jerk'].mean()
    avg_min_clearance = df['min_clearance'].mean()
    avg_stuck = df['stuck_events'].mean()
    
    # Only compute time_to_goal for successful episodes (if any)
    success_df = df[df['success'] == True]
    if len(success_df) > 0:
        avg_time = success_df['time_to_goal'].mean()
    else:
        avg_time = None
    
    results.append({
        'Algorithm': algo,
        'Representation': rep,
        'Training': training,
        'Seed': seed,
        'Success (%)': success_rate,
        'Collision (%)': collision_rate,
        'Steps': avg_steps,
        'Path Efficiency': avg_path_efficiency,
        'Jerk': avg_jerk,
        'Min Clearance (m)': avg_min_clearance,
        'Stuck Events': avg_stuck,
        'Time to Goal (s)': avg_time
    })

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Compute mean and std across seeds for each configuration
summary = df_results.groupby(['Algorithm', 'Representation', 'Training']).agg({
    'Success (%)': ['mean', 'std'],
    'Collision (%)': ['mean', 'std'],
    'Path Efficiency': ['mean', 'std'],
    'Jerk': ['mean', 'std'],
    'Min Clearance (m)': ['mean', 'std'],
    'Stuck Events': ['mean', 'std']
}).round(2)

# Flatten column names
summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
summary.reset_index(inplace=True)

# Save to CSV
summary.to_csv("rl_aggregated_summary.csv", index=False)
print("Saved aggregated summary to rl_aggregated_summary.csv")
print("\n=== Summary (mean across seeds) ===")
print(summary.to_string())