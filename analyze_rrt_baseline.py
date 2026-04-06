import pandas as pd
import numpy as np
import glob

csv_files = glob.glob("rrt_baseline_*.csv")

for file in csv_files:
    env = file.replace("rrt_baseline_", "").replace(".csv", "")
    df = pd.read_csv(file)
    
    success_rate = df['success'].mean() * 100
    collision_rate = df['collision'].mean() * 100
    avg_steps = df['steps'].mean()
    avg_path_length = df['path_length'].mean()
    avg_path_efficiency = df['path_efficiency'].mean()
    avg_min_clearance = df['min_clearance'].mean()
    avg_stuck_events = df['stuck_events'].mean()
    
    print(f"\n===== {env.upper()} Environment =====")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Collision rate: {collision_rate:.1f}%")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average path length: {avg_path_length:.3f} m")
    print(f"Average path efficiency: {avg_path_efficiency:.3f}")
    print(f"Average min clearance: {avg_min_clearance:.3f} m")
    print(f"Average stuck events: {avg_stuck_events:.2f}")