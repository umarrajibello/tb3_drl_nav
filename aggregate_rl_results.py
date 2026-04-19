import pandas as pd
import glob
import re

results = []
for f in glob.glob("eval_results/*.csv"):
    name = f.split("/")[-1].replace(".csv", "")
    # Parse filename (expected format: dense_<algo>_<rep>_seed<seed>_<training>_final)
    parts = name.split('_')
    if len(parts) < 6:
        continue
    algo = parts[1]
    rep = parts[2]
    seed = int(parts[3].replace('seed', ''))
    training = parts[4]
    df = pd.read_csv(f)
    success_rate = df['success'].mean() * 100
    collision_rate = df['collision'].mean() * 100
    path_efficiency = df['path_efficiency'].mean()
    jerk = df['smoothness_jerk'].mean()
    results.append({
        'algorithm': algo,
        'representation': rep,
        'seed': seed,
        'training': training,
        'success_rate_mean': success_rate,
        'collision_rate_mean': collision_rate,
        'path_efficiency_mean': path_efficiency,
        'jerk_mean': jerk
    })

df_out = pd.DataFrame(results)
df_out.to_csv("rl_aggregated_summary.csv", index=False)
print("Created rl_aggregated_summary.csv")
