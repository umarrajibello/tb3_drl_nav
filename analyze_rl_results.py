import pandas as pd
import numpy as np
import glob
import re
from scipy.stats import ttest_rel, ttest_ind, mannwhitneyu

# ============================================================
# 1. LOAD ALL EVALUATION CSV FILES AND EXTRACT METADATA
# ============================================================

def parse_filename(filename):
    """Extract algorithm, representation, seed, training from filename."""
    name = filename.split("/")[-1].replace(".csv", "")
    # Expected format: dense_{algorithm}_{representation}_seed{seed}_{training}_final
    # Example: dense_sac_end2end_seed42_curriculum_final
    # For lstm-sac and tx-sac, they have dashes
    parts = name.split('_')
    # Remove 'dense' prefix
    if parts[0] != 'dense':
        return None
    # Algorithm is the second part, but may be 'lstm-sac' or 'tx-sac' (dash)
    # Handle by checking the next parts
    algo = parts[1]
    if algo == 'lstm':
        algo = 'lstm-sac'
        rep = parts[3]   # after 'sac'? Actually 'lstm-sac' is two parts? Let's do robust parsing
        # Simpler: use regex
    # I'll use regex for robustness
    import re
    pattern = r'dense_([a-z\-]+)_(end2end|hybrid)_seed(\d+)_(independent|curriculum)_final'
    match = re.match(pattern, name)
    if not match:
        return None
    algo = match.group(1)
    rep = match.group(2)
    seed = int(match.group(3))
    training = match.group(4)
    return {
        'algorithm': algo,
        'representation': rep,
        'seed': seed,
        'training': training,
        'file': filename
    }

# Find all CSV files
csv_files = glob.glob("eval_results/*.csv")
print(f"Found {len(csv_files)} CSV files.")

# Load data into a list of dictionaries
all_data = []
for f in csv_files:
    meta = parse_filename(f)
    if meta is None:
        print(f"Skipping unrecognized file: {f}")
        continue
    df = pd.read_csv(f)
    # Compute per-episode metrics already present; we need per-model aggregates
    # We'll store the full dataframe for later, or compute summary now
    # For t-tests we need per-episode success rates? Actually t-test on mean success per seed is fine.
    # But for paired t-test we need per-seed means. We'll compute per-seed means first.
    meta['df'] = df
    all_data.append(meta)

# Convert to DataFrame for easier grouping
records = []
for d in all_data:
    df = d['df']
    # Compute mean metrics for this seed
    success_rate = df['success'].mean() * 100
    collision_rate = df['collision'].mean() * 100
    dynamic_collision_rate = df['dynamic_collision'].mean() * 100
    avg_steps = df['steps'].mean()
    avg_path_length = df['path_length'].mean()
    avg_path_efficiency = df['path_efficiency'].mean()
    avg_jerk = df['smoothness_jerk'].mean()
    avg_min_clearance = df['min_clearance'].mean()
    avg_stuck = df['stuck_events'].mean()
    # Time to goal only for successful episodes
    success_df = df[df['success'] == True]
    if len(success_df) > 0:
        avg_time = success_df['time_to_goal'].mean()
        time_std = success_df['time_to_goal'].std()
    else:
        avg_time = None
        time_std = None
    
    records.append({
        'algorithm': d['algorithm'],
        'representation': d['representation'],
        'seed': d['seed'],
        'training': d['training'],
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'dynamic_collision_rate': dynamic_collision_rate,
        'steps': avg_steps,
        'path_length': avg_path_length,
        'path_efficiency': avg_path_efficiency,
        'jerk': avg_jerk,
        'min_clearance': avg_min_clearance,
        'stuck_events': avg_stuck,
        'time_to_goal': avg_time,
        'time_std': time_std
    })

df_summary_seed = pd.DataFrame(records)

# ============================================================
# 2. AGGREGATE OVER SEEDS (MEAN ± STD)
# ============================================================

agg = df_summary_seed.groupby(['algorithm', 'representation', 'training']).agg({
    'success_rate': ['mean', 'std'],
    'collision_rate': ['mean', 'std'],
    'dynamic_collision_rate': ['mean', 'std'],
    'steps': ['mean', 'std'],
    'path_length': ['mean', 'std'],
    'path_efficiency': ['mean', 'std'],
    'jerk': ['mean', 'std'],
    'min_clearance': ['mean', 'std'],
    'stuck_events': ['mean', 'std'],
    'time_to_goal': ['mean', 'std']
}).round(2)

# Flatten column names
agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
agg.reset_index(inplace=True)

# Save to CSV
agg.to_csv("rl_aggregated_summary.csv", index=False)
print("\n=== Aggregated Summary (mean ± std over seeds) ===")
print(agg.to_string())

# ============================================================
# 3. STATISTICAL TESTS
# ============================================================

print("\n\n=== STATISTICAL TESTS ===")

# Helper to get per-seed success rates for a configuration
def get_success_rates(algo, rep, training):
    subset = df_summary_seed[
        (df_summary_seed['algorithm'] == algo) &
        (df_summary_seed['representation'] == rep) &
        (df_summary_seed['training'] == training)
    ]
    if len(subset) == 0:
        return []
    return subset['success_rate'].tolist()

# 3.1 Curriculum vs Independent (paired by seed)
print("\n--- Curriculum vs Independent (paired t-test, success rate) ---")
for algo in ['sac', 'ppo', 'lstm-sac', 'tx-sac']:
    for rep in ['end2end', 'hybrid']:
        curr_rates = get_success_rates(algo, rep, 'curriculum')
        indep_rates = get_success_rates(algo, rep, 'independent')
        if len(curr_rates) >= 2 and len(indep_rates) >= 2:
            # Ensure same seeds order (they should be by construction)
            t_stat, p_val = ttest_rel(curr_rates, indep_rates)
            print(f"{algo} {rep}: t={t_stat:.3f}, p={p_val:.4f} {'(p<0.05)' if p_val<0.05 else '(ns)'}")
        else:
            print(f"{algo} {rep}: insufficient data")

# 3.2 Hybrid vs End2End (paired by seed)
print("\n--- Hybrid vs End2End (paired t-test, success rate) ---")
for algo in ['sac', 'ppo', 'lstm-sac', 'tx-sac']:
    for training in ['independent', 'curriculum']:
        hybrid_rates = get_success_rates(algo, 'hybrid', training)
        e2e_rates = get_success_rates(algo, 'end2end', training)
        if len(hybrid_rates) >= 2 and len(e2e_rates) >= 2:
            t_stat, p_val = ttest_rel(hybrid_rates, e2e_rates)
            print(f"{algo} {training}: t={t_stat:.3f}, p={p_val:.4f} {'(p<0.05)' if p_val<0.05 else '(ns)'}")
        else:
            print(f"{algo} {training}: insufficient data")

# 3.3 Tx-SAC vs others (unpaired t-test, best configuration: hybrid+curriculum)
print("\n--- Tx-SAC vs Others (unpaired t-test, hybrid+curriculum, success rate) ---")
best_config = ('hybrid', 'curriculum')
tx_rates = get_success_rates('tx-sac', best_config[0], best_config[1])
sac_rates = get_success_rates('sac', best_config[0], best_config[1])
ppo_rates = get_success_rates('ppo', best_config[0], best_config[1])
lstm_rates = get_success_rates('lstm-sac', best_config[0], best_config[1])

def unpaired_test(rate1, rate2, name1, name2):
    if len(rate1) >= 2 and len(rate2) >= 2:
        t_stat, p_val = ttest_ind(rate1, rate2, equal_var=False)
        print(f"{name1} vs {name2}: t={t_stat:.3f}, p={p_val:.4f} {'(p<0.05)' if p_val<0.05 else '(ns)'}")
    else:
        print(f"{name1} vs {name2}: insufficient data")

unpaired_test(tx_rates, sac_rates, "Tx-SAC", "SAC")
unpaired_test(tx_rates, ppo_rates, "Tx-SAC", "PPO")
unpaired_test(tx_rates, lstm_rates, "Tx-SAC", "LSTM-SAC")

# 3.4 Time-to-goal comparison (only for successful episodes)
print("\n--- Time-to-Goal (seconds) for successful episodes (mean ± std over seeds) ---")
for algo in ['sac', 'ppo', 'lstm-sac', 'tx-sac']:
    for rep in ['hybrid']:
        for training in ['curriculum']:
            subset = df_summary_seed[
                (df_summary_seed['algorithm'] == algo) &
                (df_summary_seed['representation'] == rep) &
                (df_summary_seed['training'] == training)
            ]
            if len(subset) == 0:
                continue
            times = subset['time_to_goal'].dropna().tolist()
            if len(times) > 0:
                mean_t = np.mean(times)
                std_t = np.std(times)
                print(f"{algo} {rep} {training}: {mean_t:.2f} ± {std_t:.2f} s (based on {len(times)} seeds)")
            else:
                print(f"{algo} {rep} {training}: no successful episodes in any seed")

# ============================================================
# 4. OPTIONAL: Generate LaTeX table (copy to clipboard or save)
# ============================================================
# You can uncomment to generate a LaTeX table for your thesis
# from pandas import DataFrame
# latex_table = agg[['algorithm','representation','training','success_rate_mean','success_rate_std','collision_rate_mean','path_efficiency_mean','jerk_mean']].copy()
# latex_table.columns = ['Algorithm','Rep','Training','Success (%)','±','Collision (%)','Path Eff.','Jerk']
# print(latex_table.to_latex(index=False))

print("\nAll analyses complete. Results saved to rl_aggregated_summary.csv")