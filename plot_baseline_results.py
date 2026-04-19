import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
methods = ['dwa', 'purepursuit']
envs = ['static', 'mixed', 'dense']

# Mapping from filename patterns
file_patterns = {
    'dwa': 'rrt_dwa_{env}.csv',
    'purepursuit': 'rrt_purepursuit_{env}.csv'
}

# ------------------------------------------------------------
# Load and aggregate data
# ------------------------------------------------------------
results = []

for method in methods:
    for env in envs:
        fname = file_patterns[method].format(env=env)
        if not os.path.exists(fname):
            print(f"Warning: {fname} not found, skipping.")
            continue
        df = pd.read_csv(fname)
        
        # Remove rows where RRT failed (steps == 0 or time_to_goal == 0) - optional
        # df = df[df['steps'] > 0]
        
        total = len(df)
        success = df['success'].sum()
        collision = df['collision'].sum()
        
        # For steps (success only)
        success_df = df[df['success'] == True]
        avg_steps_success = success_df['steps'].mean() if len(success_df) > 0 else np.nan
        
        # For path efficiency (success only) – currently bugged, but we keep
        avg_eff_success = success_df['path_efficiency'].mean() if len(success_df) > 0 else np.nan
        
        results.append({
            'Method': method.upper(),
            'Environment': env.capitalize(),
            'Success Rate (%)': (success / total) * 100,
            'Collision Rate (%)': (collision / total) * 100,
            'Avg Steps (success)': avg_steps_success,
            'Avg Efficiency (success)': avg_eff_success,
            'Total Episodes': total
        })

# Create DataFrame
df_results = pd.DataFrame(results)
print("\n=== Classical Baseline Summary ===")
print(df_results.to_string(index=False))

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
# Set style
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Classical Baseline Performance', fontsize=14)

# 1. Bar chart – Success rate per method & environment
ax1 = axes[0, 0]
x = np.arange(len(envs))
width = 0.35
for i, method in enumerate(methods):
    subset = df_results[df_results['Method'] == method.upper()]
    success_vals = [subset[subset['Environment'] == env.capitalize()]['Success Rate (%)'].values[0] for env in envs]
    ax1.bar(x + i*width, success_vals, width, label=method.upper())
ax1.set_xticks(x + width/2)
ax1.set_xticklabels(envs)
ax1.set_ylabel('Success Rate (%)')
ax1.set_title('Success Rate')
ax1.legend()
ax1.grid(True, axis='y', alpha=0.7)

# 2. Bar chart – Collision rate
ax2 = axes[0, 1]
for i, method in enumerate(methods):
    coll_vals = [subset[subset['Environment'] == env.capitalize()]['Collision Rate (%)'].values[0] for env in envs]
    ax2.bar(x + i*width, coll_vals, width, label=method.upper())
ax2.set_xticks(x + width/2)
ax2.set_xticklabels(envs)
ax2.set_ylabel('Collision Rate (%)')
ax2.set_title('Collision Rate')
ax2.legend()
ax2.grid(True, axis='y', alpha=0.7)

# 3. Scatter plot – Success vs Collision (one point per method+env)
ax3 = axes[1, 0]
markers = {'DWA': 'o', 'PUREPURSUIT': 's'}
colors = {'DWA': 'blue', 'PUREPURSUIT': 'red'}
for method in methods:
    subset = df_results[df_results['Method'] == method.upper()]
    for _, row in subset.iterrows():
        ax3.scatter(row['Success Rate (%)'], row['Collision Rate (%)'],
                    label=f"{row['Method']} ({row['Environment']})",
                    marker=markers[method.upper()], c=colors[method.upper()], s=100)
ax3.set_xlabel('Success Rate (%)')
ax3.set_ylabel('Collision Rate (%)')
ax3.set_title('Success vs Collision')
ax3.grid(True, alpha=0.5)
# Add ideal region annotation
ax3.axhline(y=30, color='gray', linestyle='--', alpha=0.5, label='Target collision <30%')
ax3.axvline(x=37, color='gray', linestyle='--', alpha=0.5, label='Pure pursuit static success')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 4. Box plot – Steps for successful episodes (if data available)
ax4 = axes[1, 1]
steps_data = []
labels = []
for method in methods:
    for env in envs:
        fname = file_patterns[method].format(env=env)
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            success_steps = df[df['success'] == True]['steps'].dropna()
            if len(success_steps) > 0:
                steps_data.append(success_steps)
                labels.append(f"{method.upper()}\n{env}")
if steps_data:
    bp = ax4.boxplot(steps_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['blue', 'red']*3):
        patch.set_facecolor(color)
    ax4.set_ylabel('Steps (successful episodes)')
    ax4.set_title('Success Episode Length')
    ax4.grid(True, axis='y', alpha=0.7)
else:
    ax4.text(0.5, 0.5, 'No successful episodes', ha='center', va='center')
    ax4.set_title('Steps (no data)')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('baseline_results.png', dpi=150)
plt.show()