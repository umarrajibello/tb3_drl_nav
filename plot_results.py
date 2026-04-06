import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("rl_aggregated_summary.csv")

# Filter for best training (curriculum) and both representations
df_best = df[df['training'] == 'curriculum']

# Success rate bar chart
plt.figure(figsize=(10,6))
sns.barplot(data=df_best, x='algorithm', y='success_rate_mean', hue='representation', 
            errorbar=('ci', 95), capsize=0.1)
plt.title("Success Rate on Dense Environment (Curriculum Training)")
plt.ylabel("Success Rate (%)")
plt.savefig("success_rate_bars.png", dpi=300)
plt.show()

# Path efficiency bar chart
plt.figure(figsize=(10,6))
sns.barplot(data=df_best, x='algorithm', y='path_efficiency_mean', hue='representation',
            errorbar=('ci', 95), capsize=0.1)
plt.title("Path Efficiency on Dense Environment (Curriculum Training)")
plt.ylabel("Path Efficiency (higher is better)")
plt.savefig("path_efficiency_bars.png", dpi=300)
plt.show()

# Jerk (smoothness) – lower is better
plt.figure(figsize=(10,6))
sns.barplot(data=df_best, x='algorithm', y='jerk_mean', hue='representation',
            errorbar=('ci', 95), capsize=0.1)
plt.title("Motion Smoothness (Jerk) on Dense Environment")
plt.ylabel("Jerk (rad/s², lower is better)")
plt.savefig("jerk_bars.png", dpi=300)
plt.show()