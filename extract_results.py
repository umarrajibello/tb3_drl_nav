#!/usr/bin/env python3
import json
import csv
import subprocess
import sys

SWEEP_ID = "umarrajibello-harbin-institute-of-technology/tb3_drl_nav/4ofkgo57"

# Fetch data
result = subprocess.run(
    f"wandb runs --sweep {SWEEP_ID} --format json",
    shell=True, capture_output=True, text=True
)

if result.returncode != 0:
    print("Failed to fetch sweep data")
    print(result.stderr)
    sys.exit(1)

runs = json.loads(result.stdout)

# Filter finished runs
finished = [r for r in runs if r.get('state') == 'finished']

# Sort by success rate
finished.sort(key=lambda x: x.get('summary', {}).get('eval/success_rate', 0), reverse=True)

# Print to console
print("\n" + "="*70)
print("REWARD SWEEP RESULTS")
print("="*70)
print(f"{'Goal':<8} {'Progress':<10} {'Hybrid':<8} {'Success%':<10} {'Collision%':<10}")
print("-"*70)

for r in finished:
    config = r.get('config', {})
    summary = r.get('summary', {})
    goal = config.get('goal_reward', '?')
    prog = config.get('progress_weight', '?')
    hybrid = 'Yes' if config.get('hybrid') else 'No'
    success = summary.get('eval/success_rate', 0)
    collision = summary.get('eval/collision_rate', 0)
    print(f"{goal:<8} {prog:<10} {hybrid:<8} {success:<10.1f} {collision:<10.1f}")

# Save to CSV
with open('sweep_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['goal_reward', 'progress_weight', 'hybrid', 'success_rate', 'collision_rate', 'avg_steps'])
    for r in finished:
        config = r.get('config', {})
        summary = r.get('summary', {})
        writer.writerow([
            config.get('goal_reward', ''),
            config.get('progress_weight', ''),
            'hybrid' if config.get('hybrid') else 'end2end',
            summary.get('eval/success_rate', 0),
            summary.get('eval/collision_rate', 0),
            summary.get('eval/avg_steps', 0)
        ])

print("\n✅ Results saved to sweep_results.csv")

# Best configuration
if finished:
    best = finished[0]
    best_config = best.get('config', {})
    print("\n" + "="*70)
    print("BEST CONFIGURATION (use for final training)")
    print("="*70)
    print(f"  goal_reward: {best_config.get('goal_reward')}")
    print(f"  progress_weight: {best_config.get('progress_weight')}")
    print(f"  hybrid: {'true' if best_config.get('hybrid') else 'false'}")
    print(f"  collision_penalty: 150")
    print(f"  success_rate: {best.get('summary', {}).get('eval/success_rate', 0):.1f}%")
