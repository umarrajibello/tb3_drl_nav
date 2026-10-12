"""Aggregate failure categories from a motion-audit CSV."""
import csv
import sys
from collections import Counter

def analyze(csv_path):
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"No rows in {csv_path}")
        return

    terminations = Counter(row['termination_reason'] for row in rows)
    total = len(rows)

    print(f"\n=== {csv_path} ===")
    print(f"Total episodes: {total}")
    for cat, count in terminations.most_common():
        print(f"  {cat}: {count} ({100*count/total:.1f}%)")

    final_dist = [float(r['final_distance_to_goal']) for r in rows]
    path_len = [float(r['path_length']) for r in rows]
    clearance = [float(r['min_clearance']) for r in rows 
                 if r['min_clearance'] not in ('', 'nan')]
    low_motion = [float(r['low_motion_fraction']) for r in rows]
    realized_v = [float(r['mean_realized_linear_velocity']) for r in rows]

    print(f"  Final goal distance:  mean {sum(final_dist)/len(final_dist):.3f} m")
    print(f"  Path length:          mean {sum(path_len)/len(path_len):.3f} m")
    if clearance:
        print(f"  Min clearance:        mean {sum(clearance)/len(clearance):.3f} m")
    print(f"  Low-motion fraction:  mean {sum(low_motion)/len(low_motion):.3f}")
    print(f"  Realized linear v:    mean {sum(realized_v)/len(realized_v):.3f} m/s")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diag_failure_categories.py <motion_audit_csv> [more_csvs...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        analyze(path)