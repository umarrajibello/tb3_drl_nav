import torch
from gymnasium import spaces

from policies.transformer_extractor_compact_circular import (
    TransformerFeatureExtractor,
)


# ============================================================
# Configuration
# ============================================================

HISTORY_LEN = 8
OBS_DIM = 41
LIDAR_DIM = 36
NON_LIDAR_DIM = 5


# ============================================================
# Build extractor
# ============================================================

observation_space = spaces.Box(
    low=-float("inf"),
    high=float("inf"),
    shape=(HISTORY_LEN * OBS_DIM,),
    dtype=float,
)

extractor = TransformerFeatureExtractor(
    observation_space=observation_space,
    history_len=HISTORY_LEN,
    lidar_dim=LIDAR_DIM,
    non_lidar_dim=NON_LIDAR_DIM,
    cnn_channels=16,
    d_model=64,
    nhead=4,
    num_layers=2,
    dropout=0.1,
    features_dim=256,
)

extractor.eval()


# ============================================================
# Create controlled history
# ============================================================

# Start with a realistic-ish LiDAR scan.
base_lidar = torch.ones(LIDAR_DIM)

# Put an obstacle in front.
base_lidar[0] = 0.15
base_lidar[1] = 0.20
base_lidar[35] = 0.20


# State:
# [v, w, target_dx, target_dy, target_distance]
base_state = torch.tensor(
    [0.10, 0.20, 1.00, 0.50, 1.12],
    dtype=torch.float32,
)


# ============================================================
# History A: identical observation repeated 8 times
# ============================================================

base_frame = torch.cat(
    [base_lidar, base_state]
)

history_repeated = base_frame.repeat(
    HISTORY_LEN
).unsqueeze(0)


# ============================================================
# History B: changing history
# ============================================================

frames = []

for t in range(HISTORY_LEN):

    lidar = base_lidar.clone()

    state = base_state.clone()

    # Simulate the robot turning progressively.
    state[1] = 0.20 + 0.15 * t

    # Simulate forward motion changing.
    state[0] = 0.05 + 0.02 * t

    # Move the target representation gradually.
    state[2] = 1.00 - 0.04 * t
    state[3] = 0.50 + 0.03 * t

    # Move the obstacle angularly through the scan.
    lidar = torch.roll(
        lidar,
        shifts=t,
        dims=0,
    )

    frame = torch.cat(
        [lidar, state]
    )

    frames.append(frame)

history_changing = torch.stack(
    frames,
    dim=0,
).reshape(1, HISTORY_LEN * OBS_DIM)


# ============================================================
# Forward pass
# ============================================================

with torch.no_grad():

    output_repeated = extractor(
        history_repeated
    )

    output_changing = extractor(
        history_changing
    )


# ============================================================
# Results
# ============================================================

print("=" * 70)
print("COMPACT CIRCULAR TRANSFORMER — TEMPORAL HISTORY TEST")
print("=" * 70)

print()
print("History shape:")
print("  repeated :", tuple(history_repeated.shape))
print("  changing :", tuple(history_changing.shape))

print()
print("Output shape:")
print("  repeated :", tuple(output_repeated.shape))
print("  changing :", tuple(output_changing.shape))

print()
print("Finite:")
print(
    "  repeated :",
    torch.isfinite(output_repeated).all().item(),
)
print(
    "  changing :",
    torch.isfinite(output_changing).all().item(),
)


# ============================================================
# Compare temporal representations
# ============================================================

feature_difference = torch.norm(
    output_repeated - output_changing
).item()

print()
print("=" * 70)
print("TEMPORAL REPRESENTATION DIFFERENCE")
print("=" * 70)

print(
    f"Repeated history vs changing history: "
    f"{feature_difference:.8f}"
)


# ============================================================
# Frame-by-frame inspection
# ============================================================

print()
print("=" * 70)
print("HISTORY STATE CHANGES")
print("=" * 70)

for t in range(HISTORY_LEN):

    frame = history_changing[
        0,
        t * OBS_DIM:(t + 1) * OBS_DIM
    ]

    state = frame[LIDAR_DIM:]

    print(
        f"t={t}: "
        f"v={state[0].item():.4f}, "
        f"w={state[1].item():.4f}, "
        f"dx={state[2].item():.4f}, "
        f"dy={state[3].item():.4f}, "
        f"d={state[4].item():.4f}"
    )


# ============================================================
# Final assertions
# ============================================================

assert history_repeated.shape == (1, 328)
assert history_changing.shape == (1, 328)

assert output_repeated.shape == (1, 256)
assert output_changing.shape == (1, 256)

assert torch.isfinite(output_repeated).all()
assert torch.isfinite(output_changing).all()

# The changing history should not collapse to exactly the same
# representation as the repeated history.
assert feature_difference > 1e-6

print()
print("=" * 70)
print("TEMPORAL HISTORY TEST: PASS")
print("=" * 70)
