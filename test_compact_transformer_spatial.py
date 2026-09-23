import torch
import torch.nn.functional as F
from gymnasium import spaces

from policies.transformer_extractor_compact_circular import (
    TransformerFeatureExtractor,
)


# ============================================================
# Configuration
# ============================================================

OBS_DIM = 41
LIDAR_DIM = 36
NON_LIDAR_DIM = 5
HISTORY_LEN = 8


# ============================================================
# Helpers
# ============================================================

def make_observation(lidar):
    """
    Create an 8-frame observation.

    All 8 frames contain the same synthetic LiDAR pattern.
    The 5 non-LiDAR features are held constant so that measured
    differences come from the LiDAR representation.
    """

    lidar = torch.tensor(lidar, dtype=torch.float32)

    # Same robot/waypoint state for every timestep.
    state = torch.tensor(
        [0.10, 0.20, 1.00, 0.00, 1.00],
        dtype=torch.float32,
    )

    frame = torch.cat([lidar, state])

    history = frame.repeat(HISTORY_LEN)

    return history.unsqueeze(0)


def make_lidar_empty():
    """
    No obstacle:
    normalized LiDAR distance = 1.0 everywhere.
    """
    return [1.0] * LIDAR_DIM


def make_lidar_single(index, value=0.10):
    """
    One obstacle at a specified beam.
    """
    lidar = make_lidar_empty()
    lidar[index] = value
    return lidar


def make_lidar_cluster(start, width=3, value=0.10):
    """
    Obstacle cluster covering several adjacent beams.
    """
    lidar = make_lidar_empty()

    for i in range(width):
        lidar[(start + i) % LIDAR_DIM] = value

    return lidar


def feature_distance(a, b):
    return torch.norm(a - b).item()


def extract_lidar_representation(extractor, observation):
    """
    Extract the 64-D LiDAR representation before fusion with
    the 5-D robot/waypoint state.

    We deliberately expose the internal spatial representation
    here for architectural diagnosis.
    """

    with torch.no_grad():

        batch_size = observation.shape[0]

        x = observation.reshape(
            batch_size,
            HISTORY_LEN,
            OBS_DIM,
        )

        lidar = x[:, :, :LIDAR_DIM]

        lidar = lidar.reshape(
            batch_size * HISTORY_LEN,
            1,
            LIDAR_DIM,
        )

        lidar_features = extractor.lidar_cnn(lidar)

        lidar_features = extractor.lidar_projection(
            lidar_features
        )

        lidar_features = lidar_features.reshape(
            batch_size,
            HISTORY_LEN,
            64,
        )

        # Mean over the identical eight frames.
        return lidar_features.mean(dim=1)


def extract_full_output(extractor, observation):
    with torch.no_grad():
        return extractor(observation)


# ============================================================
# Model
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
# Test patterns
# ============================================================

patterns = {
    "empty": make_lidar_empty(),

    "front": make_lidar_single(0),

    "left": make_lidar_single(9),

    "rear": make_lidar_single(18),

    "right": make_lidar_single(27),

    # Boundary pair: beams 35 and 0 are physically adjacent.
    "wrap": make_lidar_single(35),

    "left_cluster": make_lidar_cluster(8, width=3),

    "right_cluster": make_lidar_cluster(26, width=3),

    # A cluster crossing the circular boundary.
    "boundary_cluster": make_lidar_cluster(35, width=3),
}


# ============================================================
# Generate representations
# ============================================================

representations = {}

print("=" * 70)
print("COMPACT CIRCULAR CNN — ANGULAR REPRESENTATION TEST")
print("=" * 70)

for name, lidar in patterns.items():

    obs = make_observation(lidar)

    representation = extract_lidar_representation(
        extractor,
        obs,
    )

    representations[name] = representation

    print(
        f"{name:20s} "
        f"shape={tuple(representation.shape)} "
        f"finite={torch.isfinite(representation).all().item()}"
    )


# ============================================================
# Pairwise angular-position tests
# ============================================================

pairs = [
    ("empty", "front"),
    ("front", "left"),
    ("front", "right"),
    ("left", "right"),
    ("left", "rear"),
    ("right", "rear"),
    ("front", "wrap"),
    ("left_cluster", "right_cluster"),
]


print()
print("=" * 70)
print("ANGULAR POSITION DIFFERENCES")
print("=" * 70)

for a, b in pairs:

    distance = feature_distance(
        representations[a],
        representations[b],
    )

    print(
        f"{a:20s} vs {b:20s}: "
        f"{distance:.8f}"
    )


# ============================================================
# Circular boundary test
# ============================================================

print()
print("=" * 70)
print("CIRCULAR BOUNDARY TEST")
print("=" * 70)

boundary_single = feature_distance(
    representations["front"],
    representations["wrap"],
)

boundary_cluster = feature_distance(
    representations["left_cluster"],
    representations["boundary_cluster"],
)

print(
    f"front vs beam-35: "
    f"{boundary_single:.8f}"
)

print(
    f"left-cluster vs boundary-cluster: "
    f"{boundary_cluster:.8f}"
)


# ============================================================
# Cyclic shift test at the LiDAR level
# ============================================================

print()
print("=" * 70)
print("CYCLIC SHIFT TEST")
print("=" * 70)


def circular_shift_lidar(lidar, shift):
    tensor = torch.tensor(
        lidar,
        dtype=torch.float32,
    )

    return torch.roll(
        tensor,
        shifts=shift,
        dims=0,
    ).tolist()


base_lidar = make_lidar_cluster(
    start=4,
    width=4,
)

base_obs = make_observation(base_lidar)

base_rep = extract_lidar_representation(
    extractor,
    base_obs,
)


for shift in [1, 3, 6, 9, 18]:

    shifted_lidar = circular_shift_lidar(
        base_lidar,
        shift,
    )

    shifted_obs = make_observation(
        shifted_lidar
    )

    shifted_rep = extract_lidar_representation(
        extractor,
        shifted_obs,
    )

    difference = feature_distance(
        base_rep,
        shifted_rep,
    )

    print(
        f"shift={shift:2d} beams: "
        f"feature difference={difference:.8f}"
    )


# ============================================================
# Complete Transformer output test
# ============================================================

print()
print("=" * 70)
print("COMPLETE TRANSFORMER OUTPUT TEST")
print("=" * 70)

for name, lidar in patterns.items():

    obs = make_observation(lidar)

    output = extract_full_output(
        extractor,
        obs,
    )

    print(
        f"{name:20s} "
        f"shape={tuple(output.shape)} "
        f"finite={torch.isfinite(output).all().item()}"
    )


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("SPATIAL TEST COMPLETE")
print("=" * 70)

print(
    """
Interpretation:

1. Non-zero pairwise distances indicate that different angular
   LiDAR configurations produce different representations.

2. The circular boundary test checks that beam 35 is processed
   as an angular neighbor of beam 0.

3. The cyclic-shift test is diagnostic rather than a requirement
   that the complete encoder be invariant to rotation. The encoder
   should retain angular position, not erase it.

4. The complete Transformer output test confirms that the revised
   spatial representation remains compatible with the temporal
   Transformer.
"""
)
