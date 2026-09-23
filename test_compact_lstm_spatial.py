import torch
import gymnasium as gym

from policies.lstm_extractor import LSTMFeatureExtractor


HISTORY_LEN = 8
LIDAR_DIM = 36
NON_LIDAR_DIM = 5
FEATURES_DIM = 256


def make_obs(lidar):
    """
    Build one flattened temporal observation.

    Each timestep contains:
        36 LiDAR values
        5 non-LiDAR values:
        [v, w, dx, dy, distance]

    The same LiDAR pattern is repeated across the 8-frame history.
    """
    state = torch.tensor(
        [0.10, 0.20, 1.0, 0.5, 1.118],
        dtype=torch.float32,
    )

    frame = torch.cat(
        [
            torch.tensor(lidar, dtype=torch.float32),
            state,
        ]
    )

    history = frame.repeat(HISTORY_LEN)

    return history.unsqueeze(0)


def make_lidar(indices, value=0.10):
    """
    Create normalized LiDAR vector.

    0.0 = very close obstacle
    1.0 = maximum/no obstacle within effective range

    The exact absolute value is less important here than the
    angular location of the perturbation.
    """
    lidar = torch.ones(LIDAR_DIM, dtype=torch.float32)

    for idx in indices:
        lidar[idx % LIDAR_DIM] = value

    return lidar


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def l2(a, b):
    return torch.norm(a - b).item()


def main():

    obs_space = gym.spaces.Box(
        low=float("-inf"),
        high=float("inf"),
        shape=(HISTORY_LEN * (LIDAR_DIM + NON_LIDAR_DIM),),
        dtype=float,
    )

    model = LSTMFeatureExtractor(
        observation_space=obs_space,
        history_len=HISTORY_LEN,
        lidar_dim=LIDAR_DIM,
        non_lidar_dim=NON_LIDAR_DIM,
        features_dim=FEATURES_DIM,
    )

    model.eval()

    print("\n=== LSTM SPATIAL / ANGULAR SENSITIVITY TEST ===")
    print(f"Parameters: {count_parameters(model):,}")

    # ------------------------------------------------------------
    # Test patterns
    #
    # Beam convention follows the project LiDAR ordering:
    # beam 0 = front
    # beam 9 = left
    # beam 18 = rear
    # beam 27 = right
    # ------------------------------------------------------------

    patterns = {
        "empty": [],
        "front": [0],
        "left": [9],
        "rear": [18],
        "right": [27],

        # Around the circular boundary:
        "wrap_front": [35],

        # Small clusters on opposite sides:
        "left_cluster": [7, 8, 9, 10, 11],
        "right_cluster": [25, 26, 27, 28, 29],

        # Cluster crossing the 0/35 boundary:
        "boundary_cluster": [34, 35, 0, 1, 2],
    }

    observations = {}
    features = {}

    with torch.no_grad():

        for name, indices in patterns.items():

            lidar = make_lidar(indices)
            obs = make_obs(lidar)

            observations[name] = obs
            out = model(obs)

            features[name] = out

            print(
                f"{name:20s} "
                f"input={tuple(obs.shape)} "
                f"output={tuple(out.shape)} "
                f"finite={torch.isfinite(out).all().item()}"
            )

    # ------------------------------------------------------------
    # Pairwise angular sensitivity
    # ------------------------------------------------------------

    print("\n=== ANGULAR FEATURE DIFFERENCES ===")

    comparisons = [
        ("empty", "front"),
        ("front", "left"),
        ("front", "right"),
        ("left", "right"),
        ("left", "rear"),
        ("right", "rear"),
        ("front", "wrap_front"),
        ("left_cluster", "right_cluster"),
    ]

    for a, b in comparisons:
        d = l2(features[a], features[b])
        print(f"{a:20s} vs {b:20s}: {d:.8f}")

    # ------------------------------------------------------------
    # Circular boundary test
    # ------------------------------------------------------------

    print("\n=== CIRCULAR BOUNDARY TEST ===")

    boundary_tests = [
        ("front", "wrap_front"),
        ("left_cluster", "boundary_cluster"),
    ]

    for a, b in boundary_tests:
        d = l2(features[a], features[b])
        print(f"{a:20s} vs {b:20s}: {d:.8f}")

    # ------------------------------------------------------------
    # Shift test
    # ------------------------------------------------------------

    print("\n=== ANGULAR SHIFT TEST ===")

    base_lidar = make_lidar([0])

    def circular_shift(x, shift):
        return torch.roll(x, shifts=shift, dims=0)

    with torch.no_grad():

        base_obs = make_obs(base_lidar)
        base_feature = model(base_obs)

        for shift in [1, 3, 6, 9, 18]:

            shifted_lidar = circular_shift(base_lidar, shift)
            shifted_obs = make_obs(shifted_lidar)
            shifted_feature = model(shifted_obs)

            d = l2(base_feature, shifted_feature)

            print(
                f"shift={shift:2d} beams "
                f"feature difference={d:.8f}"
            )

    # ------------------------------------------------------------
    # Gradient sanity check
    # ------------------------------------------------------------

    print("\n=== GRADIENT SANITY CHECK ===")

    model.train()

    test_obs = make_obs(make_lidar([0]))
    output = model(test_obs)

    loss = output.square().mean()
    loss.backward()

    missing = []

    for name, parameter in model.named_parameters():

        if parameter.requires_grad and parameter.grad is None:
            missing.append(name)

    print(f"Trainable tensors: {sum(p.requires_grad for p in model.parameters())}")
    print(f"Missing gradients: {len(missing)}")

    if missing:
        for name in missing:
            print("  MISSING:", name)

    # ------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------

    all_finite = all(
        torch.isfinite(x).all().item()
        for x in features.values()
    )

    if all_finite and not missing:
        print("\nLSTM SPATIAL TESTS PASSED")
    else:
        print("\nLSTM SPATIAL TESTS FAILED")


if __name__ == "__main__":
    main()
