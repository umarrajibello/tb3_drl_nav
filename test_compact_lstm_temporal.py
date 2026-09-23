import torch
import gymnasium as gym

from policies.lstm_extractor import LSTMFeatureExtractor


HISTORY_LEN = 8
LIDAR_DIM = 36
NON_LIDAR_DIM = 5
FEATURES_DIM = 256


def make_frame(t):
    """
    Create one timestep.

    LiDAR contains a small directional pattern that changes
    over time. The non-LiDAR state also changes over time.
    """

    lidar = torch.ones(LIDAR_DIM, dtype=torch.float32)

    # Move a small obstacle pattern around the LiDAR field.
    center = (t * 4) % LIDAR_DIM

    for offset in [-1, 0, 1]:
        lidar[(center + offset) % LIDAR_DIM] = 0.10

    state = torch.tensor(
        [
            0.05 + 0.02 * t,   # v
            0.20 + 0.15 * t,   # w
            1.00 - 0.04 * t,   # dx
            0.50 + 0.03 * t,   # dy
            1.12,              # distance
        ],
        dtype=torch.float32,
    )

    return torch.cat([lidar, state])


def make_history():
    frames = [make_frame(t) for t in range(HISTORY_LEN)]
    return torch.cat(frames).unsqueeze(0)


def make_repeated_history():
    frame = make_frame(HISTORY_LEN - 1)
    return frame.repeat(HISTORY_LEN).unsqueeze(0)


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

    changing_history = make_history()
    repeated_history = make_repeated_history()

    print("\n=== LSTM TEMPORAL HISTORY TEST ===")

    print(
        "Changing history shape:",
        tuple(changing_history.shape)
    )

    print(
        "Repeated history shape:",
        tuple(repeated_history.shape)
    )

    with torch.no_grad():

        changing_features = model(changing_history)
        repeated_features = model(repeated_history)

    print(
        "Changing output shape:",
        tuple(changing_features.shape)
    )

    print(
        "Repeated output shape:",
        tuple(repeated_features.shape)
    )

    print(
        "Changing output finite:",
        torch.isfinite(changing_features).all().item()
    )

    print(
        "Repeated output finite:",
        torch.isfinite(repeated_features).all().item()
    )

    difference = torch.norm(
        changing_features - repeated_features
    ).item()

    print(
        f"\nRepeated vs changing history "
        f"L2 difference: {difference:.8f}"
    )

    print("\n=== INPUT HISTORY ===")

    for t in range(HISTORY_LEN):

        frame = changing_history[
            0,
            t * (LIDAR_DIM + NON_LIDAR_DIM):
            (t + 1) * (LIDAR_DIM + NON_LIDAR_DIM)
        ]

        v = frame[36].item()
        w = frame[37].item()
        dx = frame[38].item()
        dy = frame[39].item()
        distance = frame[40].item()

        print(
            f"t={t}: "
            f"v={v:.4f}, "
            f"w={w:.4f}, "
            f"dx={dx:.4f}, "
            f"dy={dy:.4f}, "
            f"d={distance:.4f}"
        )

    # ------------------------------------------------------------
    # Order sensitivity test
    # ------------------------------------------------------------

    reversed_history = torch.flip(
        changing_history.reshape(
            1,
            HISTORY_LEN,
            LIDAR_DIM + NON_LIDAR_DIM
        ),
        dims=[1],
    ).reshape(1, -1)

    with torch.no_grad():

        reversed_features = model(reversed_history)

    order_difference = torch.norm(
        changing_features - reversed_features
    ).item()

    print(
        f"\nForward vs reversed history "
        f"L2 difference: {order_difference:.8f}"
    )

    # ------------------------------------------------------------
    # Gradient check
    # ------------------------------------------------------------

    model.train()

    output = model(changing_history)

    loss = output.square().mean()
    loss.backward()

    missing = []

    for name, parameter in model.named_parameters():

        if parameter.requires_grad and parameter.grad is None:
            missing.append(name)

    print("\n=== GRADIENT CHECK ===")
    print(
        f"Trainable tensors: "
        f"{sum(p.requires_grad for p in model.parameters())}"
    )
    print(f"Missing gradients: {len(missing)}")

    if missing:
        for name in missing:
            print("  MISSING:", name)

    # ------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------

    passed = (
        changing_features.shape == (1, FEATURES_DIM)
        and repeated_features.shape == (1, FEATURES_DIM)
        and torch.isfinite(changing_features).all()
        and torch.isfinite(repeated_features).all()
        and difference > 1e-6
        and order_difference > 1e-6
        and len(missing) == 0
    )

    if passed:
        print("\nLSTM TEMPORAL TESTS PASSED")
    else:
        print("\nLSTM TEMPORAL TESTS FAILED")


if __name__ == "__main__":
    main()
