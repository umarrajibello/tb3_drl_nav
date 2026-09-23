import torch
import gymnasium as gym

from policies.gru_extractor import GRUFeatureExtractor


HISTORY_LEN = 8
LIDAR_DIM = 36
NON_LIDAR_DIM = 5
FEATURES_DIM = 256


def make_obs(lidar):
    state = torch.tensor(
        [0.10, 0.20, 1.0, 0.5, 1.118],
        dtype=torch.float32,
    )

    frame = torch.cat(
        [
            torch.as_tensor(
                lidar,
                dtype=torch.float32,
            ),
            state,
        ]
    )

    history = frame.repeat(HISTORY_LEN)

    return history.unsqueeze(0)


def make_lidar(indices, value=0.10):
    lidar = torch.ones(
        LIDAR_DIM,
        dtype=torch.float32,
    )

    for idx in indices:
        lidar[idx % LIDAR_DIM] = value

    return lidar


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


def l2(a, b):
    return torch.norm(a - b).item()


def main():

    obs_space = gym.spaces.Box(
        low=float("-inf"),
        high=float("inf"),
        shape=(
            HISTORY_LEN *
            (LIDAR_DIM + NON_LIDAR_DIM),
        ),
        dtype=float,
    )

    model = GRUFeatureExtractor(
        observation_space=obs_space,
        history_len=HISTORY_LEN,
        lidar_dim=LIDAR_DIM,
        non_lidar_dim=NON_LIDAR_DIM,
        features_dim=FEATURES_DIM,
    )

    model.eval()

    print(
        "\n=== GRU SPATIAL / ANGULAR "
        "SENSITIVITY TEST ==="
    )

    print(
        f"Parameters: "
        f"{count_parameters(model):,}"
    )

    patterns = {
        "empty": [],
        "front": [0],
        "left": [9],
        "rear": [18],
        "right": [27],
        "wrap_front": [35],
        "left_cluster": [7, 8, 9, 10, 11],
        "right_cluster": [25, 26, 27, 28, 29],
        "boundary_cluster": [34, 35, 0, 1, 2],
    }

    features = {}

    with torch.no_grad():

        for name, indices in patterns.items():

            lidar = make_lidar(indices)
            obs = make_obs(lidar)

            out = model(obs)

            features[name] = out

            print(
                f"{name:20s} "
                f"input={tuple(obs.shape)} "
                f"output={tuple(out.shape)} "
                f"finite="
                f"{torch.isfinite(out).all().item()}"
            )

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

        print(
            f"{a:20s} vs {b:20s}: "
            f"{l2(features[a], features[b]):.8f}"
        )

    print("\n=== CIRCULAR BOUNDARY TEST ===")

    for a, b in [
        ("front", "wrap_front"),
        ("left_cluster", "boundary_cluster"),
    ]:

        print(
            f"{a:20s} vs {b:20s}: "
            f"{l2(features[a], features[b]):.8f}"
        )

    print("\n=== ANGULAR SHIFT TEST ===")

    base_lidar = make_lidar([0])

    with torch.no_grad():

        base_feature = model(
            make_obs(base_lidar)
        )

        for shift in [1, 3, 6, 9, 18]:

            shifted_lidar = torch.roll(
                base_lidar,
                shifts=shift,
                dims=0,
            )

            shifted_feature = model(
                make_obs(shifted_lidar)
            )

            print(
                f"shift={shift:2d} beams "
                f"feature difference="
                f"{l2(base_feature, shifted_feature):.8f}"
            )

    print("\n=== GRADIENT SANITY CHECK ===")

    model.train()

    test_obs = make_obs(
        make_lidar([0])
    )

    output = model(test_obs)

    loss = output.square().mean()
    loss.backward()

    missing = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is None
    ]

    print(
        "Trainable tensors:",
        sum(
            p.requires_grad
            for p in model.parameters()
        ),
    )

    print(
        "Missing gradients:",
        len(missing)
    )

    if missing:

        for name in missing:
            print("  MISSING:", name)

    all_finite = all(
        torch.isfinite(x).all().item()
        for x in features.values()
    )

    if all_finite and not missing:
        print("\nGRU SPATIAL TESTS PASSED")
    else:
        print("\nGRU SPATIAL TESTS FAILED")


if __name__ == "__main__":
    main()
