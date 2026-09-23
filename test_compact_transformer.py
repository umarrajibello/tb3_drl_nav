import torch
from gymnasium import spaces

from policies.transformer_extractor_compact_circular import (
    TransformerFeatureExtractor,
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():

    # ============================================================
    # 1. Create the same observation space used by the real system
    # ============================================================
    observation_space = spaces.Box(
        low=-float("inf"),
        high=float("inf"),
        shape=(328,),
        dtype=float,
    )

    print("=" * 70)
    print("TEST 1: Model construction")
    print("=" * 70)

    extractor = TransformerFeatureExtractor(
        observation_space=observation_space,
        history_len=8,
        lidar_dim=36,
        non_lidar_dim=5,
        cnn_channels=16,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        features_dim=256,
    )

    print("Model construction: PASS")

    # ============================================================
    # 2. Parameter count
    # ============================================================
    total_params = count_parameters(extractor)

    print()
    print("=" * 70)
    print("TEST 2: Parameter count")
    print("=" * 70)

    print(f"Total parameters: {total_params:,}")

    # ============================================================
    # 3. Forward pass
    # ============================================================
    print()
    print("=" * 70)
    print("TEST 3: Forward pass")
    print("=" * 70)

    observations = torch.randn(1, 328)

    with torch.no_grad():
        output = extractor(observations)

    print(f"Input shape : {tuple(observations.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output finite: {torch.isfinite(output).all().item()}")

    assert output.shape == (1, 256)
    assert torch.isfinite(output).all()

    print("Forward pass: PASS")

    # ============================================================
    # 4. Batch-size test
    # ============================================================
    print()
    print("=" * 70)
    print("TEST 4: Batch-size handling")
    print("=" * 70)

    observations = torch.randn(16, 328)

    with torch.no_grad():
        output = extractor(observations)

    print(f"Input shape : {tuple(observations.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output finite: {torch.isfinite(output).all().item()}")

    assert output.shape == (16, 256)
    assert torch.isfinite(output).all()

    print("Batch-size test: PASS")

    # ============================================================
    # 5. Gradient test
    # ============================================================
    print()
    print("=" * 70)
    print("TEST 5: Gradient propagation")
    print("=" * 70)

    extractor.train()

    observations = torch.randn(4, 328, requires_grad=True)

    output = extractor(observations)

    loss = output.mean()

    loss.backward()

    gradient_count = 0
    missing_gradients = []

    for name, parameter in extractor.named_parameters():

        if parameter.requires_grad:

            if parameter.grad is None:
                missing_gradients.append(name)
            else:
                gradient_count += 1

    print(f"Parameters with gradients: {gradient_count}")
    print(f"Parameters without gradients: {len(missing_gradients)}")

    if missing_gradients:
        print("Missing gradients:")
        for name in missing_gradients:
            print("  ", name)

    assert len(missing_gradients) == 0

    print("Gradient test: PASS")

    # ============================================================
    # Final
    # ============================================================
    print()
    print("=" * 70)
    print("ALL BASIC TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
