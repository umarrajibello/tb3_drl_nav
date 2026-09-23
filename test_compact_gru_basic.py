import torch
import gymnasium as gym

from policies.gru_extractor import GRUFeatureExtractor


HISTORY_LEN = 8
LIDAR_DIM = 36
NON_LIDAR_DIM = 5
FEATURES_DIM = 256


def count(module):
    return sum(p.numel() for p in module.parameters())


obs_space = gym.spaces.Box(
    low=float("-inf"),
    high=float("inf"),
    shape=(HISTORY_LEN * (LIDAR_DIM + NON_LIDAR_DIM),),
    dtype=float,
)

model = GRUFeatureExtractor(
    observation_space=obs_space,
    history_len=HISTORY_LEN,
    lidar_dim=LIDAR_DIM,
    non_lidar_dim=NON_LIDAR_DIM,
    features_dim=FEATURES_DIM,
)

print("\n=== GRU CONSTRUCTION ===")
print("Class:", type(model))
print("Total parameters:", count(model))

print("\n=== MODULE PARAMETERS ===")
print(
    f"{'lidar_cnn':35s}"
    f"{count(model.lidar_cnn):,}"
)
print(
    f"{'lidar_projection':35s}"
    f"{count(model.lidar_projection):,}"
)
print(
    f"{'input_projection':35s}"
    f"{count(model.input_projection):,}"
)
print(
    f"{'gru':35s}"
    f"{count(model.gru):,}"
)
print(
    f"{'output_projection':35s}"
    f"{count(model.output_projection):,}"
)

# ------------------------------------------------------------
# Single forward
# ------------------------------------------------------------

print("\n=== FORWARD TEST ===")

x = torch.randn(
    1,
    HISTORY_LEN * (LIDAR_DIM + NON_LIDAR_DIM),
)

with torch.no_grad():
    y = model(x)

print("Input shape:", tuple(x.shape))
print("Output shape:", tuple(y.shape))
print("Output finite:", torch.isfinite(y).all().item())

assert y.shape == (1, FEATURES_DIM)
assert torch.isfinite(y).all()

# ------------------------------------------------------------
# Batch forward
# ------------------------------------------------------------

print("\n=== BATCH TEST ===")

x_batch = torch.randn(
    16,
    HISTORY_LEN * (LIDAR_DIM + NON_LIDAR_DIM),
)

with torch.no_grad():
    y_batch = model(x_batch)

print("Input shape:", tuple(x_batch.shape))
print("Output shape:", tuple(y_batch.shape))
print("Output finite:", torch.isfinite(y_batch).all().item())

assert y_batch.shape == (16, FEATURES_DIM)
assert torch.isfinite(y_batch).all()

# ------------------------------------------------------------
# Gradient propagation
# ------------------------------------------------------------

print("\n=== GRADIENT TEST ===")

model.train()

x_grad = torch.randn(
    2,
    HISTORY_LEN * (LIDAR_DIM + NON_LIDAR_DIM),
)

output = model(x_grad)

loss = output.square().mean()
loss.backward()

trainable = [
    p
    for p in model.parameters()
    if p.requires_grad
]

missing_gradients = [
    name
    for name, p in model.named_parameters()
    if p.requires_grad and p.grad is None
]

print("Trainable tensors:", len(trainable))
print("Missing gradients:", len(missing_gradients))

if missing_gradients:
    for name in missing_gradients:
        print("  MISSING:", name)

assert len(missing_gradients) == 0

print("\nGRU BASIC TESTS PASSED")
