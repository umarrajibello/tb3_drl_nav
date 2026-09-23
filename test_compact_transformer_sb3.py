import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.tb3_env import TB3Env
from env.observation_wrapper import TemporalObservationWrapper


# ============================================================
# Configuration
# ============================================================

SEED = 42
HISTORY_LEN = 8


# ============================================================
# Environment factory
# ============================================================

def make_env():
    def _init():

        env = TB3Env()

        env = TemporalObservationWrapper(
            env,
            history_len=HISTORY_LEN,
        )

        return env

    return _init


# ============================================================
# TEST 1: Create real environment
# ============================================================

print("=" * 70)
print("TEST 1: REAL ENVIRONMENT")
print("=" * 70)

vec_env = DummyVecEnv([make_env()])

print("Raw VecEnv observation space:")
print(" ", vec_env.observation_space)

print("Raw VecEnv action space:")
print(" ", vec_env.action_space)


# ============================================================
# Apply VecNormalize
# ============================================================

vec_env = VecNormalize(
    vec_env,
    norm_obs=True,
    norm_reward=False,
)

print()
print("VecNormalize observation space:")
print(" ", vec_env.observation_space)

print("VecNormalize action space:")
print(" ", vec_env.action_space)


# ============================================================
# TEST 2: Environment reset
# ============================================================

obs = vec_env.reset()

print()
print("=" * 70)
print("TEST 2: ENVIRONMENT RESET")
print("=" * 70)

print("Observation shape:", obs.shape)
print("Observation dtype:", obs.dtype)
print("Observation finite:", np.isfinite(obs).all())

assert obs.shape == (1, 328)
assert np.isfinite(obs).all()

print("Environment reset: PASS")


# ============================================================
# Candidate Transformer import
# ============================================================

from policies.transformer_extractor_compact_circular import (
    TransformerFeatureExtractor,
)


# ============================================================
# TEST 3: Construct SAC policy
# ============================================================

print()
print("=" * 70)
print("TEST 3: SAC POLICY CONSTRUCTION")
print("=" * 70)

policy_kwargs = dict(

    features_extractor_class=(
        TransformerFeatureExtractor
    ),

    features_extractor_kwargs=dict(
        history_len=8,
        lidar_dim=36,
        non_lidar_dim=5,
        cnn_channels=16,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        features_dim=256,
    ),

    net_arch=dict(
        pi=[256],
        qf=[256],
    ),
)


model = SAC(

    policy="MlpPolicy",

    env=vec_env,

    learning_rate=3e-4,

    buffer_size=1000,

    batch_size=32,

    learning_starts=100,

    tau=0.005,

    gamma=0.99,

    train_freq=1,

    gradient_steps=1,

    ent_coef="auto_0.5",

    policy_kwargs=policy_kwargs,

    seed=SEED,

    verbose=0,
)


print("SAC construction: PASS")


# ============================================================
# TEST 4: Inspect actor and critic extractors
# ============================================================

print()
print("=" * 70)
print("TEST 4: FEATURE EXTRACTOR INSIDE SAC")
print("=" * 70)


actor_extractor = model.policy.actor.features_extractor

critic_extractor = model.policy.critic.features_extractor


print("Actor extractor class:")
print(" ", type(actor_extractor))

print(
    "Actor feature dimension:",
    actor_extractor.features_dim,
)


print("Critic extractor class:")
print(" ", type(critic_extractor))

print(
    "Critic feature dimension:",
    critic_extractor.features_dim,
)


assert isinstance(
    actor_extractor,
    TransformerFeatureExtractor,
)

assert isinstance(
    critic_extractor,
    TransformerFeatureExtractor,
)

assert actor_extractor.features_dim == 256
assert critic_extractor.features_dim == 256


print("Feature extractor integration: PASS")


# ============================================================
# Convert observation to PyTorch tensor
# ============================================================

obs_tensor = torch.as_tensor(
    obs,
    dtype=torch.float32,
    device=model.device,
)


# ============================================================
# TEST 5: Actor feature forward pass
# ============================================================

print()
print("=" * 70)
print("TEST 5: SB3 ACTOR FEATURE FORWARD PASS")
print("=" * 70)


with torch.no_grad():

    actor_features = actor_extractor(
        obs_tensor
    )


print(
    "Actor feature shape:",
    tuple(actor_features.shape),
)

print(
    "Actor feature finite:",
    torch.isfinite(actor_features).all().item(),
)


assert actor_features.shape == (1, 256)

assert torch.isfinite(
    actor_features
).all()


print("SB3 actor feature forward pass: PASS")


# ============================================================
# TEST 6: Critic feature forward pass
# ============================================================

print()
print("=" * 70)
print("TEST 6: SB3 CRITIC FEATURE FORWARD PASS")
print("=" * 70)


with torch.no_grad():

    critic_features = critic_extractor(
        obs_tensor
    )


print(
    "Critic feature shape:",
    tuple(critic_features.shape),
)

print(
    "Critic feature finite:",
    torch.isfinite(critic_features).all().item(),
)


assert critic_features.shape == (1, 256)

assert torch.isfinite(
    critic_features
).all()


print("SB3 critic feature forward pass: PASS")


# ============================================================
# TEST 7: SAC action prediction
# ============================================================

print()
print("=" * 70)
print("TEST 7: SAC ACTION PREDICTION")
print("=" * 70)


action, state = model.predict(
    obs,
    deterministic=True,
)


print("Action:", action)

print(
    "Action shape:",
    action.shape,
)

print(
    "Action finite:",
    np.isfinite(action).all(),
)

print(
    "Action space:",
    vec_env.action_space,
)


assert action.shape == (1, 2)

assert np.isfinite(
    action
).all()


# Check action bounds.

assert np.all(
    action >= vec_env.action_space.low - 1e-6
)

assert np.all(
    action <= vec_env.action_space.high + 1e-6
)


print("SAC action prediction: PASS")


# ============================================================
# TEST 8: Real environment step
# ============================================================

print()
print("=" * 70)
print("TEST 8: REAL ENVIRONMENT STEP")
print("=" * 70)


next_obs, reward, done, info = vec_env.step(
    action
)


print(
    "Next observation shape:",
    next_obs.shape,
)

print(
    "Next observation finite:",
    np.isfinite(next_obs).all(),
)

print(
    "Reward:",
    reward,
)

print(
    "Done:",
    done,
)


assert next_obs.shape == (1, 328)

assert np.isfinite(
    next_obs
).all()


print("Environment step: PASS")


# ============================================================
# Cleanup
# ============================================================

vec_env.close()


# ============================================================
# Final result
# ============================================================

print()
print("=" * 70)
print("ALL SB3 INTEGRATION TESTS PASSED")
print("=" * 70)