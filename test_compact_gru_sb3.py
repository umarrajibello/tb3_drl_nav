import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.tb3_env import TB3Env
from env.observation_wrapper import TemporalObservationWrapper


HISTORY_LEN = 8


def make_env():
    def _init():
        env = TB3Env()
        env = TemporalObservationWrapper(
            env,
            history_len=HISTORY_LEN,
        )
        return env

    return _init


def main():

    print("\n=== REAL ENVIRONMENT ===")

    env = DummyVecEnv([make_env()])

    print(
        "Raw VecEnv observation space:",
        env.observation_space,
    )

    print(
        "Raw VecEnv action space:",
        env.action_space,
    )

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,
        training=True,
    )

    print("\n=== VECNORMALIZE ===")

    print(
        "Observation space:",
        env.observation_space,
    )

    print(
        "Action space:",
        env.action_space,
    )

    obs = env.reset()

    print("\n=== RESET TEST ===")

    print(
        "Observation shape:",
        obs.shape,
    )

    print(
        "Observation dtype:",
        obs.dtype,
    )

    print(
        "Observation finite:",
        np.isfinite(obs).all(),
    )

    policy_module = __import__(
        "policies.gru_extractor",
        fromlist=["GRUFeatureExtractor"],
    )

    GRUFeatureExtractor = (
        policy_module.GRUFeatureExtractor
    )

    policy_kwargs = dict(
        features_extractor_class=(
            GRUFeatureExtractor
        ),
        features_extractor_kwargs=dict(
            history_len=HISTORY_LEN,
            lidar_dim=36,
            non_lidar_dim=5,
            features_dim=256,
        ),
        net_arch=dict(
            pi=[256],
            qf=[256],
        ),
    )

    print("\n=== SAC CONSTRUCTION ===")

    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=1000,
        batch_size=32,
        learning_starts=10,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto_0.5",
        gamma=0.99,
        tau=0.005,
        verbose=0,
        seed=42,
        device="cpu",
    )

    print("SAC construction: PASS")

    # ---------------------------------------------------------
    # Actor / critic extractor checks
    # ---------------------------------------------------------

    actor_extractor = (
        model.policy.actor.features_extractor
    )

    critic_extractor = (
        model.policy.critic.features_extractor
    )

    print("\n=== FEATURE EXTRACTORS ===")

    print(
        "Actor extractor:",
        type(actor_extractor),
    )

    print(
        "Critic extractor:",
        type(critic_extractor),
    )

    print(
        "Actor features_dim:",
        actor_extractor.features_dim,
    )

    print(
        "Critic features_dim:",
        critic_extractor.features_dim,
    )

    assert (
        actor_extractor.features_dim
        == 256
    )

    assert (
        critic_extractor.features_dim
        == 256
    )

    print(
        "Feature extractor integration: PASS"
    )

    # ---------------------------------------------------------
    # Actor / critic feature forward
    # ---------------------------------------------------------

    obs_tensor = torch.as_tensor(
        obs,
        dtype=torch.float32,
        device=model.device,
    )

    with torch.no_grad():

        actor_features = (
            model.policy.actor.extract_features(
                obs_tensor,
                actor_extractor,
            )
        )

        critic_features = (
            model.policy.critic.extract_features(
                obs_tensor,
                critic_extractor,
            )
        )

    print("\n=== FEATURE FORWARD ===")

    print(
        "Actor feature shape:",
        tuple(actor_features.shape),
    )

    print(
        "Actor feature finite:",
        torch.isfinite(
            actor_features
        ).all().item(),
    )

    print(
        "Critic feature shape:",
        tuple(critic_features.shape),
    )

    print(
        "Critic feature finite:",
        torch.isfinite(
            critic_features
        ).all().item(),
    )

    assert (
        actor_features.shape
        == (1, 256)
    )

    assert (
        critic_features.shape
        == (1, 256)
    )

    assert torch.isfinite(
        actor_features
    ).all()

    assert torch.isfinite(
        critic_features
    ).all()

    print(
        "Feature forward: PASS"
    )

    # ---------------------------------------------------------
    # Policy action
    # ---------------------------------------------------------

    action, _ = model.predict(
        obs,
        deterministic=True,
    )

    print("\n=== POLICY ACTION ===")

    print(
        "Action:",
        action,
    )

    print(
        "Action shape:",
        action.shape,
    )

    print(
        "Action finite:",
        np.isfinite(action).all(),
    )

    print(
        "Action bounded:",
        np.all(action >= -1.0)
        and np.all(action <= 1.0),
    )

    assert action.shape == (1, 2)
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

    print(
        "Policy action: PASS"
    )

    # ---------------------------------------------------------
    # Real environment step
    # ---------------------------------------------------------

    next_obs, reward, done, info = env.step(
        action
    )

    print("\n=== REAL ENV STEP ===")

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
    assert np.isfinite(next_obs).all()

    print(
        "Real environment step: PASS"
    )

    # ---------------------------------------------------------
    # Short SAC learning smoke test
    # ---------------------------------------------------------

    print(
        "\n=== SHORT SAC LEARNING TEST ==="
    )

    model.learn(
        total_timesteps=32,
        reset_num_timesteps=False,
    )

    print(
        "SAC learning smoke test: PASS"
    )

    print(
        "\nALL GRU SB3 INTEGRATION TESTS PASSED"
    )


if __name__ == "__main__":
    main()
