"""
Temporal observation wrapper.

Converts the base 41-dimensional navigation observation into a fixed-length
history:

    [o(t-7), ..., o(t-1), o(t)]

The newest observation is placed at the end of the vector. This ordering is
used by the CNN + LSTM/GRU/Transformer temporal feature extractors.
"""

import gymnasium as gym
import numpy as np
from collections import deque


class TemporalObservationWrapper(gym.Wrapper):
    """Stack a fixed number of consecutive observations."""

    def __init__(self, env, history_len=8):
        super().__init__(env)

        if history_len < 1:
            raise ValueError("history_len must be >= 1")

        self.history_len = int(history_len)
        self._history = deque(maxlen=self.history_len)

        # Preserve the underlying observation bounds instead of assuming
        # every observation component lies in [0, 1].  The navigation
        # observation contains signed velocity and relative-position terms.
        base_space = env.observation_space

        if not isinstance(base_space, gym.spaces.Box):
            raise TypeError(
                "TemporalObservationWrapper requires a Box observation space."
            )

        low = np.tile(
            np.asarray(base_space.low, dtype=np.float32),
            self.history_len,
        )
        high = np.tile(
            np.asarray(base_space.high, dtype=np.float32),
            self.history_len,
        )

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def _stack_history(self):
        """Return observations in chronological order, oldest -> newest."""
        return np.concatenate(list(self._history), axis=0).astype(np.float32)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        observation = np.asarray(observation, dtype=np.float32)

        self._history.clear()

        # At the beginning of an episode there are not yet history_len
        # distinct observations. Repeating the initial observation avoids
        # introducing artificial zeros that would have a different meaning
        # from real sensor measurements.
        for _ in range(self.history_len):
            self._history.append(observation.copy())

        return self._stack_history(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        observation = np.asarray(observation, dtype=np.float32)
        self._history.append(observation)

        return (
            self._stack_history(),
            reward,
            terminated,
            truncated,
            info,
        )
