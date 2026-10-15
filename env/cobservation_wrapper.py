from collections import deque
import numpy as np
import gym

class TemporalObservationWrapper(gym.Wrapper):
    def __init__(self, env, history_len=8):
        super().__init__(env)
        self.history_len = history_len
        self.buffer = deque(maxlen=history_len)
        obs_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(history_len * obs_dim,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        result = self.env.reset(seed=seed, options=options)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        self.buffer.clear()
        for _ in range(self.history_len):
            self.buffer.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.buffer.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.buffer), axis=0)