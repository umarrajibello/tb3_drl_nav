import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, history_len=8, input_dim=36, hidden_dim=128, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.history_len = history_len
        self.input_dim = input_dim

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, features_dim)

    def forward(self, observations):
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.history_len, self.input_dim)
        out, (hn, cn) = self.lstm(x)
        last_out = out[:, -1, :]        # (batch, hidden_dim)
        features = self.fc(last_out)    # (batch, features_dim)
        return features