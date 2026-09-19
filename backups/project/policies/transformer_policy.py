import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TemporalTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, history_len=8, input_dim=36, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.history_len = history_len
        self.input_dim = input_dim

        self.embedding = nn.Linear(input_dim, 64)
        self.pos_encoding = nn.Parameter(torch.randn(1, history_len, 64))
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, features_dim)

    def forward(self, observations):
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.history_len, self.input_dim)
        x = self.embedding(x)
        x = x + self.pos_encoding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        features = self.fc(x)
        return features