import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from config import OBSERVATION_HISTORY, NUM_LIDAR

class SB3_LSTM_Extractor(BaseFeaturesExtractor):
    """
    Spatio‑temporal feature extractor:
    - 1D CNN per LiDAR scan (spatial)
    - LSTM over time (temporal)
    - Concatenates other features at each timestep
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        self.history = OBSERVATION_HISTORY
        self.lidar_dim = NUM_LIDAR
        self.other_dim = observation_space.shape[0] // self.history - self.lidar_dim

        # Spatial encoder (same CNN as above)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        cnn_out = 32 * self.lidar_dim

        # LSTM over time
        self.lstm = nn.LSTM(
            input_size=cnn_out + self.other_dim,
            hidden_size=128,
            batch_first=True
        )

        self.fc = nn.Linear(128, features_dim)

    def forward(self, observations):
        batch = observations.shape[0]
        seq = observations.view(batch, self.history, self.lidar_dim + self.other_dim)

        lidar_seq = seq[:, :, :self.lidar_dim]
        other_seq = seq[:, :, self.lidar_dim:]

        # Encode each timestep spatially
        bh = batch * self.history
        lidar_flat = lidar_seq.reshape(bh, 1, self.lidar_dim)
        cnn_out = self.cnn(lidar_flat)
        cnn_out = cnn_out.reshape(batch, self.history, -1)

        combined = torch.cat([cnn_out, other_seq], dim=2)

        # LSTM
        _, (h_n, _) = self.lstm(combined)           # h_n: (1, B, 128)
        features = self.fc(h_n.squeeze(0))
        return features