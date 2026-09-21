import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from config import OBSERVATION_HISTORY, NUM_LIDAR

class SB3_CNNGRU_Extractor(BaseFeaturesExtractor):
    """
    Spatio‑temporal feature extractor:
    - 1D CNN per LiDAR scan (spatial)
    - GRU over time (temporal)
    - Concatenates other features (velocities, waypoint) at each timestep
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        self.history = OBSERVATION_HISTORY
        # Per‑timestep observation: LiDAR (NUM_LIDAR) + vel (2) + waypoint (3)
        self.lidar_dim = NUM_LIDAR
        self.other_dim = observation_space.shape[0] // self.history - self.lidar_dim
        # Should be 5 (2 velocities + 3 waypoint)

        # Spatial encoder: 1D CNN for LiDAR rays (input channels=1, length=36)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        cnn_out = 32 * self.lidar_dim   # e.g., 32 * 36 = 1152

        # GRU over time: input size = CNN features + other features
        self.gru = nn.GRU(
            input_size=cnn_out + self.other_dim,
            hidden_size=128,
            batch_first=True
        )

        # Final projection to features_dim
        self.fc = nn.Linear(128, features_dim)

    def forward(self, observations):
        """
        observations: (batch, history * (lidar_dim + other_dim))
        """
        batch = observations.shape[0]
        # Reshape to (batch, history, lidar_dim + other_dim)
        seq = observations.view(batch, self.history, self.lidar_dim + self.other_dim)

        # Split LiDAR and other features
        lidar_seq = seq[:, :, :self.lidar_dim]          # (B, H, 36)
        other_seq = seq[:, :, self.lidar_dim:]          # (B, H, other_dim)

        # Process each timestep with CNN
        # Flatten batch and time: (B*H, 1, 36)
        bh = batch * self.history
        lidar_flat = lidar_seq.reshape(bh, 1, self.lidar_dim)
        cnn_out = self.cnn(lidar_flat)                  # (B*H, cnn_out)
        cnn_out = cnn_out.reshape(batch, self.history, -1)  # (B, H, cnn_out)

        # Concatenate with other features
        combined = torch.cat([cnn_out, other_seq], dim=2)   # (B, H, cnn_out+other_dim)

        # GRU over time
        _, h_n = self.gru(combined)                     # h_n: (1, B, 128)
        features = self.fc(h_n.squeeze(0))              # (B, features_dim)
        return features