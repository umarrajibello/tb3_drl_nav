import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from config import OBSERVATION_HISTORY, NUM_LIDAR

class SB3_Transformer_Extractor(BaseFeaturesExtractor):
    """
    Spatio‑temporal feature extractor:
    - 1D CNN per LiDAR scan (spatial)
    - Transformer over time (temporal)
    - Concatenates other features at each timestep
    """
    def __init__(self, observation_space, features_dim=256, n_heads=4, n_layers=2):
        super().__init__(observation_space, features_dim)

        self.history = OBSERVATION_HISTORY
        self.lidar_dim = NUM_LIDAR
        self.other_dim = observation_space.shape[0] // self.history - self.lidar_dim

        # Spatial encoder (CNN)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        cnn_out = 32 * self.lidar_dim

        # Project combined features to transformer dimension (e.g., 64)
        self.proj = nn.Linear(cnn_out + self.other_dim, 64)

        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.history, 64))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=n_heads,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final projection to features_dim
        self.fc = nn.Linear(64, features_dim)

    def forward(self, observations):
        batch = observations.shape[0]
        seq = observations.view(batch, self.history, self.lidar_dim + self.other_dim)

        lidar_seq = seq[:, :, :self.lidar_dim]
        other_seq = seq[:, :, self.lidar_dim:]

        # CNN per timestep
        bh = batch * self.history
        lidar_flat = lidar_seq.reshape(bh, 1, self.lidar_dim)
        cnn_out = self.cnn(lidar_flat)
        cnn_out = cnn_out.reshape(batch, self.history, -1)

        # Combine with other features
        combined = torch.cat([cnn_out, other_seq], dim=2)   # (B, H, cnn_out+other_dim)

        # Project to transformer dimension
        x = self.proj(combined)                            # (B, H, 64)
        x = x + self.pos_encoding

        # Transformer
        x = self.transformer(x)                            # (B, H, 64)

        # Mean‑pool over time
        pooled = x.mean(dim=1)                             # (B, 64)

        return self.fc(pooled)