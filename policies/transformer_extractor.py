"""
Transformer feature extractor for temporal LiDAR navigation.

Expected input:
    batch x (history_len * obs_dim)

With the committed environment:
    history_len = 8
    obs_dim = 41
    LiDAR = 36
    non-LiDAR = 5

Per timestep:
    36 LiDAR values -> shared 1D CNN -> spatial feature
    + 5 robot/waypoint features
    -> timestep embedding

Across timesteps:
    positional encoding
    -> TransformerEncoder
    -> temporal mean pooling
    -> MLP
    -> SAC feature vector
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """CNN + Transformer temporal feature extractor for SAC."""

    def __init__(
        self,
        observation_space: spaces.Box,
        history_len: int = 8,
        lidar_dim: int = 36,
        non_lidar_dim: int = 5,
        cnn_channels: int = 32,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)

        self.history_len = int(history_len)
        self.lidar_dim = int(lidar_dim)
        self.non_lidar_dim = int(non_lidar_dim)
        self.obs_dim = self.lidar_dim + self.non_lidar_dim

        if self.history_len < 1:
            raise ValueError("history_len must be >= 1")

        if self.lidar_dim < 1:
            raise ValueError("lidar_dim must be >= 1")

        if self.non_lidar_dim < 0:
            raise ValueError("non_lidar_dim must be >= 0")

        if observation_space.shape is None:
            raise ValueError("Observation space must have a defined shape.")

        if len(observation_space.shape) != 1:
            raise ValueError(
                "TransformerFeatureExtractor expects a flattened 1D observation."
            )

        expected_dim = self.history_len * self.obs_dim

        if observation_space.shape[0] != expected_dim:
            raise ValueError(
                f"Expected observation dimension {expected_dim} "
                f"({self.history_len} x {self.obs_dim}), "
                f"got {observation_space.shape[0]}."
            )

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})."
            )

        # Shared spatial encoder applied independently to every LiDAR frame.
        # The LiDAR is an ordered angular scan, so Conv1d can capture local
        # relationships between neighboring rays.
        self.lidar_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        cnn_out_dim = cnn_channels

        # Combine spatial LiDAR representation with the robot/waypoint state
        # belonging to the same timestep.
        self.input_projection = nn.Sequential(
            nn.Linear(cnn_out_dim + self.non_lidar_dim, d_model),
            nn.ReLU(),
        )

        # Learnable temporal positional encoding. Shape is:
        # 1 x history_len x d_model
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.history_len, d_model)
        )
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # [B, 328] -> [B, 8, 41]
        x = observations.reshape(
            batch_size,
            self.history_len,
            self.obs_dim,
        )

        # Split each timestep into:
        # LiDAR: [B, 8, 36]
        # state: [B, 8, 5]
        lidar = x[:, :, :self.lidar_dim]
        non_lidar = x[:, :, self.lidar_dim:]

        # Process every LiDAR frame with the same CNN.
        # [B, 8, 36] -> [B*8, 1, 36]
        lidar = lidar.reshape(
            batch_size * self.history_len,
            1,
            self.lidar_dim,
        )

        lidar_features = self.lidar_cnn(lidar)

        # [B*8, cnn_channels] -> [B, 8, cnn_channels]
        lidar_features = lidar_features.reshape(
            batch_size,
            self.history_len,
            -1,
        )

        # Same timestep's LiDAR and robot/waypoint state are fused before
        # temporal attention.
        timestep_features = torch.cat(
            [lidar_features, non_lidar],
            dim=-1,
        )

        timestep_features = self.input_projection(timestep_features)

        # Add temporal position information before self-attention.
        timestep_features = (
            timestep_features + self.positional_encoding
        )

        # [B, 8, d_model] -> [B, 8, d_model]
        temporal_features = self.transformer(timestep_features)

        # Fixed-size representation of the complete observation history.
        pooled = temporal_features.mean(dim=1)

        return self.output_projection(pooled)
