"""
Transformer feature extractor for temporal LiDAR navigation.

Committed observation:
    history_len = 8
    obs_dim = 41
    LiDAR = 36
    non-LiDAR = 5

Architecture per timestep:
    36-beam circular LiDAR
        -> circular Conv1D 1 -> 16
        -> circular Conv1D 16 -> cnn_channels
        -> angular pooling 36 -> 12
        -> flatten
        -> Linear(12 * cnn_channels -> 64)
        -> 64-D LiDAR feature

    64-D LiDAR feature + 5-D robot/waypoint state
        -> 69-D timestep input
        -> temporal positional encoding
        -> TransformerEncoder
        -> temporal mean pooling
        -> 256-D SAC feature vector

The LiDAR encoder is intentionally circular because the first and last
LiDAR beams represent neighboring angular directions. Angular pooling
reduces the 36-beam scan to 12 coarse sectors (three adjacent beams per
sector) while retaining explicit angular position information.

This is a fixed-window temporal feature extractor. The complete 8-frame
history is supplied in every policy forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CircularConv1d(nn.Module):
    """1-D convolution with circular padding at both angular boundaries."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric circular padding.")

        self.padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Circular padding makes beam 0 and beam (N-1) neighbors.
        x = F.pad(x, (self.padding, self.padding), mode="circular")
        return self.conv(x)


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """Circular-LiDAR CNN + Transformer temporal feature extractor for SAC."""

    def __init__(
        self,
        observation_space: spaces.Box,
        history_len: int = 8,
        lidar_dim: int = 36,
        non_lidar_dim: int = 5,
        cnn_channels: int = 16,
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
        self.cnn_channels = int(cnn_channels)
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

        if self.lidar_dim % 3 != 0:
            raise ValueError(
                "lidar_dim must be divisible by 3 for 3-beam angular pooling."
            )

        if self.cnn_channels < 1:
            raise ValueError("cnn_channels must be >= 1")

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})."
            )

        # ------------------------------------------------------------------
        # Spatial LiDAR encoder
        # ------------------------------------------------------------------
        # The LiDAR scan is circular in angle. Circular padding prevents an
        # artificial boundary between the last and first beam.
        #
        # 36 beams
        #   -> circular Conv1D: 1 -> 16
        #   -> circular Conv1D: 16 -> cnn_channels
        #   -> 3-beam angular pooling: 36 -> 12
        #   -> flatten: cnn_channels * 12
        #   -> Linear: cnn_channels * 12 -> 64
        self.lidar_cnn = nn.Sequential(
            CircularConv1d(1, 16, kernel_size=3),
            nn.ReLU(),
            CircularConv1d(16, self.cnn_channels, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=3),
            nn.Flatten(),
        )

        self.angular_sector_count = self.lidar_dim // 3
        cnn_flat_dim = self.cnn_channels * self.angular_sector_count

        self.lidar_projection = nn.Sequential(
            nn.Linear(cnn_flat_dim, 64),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # Timestep fusion
        # ------------------------------------------------------------------
        # 64-D LiDAR feature + 5-D robot/waypoint state -> 64-D embedding.
        self.input_projection = nn.Sequential(
            nn.Linear(64 + self.non_lidar_dim, d_model),
            nn.ReLU(),
        )

        # Learnable temporal positional encoding:
        # [1, history_len, d_model]
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.history_len, d_model)
        )
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)

        # ------------------------------------------------------------------
        # Temporal encoder
        # ------------------------------------------------------------------
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

        # Final SAC feature vector.
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

        # Process every LiDAR frame with the same spatial encoder.
        # [B, 8, 36] -> [B*8, 1, 36]
        lidar = lidar.reshape(
            batch_size * self.history_len,
            1,
            self.lidar_dim,
        )

        # [B*8, 1, 36]
        #   -> [B*8, cnn_channels, 12]
        #   -> [B*8, cnn_channels*12]
        lidar_features = self.lidar_cnn(lidar)

        # Project the angularly structured LiDAR representation to 64-D.
        lidar_features = self.lidar_projection(lidar_features)

        # [B*8, 64] -> [B, 8, 64]
        lidar_features = lidar_features.reshape(
            batch_size,
            self.history_len,
            64,
        )

        # Fuse LiDAR and robot/waypoint state from the same timestep.
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
