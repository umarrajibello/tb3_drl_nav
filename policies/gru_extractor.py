"""
GRU feature extractor for temporal LiDAR navigation.

The extractor uses the same circular LiDAR spatial encoder as the
proposed Transformer and LSTM extractors so that the temporal-model
comparison is controlled.

Architecture:

    36-beam LiDAR
        -> Circular Conv1D 1 -> 16
        -> Circular Conv1D 16 -> 16
        -> 36 -> 12 angular-sector pooling
        -> 192-D spatial representation
        -> Linear 192 -> 64

    5-D non-LiDAR state
        -> concatenated with 64-D LiDAR feature
        -> Linear 69 -> 64

    8 temporal steps
        -> GRU
        -> final hidden state
        -> LayerNorm + Linear
        -> 256-D SAC feature

This is a fixed-window feature extractor, not a stateful recurrent SAC
implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CircularConv1d(nn.Module):
    """
    1-D convolution with circular padding.

    LiDAR measurements are angular and therefore beam 0 and beam
    (N-1) are adjacent. Circular padding preserves this topology.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                "CircularConv1d currently requires an odd kernel size."
            )

        self.padding = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(
            x,
            (self.padding, self.padding),
            mode="circular",
        )

        return self.conv(x)


class GRUFeatureExtractor(BaseFeaturesExtractor):
    """
    Circular-LiDAR + GRU temporal feature extractor for SAC.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        history_len: int = 8,
        lidar_dim: int = 36,
        non_lidar_dim: int = 5,
        cnn_channels: int = 16,
        gru_hidden_dim: int = 128,
        gru_layers: int = 1,
        dropout: float = 0.0,
        angular_sector_count: int = 12,
        features_dim: int = 256,
    ):
        super().__init__(
            observation_space,
            features_dim,
        )

        self.history_len = int(history_len)
        self.lidar_dim = int(lidar_dim)
        self.non_lidar_dim = int(non_lidar_dim)
        self.obs_dim = (
            self.lidar_dim +
            self.non_lidar_dim
        )

        self.angular_sector_count = int(
            angular_sector_count
        )

        if self.history_len < 1:
            raise ValueError(
                "history_len must be >= 1"
            )

        if self.lidar_dim < 1:
            raise ValueError(
                "lidar_dim must be >= 1"
            )

        if self.non_lidar_dim < 0:
            raise ValueError(
                "non_lidar_dim must be >= 0"
            )

        if observation_space.shape is None:
            raise ValueError(
                "Observation space must define a shape."
            )

        if len(observation_space.shape) != 1:
            raise ValueError(
                "GRUFeatureExtractor expects a "
                "flattened 1D observation."
            )

        expected_dim = (
            self.history_len *
            self.obs_dim
        )

        if observation_space.shape[0] != expected_dim:
            raise ValueError(
                f"Expected observation dimension "
                f"{expected_dim} "
                f"({self.history_len} x "
                f"{self.obs_dim}), "
                f"got {observation_space.shape[0]}."
            )

        if cnn_channels < 1:
            raise ValueError(
                "cnn_channels must be >= 1"
            )

        if gru_hidden_dim < 1:
            raise ValueError(
                "gru_hidden_dim must be >= 1"
            )

        if gru_layers < 1:
            raise ValueError(
                "gru_layers must be >= 1"
            )

        if self.lidar_dim % self.angular_sector_count != 0:
            raise ValueError(
                "lidar_dim must be divisible by "
                "angular_sector_count."
            )

        # ---------------------------------------------------------
        # Circular LiDAR spatial encoder
        # ---------------------------------------------------------

        self.lidar_cnn = nn.Sequential(
            CircularConv1d(
                1,
                cnn_channels,
                kernel_size=3,
            ),
            nn.ReLU(),

            CircularConv1d(
                cnn_channels,
                cnn_channels,
                kernel_size=3,
            ),
            nn.ReLU(),

            # 36 beams -> 12 angular sectors.
            #
            # Each sector contains three adjacent LiDAR beams,
            # corresponding to approximately 30 degrees for a
            # 360-degree / 36-beam LiDAR.
            nn.AvgPool1d(
                kernel_size=3,
                stride=3,
            ),

            nn.Flatten(),
        )

        cnn_flat_dim = (
            cnn_channels *
            self.angular_sector_count
        )

        # Project the spatial representation into the common
        # 64-dimensional timestep representation.
        self.lidar_projection = nn.Sequential(
            nn.Linear(
                cnn_flat_dim,
                64,
            ),
            nn.ReLU(),
        )

        # Fuse the LiDAR representation with the physical
        # non-LiDAR state.
        self.input_projection = nn.Sequential(
            nn.Linear(
                64 + self.non_lidar_dim,
                64,
            ),
            nn.ReLU(),
        )

        # ---------------------------------------------------------
        # Temporal encoder
        # ---------------------------------------------------------

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=(
                dropout
                if gru_layers > 1
                else 0.0
            ),
        )

        # Map the final GRU hidden state to the common SAC feature
        # dimension.
        self.output_projection = nn.Sequential(
            nn.LayerNorm(
                gru_hidden_dim
            ),
            nn.Linear(
                gru_hidden_dim,
                features_dim,
            ),
            nn.ReLU(),
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = observations.shape[0]

        # ---------------------------------------------------------
        # Flattened history -> timestep sequence
        # ---------------------------------------------------------

        x = observations.reshape(
            batch_size,
            self.history_len,
            self.obs_dim,
        )

        lidar = x[:, :, :self.lidar_dim]
        non_lidar = x[:, :, self.lidar_dim:]

        # ---------------------------------------------------------
        # Spatial LiDAR encoding
        # ---------------------------------------------------------

        # [B, T, 36]
        # ->
        # [B*T, 1, 36]
        lidar = lidar.reshape(
            batch_size * self.history_len,
            1,
            self.lidar_dim,
        )

        # [B*T, 1, 36]
        # ->
        # [B*T, 192]
        lidar_features = self.lidar_cnn(
            lidar
        )

        # [B*T, 192]
        # ->
        # [B*T, 64]
        lidar_features = self.lidar_projection(
            lidar_features
        )

        # ---------------------------------------------------------
        # Restore temporal sequence
        # ---------------------------------------------------------

        # [B*T, 64]
        # ->
        # [B, T, 64]
        lidar_features = lidar_features.reshape(
            batch_size,
            self.history_len,
            64,
        )

        # ---------------------------------------------------------
        # Fuse LiDAR + physical state
        # ---------------------------------------------------------

        timestep_features = torch.cat(
            [
                lidar_features,
                non_lidar,
            ],
            dim=-1,
        )

        # [B, T, 69] -> [B, T, 64]
        timestep_features = self.input_projection(
            timestep_features
        )

        # ---------------------------------------------------------
        # Temporal processing
        # ---------------------------------------------------------

        # The temporal wrapper provides chronological order:
        # oldest -> newest.
        _, hidden_state = self.gru(
            timestep_features
        )

        # Final hidden state of the final GRU layer.
        final_hidden = hidden_state[-1]

        # [B, 128] -> [B, 256]
        return self.output_projection(
            final_hidden
        )
