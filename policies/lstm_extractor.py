"""
LSTM feature extractor for temporal LiDAR navigation.

This recurrent comparison model uses the same circular LiDAR spatial
encoder and timestep embedding as the Transformer and GRU extractors.
Only the temporal modeling mechanism differs.

Architecture:
    36-beam LiDAR
        -> circular Conv1D 1 -> 16
        -> circular Conv1D 16 -> 16
        -> 3-beam angular pooling: 36 -> 12
        -> flatten: 16 x 12 = 192
        -> Linear 192 -> 64
        -> 64-D LiDAR feature

    64-D LiDAR feature + 5-D robot/waypoint state
        -> 64-D timestep embedding
        -> LSTM across 8 timesteps
        -> final hidden state
        -> 256-D SAC feature vector

This is a fixed-window feature extractor, not a stateful recurrent SAC
implementation. The complete 8-frame history is supplied in every policy
forward pass.
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
            raise ValueError(
                "kernel_size must be odd for symmetric circular padding."
            )

        self.padding = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Beam 0 and beam (N-1) represent neighboring angular directions.
        x = F.pad(
            x,
            (self.padding, self.padding),
            mode="circular",
        )
        return self.conv(x)


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """Circular-LiDAR CNN + LSTM temporal feature extractor for SAC."""

    def __init__(
        self,
        observation_space: spaces.Box,
        history_len: int = 8,
        lidar_dim: int = 36,
        non_lidar_dim: int = 5,
        cnn_channels: int = 16,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.0,
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

        if observation_space.shape is None or len(observation_space.shape) != 1:
            raise ValueError(
                "LSTMFeatureExtractor expects a flattened 1D observation."
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

        if lstm_layers < 1:
            raise ValueError("lstm_layers must be >= 1")

        # ------------------------------------------------------------------
        # Spatial LiDAR encoder
        # ------------------------------------------------------------------
        # 36 beams
        #   -> circular Conv1D: 1 -> 16
        #   -> circular Conv1D: 16 -> cnn_channels
        #   -> 3-beam angular pooling: 36 -> 12
        #   -> flatten: cnn_channels * 12
        #   -> Linear: cnn_channels * 12 -> 64
        #
        # The angular sector positions are retained explicitly instead of
        # being collapsed by global average pooling.
        self.lidar_cnn = nn.Sequential(
            CircularConv1d(
                1,
                16,
                kernel_size=3,
            ),
            nn.ReLU(),
            CircularConv1d(
                16,
                self.cnn_channels,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.AvgPool1d(
                kernel_size=3,
                stride=3,
            ),
            nn.Flatten(),
        )

        self.angular_sector_count = self.lidar_dim // 3
        cnn_flat_dim = (
            self.cnn_channels * self.angular_sector_count
        )

        self.lidar_projection = nn.Sequential(
            nn.Linear(
                cnn_flat_dim,
                64,
            ),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # Timestep fusion
        # ------------------------------------------------------------------
        # 64-D LiDAR feature + 5-D robot/waypoint state -> 64-D embedding.
        self.input_projection = nn.Sequential(
            nn.Linear(
                64 + self.non_lidar_dim,
                64,
            ),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # Temporal encoder
        # ------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.output_projection = nn.Sequential(
            nn.LayerNorm(lstm_hidden_dim),
            nn.Linear(
                lstm_hidden_dim,
                features_dim,
            ),
            nn.ReLU(),
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = observations.shape[0]

        # [B, 328] -> [B, 8, 41]
        x = observations.reshape(
            batch_size,
            self.history_len,
            self.obs_dim,
        )

        # Split each timestep into LiDAR and robot/waypoint state.
        lidar = x[:, :, :self.lidar_dim]
        non_lidar = x[:, :, self.lidar_dim:]

        # [B, 8, 36] -> [B*8, 1, 36]
        lidar = lidar.reshape(
            batch_size * self.history_len,
            1,
            self.lidar_dim,
        )

        # Process each LiDAR frame using the same spatial encoder.
        lidar_features = self.lidar_cnn(lidar)

        # [B*8, 192] -> [B*8, 64]
        lidar_features = self.lidar_projection(
            lidar_features
        )

        # [B*8, 64] -> [B, 8, 64]
        lidar_features = lidar_features.reshape(
            batch_size,
            self.history_len,
            64,
        )

        # Fuse LiDAR and robot/waypoint state from the same timestep.
        timestep_features = torch.cat(
            [
                lidar_features,
                non_lidar,
            ],
            dim=-1,
        )

        timestep_features = self.input_projection(
            timestep_features
        )

        # The wrapper supplies chronological order:
        # oldest -> newest.


        _, (hidden_state, _) = self.lstm(timestep_features)


        # Final hidden state of the final LSTM layer summarizes the
        # complete eight-frame history.

        final_hidden = hidden_state[-1]
        

        return self.output_projection(
            final_hidden
        )
    
