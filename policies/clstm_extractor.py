"""
LSTM feature extractor for temporal LiDAR navigation.

This is the recurrent comparison model corresponding to the committed
Tx-SAC representation. It uses the same per-timestep CNN and the same
8 x 41 temporal input as the Transformer extractor, then replaces
self-attention with an LSTM.

Architecture:
    LiDAR frame -> shared CNN
    CNN features + velocity/waypoint state
        -> LSTM across 8 timesteps
        -> final hidden state
        -> 256-dim SAC feature vector

This is a fixed-window feature extractor, not a stateful recurrent SAC
implementation. The complete 8-frame history is supplied in every policy
forward pass.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """CNN + LSTM temporal feature extractor for SAC."""

    def __init__(
        self,
        observation_space: spaces.Box,
        history_len: int = 8,
        lidar_dim: int = 36,
        non_lidar_dim: int = 5,
        cnn_channels: int = 32,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.0,
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

        if lstm_layers < 1:
            raise ValueError("lstm_layers must be >= 1")

        # Use the same spatial encoder as Tx-SAC so that the comparison
        # isolates the temporal modeling mechanism rather than changing
        # the LiDAR feature extractor.
        self.lidar_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        cnn_out_dim = cnn_channels

        self.input_projection = nn.Sequential(
            nn.Linear(cnn_out_dim + self.non_lidar_dim, 64),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.output_projection = nn.Sequential(
            nn.LayerNorm(lstm_hidden_dim),
            nn.Linear(lstm_hidden_dim, features_dim),
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

        lidar = x[:, :, :self.lidar_dim]
        non_lidar = x[:, :, self.lidar_dim:]

        # Apply the same CNN independently to every LiDAR timestep.
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

        # Fuse spatial LiDAR features with the robot/waypoint state from
        # the same timestep.
        timestep_features = torch.cat(
            [lidar_features, non_lidar],
            dim=-1,
        )

        timestep_features = self.input_projection(timestep_features)

        # The wrapper supplies the complete history in chronological order,
        # so the LSTM receives [oldest -> newest].
        sequence_output, (hidden_state, _) = self.lstm(
            timestep_features
        )

        # Use the final LSTM layer's hidden state, corresponding to the
        # newest timestep after processing the complete history.
        final_hidden = hidden_state[-1]

        return self.output_projection(final_hidden)
