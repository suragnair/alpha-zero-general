"""PyTorch neural network model for MiniShogi."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import NNetConfig


class MiniShogiNNet(nn.Module):
    """Convolutional neural network for MiniShogi.

    Architecture:
    - 4 convolutional layers with batch normalization
    - 2 fully connected layers with dropout
    - Policy head: outputs action probabilities (log_softmax)
    - Value head: outputs position evaluation (-1 to 1)
    """

    def __init__(self, game, config: NNetConfig | None = None):
        """Initialize the neural network.

        Args:
            game: MiniShogiGame instance for board/action size.
            config: Network configuration.
        """
        super().__init__()

        self.config = config or NNetConfig()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        num_channels = self.config.num_channels

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.bn4 = nn.BatchNorm2d(num_channels)

        # Fully connected layers
        # After 4 conv layers with padding=1, size is still board_x * board_y
        fc_input_size = num_channels * self.board_x * self.board_y
        self.fc1 = nn.Linear(fc_input_size, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        # Policy head: outputs log probabilities for each action
        self.fc_policy = nn.Linear(512, self.action_size)

        # Value head: outputs a single value
        self.fc_value = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, board_x, board_y).

        Returns:
            Tuple of (log_policy, value):
            - log_policy: Log probabilities of shape (batch_size, action_size)
            - value: Position evaluation of shape (batch_size, 1)
        """
        # Add channel dimension: (batch, board_x, board_y) -> (batch, 1, board_x, board_y)
        x = x.view(-1, 1, self.board_x, self.board_y)

        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected with dropout
        x = F.dropout(
            F.relu(self.fc_bn1(self.fc1(x))),
            p=self.config.dropout,
            training=self.training,
        )
        x = F.dropout(
            F.relu(self.fc_bn2(self.fc2(x))),
            p=self.config.dropout,
            training=self.training,
        )

        # Policy head
        policy = self.fc_policy(x)
        log_policy = F.log_softmax(policy, dim=1)

        # Value head
        value = torch.tanh(self.fc_value(x))

        return log_policy, value
