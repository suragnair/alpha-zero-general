import torch
import torch.nn as nn
import torch.nn.functional as F

from ..TicTacToe4DLogic import winning_lines

"""
Neural network for tic-tac-toe in four dimensions, with a four-dimensional convolution.

PyTorch has no Conv4d, so it is built out of Conv3d, and test_tictactoe_4d.py checks it against the definition.
"""


class Conv4d(nn.Module):
    """A 3×3×3×3 convolution. Such a kernel is three 3×3×3 kernels, one per step along the fourth axis,
    so an output layer w sums 3D convolutions of the input layers w-1, w and w+1. Input and output
    are (batch, channels, w, z, y, x), with zeros beyond the edge of the board."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 3 * out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        batch, _, layers = x.shape[:3]
        y = self.conv(x.transpose(1, 2).flatten(0, 1)).unflatten(0, (batch, layers))
        before, same, after = y.chunk(3, dim=2)
        empty = same[:, :1] * 0
        return (same + torch.cat((empty, before[:, :-1]), dim=1) + torch.cat((after[:, 1:], empty), dim=1)).transpose(1, 2)


class BatchNorm4d(nn.BatchNorm1d):
    def forward(self, x):
        return super().forward(x.flatten(2)).view(x.shape)


class ResBlock4d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv4d(channels, channels)
        self.bn1 = BatchNorm4d(channels)
        self.conv2 = Conv4d(channels, channels)
        self.bn2 = BatchNorm4d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(x + self.bn2(self.conv2(y)), inplace=True)


class TicTacToe4DNNet(nn.Module):
    def __init__(self, game, args):
        super().__init__()
        self.n = game.n
        self.board_size = game.getBoardSize()
        self.action_size = game.getActionSize()
        lines = torch.as_tensor(winning_lines(game.n))
        incidence = torch.zeros(len(lines), self.action_size)
        incidence.scatter_(1, lines, 1.0)
        self.register_buffer('incidence', incidence, persistent=False)
        self.scale = float(incidence.sum(dim=0).max())

        c = args.num_channels
        planes = 3 + 2 * (self.n - 1)  # own stones, opponent stones, hints about lines, and the board itself
        self.stem = nn.Sequential(Conv4d(planes, c), BatchNorm4d(c), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResBlock4d(c) for _ in range(args.num_blocks)])
        self.policy = nn.Sequential(nn.Conv1d(c, 16, 1, bias=False), nn.BatchNorm1d(16), nn.ReLU(inplace=True), nn.Conv1d(16, 1, 1))
        self.value_conv = nn.Conv1d(c, 16, 1)
        self.value = nn.Sequential(nn.Linear(32, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def planes(self, boards):
        """Stones as separate planes, then for each k: on how many lines through a square a side
        has k stones and the other side has none. Convolutions alone find long diagonals slowly."""
        own, opp = (boards == 1).float(), (boards == -1).float()
        flat_own, flat_opp = own.flatten(1), opp.flatten(1)
        own_count = flat_own @ self.incidence.T
        opp_count = flat_opp @ self.incidence.T
        planes = [own, opp]
        for k in range(1, self.n):
            for mine, theirs in ((own_count, opp_count), (opp_count, own_count)):
                hint = ((mine == k) & (theirs == 0)).float() @ self.incidence / self.scale
                planes.append(hint.view(own.shape))
        planes.append(torch.ones_like(own))
        return torch.stack(planes, dim=1)

    def forward(self, boards):
        # boards: batch × n × n × n × n
        h = self.blocks(self.stem(self.planes(boards))).flatten(2)
        v = F.relu(self.value_conv(h), inplace=True)
        pooled = torch.cat((v.mean(dim=2), v.amax(dim=2)), dim=1)
        return F.log_softmax(self.policy(h).squeeze(1), dim=1), torch.tanh(self.value(pooled)).squeeze(1)
