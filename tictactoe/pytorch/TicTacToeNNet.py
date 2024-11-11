import sys

sys.path.append('..')
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TicTacToeNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(TicTacToeNNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1, bias=False)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)

        # Calculate the size after convolutions
        # With padding=1, the size remains the same after convolutions
        conv_size = self.board_x * self.board_y * args.num_channels

        # Fully connected layers
        self.fc1 = nn.Linear(conv_size, 1024, bias=False)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512, bias=False)
        self.fc_bn2 = nn.BatchNorm1d(512)

        # Policy head
        self.fc3 = nn.Linear(512, self.action_size)

        # Value head
        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        # Input: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)  # batch_size x 1 x board_x x board_y

        # Convolutional layers with batch norm and ReLU
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))

        # Flatten
        s = s.view(-1, self.board_x * self.board_y * self.args.num_channels)

        # Fully connected layers with dropout
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))),
                      p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))),
                      p=self.args.dropout, training=self.training)

        # Policy and value heads
        pi = self.fc3(s)
        v = self.fc4(s)

        # Output processing
        return F.log_softmax(pi, dim=1), torch.tanh(v)