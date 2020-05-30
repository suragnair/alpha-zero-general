import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, game, args, layers: int = 4):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # nnet params
        layers = args.layers
        self.layers = layers
        assert layers > 2
        self.shrink = 2 * (self.layers - 2)
        logging.info(f"Creating NN with {layers} layers 🧮")
        self.conv = []
        self.batchnorm = []
        self.fc = []
        self.fcbn = []

        super(Model, self).__init__()

        self._setup()

    def _setup(self):
        # Create Conv layers
        in_channels = 1
        kernel_size = int(float(min(self.board_x, self.board_y)) / self.layers)
        # if kernel_size < 3:
        kernel_size = 3
        paddings = [0] * self.layers
        paddings[0] = 1
        paddings[1] = 1
        for i in range(self.layers):
            conv = nn.Conv2d(in_channels, self.args.num_channels, kernel_size, stride=1, padding=paddings[i])
            self.add_module(f'conv{i}', conv)
            self.conv.append(conv)
            in_channels = self.args.num_channels

        # Prepare Batch Normalization
        for i in range(self.layers):
            bn = nn.BatchNorm2d(self.args.num_channels)
            self.batchnorm.append(bn)
            self.add_module(f'batchnorm{i}', bn)

        # Prepare features
        in_features = self.args.num_channels * (self.board_x - self.shrink) * (self.board_y - self.shrink)
        # self.fc1 = nn.Linear(self.args.num_channels * (self.board_x-4)*(self.board_y-4), 1024)

        out_features = 512 * 2 ** (self.layers - 2)
        for i in range(self.layers - 2):
            out_features = int(out_features / 2.0)  # needs to be unchanged same outside of the loop
            linear = nn.Linear(in_features, out_features)
            self.fc.append(linear)
            self.add_module(f'fc{i}', linear)

            bn = nn.BatchNorm1d(out_features)
            self.fcbn.append(bn)
            self.add_module(f'batchnorm1d{i}', bn)

            in_features = out_features

        self.fc_pi = nn.Linear(out_features, self.action_size)
        self.fc_v = nn.Linear(out_features, 1)

    def forward(self, s: torch.Tensor):
        s = s.view(-1, 1, self.board_x, self.board_y)

        for i in range(self.layers):
            s = F.relu(self.batchnorm[i](self.conv[i](s)))

        size = self.args.num_channels * (self.board_x - self.shrink) * (self.board_y - self.shrink)
        s = s.view(-1, size)

        for i in range(self.layers - 2):
            s = F.dropout(F.relu(self.fcbn[i](self.fc[i](s))), p=self.args.dropout, training=self.training)

        pi = self.fc_pi(s)
        v = self.fc_v(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
