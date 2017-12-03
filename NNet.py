from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from pytorch_classification.utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from OthelloNNet import OthelloNNet as onnet

args = dotdict({
    'lr': 0.002,
    'momentum': 0.98,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 256,
    'cuda': True,
    'num_channels': 512
    'checkpoint': '/mnt/models/basset+/',
})

class NNetWrapper():
    def __init__(self, game):
        self.nnet = onnet(game, args)


    def trainNNet(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        for epoch in args.epochs:
            for batch in

    def predict(self, board):
        """
        board: np array with board
        """

        pi, v = onnet(board)
