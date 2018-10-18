import os
import time

import numpy as np

from NeuralNet import NeuralNet
from td2020.keras.TD2020NNet import TD2020NNet
from td2020.src.config import encoder
from utils import *

"""
args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 2,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
})
"""
args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 2,
    'batch_size': 512,
    'cuda': True,
    'num_channels': 512,
})


# noinspection PyMissingConstructor
class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = TD2020NNet(game, args)
        self.board_x, self.board_y, num_encoders = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        input_boards = encoder.encode_multiple(input_boards)

        self.nnet.model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=args.batch_size, epochs=args.epochs)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        board = encoder.encode(board)

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        self.nnet.model.load_weights(filepath)
