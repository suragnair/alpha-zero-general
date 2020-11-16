import numpy as np
import sys
import os
sys.path.append('..')
from utils import dotdict
from NeuralNet import NeuralNet

from .DotsAndBoxesNNet import DotsAndBoxesNNet as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': True,
    'num_channels': 512,
})


def normalize_score(board):
    p1_score = board[:, 0, -1]
    p2_score = board[:, 1, -1]
    score = p1_score - p2_score

    n = board.shape[-1]-1

    max_score = n ** 2
    min_score = -max_score

    min_normalized, max_normalized = 0, 1
    normalized_score = ((score - max_score) / (min_score - max_score)) * (min_normalized - max_normalized) + max_normalized

    board[:, 0, -1] = normalized_score
    board[:, 1, -1] = 0


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)

        normalize_score(input_boards)

        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=args.batch_size, epochs=args.epochs)

    def predict(self, board):
        """
        board: np array with board
        """

        board = np.copy(board)
        board = board[np.newaxis, :, :]
        normalize_score(board)

        pi, v = self.nnet.model.predict(board)

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