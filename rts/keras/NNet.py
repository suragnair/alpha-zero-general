import os
import sys

import numpy as np

sys.path.append('../..')
from NeuralNet import NeuralNet
from rts.keras.RTSNNet import RTSNNet
from rts.src.config import VERBOSE_MODEL_FIT

"""
NNet.py

NNet wrapper uses defined nnet model to train and predict

tensorflow.python.framework.errors_impl.NotFoundError: No algorithm worked! :)
"""


# noinspection PyMissingConstructor
class NNetWrapper(NeuralNet):
    def __init__(self, game, encoder=None):
        from rts.src.config_class import CONFIG

        # default
        encoder = encoder or CONFIG.nnet_args.encoder

        self.nnet = RTSNNet(game, encoder)
        self.board_x, self.board_y, num_encoders = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.encoder = encoder

    def train(self, examples):
        from rts.src.config_class import CONFIG

        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        """
        input_boards = CONFIG.nnet_args.encoder.encode_multiple(input_boards)
        """
        input_boards = self.encoder.encode_multiple(input_boards)

        self.nnet.model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=CONFIG.nnet_args.batch_size, epochs=CONFIG.nnet_args.epochs, verbose=VERBOSE_MODEL_FIT)

    def predict(self, board, player=None):

        """
        board: np array with board
        """

        # If we are learning model, use only 1 encoder on both players, else use player, specific encoder, as we might be comparing 2 different encoders using 'pit'
        """
        if CONFIG.runner == "learn":
            board = CONFIG.nnet_args.encoder.encode(board)
        else:
            if player == 1:
                board = CONFIG.player1_config.encoder.encode(board)
            else:
                board = CONFIG.player2_config.encoder.encode(board)
        """
        board = self.encoder.encode(board)

        # preparing input
        board = board[np.newaxis, :, :]

        # run
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
