import os
import time

import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.utils import plot_model

from NeuralNet import NeuralNet
from td2020.keras.TD2020NNet import TD2020NNet
from td2020.src.config import VERBOSE_MODEL_FIT

"""
NNet.py

NNet wrapper uses defined nnet model to train and predict
"""


# noinspection PyMissingConstructor
class NNetWrapper(NeuralNet):
    def __init__(self, game):

        self.nnet = TD2020NNet(game)
        self.board_x, self.board_y, num_encoders = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.tensorboard = TensorBoard(log_dir='C:\\TrumpDefense2020\\TD2020\\Content\\Scripts\\td2020\\models\\logs' + type(self.nnet).__name__, histogram_freq=0, write_graph=True, write_images=True)
        plot_model(self.nnet.model, to_file='C:\\TrumpDefense2020\\TD2020\\Content\\Scripts\\td2020\\models\\' + type(self.nnet).__name__ + '_model_plot.png', show_shapes=True, show_layer_names=True)

    def train(self, examples):
        from td2020.src.config import CONFIG
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        input_boards = CONFIG.nnet_args.encoder.encode_multiple(input_boards)

        self.nnet.model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=CONFIG.nnet_args.batch_size, epochs=CONFIG.nnet_args.epochs, verbose=VERBOSE_MODEL_FIT, callbacks=[self.tensorboard])

    def predict(self, board):
        from td2020.src.config import CONFIG
        """
        board: np array with board
        """
        # timing
        start = time.time()

        board = CONFIG.nnet_args.encoder.encode(board)

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
