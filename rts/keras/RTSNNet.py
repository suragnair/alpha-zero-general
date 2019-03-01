import os
import sys

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation, Dense, Dropout, Flatten, Reshape
from tensorflow.python.keras.optimizers import Adam

sys.path.append('../..')
from rts.src.config import USE_TF_CPU, SHOW_TENSORFLOW_GPU

if USE_TF_CPU:
    print("Using TensorFlow CPU")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if not SHOW_TENSORFLOW_GPU:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
RTSNNet

Defined NNet model used for game TD2020
"""


class RTSNNet:
    def __init__(self, game, encoder):
        """
        NNet model, copied from Othello NNet, with reduced fully connected layers fc1 and fc2 and reduced nnet_args.num_channels
        :param game: game configuration
        :param encoder: Encoder, used to encode game boards
        """
        from rts.src.config_class import CONFIG

        # game params
        self.board_x, self.board_y, num_encoders = game.getBoardSize()
        self.action_size = game.getActionSize()

        """
        num_encoders = CONFIG.nnet_args.encoder.num_encoders
        """
        num_encoders = encoder.num_encoders

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y, num_encoders))  # s: batch_size x board_x x board_y x num_encoders

        x_image = Reshape((self.board_x, self.board_y, num_encoders))(self.input_boards)  # batch_size  x board_x x board_y x num_encoders
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(CONFIG.nnet_args.num_channels, 3, padding='same', use_bias=False)(x_image)))  # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(CONFIG.nnet_args.num_channels, 3, padding='same', use_bias=False)(h_conv1)))  # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(CONFIG.nnet_args.num_channels, 3, padding='valid', use_bias=False)(h_conv2)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(CONFIG.nnet_args.num_channels, 3, padding='valid', use_bias=False)(h_conv3)))  # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(CONFIG.nnet_args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(256, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(CONFIG.nnet_args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(128, use_bias=False)(s_fc1))))  # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)  # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(CONFIG.nnet_args.lr))
