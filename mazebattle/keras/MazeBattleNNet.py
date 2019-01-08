from keras.layers import *
from keras.models import *
from keras.optimizers import *


class MazeBattleNNet:
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))  # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)  # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 3, padding='same', kernel_initializer='random_uniform')(
                x_image)))  # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 3, padding='same', kernel_initializer='random_uniform')(
                h_conv1)))  # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 3, padding='valid', kernel_initializer='random_uniform')(
            h_conv2)))  # batch_size  x (board_x) x (board_y) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 3, padding='valid', kernel_initializer='random_uniform')(
            h_conv3)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(args.dropout)(
            Activation('relu')(BatchNormalization(axis=1)(
                Dense(1024, kernel_initializer='random_uniform')(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(
            Activation('relu')(BatchNormalization(axis=1)(
                Dense(512, kernel_initializer='random_uniform')(s_fc1))))  # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi', kernel_initializer='random_uniform')(
            s_fc2)  # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v', kernel_initializer='random_uniform')(s_fc2)  # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))
