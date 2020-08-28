import sys
sys.path.append('..')
from utils import *

import argparse
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import *

def relu_bn(inputs):
    relu1 = relu(inputs)
    bn = BatchNormalization()(relu1)
    return bn

def residual_block(x, filters, kernel_size=3):
    y = Conv2D(kernel_size=kernel_size,
               strides= (1),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    out = Add()([x, y])
    out = relu_bn(out)
    return out


class Connect4NNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        # Inputs
        self.input_boards = Input(shape=(self.board_x, self.board_y))
        inputs = Reshape((self.board_x, self.board_y, 1))(self.input_boards)


        bn1 = BatchNormalization()(inputs)
        conv1 = Conv2D(args.num_channels, kernel_size=3, strides=1, padding="same")(bn1)
        t = relu_bn(conv1)

        num_blocks_list = [2, 5, 5, 2]

        for i in range(len(num_blocks_list)):
            num_blocks = num_blocks_list[i]
            for j in range(num_blocks):
                t = residual_block(t, filters=self.args.num_channels)
        
        t = AveragePooling2D(4)(t)
        t = Flatten()(t)

        dense1 = relu_bn(Dense(512)(t))

        self.pi = Dense(self.action_size, activation='softmax', name='pi')(dense1)
        self.v = Dense(1, activation='tanh', name='v')(dense1)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
