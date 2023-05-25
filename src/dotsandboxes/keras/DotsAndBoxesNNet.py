from keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Activation, Dense, Dropout, Flatten, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam


class DotsAndBoxesNNet:

    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        flat = Flatten()(self.input_boards)
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(s_fc1))))  # batch_size x 1024
        s_fc3 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(s_fc2))))  # batch_size x 1024
        s_fc4 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc3))))          # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc4)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc4)                    # batch_size x 1
        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))
