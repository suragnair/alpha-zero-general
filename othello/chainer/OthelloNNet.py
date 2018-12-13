import chainer
import chainer.functions as F  # NOQA
import chainer.links as L  # NOQA


class OthelloNNet(chainer.Chain):

    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(OthelloNNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, args.num_channels, 3, stride=1, pad=1)
            self.conv2 = L.Convolution2D(args.num_channels, args.num_channels, 3, stride=1, pad=1)
            self.conv3 = L.Convolution2D(args.num_channels, args.num_channels, 3, stride=1)
            self.conv4 = L.Convolution2D(args.num_channels, args.num_channels, 3, stride=1)

            self.bn1 = L.BatchNormalization(args.num_channels)
            self.bn2 = L.BatchNormalization(args.num_channels)
            self.bn3 = L.BatchNormalization(args.num_channels)
            self.bn4 = L.BatchNormalization(args.num_channels)

            self.fc1 = L.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
            self.fc_bn1 = L.BatchNormalization(1024)

            self.fc2 = L.Linear(1024, 512)
            self.fc_bn2 = L.BatchNormalization(512)

            self.fc3 = L.Linear(512, self.action_size)

            self.fc4 = L.Linear(512, 1)

    def forward(self, s):
        #                                                      s: batch_size x board_x x board_y
        s = F.reshape(s, (-1, 1, self.board_x, self.board_y))  # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                    # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                    # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                    # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                    # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = F.reshape(s, (-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4)))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), ratio=self.args.dropout)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), ratio=self.args.dropout)  # batch_size x 512

        pi = self.fc3(s)                                             # batch_size x action_size
        v = self.fc4(s)                                              # batch_size x 1

        return F.log_softmax(pi, axis=1), F.tanh(v)
