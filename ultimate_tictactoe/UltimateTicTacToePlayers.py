from random import random

import numpy as np

from ultimate_tictactoe.UltimateTicTacToeLogic import Board


class RandomUltimateTictacToePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanUltimateTicTacToePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i / self.game.N), int(i % self.game.N))
        while True:
            # Python 3.x
            a = input()
            # Python 2.x 
            # a = raw_input()

            x, y = [int(x) for x in a.split(' ')]
            a = self.game.N * x + y if x != -1 else self.game.N ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a


class RLUTTTPlayer:
    def __init__(self, learningModel):
        self.learningAlgo = learningModel
        super(RLUTTTPlayer, self).__init__()

    def makeNextMove(self, board):
        if not (board.is_win(1) or board.is_win(-1)):
            if random.uniform(0, 1) < 0.8:
                choices = {}
                legal_space = board.get_legal_moves()
                for lx, ly in legal_space:
                    area = board.get_area(lx, ly)
                    possible = Board(board.n)
                    possible.copy(board)
                    possible.execute_move((lx, ly), 1)
                    choices[(lx, ly)] = self.learningAlgo.getBoardStateValue(self.player, board,
                                                                             possible.pieces.reshape(81, ))
                pickOne = max(choices, key=choices.get)
            else:
                legal = board.get_legal_moves()
                pickOne = random.choice(legal)

            return board.N * pickOne[0] + pickOne[1]

    def loadLearning(self, filename):
        self.learningAlgo.loadLearning(filename)
