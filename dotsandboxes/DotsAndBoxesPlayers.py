import numpy as np


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


# Will play at random, unless there's a chance to score a square
class GreedyRandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        previous_score = board[0, -1]
        for action in np.nonzero(valids)[0]:
            new_board, _ = self.game.getNextState(board, 1, action)
            new_score = new_board[0, -1]
            if new_score > previous_score:
                return action
        a = np.random.randint(self.game.getActionSize())
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanDotsAndBoxesPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        if board[2][-1] == 1:
            # We have to pass
            return self.game.getActionSize() - 1
        valids = self.game.getValidMoves(board, 1)
        while True:
            print("Valid moves: {}".format(np.where(valids == True)[0]))
            a = int(input())
            if valids[a]:
                return a
            print('Invalid move')
