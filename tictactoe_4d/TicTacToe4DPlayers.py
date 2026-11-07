import numpy as np

from .TicTacToe4DLogic import forced_moves


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        return np.random.choice(np.flatnonzero(valids))


class OneStepPlayer():
    """Completes a line if it can, otherwise blocks one of the opponent, otherwise moves at random."""

    def __init__(self, game):
        self.game = game

    def play(self, board):
        squares, _ = forced_moves(np.asarray(board))
        if len(squares):
            return squares[0]
        return np.random.choice(np.flatnonzero(np.asarray(board).ravel() == 0))


class HumanPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        n = self.game.n
        valids = self.game.getValidMoves(board, 1)
        while True:
            parts = input('Your move as "w z y x": ').split()
            if len(parts) == 4 and all(p.isdigit() and int(p) < n for p in parts):
                action = int(np.ravel_multi_index([int(p) for p in parts], (n,) * 4))
                if valids[action]:
                    return action
            print('Invalid move')
