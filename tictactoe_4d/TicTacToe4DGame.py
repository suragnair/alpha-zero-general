import sys
sys.path.append('..')
from Game import Game
from .TicTacToe4DLogic import Board

import numpy as np

"""
Game class implementation for tic-tac-toe in four dimensions.

Based on the TicTacToeGame by Evgeny Tyurin and the OthelloGame by Surag Nair.
"""


class TicTacToe4DGame(Game):
    def __init__(self, n=4):
        self.n = n

    def getInitBoard(self):
        return np.array(Board(self.n).pieces)

    def getBoardSize(self):
        return (self.n,) * 4

    def getActionSize(self):
        # a move is a square: passing is never possible here
        return self.n ** 4

    def _board(self, pieces):
        b = Board(self.n)
        b.pieces = np.copy(pieces)
        return b

    def getNextState(self, board, player, action):
        b = self._board(board)
        b.execute_move(action, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        return (np.asarray(board).ravel() == 0).astype(int)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player won, -1 if player lost, a small value for a draw
        b = self._board(board)
        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0
        return 1e-4

    def getCanonicalForm(self, board, player):
        return player * board

    def getSymmetries(self, board, pi):
        """The 16 reflections of the hypercube. It has 384 symmetries in all, too many to store for each position."""
        assert len(pi) == self.getActionSize()
        pi_board = np.reshape(pi, self.getBoardSize())
        symmetries = []
        for flips in range(16):
            axes = tuple(axis for axis in range(4) if flips >> axis & 1)
            symmetries.append((np.flip(board, axes).copy(), np.flip(pi_board, axes).ravel().tolist()))
        return symmetries

    def stringRepresentation(self, board):
        return board.tobytes()

    @staticmethod
    def display(board):
        n = board.shape[0]
        marks = {1: 'X', -1: 'O', 0: '-'}
        for w in range(n):
            print('cube', w)
            for y in range(n):
                print('   '.join(' '.join(marks[int(board[w, z, y, x])] for x in range(n)) for z in range(n)))
            print()
