import sys

import numpy as np

from Game import Game
from ultimate_tictactoe.UltimateTicTacToeLogic import Board

sys.path.append('..')


class UltimateTicTacToeGame(Game):
    def __init__(self, n=3):
        super().__init__()
        self.n = n
        self.N = n ** 2

    def getInitBoard(self):
        return Board(self.n)

    def getBoardSize(self):
        return self.N, self.N

    def getActionSize(self):
        return (self.N ** 2) + 1

    def getNextState(self, board, player, action):
        if action == self.N ** 2:
            return board, -player

        b = Board(self.n)
        b.copy(board)

        move = (int(action / self.N), action % self.N)
        b.execute_move(move, player)

        return b, -player

    def getValidMoves(self, board, player):
        valid_move = [0] * self.getActionSize()

        b = Board(self.n)
        b.copy(board)

        legal_moves = b.get_legal_moves()

        if len(legal_moves) == 0:
            valid_move[-1] = 1
            return np.array(valid_move)

        for x, y in legal_moves:
            valid_move[self.N * x + y] = 1

        return np.array(valid_move)

    def getGameEnded(self, board, player):
        b = Board(self.n)
        b.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0

        return 1e-4

    def getCanonicalForm(self, board, player):
        b = Board(self.n)
        b.copy(board)
        b.get_canonical_form(player)
        return b

    def getSymmetries(self, board, pi):
        assert (len(pi) == self.N ** 2 + 1)
        pi_board = np.reshape(pi[:-1], (self.N, self.N))
        symmetry_list = []

        for i in range(1, 5):
            for j in [True, False]:
                new_b = board.rot90(i, copy=True)
                new_pi = np.rot90(pi_board, i)
                if j:
                    new_b = board.fliplr(copy=True)
                    new_pi = np.fliplr(new_pi)
                symmetry_list += [(new_b, list(new_pi.ravel()) + [pi[-1]])]
        return symmetry_list

    def stringRepresentation(self, board):
        return board.tostring()

    @staticmethod
    def display(board):
        value = {-1: "X", 1: 'O', 0: '.'}
        board_pieces = board.pieces
        for row in range(len(board_pieces)):
            for item in range(len(board_pieces[0])):
                if item != len(board_pieces[0]) - 1:
                    print(value[board_pieces[row][item]], end=' ')
                elif item == len(board_pieces[0]) - 1:
                    print(value[board_pieces[row][item]])
                if item in [2, 5]:
                    print('|', end=' ')
            if row in [2, 5]:
                print('------+-------+------')
