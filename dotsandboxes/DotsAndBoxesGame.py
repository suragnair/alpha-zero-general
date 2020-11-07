import sys
import numpy as np

sys.path.append('..')
from Game import Game
from .DotsAndBoxesLogic import Board


class DotsAndBoxesGame(Game):
    def __init__(self, n=3):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return 2*self.n+1, self.n+1

    def getActionSize(self):
        # return number of actions
        return 2 * (self.n + 1) * self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = Board(self.n)
        b.pieces = np.copy(board)

        if action == self.getActionSize() - 1:
            b.pieces[2, -1] = 0
        else:
            b.execute_move(action, player)

        return b.pieces, -player

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.get_legal_moves(player)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        b = Board(self.n)
        b.pieces = np.copy(board)

        if b.has_legal_moves():
            return 0

        if b.pieces[0][-1] == b.pieces[1][-1]:
            return -1 * player
        else:
            player_1_won = b.pieces[0][-1] > b.pieces[1][-1]
            return 1*player if player_1_won else -1*player

    def getCanonicalForm(self, board, player):
        board = np.copy(board)
        if player == -1:
            # swap score
            aux = board[0, -1]
            board[0, -1] = board[1, -1]
            board[1, -1] = aux
        return board

    def getSymmetries(self, board, pi):
        # mirror, rotational

        horizontal = np.copy(board[:self.n+1, :self.n])
        vertical = np.copy(board[-self.n:, :])
        t = self.n * (self.n + 1)
        pi_horizontal = np.copy(pi[:t]).reshape((self.n+1, self.n))
        pi_vertical = np.copy(pi[t:-1]).reshape((self.n, self.n+1))

        l = []

        for i in range(1, 5):
            horizontal = np.rot90(horizontal)
            vertical = np.rot90(vertical)
            pi_horizontal = np.rot90(pi_horizontal)
            pi_vertical = np.rot90(pi_vertical)

            for _ in [True, False]:
                horizontal = np.fliplr(horizontal)
                vertical = np.fliplr(vertical)
                pi_horizontal = np.fliplr(pi_horizontal)
                pi_vertical = np.fliplr(pi_vertical)

                new_board = Board(self.n)
                new_board.pieces = np.copy(board)
                new_board.pieces[:self.n + 1, :self.n] = vertical
                new_board.pieces[-self.n:, :] = horizontal

                l += [(new_board.pieces, list(pi_vertical.ravel()) + list(pi_horizontal.ravel()) + [pi[-1]])]

            aux = horizontal
            horizontal = vertical
            vertical = aux

            aux = pi_horizontal
            pi_horizontal = pi_vertical
            pi_vertical = aux
        return l

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()

    @staticmethod
    def display(board):
        n = board.shape[1]
        for i in range(n):
            for j in range(n-1):
                s = "*-x-" if board[i][j] else "*---"
                print(s, end="")
            print("*")
            if i < n-1:
                for j in range(n):
                    s = "x   " if board[i+n][j] else "|   "
                    print(s, end="")
            print("")

        print("Pass: {}".format(board[2,-1]))
        print("Score {} x {}".format(board[0, -1], board[1, -1]))