from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .UltimateTicTacToeLogic import Board
import numpy as np

class UltimateTicTacToeGame(Game):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O",
        +2: " "
    }

    @staticmethod
    def getSquarePiece(piece):
        return UltimateTicTacToeGame.square_content[piece]

    def __init__(self):
        pass

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board()
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (11, 9)

    def getActionSize(self):
        # return number of actions
        return 9*9 + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == 9*9:
            return (board, -player)
        b = Board()
        b.pieces = np.copy(board)
        move = (int(action/9), action%9)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board()
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves()
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legalMoves:
            valids[9*x + y] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board()
        b.pieces = np.copy(board)
        win = b.is_win()
        if win == player:
            return 1
        elif win == -player:
            return -1
        elif win == 2:
            return 1e-4
        return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        l = np.array([[player*p for p in r] for r in board[:9]] + \
                     [[player*p if p in (-1, 1) else p for p in board[9]]] + \
                     [board[10]])
        return l

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == 9**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (9, 9))
        piece_board = np.array(board[:9])
        wins_board = np.reshape(board[9], (3, 3))
        next_board = np.reshape(board[10], (3, 3))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(piece_board, i)
                newW = np.rot90(wins_board, i)
                newN = np.rot90(next_board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newW = np.fliplr(newW)
                    newN = np.fliplr(newN)
                    newPi = np.fliplr(newPi)
                B = np.append(newB, [newW.ravel(), newN.ravel()], axis=0)
                l += [(B, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        return board.tostring()

    @staticmethod
    def display(board):
        print("    ", end="")
        for Y in range(3):
            for y in range(3):
                print(3*Y+y, end=" ")
            print(end="  ")
        print("")
        print("    -----   -----   -----")
        for Y in range(3):
            for y in range(3):
                print(3*Y+y, "| ", end="")    # print the row #
                for X in range(3):
                    for x in range(3):
                        piece = board[3*X+x, 3*Y+y]    # get the piece to print
                        if piece in (-1, 1):
                            print(UltimateTicTacToeGame.square_content[piece], end=" ")
                        elif board[10][3*X + Y] == 0:
                            print(".", end = " ")
                        else:
                            print(" ", end = " ")
                    if X < 2:
                        print(end="  ")
                print("|", end="")
                if Y == 1:
                    for x in range(3):
                        piece = board[9][3*x + y]
                        if piece in (-1, 1):
                            print(UltimateTicTacToeGame.square_content[piece], end=" ")
                        elif piece == 2:
                            print("T", end=" ")
                        else:
                            print(" ", end=" ")
                print()
            if Y < 2:
                print("")

        print("    -----   -----   -----")
