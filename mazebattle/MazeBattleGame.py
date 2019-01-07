from __future__ import print_function

import numpy as np

from Game import Game
from .MazeBattleLogic import Board

"""
Game class implementation for the game of MazeBattleGame.

"""


class MazeBattleGame(Game):
    def __init__(self, n=3):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n, wallPercent=.35)
        return np.array(b.board)

    def getBoardSize(self):
        # (a,b) tuple
        return self.n, self.n

    def getActionSize(self):
        # return number of actions
        return 33

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n * self.n:
            return board, -player
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)
        return b.pieces, -player

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        legalMoves = board.get_legal_moves(player)
        for x, y in legalMoves:
            valids[self.n * x + y] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        # if b.has_legal_moves():
        #    return 0
        return 0  # We should not have any draw...
        # draw has a very little value 
        # return 1e-4

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return board.exchange_board(player)

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert (len(pi) == self.getActionSize())
        return []

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()


def display(board):
    n = board.shape[0]

    print("   ", end="")
    for y in range(n):
        print(y, "", end="")
    print("")
    print("  ", end="")
    for _ in range(n):
        print("-", end="-")
    print("--")
    for y in range(n):
        print(y, "|", end="")  # print the row #
        for x in range(n):
            piece = board[y][x]  # get the piece to print
            if piece == -1:
                print("X ", end="")
            elif piece == 1:
                print("O ", end="")
            else:
                if x == n:
                    print("-", end="")
                else:
                    print("- ", end="")
        print("|")

    print("  ", end="")
    for _ in range(n):
        print("-", end="-")
    print("--")
