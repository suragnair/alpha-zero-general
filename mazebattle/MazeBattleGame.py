from __future__ import print_function

import random

import numpy as np

from Game import Game
from .MazeBattleLogic import Board

"""
Game class implementation for the game of MazeBattleGame.

"""


class MazeBattleGame(Game):
    def __init__(self, n=random.randint(5, 20)):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return self.n, self.n

    def getActionSize(self):
        # return number of actions
        return 33

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = Board(self.n, initialBoard=board.copy())
        actionType = None
        direction = None
        # [stay, move1, .. , move8, build1, .., build8, break1, .., break8, shoot1, .., shoot8]
        if action == 0:
            actionType = Board.ACTION_STAY
            direction = 0
        elif 1 <= action <= 8:
            actionType = Board.ACTION_MOVE
            direction = action
        elif 9 <= action <= 16:
            actionType = Board.ACTION_BUILD_WALL
            direction = action - 8
        elif 17 <= action <= 24:
            actionType = Board.ACTION_BREAK_WALL
            direction = action - 16
        elif 25 <= action <= 32:
            actionType = Board.ACTION_SHOOT
            direction = action - 24

        move = (actionType, direction)
        b.execute_move(move, player)
        return b.pieces, -player

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        b = Board(self.n, initialBoard=board)
        legalMoves = b.get_legal_moves(player)
        return np.array(legalMoves)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n, initialBoard=board)

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
        return self.exchange_board(board, player)

    def exchange_board(self, board, color):
        copied = board.copy()
        if color != 1:
            for x in range(self.n):
                for y in range(self.n):
                    if copied[x][y] == 1:
                        copied[x][y] = -1
                    elif copied[x][y] == -1:
                        copied[x][y] = 1
                    elif copied[x][y] == Board.TAG_PLAYER2_STARTING_POINT:
                        copied[x][y] = Board.TAG_PLAYER1_STARTING_POINT
                    elif copied[x][y] == Board.TAG_PLAYER1_STARTING_POINT:
                        copied[x][y] = Board.TAG_PLAYER2_STARTING_POINT
        return copied

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert (len(pi) == self.getActionSize())
        return []

    def stringRepresentation(self, board):
        # nxn numpy array (canonical board)
        return board.tostring()


def display(board):
    n = board.shape[0]
    print()
    for x in range(n):
        for y in range(n):
            tag = board[x][y]
            if tag == Board.TAG_EMPTY:
                print(".", end='')
            elif tag == Board.TAG_WALL_0_HIT:
                print("Ñ", end='')
            elif tag == Board.TAG_WALL_1_HIT:
                print("#", end='')
            elif tag == Board.TAG_PLAYER1_STARTING_POINT:
                print("S", end='')
            elif tag == Board.TAG_PLAYER2_STARTING_POINT:
                print("E", end='')
            elif tag == Board.TAG_PLAYER1:
                print("1", end='')
            elif tag == Board.TAG_PLAYER2:
                print("2", end='')
            else:
                print("*", end='')
        print("\n")
    print("--")
