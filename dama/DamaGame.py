from __future__ import print_function
import os
import sys
from dama.DamaLogic import Board
from tafl.Digits import int2base
import numpy as np

sys.path.append('..')
from Game import Game


class Dama(Game):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """

    def __init__(self, n=8):
        self.n = n

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        # (a,b) tuple
        return self.n, self.n

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        # return number of actions
        return self.n ** 4  # damada kurallar gereği self.n * self.n * 4 olması gerekebilir

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """

        b = Board(self.n)
        b.pieces = np.copy(board)
        move = int2base(action, self.n, 4)
        b.execute_move(move, player)
        b.pieces = np.rot90(b.pieces, 2)
        return b.pieces, -player

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves(player)
        for x1, y1, x2, y2 in legalMoves:
            valids[x1 + y1 * self.n + x2 * self.n ** 2 + y2 * self.n ** 3] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        b = Board(self.n)
        b.pieces = np.copy(board)

        return b.is_game_over(player)

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return player * board

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [(board, pi)]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.tostring()

    def display(board):
        # symbols for each type of piece
        symbols = {
            0: ' ',  # Empty space
            1: '●',  # Regular piece for player 1
            -1: '○',  # Regular piece for player 2
            2: '●',  # King piece for player 1
            -2: '○'  # King piece for player 2
        }

        # ANSI escape codes for colors
        colors = {
            0: '\033[0m',  # No color
            1: '\033[34m',  # Blue for player 1
            -1: '\033[31m',  # Red  for player 2
            2: '\033[34m',  # Blue for king of player 1
            -2: '\033[31m'  # Red  for king of player 2
        }

        print("  +" + "---+" * board.shape[1])
        for i, row in enumerate(board):
            print(f"{i} | ", end='')
            for value in row:
                if value == 0:
                    print("  | ", end='')
                else:
                    print(f"{colors[value]}{symbols[value]}\033[0m | ", end='')
            print("\n  +" + "---+" * board.shape[1])
