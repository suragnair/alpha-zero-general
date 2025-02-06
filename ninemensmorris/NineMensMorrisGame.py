from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .NineMensMorrisLogic import Board
import numpy as np
import copy

'''
Author: Jonas Jakob
Created: May 31, 2023

Implementation of the Game Class for NineMensMorris
Many of these functions are based on those from OthelloGame.py:
https://github.com/suragnair/alpha-zero-general/blob/master/othello/OthelloGame.py

'''
class NineMensMorrisGame(Game):

    """
    inititalizes the list of all possible moves, the policy rotation vector and
    the number of moves without a mill to determine a draw
    """
    def __init__(self):
      self.n = 6
      self.all_moves = self.get_all_moves()
      self.policy_rotation_vector = self.get_policy_roation90()
      self.MAX_MOVES_WITHOUT_MILL = 200

    """
    Gets the list of all possible moves
    """
    def get_all_moves(self):
       """
       Returns:
          moves: A list with all possible moves for the game
       """
       moves = self.get_all_moves_phase_zero() + self.get_all_moves_phase_one_and_two()
       return list(moves)

    """
    Gets the lookup list for the rotation of the vector of legal moves
    """
    def get_policy_roation90(self):
        """
        Returns:
            rotation90: lookup list for the rotation of the legal moves vector
        """

        rotation90 = [-1] * len(self.all_moves)

        i = 0
        while i < len(self.all_moves):

            move = self.all_moves[i]
            rotatedmove = self.rotate(move)
            newindex = self.all_moves.index(rotatedmove)
            rotation90[i] = newindex

            i+=1

        return rotation90

    """
    Rotates a move by 90 degrees
    """
    def rotate(self, move):
        """
        Input:
            move: Tuple (origin, destination, piece to take)
        Returns:
            rot_move: Tuple (neworigin, newdestination, newpiece to take)
        """
        if move[0] == 'none':
            neworigin = 'none'

        elif move[0] in [6,7,14,15,22,23]:
            neworigin = move[0] - 6

        else:
            neworigin = move[0] + 2

        if move[1] in [6,7,14,15,22,23]:
            newdestination = move[1] - 6

        else:
            newdestination = move[1] + 2

        if move[2] == 'none':
            newenemy = 'none'

        elif move[2] in [6,7,14,15,22,23]:
            newenemy = move[2] - 6

        else:
            newenemy = move[2] + 2

        return (neworigin, newdestination, newenemy)

    """
    Generates all possible moves for game phase zero
    """
    def get_all_moves_phase_zero(self):
        """
        Returns:
            moves: list of all possible move Tuples
        """

        moves = []
        index = 0

        while index < 24:

            moves.append(("none",index,"none"))
            count = 0

            while count < 24:

                if count != index:

                    moves.append(("none",index,count))

                count += 1

            index += 1

        return list(moves)

    """
    Generates all possible moves for game phase one and two
    """
    def get_all_moves_phase_one_and_two(self):
        """
        Returns:
            moves: list of all possible move Tuples
        """

        moves = []
        index_origin = 0

        while index_origin < 24:

            index_move = 0

            while index_move < 24:

                if index_move != index_origin:

                    moves.append((index_origin,index_move,"none"))

                    count = 0

                    while count <24:

                        if (count != index_move)and(count != index_origin):

                            moves.append((index_origin,index_move,count))

                        count += 1

                index_move += 1

            index_origin += 1

        return list(moves)
    """
    based on Othellogame.py
    Gets the initial form of the board in game phase zero
    """
    def getInitBoard(self):
        """
        Returns:
            board: the initial board configuration
        """
        b = Board()

        return np.array(b.pieces)

    """
    based on Othellogame.py
    Gets the size of the board image in a Tuple (x, y)
    """
    def getBoardSize(self):
        """
        Returns:
            dimensions: a Tuple with the board dimensions
        """
        return (6, 6)

    """
    based on Othellogame.py
    Gets the number of all possible actions
    """
    def getActionSize(self):
        """
        Returns:
            actionssize: number of all moves
        """
        return len(self.all_moves)

    """
    based on Othellogame.py
    Returns the next state to given a board, player and move
    """
    def getNextState(self, board, player, move):
        """
        Input:
            board: current board image
            player: current player (1 or -1)
            move: move Tuple

        Returns:
            new_state: Tuple (new board, next player)
        """
        b = Board()
        b.pieces = np.copy(board)

        b.execute_move(player, move, self.all_moves)

        return (b.pieces, -player)

    """
    based on Othellogame.py
    Gets a vector of size == ActionSize that marks legal moves for the current
    board and player with 1
    """
    def getValidMoves(self, board, player):
        """
        Input:
            board: current board image
            player current player (1 or -1)
        Returns:
            valid_moves: np array of ones and zeros marking the legal moves
        """
        b = Board()
        b.pieces = np.copy(board)

        valid_moves = b.get_legal_move_vector(player, self.all_moves)

        return np.array(valid_moves)

    """
    based on Othellogame.py
    Determines if the game has ended for the given board and player.
    """
    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)
        Returns:
            game_ended: 0 if game has not ended. 1 if player won, -1 if player
            lost, small non-zero value for draw.
        """
        assert(not isinstance(board, str))

        b = Board()
        b.pieces = np.copy(board)

        if b.pieces[4][1] >= 50:
            return 0.0001
        elif not b.has_legal_moves(player):
            return -1
        elif not b.has_legal_moves(-player):
            return 1
        elif len(b.get_player_pieces(player)) < 3 and b.pieces[4][0] == 18:
            return -1
        elif len(b.get_player_pieces(-player)) < 3 and b.pieces[4][0] == 18:
            return 1
        elif b.has_legal_moves(-player) and b.has_legal_moves(player):
            return 0

    """
    Based on Othellogame.py
    Multiplies each element with the given player, resulting in a canonical
    board from the perspective of the given player. The given players pieces
    are always represented as 1 in the Canonical Form.
    Note: no true canonical form
    """
    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)
        Returns:
            b: canonical board
        """
        b = np.zeros((6,6), dtype=int)
        count_placements = copy.deepcopy(board[4][0])
        current_moves = copy.deepcopy(board[4][1])
        index = 0
        while index < 4:
          item = 0
          while item < 6:
            b[index][item] = board[index][item] * player
            item += 1
          index += 1

        b[4][0] = count_placements
        b[4][1] = current_moves
        return b
    """
    Based on Othellogame.py
    Gets some Symmetries by rotating the board three times, each time also
    adapting the legal moves vector to the new board
    """
    def getSymmetries(self, board, pi):
        """
        Input:
            board: the current board
            pi: the legal moves vector for the current board
        Returns:
            results: three board rotations
        """

        assert(len(pi) == len(self.all_moves))
        b = Board()
        b.pieces = np.copy(board)

        results = b.get_board_rotations(pi, self.all_moves, self.policy_rotation_vector)

        return results

    """
    Gets a String representation for the board, used for hashing in mcts
    """
    def stringRepresentation(self, board):
        """
        Input:
            board: the current board
        Returns:
            board_s: String representation of the board
        """
        board_s = ""
        index = 0
        i = 0
        while i < 4:
          while index < 6:
            board_s = board_s + str(board[i][index]) + ","
            index += 1
          index = 0
          i += 1
        board_s = board_s + str(board[4][0]) + ","
        board_s = board_s + str(board[4][1])

        return board_s

    """
    Gets a readable String representation for the board
    """
    def stringRepresentationReadable(self, board):
        """
        Input:
            board: the current board
        Returns:
            board_s: String representation of the board
        """
        board_s = ""
        index = 0
        i = 0
        while i < 4:
          while index < 6:
            board_s = board_s + str(board[i][index]) + ","
            index += 1
          index = 0
          i += 1
        board_s = board_s + str(board[4][0]) + ","
        board_s = board_s + str(board[4][1])

        return board_s

    @staticmethod
    def display(boardd):
        board = Board()
        board.pieces = np.copy(boardd)
        board, stuff = board.piecesToArray()
        assert(0 <= stuff[0] <= 18)
        assert(len(board) == 24)

        print('{}________ {} ________{}'.format(board[0], board[1], board[2]))
        print('|          |          | ')
        print('   {}      {}      {}    '.format(board[8], board[9], board[10]))
        print('|  |       |       |  | ')
        print('|  |  {}__ {} __{}      '.format(board[16], board[17], board[18]))
        print('|  |  |         |  |  | ')
        print('{}-{}-{}        {}-{}-{}'.format(board[7], board[15], board[23], board[19], board[11], board[3]))
        print('|  |  |         |  |  | ')
        print('|  |  {}__ {} __{}     '.format(board[22], board[21], board[20]))
        print('|  |       |       |  | ')
        print('|  {}_____ {} _____{}  '.format(board[14], board[13], board[12]))
        print('|          |          | ', )
        print('{} _______ {} ______ {} '.format(board[6], board[5], board[4]))
