from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game

import numpy as np
import chess

def to_np(board):
  a = [0]*(8*8*6)
  for sq,pc in board.piece_map().items():
    a[sq*6+pc.piece_type-1] = 1 if pc.color else -1
  return np.array(a)

def from_move(move):
  return move.from_square*64+move.to_square

def to_move(action):
  to_sq = action % 64
  from_sq = int(action / 64)
  return chess.Move(from_sq, to_sq)

def who(turn):
  return 1 if turn else -1

def mirror_move(move):
  return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square))

CHECKMATE =1
STALEMATE= 2
INSUFFICIENT_MATERIAL= 3
SEVENTYFIVE_MOVES= 4
FIVEFOLD_REPETITION= 5
FIFTY_MOVES= 6
THREEFOLD_REPETITION= 7

class ChessGame(Game):

    def __init__(self, n=8):
        pass

    def getInitBoard(self):
        # return initial board (numpy board)
        return chess.Board()

    def getBoardSize(self):
        # (a,b) tuple
        # 6 piece type
        return (8, 8, 6)

    def toArray(self, board):
        return to_np(board)

    def getActionSize(self):
        # return number of actions
        return 64*64
        # return self.n*self.n*16+1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        assert(who(board.turn) == player)
        move = to_move(action)
        if not board.turn:
          # assume the move comes from the canonical board...
          move = mirror_move(move)
        if move not in board.legal_moves:
          # could be a pawn promotion, which has an extra letter in UCI format
          move = chess.Move.from_uci(move.uci()+'q') # assume promotion to queen
          if move not in board.legal_moves:
            assert False, "%s not in %s" % (str(move), str(list(board.legal_moves)))
        board = board.copy()
        board.push(move)
        return (board, who(board.turn))

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        assert(who(board.turn) == player)
        acts = [0]*self.getActionSize()
        for move in board.legal_moves:
          acts[from_move(move)] = 1
        return np.array(acts)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner is None:
                # draw return very little value
                return 1e-4
            else:
                return who(outcome.winner)
        return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        assert(who(board.turn) == player)
        if board.turn:
          return board
        else:
          return board.mirror()

    def getSymmetries(self, board, pi):
        # mirror, rotational
        return [(board,pi)]

    def stringRepresentation(self, board):
        return board.fen()

    @staticmethod
    def display(board):
        print(board)
