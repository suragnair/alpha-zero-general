import chess
import random
import numpy as np
from _chess.ChessGame import who, from_move, mirror_move
from stockfish import Stockfish

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, who(board.turn))
        moves = np.argwhere(valids==1)
        return random.choice(moves)[0]

def move_from_uci(board, uci):
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        print('expected an UCI move')
        return None
    if move not in board.legal_moves:
        print('expected a valid move')
        return None
    return move

class HumanChessPlayer():
    def __init__(self, game):
        pass

    def play(self, board):
        mboard = board
        if board.turn:
            mboard = board.mirror()
        print('Valid Moves', end=':')
        for move in mboard.legal_moves:
            print(move.uci(), end=',')
        print()
        human_move = input()
        move = move_from_uci(mboard, human_move.strip())
        if move is None:
            print('try again, e.g., %s' % random.choice(list(mboard.legal_moves)).uci())
            return self.play(board)
        if board.turn:
            move = mirror_move(move)
        return from_move(move)

class StockFishPlayer():
    def __init__(self, game, elo=1000):
        self.stockfish = Stockfish(parameters={"Threads": 2, "Minimum Thinking Time": 30})
        self.stockfish.set_elo_rating(elo)

    def play(self, board):
        self.stockfish.set_fen_position(board.fen())
        uci_move = self.stockfish.get_best_move()
        move = move_from_uci(board, uci_move.strip())
        return from_move(move)
