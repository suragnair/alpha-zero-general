""""

    This is a Regression Test Suite to automatically test all combinations of games and ML frameworks. Each test
    plays two quick games using an untrained neural network (randomly initialized) against a random player.

    In order for the entire test suite to run successfully, all the required libraries must be installed.  They are:
    Pytorch, Keras.

     [ Games ]      Pytorch      Keras
      -----------   -------      -----
    - Othello        [Yes]       [Yes]
    - TicTacToe                  [Yes]
    - TicTacToe3D                [Yes]
    - Connect4                   [Yes]
    - Gobang                     [Yes]
    - Tafl           [Yes]       [Yes]
    - Rts                        [Yes]
    - DotsAndBoxes               [Yes]
"""

import unittest

import Arena
from MCTS import MCTS

from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import RandomPlayer
from othello.pytorch.NNet import NNetWrapper as OthelloPytorchNNet
from othello.keras.NNet import NNetWrapper as OthelloKerasNNet

from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.keras.NNet import NNetWrapper as TicTacToeKerasNNet

from tictactoe_3d.TicTacToeGame import TicTacToeGame as TicTacToe3DGame
from tictactoe_3d.keras.NNet import NNetWrapper as TicTacToe3DKerasNNet

from connect4.Connect4Game import Connect4Game
from connect4.keras.NNet import NNetWrapper as Connect4KerasNNet

from gobang.GobangGame import GobangGame
from gobang.keras.NNet import NNetWrapper as GobangKerasNNet

from tafl.TaflGame import TaflGame
from tafl.pytorch.NNet import NNetWrapper as TaflPytorchNNet
from tafl.keras.NNet import NNetWrapper as TaflKerasNNet

from rts.RTSGame import RTSGame
from rts.keras.NNet import NNetWrapper as RTSKerasNNet

from dotsandboxes.DotsAndBoxesGame import DotsAndBoxesGame
from dotsandboxes.keras.NNet import NNetWrapper as DotsAndBoxesKerasNNet

import numpy as np
from utils import *

class TestAllGames(unittest.TestCase):

    @staticmethod
    def execute_game_test(game, neural_net):
        rp = RandomPlayer(game).play

        args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
        mcts = MCTS(game, neural_net(game), args)
        n1p = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

        arena = Arena.Arena(n1p, rp, game)
        print(arena.playGames(2, verbose=False))
   
    def test_othello_pytorch(self):
        self.execute_game_test(OthelloGame(6), OthelloPytorchNNet)

    def test_othello_keras(self):
        self.execute_game_test(OthelloGame(6), OthelloKerasNNet)

    def test_tictactoe_keras(self):
        self.execute_game_test(TicTacToeGame(), TicTacToeKerasNNet)
    
    def test_tictactoe3d_keras(self):
        self.execute_game_test(TicTacToe3DGame(3), TicTacToe3DKerasNNet)
  
    def test_gobang_keras(self):
        self.execute_game_test(GobangGame(), GobangKerasNNet)

    def test_tafl_pytorch(self):
        self.execute_game_test(TaflGame(5), TaflPytorchNNet)

    def test_tafl_keras(self):
        self.execute_game_test(TaflGame(5), TaflKerasNNet)
  
    def test_connect4_keras(self):
        self.execute_game_test(Connect4Game(5), Connect4KerasNNet)

    def test_rts_keras(self):
        self.execute_game_test(RTSGame(), RTSKerasNNet)

    def test_dotsandboxes_keras(self):
        self.execute_game_test(DotsAndBoxesGame(3), DotsAndBoxesKerasNNet)

if __name__ == '__main__':
    unittest.main()
