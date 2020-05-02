import sys
sys.path = sys.path[1:] # remove current path otherwise "import tensorflow" does not work
sys.path.append('..')
import tensorflow as tf
from connect4.Connect4Game import Connect4Game as Game
from connect4.tensorflow.NNet import NNetWrapper as NNet
from connect4.Connect4Players import HumanConnect4Player
import numpy as np
from MCTS import MCTS
from Arena import Arena
from utils import dotdict
from pit_utils import *


if __name__ == "__main__":
    human_vs_cpu = False

    game = Game()
    mcts_args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
    # adapt path
    player1 = create_first_player(game, '', '', NNet, mcts_args)
    # adapt path
    player2 = create_second_player(game, HumanConnect4Player(game).play, human_vs_cpu, NNet,
                                   mcts_args, '', '')
    play(game, player1, player2, Game.display, 2)



