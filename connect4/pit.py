import sys
import tensorflow as tf
from connect4.Connect4Game import Connect4Game as Game
from connect4.tensorflow.NNet import NNetWrapper as NNet
from connect4.Connect4Players import HumanConnect4Player
import numpy as np
from MCTS import MCTS
from Arena import Arena
from utils import dotdict
from ArenaBuilder import *


class Connect4ArenaBuilder(ArenaBuilder):
    def create(self, human_vs_cpu=True):
        game = Game()
        mcts_args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
        # adapt path
        player1 = self.create_first_player(game, '', '', NNet, mcts_args)
        # adapt path
        player2 = self.create_second_player(game, HumanConnect4Player(game).play, human_vs_cpu, NNet,
                                       mcts_args, '', '')
        arena = self.set_arena(game, player1, player2, Game.display)
        return arena




