import Arena
from MCTS import MCTS
import numpy as np
from utils import *


class PitBuilder:
    def create(self):
        '''
        Creates an arena object
        :return: Arena
        '''
        raise NotImplementedError("Implement in subclass.")

    def create_first_player(self, g, path, filename, nnet, mcts_args):
        n1 = nnet(g)
        n1.load_checkpoint(path, filename)
        mcts1 = MCTS(g, n1, mcts_args)
        player1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
        return player1

    def create_second_player(self, g, hp, human_vs_cpu, nnet, mcts_args, path='', filename=''):
        if human_vs_cpu:
            player2 = hp
        else:
            n2 = nnet(g)
            n2.load_checkpoint(path, filename)
            mcts2 = MCTS(g, n2, mcts_args)
            player2 = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
        return player2

    def set_arena(self, game, player1, player2, display):
        return Arena.Arena(player1, player2, game, display=display)

    def play(self, arena, nr_games):
        print(arena.playGames(nr_games, verbose=True))
