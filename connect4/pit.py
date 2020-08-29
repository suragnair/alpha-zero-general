from connect4.Connect4Game import Connect4Game as Game
from connect4.tensorflow.NNet import NNetWrapper as NNet
from connect4.Connect4Players import HumanConnect4Player
from ArenaBuilder import *


class Connect4ArenaBuilder(ArenaBuilder):
    def create(self, human_vs_cpu=True):
        game = Game()
        mcts_args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})

        # set path and filename here
        path = ''
        filename = ''

        player1 = self.create_first_player(game, path, filename, NNet, mcts_args)
        player2 = self.create_second_player(game, HumanConnect4Player(game).play, human_vs_cpu, NNet,
                                       mcts_args, path, filename)
        arena = self.set_arena(game, player1, player2, Game.display)
        return arena




