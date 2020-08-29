from gobang.GobangGame import GobangGame as Game
from gobang.tensorflow.NNet import NNetWrapper as NNet
from gobang.GobangPlayers import HumanGobangPlayer
from ArenaBuilder import *


class GobangArenaBuilder(ArenaBuilder):
    def create(self, human_vs_cpu=True):
        game = Game()
        mcts_args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
        # adapt path
        player1 = self.create_first_player(game, '', '', NNet, mcts_args)
        # adapt path
        player2 = self.create_second_player(game, HumanGobangPlayer(game).play, human_vs_cpu, NNet,
                                       mcts_args, '', '')
        arena = self.set_arena(game, player1, player2, Game.display)
        return arena




