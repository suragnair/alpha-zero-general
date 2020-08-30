from ArenaBuilder import *
from tictactoe.TicTacToeGame import TicTacToeGame as Game
from tictactoe.keras.NNet import NNetWrapper as NNet
from tictactoe.TicTacToePlayers import HumanTicTacToePlayer

class TicTacToeBuilder(ArenaBuilder):
    def create(self, human_vs_cpu=True):
        game = Game()
        mcts_args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})

        # set path and filename here
        path = ''
        filename = ''

        player1 = self.create_first_player(game, path, filename, NNet, mcts_args)
        player2 = self.create_second_player(game, HumanTicTacToePlayer(game).play, human_vs_cpu, NNet,
                                       mcts_args, path, filename)
        arena = self.set_arena(game, player1, player2, Game.display)
        return arena
