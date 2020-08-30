from ArenaBuilder import *
from rts.src.config_class import CONFIG
import Arena
from rts.RTSGame import display, RTSGame

"""
rts/pit.py

Compares 2 players against each other and outputs num wins p1/ num wins p2/ draws
"""

class Connect4ArenaBuilder(ArenaBuilder):
    def create(self, human_vs_cpu=True):
        CONFIG.set_runner('pit')  # set visibility as pit
        g = RTSGame()
        player1, player2 = CONFIG.pit_args.create_players(g)
        self.nr_games = CONFIG.pit_args.num_games
        self.verbose = CONFIG.visibility
        return Arena.Arena(player1, player2, g, display=display)