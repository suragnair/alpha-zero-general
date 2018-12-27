import sys

from td2020.src.config_class import CONFIG

sys.path.append('..')
import Arena
from td2020.TD2020Game import display, TD2020Game

CONFIG.set_runner('pit')  # set visibility as pit
g = TD2020Game()
player1, player2 = CONFIG.pit_args.create_players(g)
CONFIG.to_file()
arena = Arena.Arena(player1, player2, g, display=display)
print(arena.playGames(CONFIG.pit_args.num_games, verbose=CONFIG.visibility))
