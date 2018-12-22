import Arena
from td2020.TD2020Game import display
from td2020.src.config import CONFIG

CONFIG.set_runner('pit')  # set visibility as pit
CONFIG.to_file()
arena = Arena.Arena(CONFIG.pit_args.player1, CONFIG.pit_args.player2, CONFIG.g, display=display)
print(arena.playGames(CONFIG.pit_args.num_games, verbose=False))