"""
rts/pit.py

Compares 2 players against each other and outputs num wins p1/ num wins p2/ draws
"""
import Arena
from rts.RTSGame import display, RTSGame
from rts.src.config_class import CONFIG


def main():
    CONFIG.set_runner('pit')  # set visibility as pit
    g = RTSGame()
    player1, player2 = CONFIG.pit_args.create_players(g)
    arena = Arena.Arena(player1, player2, g, display=display)
    print(arena.playGames(CONFIG.pit_args.num_games, verbose=CONFIG.visibility))


if __name__ == '__main__':
    main()
