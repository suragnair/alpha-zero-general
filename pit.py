import numpy as np

import Arena
from MCTS import MCTS
from othello.OthelloPlayers import HumanOthelloPlayer
from ultimate_tictactoe.UltimateTicTacToeGame import UltimateTicTacToeGame
from ultimate_tictactoe.UltimateTicTacToePlayers import RandomUltimateTictacToePlayer
from ultimate_tictactoe.keras.NNet import NNetWrapper as NNet
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

random_vs_cpu = True  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = True
num_games = 10


g = UltimateTicTacToeGame()

# all players
rp = RandomUltimateTictacToePlayer(g).play
hp = HumanOthelloPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp/', 'best.h5')
args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
elif random_vs_cpu:
    player2 = np
else:
    n2 = NNet(g)
    n2.load_checkpoint('./temp/', 'best.h5')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(player1=n1p, player2=player2, game=g, display=UltimateTicTacToeGame.display)

print(arena.playGames(num_games, verbose=True))
