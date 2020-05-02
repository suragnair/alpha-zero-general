import Arena
from MCTS import MCTS
import numpy as np
from utils import *

def create_first_player(g, path, filename, nnet):
    n1 = nnet(g)
    n1.load_checkpoint(path, filename)
    args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts1 = MCTS(g, n1, args1)
    player1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
    return player1

def create_second_player(g, hp, human_vs_cpu, nnet, path='', filename=''):
    if human_vs_cpu:
        player2 = hp
    else:
        n2 = nnet(g)
        n2.load_checkpoint(path, filename)
        args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
        mcts2 = MCTS(g, n2, args2)
        player2 = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
    return player2

def play(game, player1, player2, display, nr_games):
    arena = Arena.Arena(player1, player2, game, display=display)
    print(arena.playGames(nr_games, verbose=True))

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
