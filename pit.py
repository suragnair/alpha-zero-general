import Arena
from MCTS import MCTS
from ultimatetictactoe.UltimateTicTacToeGame import UltimateTicTacToeGame
from ultimatetictactoe.UltimateTicTacToePlayers import *
from ultimatetictactoe.keras.NNet import NNetWrapper as NNet


import numpy as np
from utils import *
"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = True

g = UltimateTicTacToeGame()

# all players
rp = RandomPlayer(g).play
hp = HumanUltimateTicTacToePlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp/','best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
def n1p(x, verbose=False):
    result = mcts1.getActionProb(x, temp=0)
    if verbose:
        s = mcts1.game.stringRepresentation(x)
        Ps = np.reshape(mcts1.Ps[s][:81], (9, 9))
        print('\n'.join(['\t'.join(['{:4}'.format(item) for item in row])
                         for row in Ps]))
    return np.argmax(result)

def n2p(x, verbose=False):
    result = mcts2.getActionProb(x, temp=0)
    if verbose:
        s = mcts2.game.stringRepresentation(x)
        Ps = np.reshape(mcts2.Ps[s][:81], (9, 9))
        print('\n'.join(['\t'.join(['{:4}'.format(item) for item in row])
                         for row in Ps]))
    return np.argmax(result)

if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g)
    n2.load_checkpoint('./temp/','best.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(n1p, player2, g, display=UltimateTicTacToeGame.display)

print(arena.playGames(2, verbose=True))
