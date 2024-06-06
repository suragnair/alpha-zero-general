import numpy as np
import sys
sys.path.append('..')
import Arena
from MCTS import MCTS
from yoxii.YoxiiGame import YoxiiGame
import yoxii.YoxiiPlayers as Yoxii
from yoxii.pytorch.NNet import NNetWrapper as NNet
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = True

g = YoxiiGame()

# all players
rp = Yoxii.RandomPlayer(g).play
hp = Yoxii.HumanPlayer(g).play


import os
current_directory = os.getcwd()
print("Current working directory:", current_directory)


# nnet players
n1 = NNet(g)
n1.load_checkpoint(current_directory+"\models\checkpoints\\",'best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0)) # exchange with rp if you dont want to lose

n2 = NNet(g)
n2.load_checkpoint(current_directory+"\models\checkpoints\\",'best.pth - copy3.6.24.tar')
args2 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0)) # exchange with rp if you dont want to lose


player2 = hp # Exchange with rp (random player), hp (human player)

arena = Arena.Arena(n1p, player2, g, display=YoxiiGame.display)

print(arena.playGames(2 if player2 == hp else 10, verbose=player2 == hp))
#print(arena.playGames(2, verbose=True))
