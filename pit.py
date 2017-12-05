import Arena
from MCTS import MCTS
from OthelloGame import OthelloGame
from NNet import NNetWrapper as NNet

import numpy as np
from utils import *

g = OthelloGame(6)

# all players
rp = Arena.RandomPlayer(g).play
gp = Arena.GreedyOthelloPlayer(g).play
hp = Arena.HumanOthelloPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('/dev/models/6x100x25/','best.pth.tar')
args1 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


#n2 = NNet(g)
#n2.load_checkpoint('/dev/8x50x25/','best.pth.tar')
#args2 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
#mcts2 = MCTS(g, n2, args2)
#n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

arena = Arena.Arena(n1p, hp, g)
print(arena.playGames(20, verbose=True))
