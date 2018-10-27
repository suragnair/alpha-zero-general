import Arena
from MCTS_NoNNet import MCTS_No_NNet
from td2020.TD2020Game import TD2020Game, display
from td2020.TD2020Players import *
from td2020.keras.NNet import NNetWrapper as NNet
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = TD2020Game(8)

# all players
rp = RandomPlayer(g).play
gp = GreedyTD2020Player(g).play
hp = HumanTD2020Player(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('.\\..\\temp\\', 'temp.pth.tar')
args1 = dotdict({'numMCTSSims': 1000, 'cpuct': 1.0})
mcts1 = MCTS_No_NNet(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

n2 = NNet(g)
n2.load_checkpoint('.\\..\\temp\\', 'temp.pth.tar')
args2 = dotdict({'numMCTSSims': 1000, 'cpuct': 1.0})
mcts2 = MCTS_No_NNet(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

arena = Arena.Arena(n1p, n2p, g, display=display)
print(arena.playGames(2, verbose=True))
