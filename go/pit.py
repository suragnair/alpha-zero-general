import Arena
from MCTS import MCTS
from go.Game import Game
from go.GoPlayers import *
from go.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use thisss script to play any two agents against each other, or play manually with
any agent.
"""
args = dotdict({
    'size': 9,                  #board size
    'numMCTSSims': 200,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 2,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1.1,
    'arenaNumMCTSSims': 20,      # simulations for arena
    'instinctArena': False,      # if set true reset Arena's MTCL tree each time
    'balancedGame': True,      # if balanced, black should win over 6 scores
    'resignThreshold': -0.9, # No Use. Resign Only in self-play Training
    'maxLevel': 7,
    'levelBased': True,
    'maxLeaves': 4,
})
args2 = dotdict({
    'size': 9,                  #board size
    'numMCTSSims': 200,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 2,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1.1,
    'arenaNumMCTSSims': 20,      # simulations for arena
    'instinctArena': False,      # if set true reset Arena's MTCL tree each time
    'balancedGame': True,      # if balanced, black should win over 6 scores
    'resignThreshold': -0.9, # No Use. Resign Only in self-play Training
    'maxLevel': 7,
    'levelBased': True,
    'maxLeaves': 10,
})
human_vs_cpu = True


g = Game(args)

# all players
rp = RandomPlayer(g).play
#gp = GreedyGoPlayer(g).play
hp = HumanGoPlayer(g).play



# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp/','9*9aug16th.tar')
mcts1 = MCTS(g, n1, args)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, arena=1, temp=0, ew=-1,instinctPlay=args.instinctArena, levelBased=args.levelBased)[0])

# nnet players
n2 = NNet(g)
n2.load_checkpoint('./temp/','9*9aug16th.tar')
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, arena=1, temp=0,instinctPlay=args.instinctArena, levelBased=args.levelBased)[0])

player2 = hp


arena = Arena.Arena(n1p, n1p, g, display=Game.display)
x, y, z, xb = arena.playGames(2, verbose=True)
print("94 win: ", x)
print("710 win: ", y)
print("Draw: ", z)
print("Bot Win with Black: ", xb )

