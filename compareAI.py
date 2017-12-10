import Arena
from MCTS import MCTS
from OthelloGame import OthelloGame
from NNet import NNetWrapper as NNet
from OthelloGame import OthelloGame
import game.controllers as controller
from game.board import Board
import numpy as np

from utils import *

size=8
g = OthelloGame(size)

# all players
rp = Arena.RandomPlayer(g).play
gp = Arena.GreedyOthelloPlayer(g).play
hp = Arena.HumanOthelloPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('/dev/models/8x100x50/','best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
#n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0.5))
n1p = lambda x: np.random.choice(list(range(size*size+1)), p=mcts1.getActionProb(x, temp=0.5))
AI = controller.AiController(1, 'WHITE', 1000)


#n2 = NNet(g)
#n2.load_checkpoint('/dev/8x50x25/','best.pth.tar')
#args2 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
#mcts2 = MCTS(g, n2, args2)
#n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

arena = Arena.Arena(AI.next_move, n1p, g)
x = 0
num = 30
for _ in range(num):
    y=(arena.playGame(verbose=False))
    if y==1:
        x+=1

print("AI won ",x," out of ",num)
