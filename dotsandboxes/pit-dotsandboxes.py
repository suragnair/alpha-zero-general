import os
import numpy as np
import Arena
from MCTS import MCTS
from utils import dotdict
from dotsandboxes.DotsAndBoxesGame import DotsAndBoxesGame
from dotsandboxes.DotsAndBoxesPlayers import HumanDotsAndBoxesPlayer, RandomPlayer, GreedyRandomPlayer
from dotsandboxes.keras.NNet import NNetWrapper


g = DotsAndBoxesGame(n=3)

hp1 = HumanDotsAndBoxesPlayer(g).play
hp2 = HumanDotsAndBoxesPlayer(g).play

rp1 = RandomPlayer(g).play
rp2 = RandomPlayer(g).play

grp1 = GreedyRandomPlayer(g).play
grp2 = GreedyRandomPlayer(g).play

numMCTSSims = 50
n1 = NNetWrapper(g)
n1.load_checkpoint(os.path.join('../', 'pretrained_models', 'dotsandboxes', 'keras', '3x3'), 'best.pth.tar')
args1 = dotdict({'numMCTSSims': numMCTSSims, 'cpuct': 1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

n2 = NNetWrapper(g)
n2.load_checkpoint(os.path.join('../', 'pretrained_models', 'dotsandboxes', 'keras', '3x3'), 'best.pth.tar')
args2 = dotdict({'numMCTSSims': numMCTSSims, 'cpuct': 1.0})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

# Play AlphaZero versus Human
p1 = n1p
p2 = hp1
arena = Arena.Arena(p1, p2, g, display=DotsAndBoxesGame.display)
oneWon, twoWon, draws = arena.playGames(2, verbose=True)
print("oneWon: {}, twoWon: {}, draws: {}".format(oneWon, twoWon, draws))

# # Play Greedy vs Greedy
# p1 = grp1
# p2 = grp2
# arena = Arena.Arena(p1, p2, g, display=DotsAndBoxesGame.display)
# oneWon, twoWon, draws = arena.playGames(100, verbose=False)
# print("oneWon: {}, twoWon: {}, draws: {}".format(oneWon, twoWon, draws))

# # Play AlphaZero vs Greedy
# p1 = n1p
# p2 = grp2
# arena = Arena.Arena(p1, p2, g, display=DotsAndBoxesGame.display)
# oneWon, twoWon, draws = arena.playGames(2, verbose=False)
# print("oneWon: {}, twoWon: {}, draws: {}".format(oneWon, twoWon, draws))