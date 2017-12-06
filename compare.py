from Arena import Arena, RandomPlayer, GreedyOthelloPlayer, HumanOthelloPlayer
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, Logger, AverageMeter
from NNet import NNetWrapper as NNet
from OthelloGame import OthelloGame
import game.controllers as controller
from game.board import Board

# nnet = NNet(self.game)
# nnet.load_checkpoint(folder=self.args.checkpoint, filename='chec0kpoint_' + str(i) + '.pth.tar')
# mcts = MCTS(self.game, pnet, self.args)


curGame = OthelloGame(8)
p2 = RandomPlayer(curGame)
AI = controller.AiController(1, 'WHITE', 5000000)
p1 = GreedyOthelloPlayer(curGame)
# p1 = HumanOthelloPlayer(curGame)
# p2 = HumanOthelloPlayer(curGame)
arena = Arena( AI.next_move,p1.play,curGame)
c = 0
# for i in xrange(100):
print(arena.playGame(verbose=False))
# 		c+=1
# 	print c
# print c
# print(pwins,nwins)