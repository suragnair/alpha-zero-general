from Arena import Arena, RandomPlayer, GreedyOthelloPlayer
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
p1 = RandomPlayer(curGame)
# p2 = GreedyOthelloPlayer(curGame)
AI = controller.AiController(1, 'BLACK', 1000)
arena = Arena(p1.play, AI.next_move, curGame)
pwins, nwins = arena.playGames(2)
print(pwins,nwins)