import sys
sys.path = sys.path[1:] # remove current path otherwise "import tensorflow" does not work
sys.path.append('..')
import tensorflow as tf
from connect4.Connect4Game import Connect4Game as Game
from connect4.tensorflow.NNet import NNetWrapper as nn
from connect4.Connect4Players import HumanConnect4Player
import numpy as np
from MCTS import MCTS
from Arena import Arena
from utils import dotdict


if __name__ == "__main__":
    human_vs_cpu = True

    game = Game()
    net = nn(game)
    net.load_checkpoint('./pretrained_models/connect4/keras/', 'best.pth.tar')
    args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts1 = MCTS(game, net, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    hp = HumanConnect4Player(game).play

    if human_vs_cpu:
        player2 = hp
    else:
        n2 = nn(game)
        #todo adapt
        n2.load_checkpoint('./pretrained_models/connect4/keras/', 'best.pth.tar')
        args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
        mcts2 = MCTS(game, n2, args2)
        n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
        player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

    arena = Arena(n1p, player2, game, display=Game.display)

    print(arena.playGames(2, verbose=True))



