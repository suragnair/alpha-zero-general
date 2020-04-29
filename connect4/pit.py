import sys
sys.path = sys.path[1:] # remove current path otherwise "import tensorflow" does not work
sys.path.append('..')
import tensorflow as tf
from connect4.Connect4Game import Connect4Game as Game
from connect4.tensorflow.NNet import NNetWrapper as nn
import numpy as np
from MCTS import MCTS
from utils import dotdict


if __name__ == "__main__":
    human_vs_cpu = True

    game = Game()
    net = nn(game)
    # todo adapt
    net.load_checkpoint('home/nmo/Documents/alpha-zero-general/temp/best.pth.tar')
    args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts1 = MCTS(g, net, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    hp = HumanOthelloPlayer(g).play

    if human_vs_cpu:
        player2 = hp
    else:
        n2 = NNet(g)
        #todo adapt
        n2.load_checkpoint('home/nmo/Documents/alpha-zero-general/temp/best.pth.tar')
        args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
        mcts2 = MCTS(g, n2, args2)
        n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
        player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

    arena = Arena.Arena(n1p, player2, g, display=OthelloGame.display)

    print(arena.playGames(2, verbose=True))



