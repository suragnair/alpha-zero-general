import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = True  # Play in 6x6 instead of the normal 8x8.
opponent_type = "random"  # Possible opponent types: "random", "greedy", "human", "neural-net"


def create_nn_player(mini):
    # Create a trained neural network player
    nn = NNet(g)
    if mini:
        nn.load_checkpoint('./pretrained_models/othello/pytorch/', '6x100x25_best.pth.tar')
    else:
        nn.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
    args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts = MCTS(g, nn, args)
    player = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
    return player


if mini_othello:
    g = OthelloGame(6)
else:
    g = OthelloGame(8)

player1 = create_nn_player(mini_othello)

if opponent_type == "human":
    player2 = HumanOthelloPlayer(g).play
elif opponent_type == "greedy":
    player2 = GreedyOthelloPlayer(g).play
elif opponent_type == "random":
    player2 = RandomPlayer(g).play
elif opponent_type == "neural-net":
    player2 = create_nn_player(mini_othello)
else:
    player2 = HumanOthelloPlayer(g).play

arena = Arena.Arena(player1, player2, g, display=OthelloGame.display)

print(arena.playGames(2, verbose=True))
