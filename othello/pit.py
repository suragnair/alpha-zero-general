import sys
sys.path.append('..')
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet

from pit_utils import *

if __name__ == "__main__":
    mini_othello = False  # Play in 6x6 instead of the normal 8x8.
    human_vs_cpu = True

    if mini_othello:
        g = OthelloGame(6)
    else:
        g = OthelloGame(8)

    # all players
    rp = RandomPlayer(g).play
    gp = GreedyOthelloPlayer(g).play
    hp = HumanOthelloPlayer(g).play

    # nnet players
    path = '../pretrained_models/othello/pytorch/'
    if mini_othello:
        filename = '6x100x25_best.pth.tar'
    else:
        filename = '8x8_100checkpoints_best.pth.tar'

    player1 = create_first_player(g, path, filename, NNet)
    player2 = create_second_player(g, hp, human_vs_cpu, NNet, path=path,
                                   filename=filename)
    play(g, player1, player2, OthelloGame.display, 2)