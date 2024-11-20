import Arena
from MCTS import MCTS
from gomoku.GomokuGame import GomokuGame
from gomoku.GomokuPlayers import *
from gomoku.pytorch.NNet import NNetWrapper as NNet
from pickle import Pickler, Unpickler
import os


import numpy as np
from utils import *
import logging
import coloredlogs

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

def main():
    human_vs_cpu = True

    g = GomokuGame(6)

    # all players
    rp = RandomPlayer(g).play
    gp = GreedyGomokuPlayer(g).play
    hp = HumanGomokuPlayer(g).play



    # nnet players
    n1 = NNet(g)
    n1.load_checkpoint('./checkpoints/gomoku/6*6_numeps_100_num_mcts_sims_25_2_input_channels','best.pth.tar')

    args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    if human_vs_cpu:
        player2 = hp
    else:
        n2 = NNet(g)
        n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
        args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
        mcts2 = MCTS(g, n2, args2)
        n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

        player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

    arena = Arena.Arena(n1p, player2, g, display=GomokuGame.display)

    print(arena.playGames(2, verbose=True))


def two_model_compete():
    g = GomokuGame(6)
    n1 = NNet(g, input_channels = 2)
    n1.load_checkpoint('./checkpoints/gomoku/6*6_numeps_100_num_mcts_sims_100_2_input_channels','best.pth.tar')

    args1 = dotdict({'numMCTSSims': 100, 'cpuct':1.0})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    n2 = NNet(g)
    n2.load_checkpoint('./checkpoints/gomoku/6*6_numeps_100_num_mcts_sims_25_2_input_channels','best.pth.tar')
    args2 = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    arena = Arena.Arena(n1p, n2p, g, display=GomokuGame.display)
    p1wins, p2wins, draws = arena.playGames(20)

    log.info(f"player 1 wins / loss /draw: {p1wins} / {p2wins} / {draws}")


def print_train_example():
    modelFile = os.path.join('./checkpoints/gomoku/6*6_numeps_100_num_mcts_sims_25_2_input_channels/', 'checkpoint_18.pth.tar')
    examplesFile = modelFile + ".examples"
    if not os.path.isfile(examplesFile):
        log.error(f'File "{examplesFile}" with trainExamples not found!')
        return
    log.info("File with trainExamples found. Loading it...")
    with open(examplesFile, "rb") as f:
        trainExamplesHistory = Unpickler(f).load()
        draw_count = 0
        total_count = 0
        for episode in trainExamplesHistory:
            if episode[0][2] == 0:
                draw_count += 1
            total_count += 1
        log.info(f"draw {draw_count}; total {total_count}")
    log.info('Loading done!')


# TODO
# P2 - remove duplicated training examples from training history
# 
if __name__ == "__main__":
    # print_train_example()
    two_model_compete()
    # main()