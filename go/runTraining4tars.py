import logging

import coloredlogs

from Coach_update4tar import Coach_update4tar
from go.Game import Game as Game
from go.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'size': 9,                  #board size
    'numIters': 1000,
    'tempThreshold': 0,        # zero
    'numMCTSSims': 200,          # Number of games moves for MCTS to simulate.
    'arenaNumMCTSSims': 200,      #simulation for arena
    'arenaCompare': 2,         # Tornament version alsways 2
    'cpuct': 1.1,
    'instinctArena': False,     #if set true reset Arena's MTCL tree each time
    'balancedGame': True,      # if balanced, black should win over 6 scores
    'firstIter': True,        #set true if it produce first chechpoint to save, for multuprocess, the followings has to be FALSE
    'checkpoint': './temp/',
    'load_folder_file': ('./temp/','9*9aug16th.tar'),
    'numItersForTrainExamplesHistory': 1200,
    'resignThreshold': -2,   #resign does not work for Arena
    'maxLevel': 9,
    'levelBased': True,
    'maxLeaves': 4,
})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(args)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    log.info('Loading the Coach_update4tar...')
    c = Coach_update4tar(g, nnet, args)

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()

