import logging

import coloredlogs

from Coach_selfplay import Coach_selfplay
from go.Game import Game as Game
from go.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'size': 9,                  #board size
    'numIters': 5000,
    'numEps': 1,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 0,        # zero
    'updateThreshold': 0.51,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 400,          # levelBased.
    'arenaCompare': 0,         # No Arena
    'cpuct': 1.1,
    'arenaNumMCTSSims': 0,      #No USE
    'instinctArena': False,     #if set true reset Arena's MTCL tree each time
    'balancedGame': True,      # if balanced, black should win over 6 scores
    'firstIter': False,        # No checkpoint for self-play
    'checkpoint': './temp_level/',
    'load_model': False,
    'load_folder_file': ('./temp_level','best.pth.tar'),
    'numItersForTrainExamplesHistory': 0,
    'resignThreshold': -0.9999,   #resign when best Q value less than threshold Q[-1, 1]
    'levelBased': True,
    'maxLevel' : 14,
    'maxLeaves': 4,

})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(args)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach_selfplay...')
    c = Coach_selfplay(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()