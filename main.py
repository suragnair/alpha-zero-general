import logging

import coloredlogs

from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,           # Number of training iterations
    'numEps': 100,              # Number of self-play games per training iteration
    'tempThreshold': 15,        # Number of iterations to pass before increasing MCTS temp by 1
    'updateThreshold': 0.6,     # Threshold win percentage of arena games to accept a new neural network
    'maxlenOfQueue': 200_000,   # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of arena games to assess neural network for acceptance
    'cpuct': 1,                 # PUCT exploration constant

    'checkpoint': './temp/',    # Folder name to save checkpoints
    # Set True to load in the model weights from checkpoint and training
    # examples from the load_folder_file
    'load_model': False,
    # Two-tuple of folder and filename where training examples are housed
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    # Max amount of training examples to keep in the history, dropping the
    # oldest example beyond that before adding a new one (like a FIFO queue)
    'numItersForTrainExamplesHistory': 20,
})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(6)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
