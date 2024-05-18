import logging
import coloredlogs
from Coach import Coach
from ninemensmorris.NineMensMorrisGame import NineMensMorrisGame as Game
from ninemensmorris.pytorch.NNet import NNetWrapper as nn
from utils import *

#NOTE -> TO SWITCH BETWEEN KERAS AND PYTORCH, CHANGE NAMES FROM NNET AND NNETWRAPPER

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 10,             # default 1000 -> takes too long
    'numEps': 1000,              # Number of complete self-play games to simulate during a new iteration. default 100
    'tempThreshold': 5,        # default 15
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won. default 0.6
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks. default 200000
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate. default 25
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted. default 40
    'cpuct': 1,                 # default 1

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    'lr': 0.005, #default 0.001
    'dropout': 0.3,
    'epochs': 10, #default 10 -> try 15 or 20
    'batch_size': 64,
    #'cuda': False,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,

})


def main():
    log.info('Loading %s...', NineMensMorrisGame.__name__)
    g = NineMensMorrisGame()

    log.info('Loading %s...', NNetWrapper.__name__)
    nnet = NNetWrapper(g)
    log.info('cuda available "%s"', torch.cuda.is_available())
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
