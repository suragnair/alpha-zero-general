import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from Coach import Coach
from ultimate_tictactoe.UltimateTicTacToeGame import UltimateTicTacToeGame
from ultimate_tictactoe.keras.NNet import NNetWrapper as nn
from utils import *

print('Loading %s...', UltimateTicTacToeGame.__name__)

g = UltimateTicTacToeGame()

print('Loading %s...', nn.__name__)
nnet = nn(g)
#

#  log.info('Loading the Coach...')

args = dotdict({
    'numIters': 15,
    'numEps': 100,  # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,
    'updateThreshold': 0.6,  # During arena playoff, new neural net will be accepted if threshold or more of games
    # are won.
    'maxlenOfQueue': 2000,  # Number of game examples to train the neural networks.
    'numMCTSSims': 25,  # Number of games moves for MCTS to simulate.
    'arenaCompare': 10,  # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp_2/',
    'load_model': False,
    'load_folder_file': ('./temp/', 'checkpoint_5.h5'),
    'numItersForTrainExamplesHistory': 20,

})

if args.load_model:
    print('Loading checkpoint "%s/%s"...', args.load_folder_file)
    nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
else:
    print('Not loading a checkpoint!')

c = Coach(g, nnet, args)

if args.load_model:
    print("Loading 'trainExamples' from file...")
    c.loadTrainExamples()

print('Starting the learning process 🎉')
c.learn()
