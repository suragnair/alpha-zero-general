
import logging

import coloredlogs

from Coach import Coach
from gomoku.GomokuGame import GomokuGame as Game
from gomoku.pytorch.NNet import NNetWrapper as nn
from utils import *
 
import timeit

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 35,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 30,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './checkpoints/gomoku/15*15_numeps_100_num_mcts_sims_25_temp_15_input_channels_2_channels_128',
    'load_model': True,
    'load_folder_file': ('checkpoints/gomoku/15*15_numeps_100_num_mcts_sims_25_temp_15_input_channels_2_channels_128','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'num_channels': 128,
    'input_channels': 2,
    'verbose': 0,
    # parallism params
    'num_workers': 1,
})

g = Game(15)

nnet = nn(g, input_channels=args.input_channels, num_channels=args.num_channels)

nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
nnet.nnet.to("cpu")

board = g.getInitBoard()
def predict_wrapper():
	nnet.predict(board)


time = timeit.timeit(predict_wrapper, number = 100)
print(f"Average prediction time: {time} seconds")