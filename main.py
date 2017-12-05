from Coach import Coach
from OthelloGame import OthelloGame
from NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 25,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 100000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': '/dev/8x100x25/',
    'load_model': False,
    'load_folder_file': ('/mnt/lol50x20','checkpoint_4.pth.tar'),
})

if __name__=="__main__":
    g = OthelloGame(8)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        
    c = Coach(g, nnet, args)
    c.learn()
