from Coach import Coach
from OthelloGame import OthelloGame
from NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 1000,
    'numEps': 50,
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 50000,
    'numMCTSSims': 20,
    'arenaCompare': 20,
    'cpuct': 1,

    'checkpoint': '/mnt/lol50x20/',
    'load_model': True,
    'load_folder_file': ('/mnt/lol50x20','checkpoint_4.pth.tar'),
})

if __name__=="__main__":
    g = OthelloGame(6)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        
    c = Coach(g, nnet, args)
    c.learn()
