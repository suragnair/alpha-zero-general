from Coach import Coach
from OthelloGame import OthelloGame
from NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 1000,
    'numEps': 50,
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 25000,
    'numMCTSSims': 20,
    'arenaCompare': 20,
    'cpuct': 1,

    'checkpoint': 'checkpoints',
})

if __name__=="__main__":
    g = OthelloGame(6)
    nnet = nn(g)
    c = Coach(g, nnet, args)
    c.learn()
