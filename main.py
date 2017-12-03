from Coach import Coach
from OthelloGame import OthelloGame
from NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters' = 1000,
    'numEps' = 100,
    'tempThreshold = 15,
    'updateThreshold' = 0.6,
    'maxlenOfQueue' = 5000,
    'numMCSTSSims' = 100,
    'arenaCompare' = 100,
    'cpuct' = 1,

    'checkpoint': 'checkpoints',
})

if __name__=="__main__":
    g = OthelloGame(6)
    nnet = nn(g)
    c = Coach(g, nn)
    c.learn()
