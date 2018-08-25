from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': '/workspace/checkpoints/',
    'load_model': True,
    'load_folder_file': ('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__=="__main__":
    g = Game(8)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
