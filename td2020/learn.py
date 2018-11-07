from td2020.Coach_Fixed import Coach
from td2020.TD2020Game import TD2020Game as Game
from td2020.keras.NNet import NNetWrapper as nn
from td2020.src.config import MAKE_STATS
from td2020.stats.files import Stats
from utils import *

args = dotdict({
    'numIters': 40,
    'numEps': 10,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 8000,  # THIS SHOULD BE NUM_EPS * 800
    'numMCTSSims': 100,
    'arenaCompare': 20,
    'cpuct': 1,

    'checkpoint': '.\\..\\temp\\',
    'load_model': False,
    'load_folder_file': ('.\\..\\temp\\', 'checkpoint_4.pth.tar'),
    'numItersForTrainExamplesHistory': 99999, # This can be unlimited, as i am removing train examples on dump crash

})

if __name__ == "__main__":
    g = Game(8)
    nnet = nn(g)

    if MAKE_STATS:
        Stats.clear()

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
