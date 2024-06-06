import logging
import coloredlogs
#import winsound
import sys
sys.path.append('..')
from Coach import Coach  # Main class for training the ai
from yoxii.YoxiiGame import YoxiiGame as Game # Import the specific game rules
from yoxii.pytorch.NNet import NNetWrapper as nn  # Neuronal Network Class itself
from utils import *

log = logging.getLogger(__name__)  # Creates a quickly accessible log class for the terminal logger

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

import os
current_directory = os.getcwd()

args = dotdict({
    'numIters': 1000,  #originally 1000          # Number of iterations to train the neural networks.
    'numEps': 100, #originally 100              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        # Number of simulations with temp = 1 (MCTS focuses on exploration) then temp = 0 (MCTS focuses on exploitation)
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 50,   # originally 25       # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,   # originally 40      # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': current_directory+'/models/checkpoints/',
    'load_model': True,        # Loading a pretrained model? 
    'load_folder_file': (current_directory+'/models/checkpoints/','checkpoint_76.pth.tar'), 
    'numItersForTrainExamplesHistory': 30,     # Model is not only trained on the new data but also on the past games of the previous iterations. 

})

"""
As aid, the following notation shall be introduced: AlgX.Y for Algorithm X, Line Y in the paper. 
E.g. Alg2.3 is the 3rd line of 2nd algorithm in the paper.
"""


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game()                                    # Initialise game as Yoxii Game

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)                                  # Initialise neuronal net (Alg2.2)

    if args.load_model:                           # Load a model if args are configured this way
        # Loginfo only
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        # actual loading in the checkpoint
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)                      # Initialise the Coach class

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()                                     # This process may take a while!

    log.info('Saving the model...')
    nnet.save_checkpoint(args.checkpoint,"trainedFinal2.pth.tar")        # Save the model!


    log.info('Done!')

# Ensures that the main function is only executed when the script is run directly, not when imported as a module:
if __name__ == "__main__":
    main()
    """
    try:
        main()
    except Exception as e: 
        for i in range(5):
            winsound.Beep(frequency=2400, duration=300)
        print("An error occurred:", str(e))
    """
