import sys

from rts.src.config_class import CONFIG

sys.path.append('..')
from Coach import Coach
# from rts.configurations.ConfigWrapper import LearnArgs
from rts.RTSGame import RTSGame as Game
from rts.keras.NNet import NNetWrapper as nn

# from rts.src.config import grid_size

"""
rts/learn.py

Teaches neural network playing of specified game configuration using self play
This configuration needs to be kept seperate, as different nnet and game configs are set
"""

if __name__ == "__main__":

    CONFIG.set_runner('learn')  # set visibility as learn

    # create nnet for this game
    g = Game()
    nnet = nn(g, CONFIG.nnet_args.encoder)

    # If training examples should be loaded from file
    if CONFIG.learn_args.load_model:
        nnet.load_checkpoint(CONFIG.learn_args.load_folder_file[0], CONFIG.learn_args.load_folder_file[1])

    # Create coach instance that starts teaching nnet on newly created game using self-play
    c = Coach(g, nnet, CONFIG.learn_args)
    if CONFIG.learn_args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
