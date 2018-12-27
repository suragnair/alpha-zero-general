import sys

from td2020.src.config_class import CONFIG

sys.path.append('..')
from td2020.Coach_Fixed import Coach
# from td2020.configurations.ConfigWrapper import LearnArgs
from td2020.TD2020Game import TD2020Game
from td2020.keras.NNet import NNetWrapper as NN

# from td2020.src.config import grid_size

"""
learn.py

Teaches neural network playing of specified game configuration using self play
"""

if __name__ == "__main__":

    CONFIG.set_runner('learn')  # set visibility as learn
    CONFIG.to_file()

    # create nnet for this game
    g = TD2020Game()
    nnet = NN(g, CONFIG.nnet_args.encoder)

    # If training examples should be loaded from file
    if CONFIG.learn_args.load_model:
        nnet.load_checkpoint(CONFIG.learn_args.load_folder_file[0], CONFIG.learn_args.load_folder_file[1])

    # Create coach instance that starts teaching nnet on newly created game using self-play
    c = Coach(g, nnet, CONFIG.learn_args)
    if CONFIG.learn_args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
