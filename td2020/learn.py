import datetime

from td2020.Coach_Fixed import Coach
from td2020.TD2020Game import TD2020Game as Game
from td2020.keras.NNet import NNetWrapper as nn
from td2020.keras.NNet import args as nnet_args
from td2020.src.config import INITIAL_GOLD, acts_enabled, TIMEOUT, game_stats_file
from td2020.src.config import MAKE_STATS, learn_file
from td2020.src.config import MONEY_INC, USE_TIMEOUT, HEAL_AMOUNT, HEAL_COST, DAMAGE, USE_ONE_HOT_ENCODER
from td2020.src.config import grid_size
from td2020.stats.files import Stats
from utils import *

"""
learn.py

Teaches neural network playing of specified game configuration using self play
"""

args = dotdict({
    'numIters': 4,  # total number of games played from start to finish is numIters * numEps
    'numEps': 4,  # How may game is played in this episode
    'tempThreshold': 15,
    'updateThreshold': 0.6,  # Percentage that new model has to surpass by win rate to replace old model
    'maxlenOfQueue': 6400,
    'numMCTSSims': 10,  # How many MCTS tree searches are performing (mind that this MCTS doesnt use simulations)
    'arenaCompare': 5,  # How many comparisons are made between old and new model
    'cpuct': 1,  # search parameter for MCTS

    'checkpoint': '.\\..\\temp\\',
    'load_model': False,  # Load training examples from file - WARNING - this is disabled in TD2020Players.py because of memory errors received when loading data from file
    'load_folder_file': ('.\\..\\temp\\', 'checkpoint_13.pth.tar'),
    'numItersForTrainExamplesHistory': 8,  # maximum number of 'iterations' that game episodes are kept in queue. After that last is popped and new one is added.
})

if __name__ == "__main__":

    # Write learning configuration to file
    print(datetime.datetime.now())
    with open(learn_file, "w") as f:
        f.write("Args\n"
                + "\n"
                + "Coach args\n"
                + "numIters: " + str(args.numIters) + "\n"
                + "numEps: " + str(args.numEps) + "\n"
                + "maxlenOfQueue: " + str(args.maxlenOfQueue) + "\n"
                + "numMCTSSims: " + str(args.numMCTSSims) + "\n"
                + "arenaCompare: " + str(args.arenaCompare) + "\n"
                + "numItersForTrainExamplesHistory: " + str(args.numItersForTrainExamplesHistory) + "\n"
                + "\n"
                + "Learn args\n"
                + "epochs: " + str(nnet_args.epochs) + "\n"
                + "batch_size: " + str(nnet_args.batch_size) + "\n"
                + "\n"
                + "Game Config\n"
                + "USE_ONE_HOT_ENCODER: " + str(USE_ONE_HOT_ENCODER) + "\n"
                + "INITIAL_GOLD: " + str(INITIAL_GOLD) + "\n"
                + "MONEY_INC: " + str(MONEY_INC) + "\n"
                + "DAMAGE: " + str(DAMAGE) + "\n"
                + "HEAL_AMOUNT: " + str(HEAL_AMOUNT) + "\n"
                + "HEAL_COST: " + str(HEAL_COST) + "\n"
                + "USE_TIMEOUT: " + str("Using timeout" if USE_TIMEOUT else "Using Kill function") + "\n"
                + "timeout: " + str(TIMEOUT) + "\n"
                + "grid_size: " + str(grid_size) + "\n"
                + "\n"
                + "Actions\n"
                + "idle: " + str(acts_enabled.idle) + "\n"
                + "up: " + str(acts_enabled.up) + "\n"
                + "down: " + str(acts_enabled.down) + "\n"
                + "right: " + str(acts_enabled.right) + "\n"
                + "left: " + str(acts_enabled.left) + "\n"
                + "mine_resources: " + str(acts_enabled.mine_resources) + "\n"
                + "return_resources: " + str(acts_enabled.return_resources) + "\n"
                + "attack: " + str(acts_enabled.attack) + "\n"
                + "npc: " + str(acts_enabled.npc) + "\n"
                + "rifle_infantry: " + str(acts_enabled.rifle_infantry) + "\n"
                + "barracks: " + str(acts_enabled.barracks) + "\n"
                + "town_hall: " + str(acts_enabled.town_hall) + "\n"
                + "heal: " + str(acts_enabled.heal) + "\n"
                + "\n"
                + "\n"
                + "\n"
                )
        f.write("Learning started " + str(datetime.datetime.now()) + "\n")

    # create game
    g = Game(grid_size)

    # create nnet for this game
    nnet = nn(g)

    if MAKE_STATS:
        Stats.clear()

    # If training examples should be loaded from file
    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    with open(game_stats_file, "w") as f:
        param_line = "LEARNING\nmoney_inc:" + str(MONEY_INC) + " use_timeout:" + str(USE_TIMEOUT) + " heal_amount:" + str(HEAL_AMOUNT) + " heal_cost:" + str(HEAL_COST) + " damage:" + str(DAMAGE) + " use_onehot_encoder" + str(USE_ONE_HOT_ENCODER) + " num_iters:"
        param_line += str(args.numIters) + " num_eps:" + str(args.numEps) + " learn_num_mcts:" + str(args.numMCTSSims) + " learn_arena_compare:" + str(args.arenaCompare) + " cpuct:" + str(args.cpuct) + " learning_rate:"
        param_line += str(nnet_args.lr) + " epochs:" + str(nnet_args.epochs) + " batch_size:" + str(nnet_args.batch_size)
        f.write("iteration,game_ep,player,x,y,action_index,act_rev,output_direction,score,iteration,\n")
        f.write(param_line + "\n")

    # Create coach instance that starts teaching nnet on newly created game using self-play
    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
