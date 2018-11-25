from td2020.Coach_Fixed import Coach
from td2020.TD2020Game import TD2020Game as Game
from td2020.keras.NNet import NNetWrapper as nn
from td2020.src.config import MAKE_STATS, USE_ONE_HOT_ENCODER, INITIAL_GOLD, USE_TIMEOUT, HEAL_COST, HEAL_AMOUNT, MONEY_INC, DAMAGE, acts_enabled, learn_file, TIMEOUT, grid_size
from td2020.stats.files import Stats
from utils import *
import datetime

args = dotdict({
    'numIters': 100,
    'numEps': 8,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 6400,  # THIS SHOULD BE NUM_EPS * 800
    'numMCTSSims': 30,
    'arenaCompare': 10,
    'cpuct': 1,

    'checkpoint': '.\\..\\temp\\',
    'load_model': False,
    'load_folder_file': ('.\\..\\temp\\', 'checkpoint_13.pth.tar'),
    'numItersForTrainExamplesHistory': 8,
})
from td2020.keras.NNet import args as nnet_args

if __name__ == "__main__":

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

    g = Game(grid_size)
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
