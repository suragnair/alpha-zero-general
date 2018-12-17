import datetime

import Arena
from MCTS import MCTS
from td2020.TD2020Game import TD2020Game, display
from td2020.TD2020Players import *
from td2020.keras.NNet import NNetWrapper as NNet
from td2020.keras.NNet import args as nnet_args
from td2020.learn import args as learn_args
from td2020.src.config import INITIAL_GOLD, acts_enabled, TIMEOUT, pit_file, game_stats_file
from td2020.src.config import MONEY_INC, USE_TIMEOUT, HEAL_AMOUNT, HEAL_COST, DAMAGE, USE_ONE_HOT_ENCODER
from td2020.src.config import grid_size
from utils import *

"""
pit.py

use this script to play any two agents against each other, or play manually with any agent as human player.
"""
# Create game of size grid_size (usually 8)
g = TD2020Game(grid_size)

# defining random player
rp = RandomPlayer(g).play
# defining greedy player
gp = GreedyTD2020Player(g).play
# defining human player
hp = HumanTD2020Player(g).play

# defining nnet player 1
n1 = NNet(g)
n1.load_checkpoint('.\\..\\temp\\', 'best.pth.tar')
args1 = dotdict({'numMCTSSims': 2, 'cpuct': 1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

# defining nnet player 2
n2 = NNet(g)
n2.load_checkpoint('.\\..\\temp\\', 'best.pth.tar')
args2 = dotdict({'numMCTSSims': 2, 'cpuct': 1.0})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

# define players
p1 = rp
p2 = rp
# Arena receives players (can be human player, nnet player, random player or greedy player

arena = Arena.Arena(p1, p2, g, display=display)

# Write learning configuration to file
print(datetime.datetime.now())
with open(pit_file, "w") as f:
    f.write("Args\n"
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
    f.write("Pitting started " + str(datetime.datetime.now()) + "\n")

with open(game_stats_file, "w") as f:
    param_line = "PITTING\nmoney_inc:" + str(MONEY_INC) + " use_timeout:" + str(USE_TIMEOUT) + " heal_amount:" + str(HEAL_AMOUNT) + " heal_cost:" + str(HEAL_COST) + " damage:" + str(DAMAGE) + " use_onehot_encoder" + str(USE_ONE_HOT_ENCODER) + " num_iters:"
    param_line += str(learn_args.numIters) + " num_eps:" + str(learn_args.numEps) + " learn_num_mcts:" + str(learn_args.numMCTSSims) + " learn_arena_compare:" + str(learn_args.arenaCompare) + " cpuct:" + str(learn_args.cpuct) + " learning_rate:"
    param_line += str(nnet_args.lr) + " epochs:" + str(nnet_args.epochs) + " batch_size:" + str(nnet_args.batch_size)
    param_line += " player1:" + p1.__str__() + " player2:" + p2.__str__()
    f.write(param_line + "\n")
    f.write("iteration,game_ep,player,x,y,action_index,act_rev,output_direction,score,iteration,\n")

# Print number of wins for each player
print(arena.playGames(2, verbose=False))
