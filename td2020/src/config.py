import os

from td2020.src.encoders import OneHotEncoder, NumericEncoder
from utils import dotdict

# ####################################################################################
# ###################### INITIAL CONFIGS AND OUTPUTS ##################################
# ####################################################################################

# specifically choose TF cpu if needed. This will have no effect if GPU is not present
USE_TF_CPU = False

# helper path so model weights are imported and exported correctly when transferring project
PATH = os.path.dirname(os.path.realpath(__file__))

# output of learn log
learn_file = ".\\..\\temp\\learning.txt"

# output of pit log
pit_file = ".\\..\\temp\\pitting.txt"

# output for game stats during playing games (game_episode, game iteration, player name, action executed, action_name, action_direction, player_score...
game_stats_file = ".\\..\\temp\\game_stats.csv"
game_stats_file_player2 = ".\\..\\temp\\game_stats_player2.csv"

# Show initial TF configuration when TF is getting initialized
SHOW_TENSORFLOW_GPU = True

# Show initial Pygame welcome message when Pygame is getting initialized
SHOW_PYGAME_WELCOME = False

# Should logger write learning outputs to files for later plotting
MAKE_STATS = False

# change output log verbosity. If visibility.verbose > 3, Pygame is shown
visibility = dotdict({
    'verbose': 0,
    'verbose_learn': 0
})

# Maximum number of fps Pygame will render game at. Only relevant when running with verbose > 3
FPS = 1000

# ####################################################################################
# ################################## GAME RULES ######################################
# ####################################################################################

# board size (square)
grid_size = 8

# ##################################
# ############# GOLD ###############
# ##################################

# how much money is returned when returned resources
MONEY_INC = 3

# how much initial gold do players get at game begining
INITIAL_GOLD = 1

# Maximum gold that players can have - It is limited to 8 bits for one-hot encoder
MAX_GOLD = 255

# ##################################
# ############# HEAL ###############
# ##################################


# Game mechanic where actors can damage themselves to heal friendly unit. This is only used when player doesn't have any money to pay for heal action
SACRIFICIAL_HEAL = False

# How much friendly unit is healed when executing heal action
HEAL_AMOUNT = 5
# how much money should player pay when heal action is getting executed.
HEAL_COST = 1

# ##################################
# ########### TIMEOUT ##############
# ##################################

# If timeout should be used. This causes game to finish after TIMEOUT number of actions. If timeout isnt used, Kill function is used, which is reducing number of hitpoints of units as seen in file "Graph.py"
USE_TIMEOUT = True

# this gets used by kill function that determines the end point
MAX_TIME = 2048

# Check if timeout is being used. Alternatively Kill function is used
if USE_TIMEOUT:
    print("Using Timeout")
    # how many turns until game end - this gets reduced when each turn is executed
    TIMEOUT = 200
else:
    print("Using Kill Function")
    # sets initial tick to 0 and then in getGameEnded it gets incremented unitl number 8191
    TIMEOUT = 0
# ##################################
# ########## ATTACKING #############
# ##################################

# how much damage is dealt to attacked actor
DAMAGE = 20
# when attacking, all enemy units are destroyed, resulting in victory for the attacking player
DESTROY_ALL = False
if DESTROY_ALL:
    DAMAGE = 10000

# ##################################
# ########## ENCODERS ##############
# ##################################

# Should one-hot encoder be used (recommended)
USE_ONE_HOT_ENCODER = True
if USE_ONE_HOT_ENCODER:
    print("Using One hot encoder")
    encoder = OneHotEncoder()
else:
    print("Using Numeric encoder")
    encoder = NumericEncoder()

# Defining number of encoders
NUM_ENCODERS = 6  # player_name, act_type, health, carrying, money, remaining_time

# Setting indexes to each encoder
P_NAME_IDX = 0
A_TYPE_IDX = 1
HEALTH_IDX = 2
CARRY_IDX = 3
MONEY_IDX = 4
TIME_IDX = 5

# ##################################
# ########### ACTORS ###############
# ##################################

# Dictionary for actors
d_a_type = dotdict({
    'Gold': 1,
    'Work': 2,
    'Barr': 3,
    'Rifl': 4,
    'Hall': 5,
})

# Reverse dictionary for actors
d_type_rev = dotdict({
    1: 'Gold',
    2: 'Work',
    3: 'Barr',
    4: 'Rifl',
    5: 'Hall',
})

# Maximum health that actor can have - this is also initial health that actor has.
a_max_health = dotdict({
    1: 10,  # Gold
    2: 10,  # Work
    3: 20,  # Barr
    4: 20,  # Rifl
    5: 30,  # Hall
})

# Cost of actor to produce (key - actor type, value - number of gold coins to pay)
a_cost = dotdict({
    1: 0,  # Gold
    2: 1,  # Work
    3: 4,  # Barr
    4: 2,  # Rifl
    5: 7,  # Hall
})

# ##################################
# ########## ACTIONS ###############
# ##################################

# Dictionary for actions and which actor can execute them
d_acts = dotdict({
    1: [],  # Gold
    2: ['up', 'down', 'left', 'right',
        'mine_resources', 'return_resources',
        'barracks_up', 'barracks_down', 'barracks_right', 'barracks_left',
        'town_hall_up', 'town_hall_down', 'town_hall_right', 'town_hall_left',
        ],  # Work #'idle','heal_up', 'heal_down', 'heal_right', 'heal_left'
    3: ['rifle_infantry_up', 'rifle_infantry_down', 'rifle_infantry_right', 'rifle_infantry_left',
        ],  # Barr #'idle','heal_up', 'heal_down', 'heal_right', 'heal_left'
    4: ['up', 'down', 'left', 'right',
        'attack_up', 'attack_down', 'attack_right', 'attack_left',
        ],  # Rifl # 'idle','heal_up', 'heal_down', 'heal_right', 'heal_left'
    5: ['npc_up', 'npc_down', 'npc_right', 'npc_left',
        ],  # Hall #'idle','heal_up', 'heal_down', 'heal_right', 'heal_left'
})

# Reverse dictionary for actions
d_acts_int = dotdict({
    1: [],  # Gold
    2: [1, 2, 3, 4,
        5, 6,
        19, 20, 21, 22,
        23, 24, 25, 26,
        ],  # Work #0, 27, 28, 29, 30
    3: [15, 16, 17, 18,
        ],  # Barr #0, 27, 28, 29, 30
    4: [1, 2, 3, 4,
        7, 8, 9, 10,
        ],  # Rifl #0, 27, 28, 29, 30
    5: [11, 12, 13, 14,
        ],  # Hall #0, 27, 28, 29, 30
})

#  Disabling actions
acts_enabled = dotdict({
    "idle": False,
    "up": True,
    "down": True,
    "right": True,
    "left": True,
    "mine_resources": True,
    "return_resources": True,
    "attack": True,
    "npc": True,
    "rifle_infantry": True,
    "barracks": True,
    "town_hall": True,
    "heal": True
})

# Defining all actions
ACTS = {
    "idle": 0,

    "up": 1,
    "down": 2,
    "right": 3,
    "left": 4,

    "mine_resources": 5,
    "return_resources": 6,

    "attack_up": 7,
    "attack_down": 8,
    "attack_right": 9,
    "attack_left": 10,

    "npc_up": 11,
    "npc_down": 12,
    "npc_right": 13,
    "npc_left": 14,

    "rifle_infantry_up": 15,
    "rifle_infantry_down": 16,
    "rifle_infantry_right": 17,
    "rifle_infantry_left": 18,

    "barracks_up": 19,
    "barracks_down": 20,
    "barracks_right": 21,
    "barracks_left": 22,

    "town_hall_up": 23,
    "town_hall_down": 24,
    "town_hall_right": 25,
    "town_hall_left": 26,

    "heal_up": 27,
    "heal_down": 28,
    "heal_right": 29,
    "heal_left": 30

}

# Reverse dictionary for all actions
ACTS_REV = {
    0: "idle",

    1: "up",
    2: "down",
    3: "right",
    4: "left",

    5: "mine_resources",
    6: "return_resources",

    7: "attack_up",
    8: "attack_down",
    9: "attack_right",
    10: "attack_left",

    11: "npc_up",
    12: "npc_down",
    13: "npc_right",
    14: "npc_left",

    15: "rifle_infantry_up",
    16: "rifle_infantry_down",
    17: "rifle_infantry_right",
    18: "rifle_infantry_left",

    19: "barracks_up",
    20: "barracks_down",
    21: "barracks_right",
    22: "barracks_left",

    23: "town_hall_up",
    24: "town_hall_down",
    25: "town_hall_right",
    26: "town_hall_left",

    27: "heal_up",
    28: "heal_down",
    29: "heal_right",
    30: "heal_left"
}

# Cound of all actions
NUM_ACTS = len(ACTS)

# ####################################################################################
# ################################## PLAYING #########################################
# ####################################################################################

# User shortcuts that player can use using Pygame
d_user_shortcuts = dotdict({
    ' ': 0,  # idle
    'w': 1,  # up
    's': 2,  # down
    'd': 3,  # right
    'a': 4,  # left
    'q': 5,  # mine_resources
    'e': 6,  # return_resources
    '1': 7,  # attack_up
    '2': 8,  # attack_down
    '3': 9,  # attack_right
    '4': 10,  # attack_left
    '6': 11,  # npc_up
    '7': 12,  # npc_down
    '8': 13,  # npc_right
    '9': 14,  # npc_left
    't': 15,  # rifle_infantry_up
    'z': 16,  # rifle_infantry_down
    'u': 17,  # rifle_infantry_right
    'i': 18,  # rifle_infantry_left
    'f': 19,  # barracks_up
    'g': 20,  # barracks_down
    'h': 21,  # barracks_right
    'j': 22,  # barracks_left
    'y': 23,  # town_hall_up
    'x': 24,  # town_hall_down
    'c': 25,  # town_hall_right
    'v': 26,  # town_hall_left
    'b': 27,  # heal_up
    'n': 28,  # heal_down
    'm': 29,  # heal_right
    ',': 30,  # heal_left
})

# Reverse dictionary for user shortcuts
d_user_shortcuts_rev = dotdict({
    0: ' ',  # idle

    1: 'w',  # up
    2: 's',  # down
    3: 'd',  # right
    4: 'a',  # left

    5: 'q',  # mine_resources
    6: 'e',  # return_resources

    7: '1',  # attack_up
    8: '2',  # attack_down
    9: '3',  # attack_right
    10: '4',  # attack_left

    11: '6',  # npc_up
    12: '7',  # npc_down
    13: '8',  # npc_right
    14: '9',  # npc_left

    15: 't',  # rifle_infantry_up
    16: 'z',  # rifle_infantry_down
    17: 'u',  # rifle_infantry_right
    18: 'i',  # rifle_infantry_left

    19: 'f',  # barracks_up
    20: 'g',  # barracks_down
    21: 'h',  # barracks_right
    22: 'j',  # barracks_left

    23: 'y',  # town_hall_up
    24: 'x',  # town_hall_down
    25: 'c',  # town_hall_right
    26: 'v',  # town_hall_left

    27: 'b',  # heal_up
    28: 'n',  # heal_down
    29: 'm',  # heal_right
    30: ',',  # heal_left
})

# Colors of actors displayed in Pygame
d_a_color = dotdict({
    1: (230, 0, 50),  # Gold
    2: (0, 165, 208),  # Work
    3: (255, 156, 255),  # Barr
    4: (152, 0, 136),  # Rifl
    5: (235, 255, 0),  # Hall
})
