import os

from td2020.src.encoders import OneHotEncoder, NumericEncoder
from utils import dotdict

USE_TF_CPU = False

PATH = os.path.dirname(os.path.realpath(__file__))

SHOW_TENSORFLOW_GPU = True
SHOW_PYGAME_WELCOME = False
VERBOSE = 4
FPS = 10  # only relevant when pygame

#############################################
#############################################

MAKE_STATS = False
MONEY_INC = 5  # how much money is returned when returned resources

INITIAL_GOLD = 1  # how much initial gold do players get at game begining
MAX_GOLD = 255  # to encode with onehot encoder in 8 bits

SACRIFICIAL_HEAL = False
HEAL_AMOUNT = 20
HEAL_COST = 5

USE_TIMEOUT = True
MAX_TIME = 2048  # this gets used by kill function that determines the end point
if USE_TIMEOUT:
    print("Using Timeout")
    TIMEOUT = 100  # how many turns until game end - this gets reduced when each turn is executed
else:
    print("Using Kill Function")
    TIMEOUT = 0  # sets initial tick to 0 and then in getGameEnded it gets incremented unitl number 8191

DAMAGE = 20  # how much damage is dealt to attacked actor
DESTROY_ALL = False  # when attacking, all enemy units are destroyed, resulting in victory for the attacking player
if DESTROY_ALL:
    DAMAGE = 10000

############################################
#############################################
USE_ONE_HOT_ENCODER = False
if USE_ONE_HOT_ENCODER:
    print("Using One hot encoder")
    encoder = OneHotEncoder()
else:
    print("Using Numeric encoder")
    encoder = NumericEncoder()

NUM_ENCODERS = 6  # player_name, act_type, health, carrying, money, remaining_time
P_NAME_IDX = 0
A_TYPE_IDX = 1
HEALTH_IDX = 2
CARRY_IDX = 3
MONEY_IDX = 4
TIME_IDX = 5

#############################################
#############################################

d_a_type = dotdict({
    'Gold': 1,
    'Work': 2,
    'Barr': 3,
    'Rifl': 4,
    'Hall': 5,
})
d_acts = dotdict({
    1: [],  # Gold
    2: ['idle',
        'up', 'down', 'left', 'right',
        'mine_resources', 'return_resources',
        'barracks_up', 'barracks_down', 'barracks_right', 'barracks_left',
        'town_hall_up', 'town_hall_down', 'town_hall_right', 'town_hall_left',
        'heal_up', 'heal_down', 'heal_right', 'heal_left'],  # Work
    3: ['idle',
        'rifle_infantry_up', 'rifle_infantry_down', 'rifle_infantry_right', 'rifle_infantry_left',
        'heal_up', 'heal_down', 'heal_right', 'heal_left'],  # Barr
    4: ['idle',
        'up', 'down', 'left', 'right',
        'attack_up', 'attack_down', 'attack_right', 'attack_left',
        'heal_up', 'heal_down', 'heal_right', 'heal_left'],  # Rifl
    5: ['idle',
        'npc_up', 'npc_down', 'npc_right', 'npc_left',
        'heal_up', 'heal_down', 'heal_right', 'heal_left'],  # Hall
})

d_acts_int = dotdict({
    1: [],  # Gold
    2: [0,
        1, 2, 3, 4,
        5, 6,
        19, 20, 21, 22,
        23, 24, 25, 26,
        27, 28, 29, 30],  # Work
    3: [0,
        15, 16, 17, 18,
        27, 28, 29, 30],  # Barr
    4: [0,
        1, 2, 3, 4,
        7, 8, 9, 10,
        27, 28, 29, 30],  # Rifl
    5: [0,
        11, 12, 13, 14,
        27, 28, 29, 30],  # Hall
})

d_type_rev = dotdict({
    1: 'Gold',
    2: 'Work',
    3: 'Barr',
    4: 'Rifl',
    5: 'Hall',
})

a_max_health = dotdict({  # MAX HEALTH THAT UNIT CAN HAVE - this gets in use when ill be implementing healing
    1: 10,  # Gold
    2: 10,  # Work
    3: 20,  # Barr
    4: 20,  # Rifl
    5: 30,  # Hall
})

a_m_health = dotdict({  # INITIAL HEALTH THAT UNIT HAS
    1: 10,  # Gold
    2: 10,  # Work
    3: 20,  # Barr
    4: 20,  # Rifl
    5: 30,  # Hall
})
a_cost = dotdict({
    1: 0,  # Gold
    2: 1,  # Work
    3: 4,  # Barr
    4: 2,  # Rifl
    5: 7,  # Hall
})

acts_enabled = dotdict({  # mine and return resources only
    "idle": False,
    "up": True,
    "down": True,
    "right": True,
    "left": True,
    "mine_resources": True,
    "return_resources": True,
    "attack": False,
    "npc": False,
    "rifle_infantry": False,
    "barracks": False,
    "town_hall": False,
    "heal": False
})

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
NUM_ACTS = len(ACTS)

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

d_user_shortcuts = dotdict({
    ' ': 0,  # idle
    'w': 1,  # up
    's': 2,  # down
    'd': 3,  # right
    'a': 4,  # left
    'g': 5,  # mine_resources
    'r': 6,  # return_resources
    'q': 7,  # attack
    'x': 8,  # npc
    'c': 9,  # rifle_infantry
    'v': 10,  # barracks
    'b': 11,  # town_hall
    'e': 12  # heal
})
d_user_shortcuts_rev = dotdict({
    0: ' ',  # idle
    1: 'w',  # up
    2: 's',  # down
    3: 'd',  # right
    4: 'a',  # left
    5: 'g',  # mine_resources
    6: 'r',  # return_resources
    7: 'q',  # attack
    8: 'x',  # npc
    9: 'c',  # rifle_infantry
    10: 'v',  # barracks
    11: 'b',  # town_hall
    12: 'e'  # heal
})

######################################################
# ################# PYGAME ###########################
######################################################
d_a_color = dotdict({
    1: (230, 0, 50),  # Gold
    2: (0, 165, 208),  # Work
    3: (255, 156, 255),  # Barr
    4: (152, 0, 136),  # Rifl
    5: (235, 255, 0),  # Hall
})
