import os

from td2020.src.encoders import OneHotEncoder, NumericEncoder
from utils import dotdict

USE_TF_CPU = False

PATH: str = os.path.dirname(os.path.realpath(__file__))

SHOW_TENSORFLOW_GPU: bool = True
SHOW_PYGAME_WELCOME: bool = False
VERBOSE: int = 4
FPS: int = 100  # only relevant when pygame

#############################################
#############################################

MAKE_STATS = False
MONEY_INC = 10  # how much money is returned when returned resources

INITIAL_GOLD = 20  # how much initial gold do players get at game begining
MAX_GOLD = 31  # to encode with onehot encoder in 5 bits

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

print("TODO - check if heal cheker works and if heal executor works - for both sacrificial and non-sacrificial")

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
    2: ['idle', 'up', 'down', 'left', 'right', 'barracks', 'town_hall', 'mine_resources', 'return_resources', 'heal'],  # Work
    3: ['idle', 'rifle_infantry', 'heal'],  # Barr
    4: ['idle', 'up', 'down', 'left', 'right', 'attack', 'heal'],  # Rifl
    5: ['idle', 'npc', 'heal'],  # Hall
})

d_acts_int = dotdict({
    1: [],  # Gold
    2: [0, 1, 2, 3, 4, 5, 6, 10, 11, 12],  # Work
    3: [0, 9, 12],  # Barr
    4: [0, 1, 2, 3, 4, 7, 12],  # Rifl
    5: [0, 8, 12],  # Hall
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

"""
acts_enabled = dotdict({
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
"""

acts_enabled = dotdict({
    "idle": True,
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


ACTS = {
    "idle": 0,
    "up": 1,
    "down": 2,
    "right": 3,
    "left": 4,
    "mine_resources": 5,
    "return_resources": 6,
    "attack": 7,
    "npc": 8,
    "rifle_infantry": 9,
    "barracks": 10,
    "town_hall": 11,
    "heal": 12
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
    7: "attack",
    8: "npc",
    9: "rifle_infantry",
    10: "barracks",
    11: "town_hall",
    12: "heal"
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
