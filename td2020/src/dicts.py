import os

from utils import dotdict

USE_TF_CPU = False

PATH: str = os.path.dirname(os.path.realpath(__file__))

SHOW_TENSORFLOW_GPU: bool = False
SHOW_PYGAME_WELCOME: bool = False
VERBOSE: int = 4
FPS: int = 100  # only relevant when pygame

#############################################
#############################################

USER_PLAYER = 1  # used by Human Player - this does not change if human pit player is 1 or -1

NUM_ACTS = 12  # num all actions
NUM_ENCODERS = 6  # player_name, act_type, health, carrying, money, remaining_time
EXCLUDE_IDLE = True  # exclude idle action from all actions
MONEY_INC = 1  # how much money is returned when returned resources

INITIAL_GOLD = 0  # how much initial gold do players get at game begining
if EXCLUDE_IDLE and INITIAL_GOLD == 0:
    # let players have at least some gold so they have any valid moves
    INITIAL_GOLD = 1
TIMEOUT = 100  # how many turns until game end - this gets reduced when each turn is executed
DAMAGE = 1  # how much damage is dealt to attacked actor
DAMAGE_ANYWHERE = False  # allows infantry to attack any unit on grid
DESTROY_ALL = False  # when attacking, all enemy units are destroyed, resulting in victory for the attacking player
if DESTROY_ALL:
    DAMAGE_ANYWHERE = True
    DAMAGE = 10000

############################################
#############################################
P_NAME_IDX_INC = 1
A_TYPE_IDX_INC = 1
HEALTH_IDX_INC = 1
CARRY_IDX_INC = 1
MONEY_IDX_INC = 1
REMAIN_IDX_INC = 1

USE_ONE_HOT = False

if USE_ONE_HOT:
    P_NAME_IDX_INC = 2   # playerName 2 bit - 00(neutral), 01(1) or 10(-1),
    A_TYPE_IDX_INC = 4   # actor type -> 4 bit,
    HEALTH_IDX_INC = 2   # health-> 2 bit,
    CARRY_IDX_INC = 1    # carrying-> 1 bit,
    MONEY_IDX_INC = 5    # money-> 5 bits (32 aka 4 town halls or 32 workers) [every unit has the same for player]
    REMAIN_IDX_INC = 13  # 2^13 8192(za total annihilation)

#############################################
#############################################

# builds indexes for character encoding - if not using one hot encoding, max indexes are incremented by 1 from previous index, but for one hot encoding, its incremented by num bits
P_NAME_IDX = 0
P_NAME_IDX_MAX = P_NAME_IDX_INC

A_TYPE_IDX = P_NAME_IDX_MAX
A_TYPE_IDX_MAX = A_TYPE_IDX + A_TYPE_IDX_INC

HEALTH_IDX = A_TYPE_IDX_MAX
HEALTH_IDX_MAX = HEALTH_IDX + HEALTH_IDX_INC

CARRY_IDX = HEALTH_IDX_MAX
CARRY_IDX_MAX = CARRY_IDX + CARRY_IDX_INC

MONEY_IDX = CARRY_IDX_MAX
MONEY_IDX_MAX = MONEY_IDX + MONEY_IDX_INC

REMAIN_IDX = MONEY_IDX_MAX
REMAIN_IDX_MAX = REMAIN_IDX + REMAIN_IDX_INC

print("ONE HOT ENCODING")

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
    2: ['idle', 'up', 'down', 'left', 'right', 'barracks', 'town_hall', 'mine_resources', 'return_resources'],  # Work
    3: ['idle', 'rifle_infantry'],  # Barr
    4: ['idle', 'up', 'down', 'left', 'right', 'attack'],  # Rifl
    5: ['idle', 'npc'],  # Hall
})

d_acts_int = dotdict({
    1: [],  # Gold
    2: [0, 1, 2, 3, 4, 5, 6, 10, 11],  # Work
    3: [0, 9],  # Barr
    4: [0, 1, 2, 3, 4, 7],  # Rifl
    5: [0, 8],  # Hall
})

d_type_rev = dotdict({
    1: 'Gold',
    2: 'Work',
    3: 'Barr',
    4: 'Rifl',
    5: 'Hall',
})

a_m_health = dotdict({
    1: 1,  # Gold
    2: 1,  # Work
    3: 3,  # Barr
    4: 2,  # Rifl
    5: 4,  # Hall
})
a_cost = dotdict({
    1: 0,  # Gold
    2: 1,  # Work
    3: 4,  # Barr
    4: 2,  # Rifl
    5: 7,  # Hall
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
}

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
    'b': 11  # town_hall
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
    11: 'b'  # town_hall
})

######################################################
################### PYGAME ###########################
######################################################
d_a_color = dotdict({
    1: (230, 0, 50),  # Gold
    2: (0, 165, 208),  # Work
    3: (255, 156, 255),  # Barr
    4: (152, 0, 136),  # Rifl
    5: (235, 255, 0),  # Hall
})