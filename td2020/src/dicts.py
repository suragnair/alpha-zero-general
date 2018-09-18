import os

from utils import dotdict

USE_TF_CPU = False

PATH: str = os.path.dirname(os.path.realpath(__file__))

SHOW_TENSORFLOW_GPU: bool = False
SHOW_PYGAME_WELCOME: bool = False
VERBOSE: int = 2
FPS: int = 10  # only relevant when pygame

#############################################
#############################################


NUM_ACTS = 12
NUM_ENCODERS = 6  # player_name, act_type, health, carrying, money, remaining_time
EXCLUDE_IDLE = True
MONEY_INC = 1

INITIAL_GOLD = 200
TIMEOUT = 100
DAMAGE = 5
DAMAGE_ANYWHERE = True
KILL_ALL = True
#############################################
#############################################



d_a_type = dotdict({
    'Gold': 1,
    'Work': 2,
    'Barr': 3,
    'Rifl': 4,
    'Hall': 5,
})
# d_acts = dotdict({
#     1: [],  # Gold
#     2: ['idle', 'up', 'down', 'left', 'right', 'barracks', 'town_hall', 'mine_resources', 'return_resources'],  # Work
#     3: ['idle', 'rifle_infantry'],  # Barr
#     4: ['idle', 'up', 'down', 'left', 'right', 'attack'],  # Rifl
#     5: ['idle', 'npc'],  # Hall
# })



# d_acts_int = dotdict({
#     1: [],  # Gold
#     2: [0, 1, 2, 3, 4, 5, 6, 10, 11],  # Work
#     3: [0, 9],  # Barr
#     4: [0, 1, 2, 3, 4, 7],  # Rifl
#     5: [0, 8],  # Hall
# })

d_acts = dotdict({
    1: [],  # Gold
    2: ['idle', 'up', 'down', 'left', 'right', 'barracks','npc'],  # Work
    3: ['idle', 'rifle_infantry','npc'],  # Barr
    4: ['idle', 'up', 'down', 'left', 'right', 'attack','npc'],  # Rifl
    5: ['idle', 'npc'],  # Hall
})


d_acts_int = dotdict({
    1: [],  # Gold
    2: [0, 1, 2, 3, 4,  10, 8],  # Work
    3: [0, 9, 8],  # Barr
    4: [0, 1, 2, 3, 4, 7, 8],  # Rifl
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

#
P_NAME_IDX = 0
A_TYPE_IDX = 1
HEALTH_IDX = 2
CARRY_IDX = 3
MONEY_IDX = 4
REMAIN_IDX = 5

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
