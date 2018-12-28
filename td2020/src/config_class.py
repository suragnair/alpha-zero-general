# noinspection PyUnresolvedReferences
from td2020.src.config import Configuration

# Basic config:
# CONFIG = Configuration()

# ################################# OVERRIDES #######################################
# Two different encoders
"""
# First learning model:
# CONFIG = Configuration(use_one_hot_encoder=True, onehot_encoder_player1=True, onehot_encoder_player2=False)
# Run learn
# Dont forget to rename it to "best_player1.pth.tar" after learning 
# Second learning model:
# CONFIG = Configuration(use_one_hot_encoder=True, onehot_encoder_player1=True, onehot_encoder_player2=False) 
# Run learn
# Dont forget to rename it to "best_player2.pth.tar" after learning
# Run pit
"""

# ################################# RUN 1 ##############################################
# Onehot encoder
CONFIG = Configuration(num_iters=10,
                       num_eps=10,
                       num_mcts_sims=30,
                       epochs=100)
# ################################# OLD RUNS ###########################################
# Sample Health Task
"""
CONFIG = Configuration(num_iters=20, 
                       num_eps=10,
                       num_mcts_sims=30,
                       epochs=100)
"""

# Model Gathering Task
"""
CONFIG = Configuration(num_iters=10,
                       num_eps=10,
                       num_mcts_sims=30,
                       epochs=100,
                       timeout_player1=100,
                       timeout_player2=100,
                       acts_enabled_player1={
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
                       },
                       acts_enabled_player2={
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
                       },
                       score_function_player1=1,
                       score_function_player2=1,
                       initial_board_config=[
                           Configuration.BoardTile(1, 6, 4, 'Work'),
                           Configuration.BoardTile(-1, 6, 5, 'Work'),
                           Configuration.BoardTile(1, 4, 4, 'Gold'),
                           Configuration.BoardTile(-1, 4, 5, 'Gold'),
                           Configuration.BoardTile(1, 5, 4, 'Hall'),
                           Configuration.BoardTile(-1, 5, 5, 'Hall')])
"""
