# noinspection PyUnresolvedReferences
from rts.src.config import Configuration

# Basic config:
# CONFIG = Configuration()

# ################################# Examples #######################################
# Two different encoders
# Release:
"""
https://github.com/JernejHabjan/alpha-zero-general/releases/tag/0.3.0
"""
# Example
"""
First learning model:
CONFIG = Configuration(use_one_hot_encoder=True, onehot_encoder_player1=True, onehot_encoder_player2=False)
Run learn
Dont forget to rename it to "best_player1.pth.tar" after learning 
Second learning model:
CONFIG = Configuration(use_one_hot_encoder=False, onehot_encoder_player1=True, onehot_encoder_player2=False) 
Run learn
Dont forget to rename it to "best_player2.pth.tar" after learning
Run pit
"""

# ######## Example learning ##################
# Release:
"""
https://github.com/JernejHabjan/alpha-zero-general/releases/tag/0.4.0
"""
"""
CONFIG = Configuration(num_iters=10,
                       num_eps=10,
                       num_mcts_sims=30,
                       epochs=100)
"""
# Description
"""
Example of longer learning with high number of eps and mcts sims.
"""

# ################################# RUN 1 ##############################################

CONFIG = Configuration(num_iters=100,
                       num_iters_for_train_examples_history=30,
                       num_eps=4,
                       num_mcts_sims=5,
                       arena_compare=7,
                       epochs=100,
                       initial_gold_player1=10,
                       initial_gold_player2=10)
# Release:
"""
https://github.com/JernejHabjan/alpha-zero-general/releases/tag/1.0.0
"""

# Description
"""
Num iterations: Increased to 100, so graphing can be done correctly and multiple comparisons between models are done.
Train examples history: Increased to 30, because of high number of iterations. After 30 iterations, learning process becomes quite slow but efficient
Num eps: Decreased to 4, so multiple iterations can be triggered faster
Num mcts sims: Decreased to 5, because game is not played to end, it doesnt really contribute that much
Arena compare: 7 so comparisons between old and new model are quick but not resulting in overwriting better model
Epochs: Increased to 100, because of GPU, where learning is done relatively fast even with such a number
Initial gold: Increased for both players to 10. This is most important parameter change here, because it gives players enough money to start constructing different actors, which results in non-tie games and forces players to keep creating new actors and attacking with rifle units.
"""
# Results
"""
Workers are very frequently gathering gold when near that actor. 
Random movement has been greatly decreased over learning period, resulting in less time wasted.
Rifle units are also produced later in the game, which successfully attack enemy units when they are placed near them. Hunting for enemy actors doesn't occur, where player would try to annihilate enemy player.
Players mostly gather gold and construct new actors with occasional attacks.
"""

# ######## Pit with different board setup ###########
"""
CONFIG = Configuration(num_iters=100,
                       num_iters_for_train_examples_history=30,
                       num_eps=4,
                       num_mcts_sims=5,
                       arena_compare=7,
                       epochs=100,
                       initial_gold_player1=10,
                       initial_gold_player2=10,
                       initial_board_config=[
                           Configuration.BoardTile(1, 0, 4, 'Gold'),
                           Configuration.BoardTile(-1, 7, 4, 'Gold'),
                           Configuration.BoardTile(1, 3, 5, 'Hall'),
                           Configuration.BoardTile(-1, 4, 5, 'Hall')]
                       )

"""
# Release:
"""
https://github.com/JernejHabjan/alpha-zero-general/releases/tag/1.0.0
"""
# Description
"""
Initial board config: players have gold actors on edges of map
"""

# Results
"""
Players start game by constructing as much actors as they can with provided gold.
Players continue to successfully gather gold when they get near gold minerals, but randomly walk around when they are not.
Attacking units continue to damage and destroy enemy units when nearby, but attacks on enemy base are not initiated, resulting in annihilation
"""

# ################################# RUN 2 ##############################################

# First learning model (best_player1.pth.tar):
"""
CONFIG = Configuration(use_one_hot_encoder=True,
                       onehot_encoder_player1=True,
                       onehot_encoder_player2=False,

                       num_iters=20,
                       num_iters_for_train_examples_history=5,
                       num_eps=4,
                       num_mcts_sims=5,
                       arena_compare=7,
                       epochs=100,
                       initial_gold_player1=10,
                       initial_gold_player2=10,
                       
                       num_games=100,
                       pit_visibility=0)
"""
# Second learning model (best_player2.pth.tar):
"""
CONFIG = Configuration(use_one_hot_encoder=False,
                       onehot_encoder_player1=True,
                       onehot_encoder_player2=False,

                       num_iters=20,
                       num_iters_for_train_examples_history=5,
                       num_eps=4,
                       num_mcts_sims=5,
                       arena_compare=7,
                       epochs=100,
                       initial_gold_player1=10,
                       initial_gold_player2=10,

                       num_games=100,
                       pit_visibility=0)
"""
# Release
"""
https://github.com/JernejHabjan/alpha-zero-general/releases/tag/1.0.1
"""

# Description
"""
Comparing model encoded using one-hot encoder against numeric encoder
"""

# Results:
"""
(62, 32, 6) (onehot, numeric, ties)
"""

# ################################# OLD RUNS (Deprecated) ###########################################

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
