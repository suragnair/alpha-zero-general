import sys
from typing import Any

import numpy as np

sys.path.append('../..')
from rts.src.config import d_a_type, d_acts, A_TYPE_IDX, P_NAME_IDX, CARRY_IDX, MONEY_IDX, NUM_ACTS, ACTS_REV, NUM_ENCODERS, HEALTH_IDX, TIME_IDX

"""
Board.py

Defines game rules (action checking, end-game conditions)
can_execute_move is checking if move can be executed and execute_move is applying this move to new board
"""


class Board:

    def __init__(self, n) -> None:
        self.n = n
        self.pieces = np.zeros((self.n, self.n, NUM_ENCODERS))

    def __getitem__(self, index: int) -> np.array:
        return self.pieces[index]

    def execute_move(self, move, player) -> None:
        """
        Executes move on this board for specified player
        :param move: (x, y, action_index), that define which action should be executed on which tile
        :param player: int - player that is executing action
        :return: /
        """
        from rts.src.config_class import CONFIG

        if player == 1:
            config = CONFIG.player1_config
        else:
            config = CONFIG.player2_config

        x, y, action_index = move
        act = ACTS_REV[action_index]
        if act == "idle":
            return
        if act == "up":
            new_x, new_y = x, y - 1
            self._move(x, y, new_x, new_y)
            return
        if act == "down":
            new_x, new_y = x, y + 1
            self._move(x, y, new_x, new_y)
            return
        if act == "right":
            new_x, new_y = x + 1, y
            self._move(x, y, new_x, new_y)
            return
        if act == "left":
            new_x, new_y = x - 1, y
            self._move(x, y, new_x, new_y)
            return
        if act == "mine_resources":
            self[x][y][CARRY_IDX] = 1
            return
        if act == "return_resources":
            self[x][y][CARRY_IDX] = 0
            self._update_money(player, config.MONEY_INC)
            return
        if act == "attack_up":
            self._attack(x, y, x, y - 1, config=config)
            return
        if act == "attack_down":
            self._attack(x, y, x, y + 1, config=config)
            return
        if act == "attack_left":
            self._attack(x, y, x - 1, y, config=config)
            return
        if act == "attack_right":
            self._attack(x, y, x + 1, y, config=config)
            return

        if act == "heal_up":
            self._heal(x, y, x, y - 1, config=config)
            return
        if act == "heal_down":
            self._heal(x, y, x, y + 1, config=config)
            return
        if act == "heal_left":
            self._heal(x, y, x - 1, y, config=config)
            return
        if act == "heal_right":
            self._heal(x, y, x + 1, y, config=config)
            return

        if act == "npc_up":
            self._update_money(player, -config.a_cost[2])
            self._spawn(x, y, x, y - 1, 2, config=config)
            return
        if act == "npc_down":
            self._update_money(player, -config.a_cost[2])
            self._spawn(x, y, x, y + 1, 2, config=config)
            return
        if act == "npc_left":
            self._update_money(player, -config.a_cost[2])
            self._spawn(x, y, x - 1, y, 2, config=config)
            return
        if act == "npc_right":
            self._update_money(player, -config.a_cost[2])
            self._spawn(x, y, x + 1, y, 2, config=config)
            return

        if act == "barracks_up":
            self._update_money(player, -config.a_cost[3])
            self._spawn(x, y, x, y - 1, 3, config=config)
            return
        if act == "barracks_down":
            self._update_money(player, -config.a_cost[3])
            self._spawn(x, y, x, y + 1, 3, config=config)
            return
        if act == "barracks_left":
            self._update_money(player, -config.a_cost[3])
            self._spawn(x, y, x - 1, y, 3, config=config)
            return
        if act == "barracks_right":
            self._update_money(player, -config.a_cost[3])
            self._spawn(x, y, x + 1, y, 3, config=config)
            return

        if act == "rifle_infantry_up":
            self._update_money(player, -config.a_cost[4])
            self._spawn(x, y, x, y - 1, 4, config=config)
            return
        if act == "rifle_infantry_down":
            self._update_money(player, -config.a_cost[4])
            self._spawn(x, y, x, y + 1, 4, config=config)
            return
        if act == "rifle_infantry_left":
            self._update_money(player, -config.a_cost[4])
            self._spawn(x, y, x - 1, y, 4, config=config)
            return
        if act == "rifle_infantry_right":
            self._update_money(player, -config.a_cost[4])
            self._spawn(x, y, x + 1, y, 4, config=config)
            return

        if act == "town_hall_up":
            self._update_money(player, -config.a_cost[5])
            self._spawn(x, y, x, y - 1, 5, config=config)
            return
        if act == "town_hall_down":
            self._update_money(player, -config.a_cost[5])
            self._spawn(x, y, x, y + 1, 5, config=config)
            return
        if act == "town_hall_left":
            self._update_money(player, -config.a_cost[5])
            self._spawn(x, y, x - 1, y, 5, config=config)
            return
        if act == "town_hall_right":
            self._update_money(player, -config.a_cost[5])
            self._spawn(x, y, x + 1, y, 5, config=config)
            return

    def _move(self, x, y, new_x, new_y):
        """
        Move actor to new location
        :param x: int - coordinate x where actor is located
        :param y: int - coordinate y where actor is located
        :param new_x: int - coordinate x where actor needs to be moved to
        :param new_y: int - coordinate y where actor needs to be moved to
        """
        self[new_x][new_y] = self[x][y]
        self[x][y] = [0] * NUM_ENCODERS
        self[x][y][TIME_IDX] = self[new_x][new_y][TIME_IDX]  # set time back to empty tile

    def _update_money(self, player, money_update):
        """
        :param player: int - player to which money gets appended/ decreased
        :param money_update: int - amount of money
        """
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y][P_NAME_IDX] == player:
                    assert self[x][y][MONEY_IDX] + money_update >= 0
                    self[x][y][MONEY_IDX] = self[x][y][MONEY_IDX] + money_update

    def _attack(self, x, y, n_x, n_y, config):
        """
        Actor attacks new actor on other coordinate
        :param x: actor on coordinate x
        :param y: actor on coordinate y
        :param n_x: attack new actor on coordinate n_x
        :param n_y: attack new actor on coordinate n_x
        :param config: config that specifies damage - different config can be used for each player
        """
        self[n_x][n_y][HEALTH_IDX] -= config.DAMAGE
        if self[n_x][n_y][HEALTH_IDX] <= 0:
            self[n_x][n_y] = [0] * NUM_ENCODERS
            self[n_x][n_y][TIME_IDX] = self[x][y][TIME_IDX]  # set time back to empty tile just in case

    def _spawn(self, x, y, n_x, n_y, a_type, config):
        """
        Actor spawns actor on other coordinate
        :param x: coordinate of building that is spawning new actor
        :param y: coordinate of building that is spawning new actor
        :param n_x: coordinate where new actor will spawn to
        :param n_y: coordinate where new actor will spawn to
        :param a_type: type of unit to spawn on new coordinate
        :param config: additional config that is separate for each player (maximum actor health for this type)
        """
        self[n_x][n_y] = [self[x][y][P_NAME_IDX], a_type, config.a_max_health[a_type], 0, self[x][y][MONEY_IDX], self[x][y][TIME_IDX]]

    def _heal(self, x, y, n_x, n_y, config):
        """
        Actor heals actor on other coordinate
        :param x: coordinate of actor executing heal action
        :param y: oordinate of actor executing heal action
        :param n_x: coordinate of actor that will receive heal
        :param n_y: coordinate of actor that will receive heal
        :param config: additional config that is separate for each player (heal_cost, heal_amount, max_actor_health)
        """
        if config.SACRIFICIAL_HEAL:
            self[x][x][HEALTH_IDX] -= config.HEAL_COST
            if self[x][y][HEALTH_IDX] <= 0:
                self[x][y] = [0] * NUM_ENCODERS
                self[x][y][TIME_IDX] = self[x][y][TIME_IDX]
        elif self[n_x][n_y][MONEY_IDX] - config.HEAL_AMOUNT >= 0:
            self[n_x][n_y][HEALTH_IDX] += config.HEAL_AMOUNT
            self._update_money(self[n_x][n_y][P_NAME_IDX], -config.HEAL_COST)

        # clamp value to max
        self[n_x][n_y][HEALTH_IDX] = self.clamp(self[n_x][n_y][HEALTH_IDX] + config.HEAL_AMOUNT, 0, config.a_max_health[self[n_x][n_y][A_TYPE_IDX]])

    def get_moves_for_square(self, x, y, config) -> Any:
        """
        Returns all valid actions for specific tile
        :param x: x coordinate of tile
        :param y: y coordinate of tile
        :param config: additional config that is separate for each player
        :return: array of valid actions
        """
        # determine the color of the piece.
        player = self[x][y][P_NAME_IDX]

        if player == 0:
            return None
        a_type = self[x][y][A_TYPE_IDX]
        acts = d_acts[a_type]
        moves = [0] * NUM_ACTS
        for i in range(NUM_ACTS):
            act = ACTS_REV[i]

            if act in acts:
                # a is now string action
                move = self._valid_act(x, y, act, config=config) * 1
                if move:
                    moves[i] = move
        # return the generated move list
        return moves

    def _valid_act(self, x, y, act, config):
        """
        Returns true if action on specific tile is valid, false otherwise
        :param x: tile x that action will be executing upon
        :param y: tile y that action will be executing upon
        :param act: str: action that will be executing on this tile
        :param config: additional config that gets passed to functions
        :return: true/false
        """
        money = self[x][y][MONEY_IDX]
        if act == "idle":
            return config.acts_enabled.idle
        if act == "up":
            return config.acts_enabled.up and self._check_if_empty(x, y - 1)
        if act == "down":
            return config.acts_enabled.down and self._check_if_empty(x, y + 1)
        if act == "right":
            return config.acts_enabled.right and self._check_if_empty(x + 1, y)
        if act == "left":
            return config.acts_enabled.left and self._check_if_empty(x - 1, y)

        if act == "mine_resources":
            return config.acts_enabled.mine_resources and self[x][y][CARRY_IDX] == 0 and self._check_if_nearby(x, y, d_a_type['Gold'])
        if act == "return_resources":
            return config.acts_enabled.return_resources and self[x][y][CARRY_IDX] == 1 and self._check_if_nearby(x, y, d_a_type['Hall'], check_friendly=True) and (config.MAX_GOLD >= self[x][y][MONEY_IDX] + config.MONEY_INC)

        if act == "attack_up":
            return config.acts_enabled.attack and self._check_if_attack(x, y, x, y - 1)
        if act == "attack_down":
            return config.acts_enabled.attack and self._check_if_attack(x, y, x, y + 1)
        if act == "attack_right":
            return config.acts_enabled.attack and self._check_if_attack(x, y, x + 1, y)
        if act == "attack_left":
            return config.acts_enabled.attack and self._check_if_attack(x, y, x - 1, y)

        if act == "heal_up":
            return config.acts_enabled.heal and self._check_if_heal(x, y - 1, config=config)
        if act == "heal_down":
            return config.acts_enabled.heal and self._check_if_heal(x, y + 1, config=config)
        if act == "heal_right":
            return config.acts_enabled.heal and self._check_if_heal(x + 1, y, config=config)
        if act == "heal_left":
            return config.acts_enabled.heal and self._check_if_heal(x - 1, y, config=config)

        if act == "npc_up":
            return config.acts_enabled.npc and config.a_cost[2] <= money and self._check_if_empty(x, y - 1)
        if act == "npc_down":
            return config.acts_enabled.npc and config.a_cost[2] <= money and self._check_if_empty(x, y + 1)
        if act == "npc_right":
            return config.acts_enabled.npc and config.a_cost[2] <= money and self._check_if_empty(x + 1, y)
        if act == "npc_left":
            return config.acts_enabled.npc and config.a_cost[2] <= money and self._check_if_empty(x - 1, y)

        if act == "barracks_up":
            return config.acts_enabled.barracks and config.a_cost[3] <= money and self._check_if_empty(x, y - 1)
        if act == "barracks_down":
            return config.acts_enabled.barracks and config.a_cost[3] <= money and self._check_if_empty(x, y + 1)
        if act == "barracks_right":
            return config.acts_enabled.barracks and config.a_cost[3] <= money and self._check_if_empty(x + 1, y)
        if act == "barracks_left":
            return config.acts_enabled.barracks and config.a_cost[3] <= money and self._check_if_empty(x - 1, y)

        if act == "rifle_infantry_up":
            return config.acts_enabled.rifle_infantry and config.a_cost[4] <= money and self._check_if_empty(x, y - 1)
        if act == "rifle_infantry_down":
            return config.acts_enabled.rifle_infantry and config.a_cost[4] <= money and self._check_if_empty(x, y + 1)
        if act == "rifle_infantry_right":
            return config.acts_enabled.rifle_infantry and config.a_cost[4] <= money and self._check_if_empty(x + 1, y)
        if act == "rifle_infantry_left":
            return config.acts_enabled.rifle_infantry and config.a_cost[4] <= money and self._check_if_empty(x - 1, y)

        if act == "town_hall_up":
            return config.acts_enabled.town_hall and config.a_cost[5] <= money and self._check_if_empty(x, y - 1)
        if act == "town_hall_down":
            return config.acts_enabled.town_hall and config.a_cost[5] <= money and self._check_if_empty(x, y + 1)
        if act == "town_hall_right":
            return config.acts_enabled.town_hall and config.a_cost[5] <= money and self._check_if_empty(x + 1, y)
        if act == "town_hall_left":
            return config.acts_enabled.town_hall and config.a_cost[5] <= money and self._check_if_empty(x - 1, y)
        print("Unrecognised action", act)
        sys.exit(0)

    def _check_if_empty(self, x, y):
        """
        Checks if tile is empty
        :param x: if x coordinate is empty
        :param y: if y coordinate is empty
        :return: true/false
        """
        # noinspection PyChainedComparisons
        return self.n > x >= 0 and 0 <= y < self.n and self[x][y][P_NAME_IDX] == 0

    def _check_if_attack(self, x, y, n_x, n_y):
        """
        Check if actor on x,y can attack actor on n_x,n_y
        :param x: actor on coordinate x
        :param y: actor on coordinate y
        :param n_x: can attack actor on coordinate n_x
        :param n_y: can attack actor on coordinate n_y
        :return: true/false
        """
        return 0 <= n_x < self.n and 0 <= n_y < self.n and self[x][y][P_NAME_IDX] == -self[n_x][n_y][P_NAME_IDX] and self[n_x][n_y][A_TYPE_IDX] != d_a_type['Gold']

    def _check_if_heal(self, x, y, config):
        """
        Check if actor on x,y can be healed
        :param x: coordinate of actor that is getting to be healed
        :param y: coordinate of actor that is getting to be healed
        :param config: special config specific for each player (max_health, heal_cost)
        :return: true/false
        """
        return 0 <= x < self.n and 0 <= y < self.n and self[x][y][P_NAME_IDX] == self[x][y][P_NAME_IDX] and self[x][y][A_TYPE_IDX] != d_a_type['Gold'] and self[x][y][A_TYPE_IDX] > 0 and self[x][y][HEALTH_IDX] < config.a_max_health[self[x][y][A_TYPE_IDX]] and (
                config.SACRIFICIAL_HEAL or self[x][y][MONEY_IDX] - config.HEAL_COST >= 0)

    def _check_if_nearby(self, x, y, a_type, check_friendly=False):
        """
        Checks if actor is nearby - friendly or foe
        :param x: coordinate of current actor
        :param y: coordinate of current actor
        :param a_type: type of nearby actor
        :param check_friendly: check if nearby actor should be friendly
        :return: true/false
        """
        coordinates = [(x - 1, y + 1),
                       (x, y + 1),
                       (x + 1, y + 1),
                       (x - 1, y),
                       (x + 1, y),
                       (x - 1, y - 1),
                       (x, y - 1),
                       (x + 1, y - 1)]
        for n_x, n_y in coordinates:
            if 0 <= n_x < self.n and 0 <= n_y < self.n:
                if self[n_x][n_y][A_TYPE_IDX] == a_type:
                    if not check_friendly:
                        return True
                    if self[n_x][n_y][P_NAME_IDX] == self[x][y][P_NAME_IDX]:
                        return True
        return False

    @staticmethod
    def _num_destroys(time):
        """
        Defines parameter in time_killer function, how many actors should be damaged each round
        :param time: current time frame
        :return: amount of damaged actors
        """
        return int((time / 256) ** 2 + 1)

    @staticmethod
    def _damage(time):
        """
        Defines parameter in time_killer function, how much each damaged actor should receive damage
        :param time: current time frame
        :return: damage to each actor
        """
        return int((time / 8) ** 2.718 / (time * 8))

    def time_killer(self, player):
        """
        Additional function that can be used to stop players from looping in game. It is defined using _damage and _num_destroys functions, which define how much and how many actors should be damaged each round.
        This prevents players playing games too long, as its starting to kill them. Better players should gather more gold and therefore create more advanced actors prolonging their lives.
        :param player: which player is currently executing action
        :return: /
        """
        # I can pass player through, because this board is canonical board that this action gets executed upon

        current_time = self[0][0][TIME_IDX]

        destroys_per_round = self._num_destroys(current_time)
        damage_amount = self._damage(current_time)

        # Damage as many actors as "damage_amount" parameter provides
        currently_damaged_actors = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y][P_NAME_IDX] == player and self[x][y][A_TYPE_IDX] != 1:  # for current player and not gold
                    if currently_damaged_actors >= destroys_per_round:
                        return
                    self[x][y][HEALTH_IDX] -= damage_amount

                    if self[x][y][HEALTH_IDX] <= 0:
                        time = self[x][y][TIME_IDX]
                        self[x][y] = [0] * NUM_ENCODERS
                        self[x][y][TIME_IDX] = time
                    currently_damaged_actors += 1

    @staticmethod
    def clamp(num, min_value, max_value):
        return max(min(num, max_value), min_value)

    def get_money_score(self, player) -> int:
        """
        1. of 3 functions that define elo rating of specified player. This one takes into account only players money count
        :param player: player that requires to know his money count
        :return: money count for specified player
        """
        return sum([self[x][y][MONEY_IDX] for x in range(self.n) for y in range(self.n) if self[x][y][P_NAME_IDX] == player])

    def get_health_score(self, player) -> int:
        """
        2. of 3 functions that define elo rating for specified player. This one takes into account only total current health of units. Players with more units, which have more health should win.
        :param player: player that requires to know sum of health for his units
        :return: sum of health for specified player
        """
        return sum([self[x][y][HEALTH_IDX] for x in range(self.n) for y in range(self.n) if self[x][y][P_NAME_IDX] == player])

    def get_combined_score(self, player) -> int:
        """
        3. of 3 functions that define elo rating for specified player. This takes into account both 1. and 2. functions and joins them together.
        :param player: player that requires to know his money count + sum of health of his units
        :return: count of money + sum of health of specified players' units
        """
        # money is not worth more than 1hp because this forces players to spend money in order to create new units
        return sum([self[x][y][HEALTH_IDX] + self[x][y][MONEY_IDX] for x in range(self.n) for y in range(self.n) if self[x][y][P_NAME_IDX] == player])
