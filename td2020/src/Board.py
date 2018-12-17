import sys
from typing import Any

import numpy as np

from td2020.src.Graph import num_destroys, damage
from td2020.src.config import d_a_type, d_acts, A_TYPE_IDX, P_NAME_IDX, CARRY_IDX, MONEY_IDX, a_cost, NUM_ACTS, ACTS_REV, NUM_ENCODERS, MONEY_INC, HEALTH_IDX, TIME_IDX, DAMAGE, MAX_GOLD, HEAL_AMOUNT, \
    a_max_health, SACRIFICIAL_HEAL, HEAL_COST, acts_enabled, MAKE_STATS, visibility
from td2020.stats.files import Stats

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
        x, y, action_index = move
        act = ACTS_REV[action_index]
        if MAKE_STATS:
            Stats.action(self[x][y][TIME_IDX], act)
            Stats.sum(self[x][y][TIME_IDX], self)
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
            self._update_money(player, MONEY_INC)
            return
        if act == "attack_up":
            self._attack(x, y, x, y - 1)
            return
        if act == "attack_down":
            self._attack(x, y, x, y + 1)
            return
        if act == "attack_left":
            self._attack(x, y, x - 1, y)
            return
        if act == "attack_right":
            self._attack(x, y, x + 1, y)
            return

        if act == "heal_up":
            self._heal(x, y, x, y - 1)
            return
        if act == "heal_down":
            self._heal(x, y, x, y + 1)
            return
        if act == "heal_left":
            self._heal(x, y, x - 1, y)
            return
        if act == "heal_right":
            self._heal(x, y, x + 1, y)
            return

        if act == "npc_up":
            self._update_money(player, -a_cost[2])
            self._spawn(x, y, x, y - 1, 2)
            return
        if act == "npc_down":
            self._update_money(player, -a_cost[2])
            self._spawn(x, y, x, y + 1, 2)
            return
        if act == "npc_left":
            self._update_money(player, -a_cost[2])
            self._spawn(x, y, x - 1, y, 2)
            return
        if act == "npc_right":
            self._update_money(player, -a_cost[2])
            self._spawn(x, y, x + 1, y, 2)
            return

        if act == "barracks_up":
            self._update_money(player, -a_cost[3])
            self._spawn(x, y, x, y - 1, 3)
            return
        if act == "barracks_down":
            self._update_money(player, -a_cost[3])
            self._spawn(x, y, x, y + 1, 3)
            return
        if act == "barracks_left":
            self._update_money(player, -a_cost[3])
            self._spawn(x, y, x - 1, y, 3)
            return
        if act == "barracks_right":
            self._update_money(player, -a_cost[3])
            self._spawn(x, y, x + 1, y, 3)
            return

        if act == "rifle_infantry_up":
            self._update_money(player, -a_cost[4])
            self._spawn(x, y, x, y - 1, 4)
            return
        if act == "rifle_infantry_down":
            self._update_money(player, -a_cost[4])
            self._spawn(x, y, x, y + 1, 4)
            return
        if act == "rifle_infantry_left":
            self._update_money(player, -a_cost[4])
            self._spawn(x, y, x - 1, y, 4)
            return
        if act == "rifle_infantry_right":
            self._update_money(player, -a_cost[4])
            self._spawn(x, y, x + 1, y, 4)
            return

        if act == "town_hall_up":
            self._update_money(player, -a_cost[5])
            self._spawn(x, y, x, y - 1, 5)
            return
        if act == "town_hall_down":
            self._update_money(player, -a_cost[5])
            self._spawn(x, y, x, y + 1, 5)
            return
        if act == "town_hall_left":
            self._update_money(player, -a_cost[5])
            self._spawn(x, y, x - 1, y, 5)
            return
        if act == "town_hall_right":
            self._update_money(player, -a_cost[5])
            self._spawn(x, y, x + 1, y, 5)
            return

    def _move(self, x, y, new_x, new_y):
        self[new_x][new_y] = self[x][y]
        self[x][y] = [0] * NUM_ENCODERS
        self[x][y][TIME_IDX] = self[new_x][new_y][TIME_IDX]  # set time back to empty tile

    def _update_money(self, player, money_update):
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y][P_NAME_IDX] == player:
                    assert self[x][y][MONEY_IDX] + money_update >= 0
                    self[x][y][MONEY_IDX] = self[x][y][MONEY_IDX] + money_update

    def _attack(self, x, y, n_x, n_y):
        self[n_x][n_y][HEALTH_IDX] -= DAMAGE
        if self[n_x][n_y][HEALTH_IDX] <= 0:
            Stats.killed_by(self[n_x][n_x][TIME_IDX], self[n_x][n_x][A_TYPE_IDX], "attack")
            if visibility.verbose:
                print("destroyed unit type", self[n_x][n_y][A_TYPE_IDX], "on", n_x, n_y, "and destroyer of type", self[x][y][A_TYPE_IDX], "on", x, y)

            self[n_x][n_y] = [0] * NUM_ENCODERS
            self[n_x][n_y][TIME_IDX] = self[x][y][TIME_IDX]  # set time back to empty tile just in case

        if visibility.verbose:
            print("damaged unit type", self[n_x][n_y][A_TYPE_IDX], "on", n_x, n_y, "and damage initiator of type", self[x][y][A_TYPE_IDX], "on", x, y)

    def _spawn(self, x, y, n_x, n_y, a_type):
        self[n_x][n_y] = [self[x][y][P_NAME_IDX], a_type, a_max_health[a_type], 0, self[x][y][MONEY_IDX], self[x][y][TIME_IDX]]

    def _heal(self, x, y, n_x, n_y):

        if SACRIFICIAL_HEAL:
            self[x][x][HEALTH_IDX] -= HEAL_COST
            if self[x][y][HEALTH_IDX] <= 0:
                self[x][y] = [0] * NUM_ENCODERS
                self[x][y][TIME_IDX] = self[x][y][TIME_IDX]
        elif self[n_x][n_y][MONEY_IDX] - HEAL_AMOUNT >= 0:
            self[n_x][n_y][HEALTH_IDX] += HEAL_AMOUNT
            self._update_money(self[n_x][n_y][P_NAME_IDX], -HEAL_COST)

        # clamp value to max
        self[n_x][n_y][HEALTH_IDX] = self.clamp(self[n_x][n_y][HEALTH_IDX] + HEAL_AMOUNT, 0, a_max_health[self[n_x][n_y][A_TYPE_IDX]])

    def get_moves_for_square(self, x, y) -> Any:

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
                move = self._valid_act(x, y, act) * 1
                if move:
                    moves[i] = move
        # return the generated move list
        return moves

    def _valid_act(self, x, y, act):
        money = self[x][y][MONEY_IDX]
        if act == "idle":
            return acts_enabled.idle
        if act == "up":
            return acts_enabled.up and self._check_if_empty(x, y - 1)
        if act == "down":
            return acts_enabled.down and self._check_if_empty(x, y + 1)
        if act == "right":
            return acts_enabled.right and self._check_if_empty(x + 1, y)
        if act == "left":
            return acts_enabled.left and self._check_if_empty(x - 1, y)

        if act == "mine_resources":
            return acts_enabled.mine_resources and self[x][y][CARRY_IDX] == 0 and self._check_if_nearby(x, y, d_a_type['Gold'])
        if act == "return_resources":
            return acts_enabled.return_resources and self[x][y][CARRY_IDX] == 1 and self._check_if_nearby(x, y, d_a_type['Hall'], check_friendly=True) and (MAX_GOLD >= self[x][y][MONEY_IDX] + MONEY_INC)

        if act == "attack_up":
            return acts_enabled.attack and self._check_if_attack(x, y, x, y - 1)
        if act == "attack_down":
            return acts_enabled.attack and self._check_if_attack(x, y, x, y + 1)
        if act == "attack_right":
            return acts_enabled.attack and self._check_if_attack(x, y, x + 1, y)
        if act == "attack_left":
            return acts_enabled.attack and self._check_if_attack(x, y, x - 1, y)

        if act == "heal_up":
            return acts_enabled.heal and self._check_if_heal(x, y - 1)
        if act == "heal_down":
            return acts_enabled.heal and self._check_if_heal(x, y + 1)
        if act == "heal_right":
            return acts_enabled.heal and self._check_if_heal(x + 1, y)
        if act == "heal_left":
            return acts_enabled.heal and self._check_if_heal(x - 1, y)

        if act == "npc_up":
            return acts_enabled.npc and a_cost[2] <= money and self._check_if_empty(x, y - 1)
        if act == "npc_down":
            return acts_enabled.npc and a_cost[2] <= money and self._check_if_empty(x, y + 1)
        if act == "npc_right":
            return acts_enabled.npc and a_cost[2] <= money and self._check_if_empty(x + 1, y)
        if act == "npc_left":
            return acts_enabled.npc and a_cost[2] <= money and self._check_if_empty(x - 1, y)

        if act == "barracks_up":
            return acts_enabled.barracks and a_cost[3] <= money and self._check_if_empty(x, y - 1)
        if act == "barracks_down":
            return acts_enabled.barracks and a_cost[3] <= money and self._check_if_empty(x, y + 1)
        if act == "barracks_right":
            return acts_enabled.barracks and a_cost[3] <= money and self._check_if_empty(x + 1, y)
        if act == "barracks_left":
            return acts_enabled.barracks and a_cost[3] <= money and self._check_if_empty(x - 1, y)

        if act == "rifle_infantry_up":
            return acts_enabled.rifle_infantry and a_cost[4] <= money and self._check_if_empty(x, y - 1)
        if act == "rifle_infantry_down":
            return acts_enabled.rifle_infantry and a_cost[4] <= money and self._check_if_empty(x, y + 1)
        if act == "rifle_infantry_right":
            return acts_enabled.rifle_infantry and a_cost[4] <= money and self._check_if_empty(x + 1, y)
        if act == "rifle_infantry_left":
            return acts_enabled.rifle_infantry and a_cost[4] <= money and self._check_if_empty(x - 1, y)

        if act == "town_hall_up":
            return acts_enabled.town_hall and a_cost[5] <= money and self._check_if_empty(x, y - 1)
        if act == "town_hall_down":
            return acts_enabled.town_hall and a_cost[5] <= money and self._check_if_empty(x, y + 1)
        if act == "town_hall_right":
            return acts_enabled.town_hall and a_cost[5] <= money and self._check_if_empty(x + 1, y)
        if act == "town_hall_left":
            return acts_enabled.town_hall and a_cost[5] <= money and self._check_if_empty(x - 1, y)
        print("Unrecognised action", act)
        sys.exit(0)

    def _check_if_empty(self, x, y):
        # noinspection PyChainedComparisons
        return self.n > x >= 0 and 0 <= y < self.n and self[x][y][P_NAME_IDX] == 0

    def _check_if_attack(self, x, y, n_x, n_y):
        return 0 <= n_x < self.n and 0 <= n_y < self.n and self[x][y][P_NAME_IDX] == -self[n_x][n_y][P_NAME_IDX] and self[n_x][n_y][A_TYPE_IDX] != d_a_type['Gold']

    def _check_if_heal(self, x, y):
        return 0 <= x < self.n and 0 <= y < self.n and self[x][y][P_NAME_IDX] == self[x][y][P_NAME_IDX] and self[x][y][A_TYPE_IDX] != d_a_type['Gold'] and self[x][y][A_TYPE_IDX] > 0 and self[x][y][HEALTH_IDX] < a_max_health[self[x][y][A_TYPE_IDX]] and (
                SACRIFICIAL_HEAL or self[x][y][MONEY_IDX] - HEAL_COST >= 0)

    def _check_if_nearby(self, x, y, a_type, check_friendly=False):
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

    def time_killer(self, player):
        # I can pass player through, because this board is canonical board that this action gets executed upon

        current_time = self[0][0][TIME_IDX]

        destroys_per_round = num_destroys(current_time)
        damage_amount = damage(current_time)

        # Damage as many actors as "damage_amount" parameter provides
        currently_damaged_actors = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y][P_NAME_IDX] == player and self[x][y][A_TYPE_IDX] != 1:  # for current player and not gold
                    if currently_damaged_actors >= destroys_per_round:
                        return
                    self[x][y][HEALTH_IDX] -= damage_amount

                    if self[x][y][HEALTH_IDX] <= 0:
                        if MAKE_STATS:
                            Stats.killed_by(self[x][y][TIME_IDX], self[x][y][A_TYPE_IDX], "time_function")
                        if visibility.verbose:
                            print("actor died because of timer kill function", self[x][y][A_TYPE_IDX], "for player", player, "in round", self[0][0][TIME_IDX])
                        time = self[x][y][TIME_IDX]
                        self[x][y] = [0] * NUM_ENCODERS
                        self[x][y][TIME_IDX] = time
                    currently_damaged_actors += 1

    @staticmethod
    def clamp(num, min_value, max_value):
        return max(min(num, max_value), min_value)

    def get_money_score(self, player) -> int:
        return sum([self[x][y][MONEY_IDX] for x in range(self.n) for y in range(self.n) if self[x][y][P_NAME_IDX] == player])

    def get_health_score(self, player) -> int:
        return sum([self[x][y][HEALTH_IDX] for x in range(self.n) for y in range(self.n) if self[x][y][P_NAME_IDX] == player])

    def get_combined_score(self, player) -> int:
        # money is not worth more than 1hp because this forces players to spend money in order to create new units
        return sum([self[x][y][HEALTH_IDX] + self[x][y][MONEY_IDX] for x in range(self.n) for y in range(self.n) if self[x][y][P_NAME_IDX] == player])
