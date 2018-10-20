from typing import List, Any

import numpy as np

from td2020.src.Graph import num_destroys, damage
from td2020.src.config import d_a_type, a_m_health, d_acts, EXCLUDE_IDLE, A_TYPE_IDX, P_NAME_IDX, CARRY_IDX, MONEY_IDX, a_cost, NUM_ACTS, ACTS_REV, NUM_ENCODERS, MONEY_INC, HEALTH_IDX, TIME_IDX, DAMAGE, DAMAGE_ANYWHERE, DESTROY_ALL, VERBOSE, MAX_GOLD, HEAL_AMOUNT, \
    a_max_health, SACRIFICIAL_HEAL,  HEAL_COST


class Board:

    def __init__(self, n) -> None:
        self.n = n
        self.pieces = []
        for i in range(self.n):
            rows = []
            for j in range(self.n):
                rows.append([0] * NUM_ENCODERS)
            self.pieces.append(rows)
        self.pieces = np.array(self.pieces)

    def __getitem__(self, index: int) -> List[List[int]]:
        return self.pieces[index]

    def execute_move(self, move, player) -> None:
        x, y, action_index = move
        act = ACTS_REV[action_index]
        if act == "idle":
            pass
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
            # if VERBOSE:
            #    print("RETURNED RESOURCES - new money->", self[x][y][MONEY_IDX] + MONEY_INC)
            self[x][y][CARRY_IDX] = 0
            self._update_money(player, MONEY_INC)
            return
        if act == "attack":
            self._attack_nearby((x, y))
            return
        if act == "heal":
            self._heal_nearby((x, y))
            return
        if act == "npc":
            self._update_money(player, -a_cost[2])
            self._spawn_nearby((x, y), 2)
            return
        if act == "barracks":
            #if VERBOSE:
            print("spawned barracks")
            self._update_money(player, -a_cost[3])
            self._spawn_nearby((x, y), 3)
            return
        if act == "rifle_infantry":
            # if VERBOSE:
            print("spawned rifle inf")
            self._update_money(player, -a_cost[4])
            self._spawn_nearby((x, y), 4)
            return
        if act == "town_hall":
            # if VERBOSE:
            print("spawned town hall")
            self._update_money(player, -a_cost[5])
            self._spawn_nearby((x, y), 5)
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

    def _heal_nearby(self, square):
        (x, y) = square
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
                if (self[n_x][n_y][P_NAME_IDX] == self[x][y][P_NAME_IDX]) and self[n_x][n_y][A_TYPE_IDX] != d_a_type['Gold'] and self[n_x][n_y][HEALTH_IDX] < a_max_health[self[n_x][n_y][A_TYPE_IDX]]:
                    if SACRIFICIAL_HEAL:
                        self[x][x][HEALTH_IDX] -= HEAL_COST
                        if self[x][y][HEALTH_IDX] <= 0:
                            # if VERBOSE:
                            #     print("unit sacrificed itself to heal other unit")
                            self[x][y] = [0] * NUM_ENCODERS
                            self[x][y][TIME_IDX] = self[x][y][TIME_IDX]
                    elif self[n_x][n_y][MONEY_IDX] - HEAL_AMOUNT >= 0:
                        self[n_x][n_y][HEALTH_IDX] += HEAL_AMOUNT
                        self._update_money(self[n_x][n_y][P_NAME_IDX], -HEAL_COST)

                    # print("healed")

                    # clamp value to max
                    self[n_x][n_y][HEALTH_IDX] = self.clamp(self[n_x][n_y][HEALTH_IDX] + HEAL_AMOUNT, 0, a_max_health[self[n_x][n_y][A_TYPE_IDX]])
                    # if VERBOSE:
                    #     print("new health of this unit is ", self[n_x][n_y][HEALTH_IDX])
                    return

    def _attack_nearby(self, square):
        (x, y) = square

        if DAMAGE_ANYWHERE:
            coordinates = [(n_x, n_y) for n_x in range(self.n) for n_y in range(self.n) if (n_x, n_y) != (x, y)]
        else:
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
                if (self[n_x][n_y][P_NAME_IDX] == -self[x][y][P_NAME_IDX]) and self[n_x][n_y][A_TYPE_IDX] != d_a_type['Gold']:
                    self[n_x][n_y][HEALTH_IDX] -= DAMAGE

                    if self[n_x][n_y][HEALTH_IDX] <= 0:
                        if VERBOSE:
                            print("destroyed unit type", self[n_x][n_y][A_TYPE_IDX], "on", n_x, n_y, "and destroyer of type", self[x][y][A_TYPE_IDX], "on", x, y)

                        self[n_x][n_y] = [0] * NUM_ENCODERS
                        self[n_x][n_y][TIME_IDX] = self[x][y][TIME_IDX]  # set time back to empty tile just in case
                        if not DESTROY_ALL:
                            return

                    if VERBOSE:
                        print("damaged unit type", self[n_x][n_y][A_TYPE_IDX], "on", n_x, n_y, "and damage initiator of type", self[x][y][A_TYPE_IDX], "on", x, y)
                    if not DESTROY_ALL:
                        return

    def _spawn_nearby(self, square, a_type):
        (x, y) = square

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
                if self[n_x][n_y][P_NAME_IDX] == 0:
                    self[n_x][n_y] = [self[x][y][P_NAME_IDX], a_type, a_m_health[a_type], 0, self[x][y][MONEY_IDX], self[x][y][TIME_IDX]]
                    return

        ##############################################
        ##############################################
        ##############################################
        ##############################################
        ##############################################
        ##############################################

    def get_moves_for_square(self, square) -> Any:
        (x, y) = square

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
                move = self._valid_act(square, act) * 1

                if move:
                    moves[i] = move
        # return the generated move list
        return moves

    def _valid_act(self, square, act):
        (x, y) = square
        money = self[x][y][MONEY_IDX]
        if act == "idle":
            return not EXCLUDE_IDLE
        if act == "up":
            new_x, new_y = x, y - 1
            return self._check_if_empty(new_x, new_y)
        if act == "down":
            new_x, new_y = x, y + 1
            return self._check_if_empty(new_x, new_y)
        if act == "right":
            new_x, new_y = x + 1, y
            return self._check_if_empty(new_x, new_y)
        if act == "left":
            new_x, new_y = x - 1, y
            return self._check_if_empty(new_x, new_y)
        if act == "mine_resources":
            return self[x][y][CARRY_IDX] == 0 and self._check_if_nearby(square, d_a_type['Gold'])
        if act == "return_resources":
            return self[x][y][CARRY_IDX] == 1 and self._check_if_nearby(square, d_a_type['Hall'], check_friendly=True) and (MAX_GOLD >= self[x][y][MONEY_IDX] + MONEY_INC)
        if act == "attack":
            return self._check_if_nearby_attack(square)
        if act == "heal":
            return self._check_if_nearby_heal(square)
        if act == "npc":
            return a_cost[2] <= money and self._check_if_nearby_empty(square)
        if act == "barracks":
            return a_cost[3] <= money and self._check_if_nearby_empty(square)
        if act == "rifle_infantry":
            return a_cost[4] <= money and self._check_if_nearby_empty(square)
        if act == "town_hall":
            return a_cost[5] <= money and self._check_if_nearby_empty(square)

    def _check_if_empty(self, new_x, new_y):
        # noinspection PyChainedComparisons
        return 0 <= new_x < self.n and 0 <= new_y < self.n and self[new_x][new_y][P_NAME_IDX] == 0

    def _check_if_nearby_attack(self, square):
        (x, y) = square
        if DAMAGE_ANYWHERE:
            coordinates = [(n_x, n_y) for n_x in range(self.n) for n_y in range(self.n) if (n_x, n_y) != (x, y)]
        else:
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
                if (self[n_x][n_y][P_NAME_IDX] == -self[x][y][P_NAME_IDX]) and self[n_x][n_y][A_TYPE_IDX] != d_a_type['Gold']:
                    return True
        return False

    def _check_if_nearby_heal(self, square):
        (x, y) = square
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
                if (self[n_x][n_y][P_NAME_IDX] == self[x][y][P_NAME_IDX]) and self[n_x][n_y][A_TYPE_IDX] != d_a_type['Gold'] and self[n_x][n_y][HEALTH_IDX] < a_max_health[self[n_x][n_y][A_TYPE_IDX]]:
                    if SACRIFICIAL_HEAL:
                        return True
                    elif self[n_x][n_y][MONEY_IDX] - HEAL_COST >= 0:
                        return True
        return False

    def _check_if_nearby_empty(self, square):
        (x, y) = square
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
                if self[n_x][n_y][P_NAME_IDX] == 0:
                    return True
        return False

    def _check_if_nearby(self, square, a_type, check_friendly=False):
        (x, y) = square
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
                        if VERBOSE:
                            print("actor died because of timer kill function", self[x][y][A_TYPE_IDX], "for player", player, "in round", self[0][0][TIME_IDX])
                        time = self[x][y][TIME_IDX]
                        self[x][y] = [0] * NUM_ENCODERS
                        self[x][y][TIME_IDX] = time
                    currently_damaged_actors += 1

    @staticmethod
    def clamp(num, min_value, max_value):
        return max(min(num, max_value), min_value)
