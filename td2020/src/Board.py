from typing import List, Any

import numpy as np

from td2020.src.dicts import d_a_type, a_m_health, d_acts, EXCLUDE_IDLE, A_TYPE_IDX, P_NAME_IDX, CARRY_IDX, MONEY_IDX, a_cost, NUM_ACTS, ACTS_REV, NUM_ENCODERS, MONEY_INC, HEALTH_IDX, TIMEOUT, REMAIN_IDX, INITIAL_GOLD, DAMAGE, DAMAGE_ANYWHERE, DESTROY_ALL, VERBOSE


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

        t_hall = d_a_type['Hall']
        h_hall = a_m_health[t_hall]
        t_gold = d_a_type['Gold']
        h_gold = a_m_health[t_gold]
        self[int(self.n / 2) - 1][int(self.n / 2)] = [1, t_gold, h_gold, 0, INITIAL_GOLD, TIMEOUT]
        self[int(self.n / 2)][int(self.n / 2) - 1] = [-1, t_gold, h_gold, 0, INITIAL_GOLD, TIMEOUT]
        self[int(self.n / 2) - 1][int(self.n / 2) - 1] = [1, t_hall, h_hall, 0, INITIAL_GOLD, TIMEOUT]
        self[int(self.n / 2)][int(self.n / 2)] = [-1, t_hall, h_hall, 0, INITIAL_GOLD, TIMEOUT]

        # remaining time is stored in all squares
        self.pieces[:, :, REMAIN_IDX] = TIMEOUT

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
            # print("RETURNED RESOURCES - old money->", self[x][y][MONEY_IDX])
            self[x][y][CARRY_IDX] = 0
            self._update_money(player, MONEY_INC)
            return
        if act == "attack":
            self._attack_nearby((x, y))
            return
        if act == "npc":
            self._update_money(player, -a_cost[2])
            self._spawn_nearby((x, y), 2)
            return
        if act == "barracks":
            if VERBOSE:
                pass
                # print("spawned barracks")
            self._update_money(player, -a_cost[3])
            self._spawn_nearby((x, y), 3)
            return
        if act == "rifle_infantry":
            if VERBOSE:
                pass

                # print("spawned rifle inf")
            self._update_money(player, -a_cost[4])
            self._spawn_nearby((x, y), 4)
            return
        if act == "town_hall":
            if VERBOSE:
                pass
                # print("spawned town hall")
            self._update_money(player, -a_cost[5])
            self._spawn_nearby((x, y), 5)
            return

    def _move(self, x, y, new_x, new_y):
        self[new_x][new_y] = self[x][y]
        self[x][y] = [0] * NUM_ENCODERS
        self[x][y][REMAIN_IDX] = self[new_x][new_y][REMAIN_IDX]  # set time back to empty tile

    def _update_money(self, player, money_update):
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y][P_NAME_IDX] == player:
                    assert self[x][y][MONEY_IDX] + money_update >= 0
                    self[x][y][MONEY_IDX] = self[x][y][MONEY_IDX] + money_update

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
                        self[n_x][n_y][REMAIN_IDX] = self[x][y][REMAIN_IDX]  # set time back to empty tile just in case
                        if not DESTROY_ALL:
                            return

                    if VERBOSE:
                        print("damaged unit type", self[n_x][n_y][A_TYPE_IDX], "on", n_x, n_y, "and damage initiator of type", self[x][y][A_TYPE_IDX], "on", x, y)
                    if not DESTROY_ALL:
                        return
        print("returning")

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
                    self[n_x][n_y] = [self[x][y][P_NAME_IDX], a_type, a_m_health[a_type], 0, self[x][y][MONEY_IDX], self[x][y][REMAIN_IDX]]
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
            return self[x][y][CARRY_IDX] == 1 and self._check_if_nearby(square, d_a_type['Hall'], check_friendly=True)
        if act == "attack":
            return self._check_if_nearby_attack(square)
        if act == "npc":
            return a_cost[2] <= money and self._check_if_nearby_empty(square)
        if act == "barracks":
            return a_cost[3] <= money and self._check_if_nearby_empty(square)
        if act == "rifle_infantry":
            return a_cost[4] <= money and self._check_if_nearby_empty(square)
        if act == "town_hall":
            return a_cost[5] <= money and self._check_if_nearby_empty(square)

    # noinspection PyChainedComparisons
    def _check_if_empty(self, new_x, new_y):
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
