from typing import List, Any

import numpy as np

from td2020.src.config import d_a_type, a_m_health, d_acts, EXCLUDE_IDLE, A_TYPE_IDX, P_NAME_IDX, CARRY_IDX, MONEY_IDX, a_cost, NUM_ACTS, ACTS_REV, NUM_ENCODERS, MONEY_INC, HEALTH_IDX, TIMEOUT, TIME_IDX, INITIAL_GOLD, DAMAGE, DAMAGE_ANYWHERE, DESTROY_ALL, VERBOSE, MAX_GOLD, HEAL_AMOUNT, \
    a_max_health, SACRIFICIAL_HEAL, SHOW_TIME_GRAPH


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
            if VERBOSE:
                print("RETURNED RESOURCES - new money->", self[x][y][MONEY_IDX] + MONEY_INC)
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
            if VERBOSE:
                print("spawned barracks")
            self._update_money(player, -a_cost[3])
            self._spawn_nearby((x, y), 3)
            return
        if act == "rifle_infantry":
            if VERBOSE:
                print("spawned rifle inf")
            self._update_money(player, -a_cost[4])
            self._spawn_nearby((x, y), 4)
            return
        if act == "town_hall":
            if VERBOSE:
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
                if (self[n_x][n_y][P_NAME_IDX] == self[x][y][P_NAME_IDX]) and self[n_x][n_y][A_TYPE_IDX] != d_a_type['Gold'] and a_max_health[self[n_x][n_y][A_TYPE_IDX]] >= self[n_x][n_y][HEALTH_IDX] + HEAL_AMOUNT:
                    if SACRIFICIAL_HEAL:
                        self[x][x][HEALTH_IDX] -= HEAL_AMOUNT
                        if self[x][y][HEALTH_IDX] <= 0:
                            if VERBOSE:
                                print("unit sacrificed itself to heal other unit")
                            self[x][y] = [0] * NUM_ENCODERS
                            self[x][y][TIME_IDX] = self[x][y][TIME_IDX]  # set time back to empty tile just in case

                        self[n_x][n_y][HEALTH_IDX] += HEAL_AMOUNT
                        print("healed")
                        return
                    elif self[n_x][n_y][MONEY_IDX] - HEAL_AMOUNT >= 0:
                        self[n_x][n_y][HEALTH_IDX] += HEAL_AMOUNT
                        self._update_money(self[n_x][n_y][P_NAME_IDX], -HEAL_AMOUNT)
                        print("healed")
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
                if (self[n_x][n_y][P_NAME_IDX] == self[x][y][P_NAME_IDX]) and self[n_x][n_y][A_TYPE_IDX] != d_a_type['Gold'] and a_max_health[self[n_x][n_y][A_TYPE_IDX]] >= self[n_x][n_y][HEALTH_IDX] + HEAL_AMOUNT:
                    if SACRIFICIAL_HEAL:
                        return True
                    elif self[n_x][n_y][MONEY_IDX] - HEAL_AMOUNT >= 0:
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

        def num_destroys(time):
            return int((time / 1024) ** 2 + 1)

        def damage(time):
            return int((time / 2) ** 2.718 / (time * 4096) + 1)

        if SHOW_TIME_GRAPH:
            import matplotlib.pyplot as plt

            all_damage_amounts = []
            all_num_destoys_per_turn = []
            for current_time_local in range(1, 8192):
                all_num_destoys_per_turn.append(num_destroys(current_time_local))
                all_damage_amounts.append(damage(current_time_local))

            plt.plot(all_damage_amounts)
            plt.plot(all_num_destoys_per_turn, 'r--')
            plt.title("")
            plt.xlabel('Game round')
            plt.ylabel('Damaged actors / Damage per actors')
            plt.axis([0, 8000, 0, 64])  # 64 as in max actors on field
            plt.annotate('Destroyed figures', xy=(5500, num_destroys(5500)), xytext=(6000, 20), arrowprops=dict(facecolor='black', shrink=0.1), )
            plt.annotate('Damage dealt', xy=(3000, damage(3000)), xytext=(1000, 40), arrowprops=dict(facecolor='black', shrink=0.1), )
            plt.show()

        destroys_per_round = num_destroys(current_time)
        damage_amount = damage(current_time)

        print("neki sm se spovnu - da je ta heal preveč rough -> rabš plačat 5 coinsu za 5 lajfa- toj preveč - recmo 1 coin za 10 lajfa right? - oziroma vč.. "
              "pa spremen tko da ga heala do maxa če nima povhnga lajfa && sum(current_life, heal_amount) > max_life - pač nared clamp max life pa checki "
              "če ga slučajn nima že tko max - drgač nemore healat")

        def damage_single_actor(board: Board, damage_amount: int):
            for y in range(board.n):
                for x in range(board.n):
                    if board[x][y][P_NAME_IDX] == player and board[x][y][A_TYPE_IDX] != 1:  # for current player and not gold
                        board[x][y][HEALTH_IDX] -= damage_amount

                        if board[x][y][HEALTH_IDX] <= 0:
                            if VERBOSE:
                                print("actor died because of timer kill function", board[x][y][A_TYPE_IDX], "for player", player, "in round", board[0][0][TIME_IDX])

                            board[x][y] = [0] * NUM_ENCODERS
                            board[x][y][TIME_IDX] = board[x][y][TIME_IDX]  # set time back to empty tile just in case
                        return
        print("DUBU SM IDEJO - DESTROYS_PER_ROUND JE KOK ACTORJU JE DAMAGANIH VSAKO RUNDO- ČE JIH JE VČ KOKR JIH PLAYER IMA, NIMA UČINKA VEČ")
        # I have to figure out pattern which actor will be damaged,
        #  and so not the same actor is damaged multiple times if there are more destroys per round
        # for now i am damaging first actor that I found
        # Todo
        for i in range(destroys_per_round):
            damage_single_actor(self, damage_amount)
