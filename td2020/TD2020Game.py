from typing import Tuple

import numpy as np

from td2020.src.Board import Board
from td2020.src.config import NUM_ENCODERS, NUM_ACTS, P_NAME_IDX, A_TYPE_IDX, HEALTH_IDX, TIME_IDX, VERBOSE, FPS, USE_TIMEOUT, MAX_TIME, d_a_type, a_m_health, INITIAL_GOLD, TIMEOUT
from td2020.stats.files import Stats
from utils import dotdict


# noinspection PyPep8Naming,PyMethodMayBeStatic
class TD2020Game:
    def __init__(self, n) -> None:
        self.n = n

        self.initial_board_config = [
            dotdict({
                'x': int(self.n / 2) - 1,
                'y': int(self.n / 2),
                'player': 1,
                'a_type': d_a_type['Gold'],
                'health': a_m_health[d_a_type['Gold']],
                'carry': 0,
                'gold': INITIAL_GOLD,
                'timeout': TIMEOUT
            }),
            dotdict({
                'x': int(self.n / 2),
                'y': int(self.n / 2) - 1,
                'player': -1,
                'a_type': d_a_type['Gold'],
                'health': a_m_health[d_a_type['Gold']],
                'carry': 0,
                'gold': INITIAL_GOLD,
                'timeout': TIMEOUT
            }),
            dotdict({
                'x': int(self.n / 2) - 1,
                'y': int(self.n / 2) - 1,
                'player': 1,
                'a_type': d_a_type['Hall'],
                'health': a_m_health[d_a_type['Hall']],
                'carry': 0,
                'gold': INITIAL_GOLD,
                'timeout': TIMEOUT
            }),
            dotdict({
                'x': int(self.n / 2),
                'y': int(self.n / 2),
                'player': -1,
                'a_type': d_a_type['Hall'],
                'health': a_m_health[d_a_type['Hall']],
                'carry': 0,
                'gold': INITIAL_GOLD,
                'timeout': TIMEOUT
            }),
        ]

    def setInitBoard(self, board_config) -> None:
        self.initial_board_config = board_config

    def getInitBoard(self) -> np.ndarray:
        b = Board(self.n)

        remaining_time = None  # when setting initial board, remaining time might be different
        for e in self.initial_board_config:
            b.pieces[e.x, e.y] = [e.player, e.a_type, e.health, e.carry, e.gold, e.timeout]
            remaining_time = e.timeout
        # remaining time is stored in all squares
        b.pieces[:, :, TIME_IDX] = remaining_time
        return np.array(b.pieces)

    def getBoardSize(self) -> Tuple[int, int, int]:
        # (a,b) tuple
        return self.n, self.n, NUM_ENCODERS

    def getActionSize(self) -> int:
        return self.n * self.n * NUM_ACTS + 1

    def getNextState(self, board: np.ndarray, player: int, action: int) -> Tuple[np.ndarray, int]:
        b = Board(self.n)
        b.pieces = np.copy(board)

        y, x, action_index = np.unravel_index(action, [self.n, self.n, NUM_ACTS])
        move = (x, y, action_index)

        # first execute move, then run time function to destroy any actors if needed
        b.execute_move(move, player)

        # update timer on every tile:
        if USE_TIMEOUT:
            b.pieces[:, :, TIME_IDX] -= 1
        else:
            b.pieces[:, :, TIME_IDX] += 1
            b.time_killer(player)

        return b.pieces, -player

    def getValidMoves(self, board: np.ndarray, player: int):
        valids = []
        b = Board(self.n)
        b.pieces = np.copy(board)

        for y in range(self.n):
            for x in range(self.n):
                if b[x][y][P_NAME_IDX] == player and b[x][y][A_TYPE_IDX] != 1:  # for this player and not Gold
                    valids.extend(b.get_moves_for_square((x, y)))
                else:
                    valids.extend([0] * NUM_ACTS)
        valids.append(0)  # because of that +1 in action Size

        return np.array(valids)

    # noinspection PyUnusedLocal
    def getGameEnded(self, board: np.ndarray, player) -> float:
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1

        n = board.shape[0]

        # detect timeout
        if USE_TIMEOUT:
            if board[0, 0, TIME_IDX] < 1:
                if VERBOSE:
                    print("timeout")
                Stats.game_end(board[0, 0, TIME_IDX], 0, 'timeout')
                return 0.001
        else:
            if board[0, 0, TIME_IDX] >= MAX_TIME:
                print("######################################## ERROR ####################################")
                print("################ YOU HAVE TIMEOUTED BECAUSE NO PLAYER HAS LOST YET #################")
                print("###################################### END ERROR ##################################")

                Stats.game_end(board[0, 0, TIME_IDX], 0, 'timeout')
                return 0.001

        # detect win condition
        sum_p1 = 0
        sum_p2 = 0
        for y in range(n):
            for x in range(n):
                if board[x][y][P_NAME_IDX] == 1:
                    sum_p1 += 1
                if board[x][y][P_NAME_IDX] == -1:
                    sum_p2 += 1
        """
        if sum_p1 > 3:
            if VERBOSE:
                print("#############################################################")
                print("game end player +1, tick", board[0, 0, TIME_IDX])
                print("#############################################################")

            return 1
        if sum_p2 > 3:
            if VERBOSE:
                print("#############################################################")
                print("game end player -1,tick", board[0, 0, TIME_IDX])
                print("#############################################################")

            return -1
        """

        # print("sump", sum_p1,sum_p2)

        if sum_p1 < 2:  # SUM IS 1 WHEN PLAYER ONLY HAS MINERALS LEFT
            if VERBOSE:
                print("#############################################################")
                print("game end player -1, tick", board[0, 0, TIME_IDX])
                print("#############################################################")
            Stats.game_end(board[0, 0, TIME_IDX], -1, 'no_actors')
            return -1
        if sum_p2 < 2:  # SUM IS 1 WHEN PLAYER ONLY HAS MINERALS LEFT
            if VERBOSE:
                print("#############################################################")
                print("game end player +1,tick", board[0, 0, TIME_IDX])
                print("#############################################################")
            Stats.game_end(board[0, 0, TIME_IDX], +1, 'no_actors')

            return +1

        # detect no valid actions - possible tie by overpopulating on non-attacking units and buildings - all fields are full or one player is surrounded:
        if sum(self.getValidMoves(board, 1)) == 0:
            if VERBOSE:
                print("#############################################################")
                print("game end player +1,tick", board[0, 0, TIME_IDX])
                print("No valid moves for player", 1)
                print("#############################################################")
            Stats.game_end(board[0, 0, TIME_IDX], -1, 'no_valids')

            return -1

        if sum(self.getValidMoves(board, -1)) == 0:
            if VERBOSE:
                print("#############################################################")
                print("game end player +1,tick", board[0, 0, TIME_IDX])
                print("No valid moves for player", -1)
                print("#############################################################")
            Stats.game_end(board[0, 0, TIME_IDX], +1, 'no_valids')
            return 1
        # continue game
        return 0

    def getCanonicalForm(self, board: np.ndarray, player: int):
        b = np.copy(board)
        n = b.shape[0]
        for y in range(n):
            for x in range(n):
                act_encode = b[x][y]
                act_encode[P_NAME_IDX] = act_encode[P_NAME_IDX] * player
        return b

    def getSymmetries(self, board: np.ndarray, pi):
        # mirror, rotational
        assert (len(pi) == self.n * self.n * NUM_ACTS + 1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n, NUM_ACTS))
        return_list = []
        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                return_list += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return return_list

    def stringRepresentation(self, board: np.ndarray):
        return board.tostring()

    def getScore(self, board: np.array, player: int):
        n = board.shape[0]
        return sum([board[x][y][HEALTH_IDX] for x in range(n) for y in range(n) if board[x][y][P_NAME_IDX] == player])


def display(board):
    from td2020.visualization.Graphics import init_visuals, update_graphics

    if not VERBOSE:
        return

    n = board.shape[0]
    if VERBOSE > 3:
        game_display, clock = init_visuals(n, n, VERBOSE)
        update_graphics(board, game_display, clock, FPS)
    else:
        for y in range(n):
            print('-' * (n * 6 + 1))
            for x in range(n):
                a_player = board[x][y][P_NAME_IDX]
                if a_player == 1:
                    a_player = '+1'
                if a_player == -1:
                    a_player = '-1'
                if a_player == 0:
                    a_player = ' 0'
                print("|" + a_player + " " + str(board[x][y][A_TYPE_IDX]) + " ", end="")
            print("|")
        print('-' * (n * 6 + 1))
