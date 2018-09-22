from typing import Tuple

import numpy as np

from td2020.src.Board import Board
from td2020.src.dicts import NUM_ENCODERS, NUM_ACTS, P_NAME_IDX, A_TYPE_IDX, HEALTH_IDX, REMAIN_IDX, VERBOSE, FPS


class TD2020Game:
    def __init__(self, n) -> None:
        self.n = n

    def getInitBoard(self) -> np.ndarray:
        b = Board(self.n)
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

        # update timer on every tile:
        b.pieces[:, :, REMAIN_IDX] -= 1

        b.execute_move(move, player)
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

    def getGameEnded(self, board: np.ndarray, player) -> float:

        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1

        n = board.shape[0]

        # detect timeout
        if board[0, 0, REMAIN_IDX] < 1:
            if VERBOSE:
                print("timeout")
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
                print("game end player +1, tick", board[0, 0, REMAIN_IDX])
                print("#############################################################")

            return 1
        if sum_p2 > 3:
            if VERBOSE:
                print("#############################################################")
                print("game end player -1,tick", board[0, 0, REMAIN_IDX])
                print("#############################################################")

            return -1
        """

        # print("sump", sum_p1,sum_p2)

        if sum_p1 < 2:  # SUM IS 1 WHEN PLAYER ONLY HAS MINERALS LEFT
            if VERBOSE:
                print("#############################################################")
                print("game end player -1, tick", board[0, 0, REMAIN_IDX])
                print("#############################################################")

            return -1
        if sum_p2 < 2:  # SUM IS 1 WHEN PLAYER ONLY HAS MINERALS LEFT
            if VERBOSE:
                print("#############################################################")
                print("game end player +1,tick", board[0, 0, REMAIN_IDX])
                print("#############################################################")
            return +1

        # detect no valid actions - possible tie by overpopulating on non-attacking units and buildings - all fields are full or one player is surrounded:
        if sum(self.getValidMoves(board, 1)) == 0:
            if VERBOSE:
                print("no valid moves for player", 1)
            # return -0.1
            return -1

        if sum(self.getValidMoves(board, -1)) == 0:
            if VERBOSE:
                print("no valid moves for player", -1)
            # return 0.1
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
        l = []
        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board: np.ndarray):
        return board.tostring()

    def getScore(self, board: np.array, player: int):
        n = board.shape[0]
        return sum([board[x][y][HEALTH_IDX] for x in range(n) for y in range(n) if board[x][y][P_NAME_IDX] == player])


def display(board):
    from td2020.src.Graphics import init_visuals, update_graphics

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
