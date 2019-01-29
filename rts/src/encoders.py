from typing import List

import numpy as np

"""
encoders.py

Defines 'numeric' and one-hot encoder

Numeric encoder uses integers to encode game state (also negative numbers for player names)
One-hot encoder uses binary representation of integer numbers, with exception of player name, which is processed seperately
"""


class Encoder:

    def __init__(self):
        self.NUM_ENCODERS = None

    def encode(self, board) -> np.ndarray:
        pass

    def encode_multiple(self, boards: np.ndarray) -> np.ndarray:
        pass

    @property
    def num_encoders(self):
        return self.NUM_ENCODERS


class NumericEncoder(Encoder):

    def __init__(self) -> None:
        super().__init__()
        self.NUM_ENCODERS = 6  # player_name, act_type, health, carrying, money, remaining_time

    def encode_multiple(self, boards: np.ndarray) -> np.ndarray:
        return boards

    def encode(self, board) -> np.ndarray:
        return board


class OneHotEncoder(Encoder):
    def __init__(self) -> None:
        super().__init__()
        self._build_indexes()

    def _build_indexes(self):

        self.P_NAME_IDX_INC_OH = 2  # playerName 2 bit - 00(neutral), 01(1) or 10(-1),
        self.A_TYPE_IDX_INC_OH = 3  # actor type -> 3 bit,
        self.HEALTH_IDX_INC_OH = 5  # health-> 5 bit,
        self.CARRY_IDX_INC_OH = 1  # carrying-> 1 bit,
        self.MONEY_IDX_INC_OH = 8  # money-> 8 bits (255) [every unit has the same for player]
        self.REMAIN_IDX_INC_OH = 11  # 2^11 2048(za total annihilation)

        # builds indexes for character encoding - if not using one hot encoding, max indexes are incremented by 1 from previous index, but for one hot encoding, its incremented by num bits
        self.P_NAME_IDX_OH = 0
        self.P_NAME_IDX_MAX_OH = self.P_NAME_IDX_INC_OH

        self.A_TYPE_IDX_OH = self.P_NAME_IDX_MAX_OH
        self.A_TYPE_IDX_MAX_OH = self.A_TYPE_IDX_OH + self.A_TYPE_IDX_INC_OH

        self.HEALTH_IDX_OH = self.A_TYPE_IDX_MAX_OH
        self.HEALTH_IDX_MAX_OH = self.HEALTH_IDX_OH + self.HEALTH_IDX_INC_OH

        self.CARRY_IDX_OH = self.HEALTH_IDX_MAX_OH
        self.CARRY_IDX_MAX_OH = self.CARRY_IDX_OH + self.CARRY_IDX_INC_OH

        self.MONEY_IDX_OH = self.CARRY_IDX_MAX_OH
        self.MONEY_IDX_MAX_OH = self.MONEY_IDX_OH + self.MONEY_IDX_INC_OH

        self.REMAIN_IDX_OH = self.MONEY_IDX_MAX_OH
        self.REMAIN_IDX_MAX_OH = self.REMAIN_IDX_OH + self.REMAIN_IDX_INC_OH

        self.NUM_ENCODERS = self.REMAIN_IDX_MAX_OH

    @staticmethod
    def bti(num: List[int]) -> int:
        """
        example ->  print(bti(itb(6,13)))
        """
        return int("".join([str(i) for i in num]), 2)

    @staticmethod
    def itb(num: int, length: int) -> List[int]:
        num = int(num)
        if length == 1:
            return [int(i) for i in '{0:01b}'.format(num)]
        if length == 2:
            return [int(i) for i in '{0:02b}'.format(num)]
        if length == 3:
            return [int(i) for i in '{0:03b}'.format(num)]
        if length == 4:
            return [int(i) for i in '{0:04b}'.format(num)]
        if length == 5:
            return [int(i) for i in '{0:05b}'.format(num)]
        if length == 8:
            return [int(i) for i in '{0:08b}'.format(num)]
        if length == 11:
            return [int(i) for i in '{0:011b}'.format(num)]
        raise TypeError("Length not supported:", length)

    def encode_multiple(self, boards: np.ndarray) -> np.ndarray:
        new_boards = []
        for board in boards:
            new_boards.append(self.encode(board))
        return np.asarray(new_boards)

    def encode(self, board) -> np.ndarray:
        from rts.src.config import P_NAME_IDX, A_TYPE_IDX, HEALTH_IDX, CARRY_IDX, MONEY_IDX, TIME_IDX

        n = board.shape[0]

        b = np.zeros((n, n, self.NUM_ENCODERS))
        for y in range(n):
            for x in range(n):
                # switch player from -1 to 2
                player = 0
                if board[x, y, P_NAME_IDX] == 1:
                    player = 1
                elif board[x, y, P_NAME_IDX] == -1:
                    player = 2

                b[x, y][self.P_NAME_IDX_OH:self.P_NAME_IDX_MAX_OH] = self.itb(player, self.P_NAME_IDX_INC_OH)
                b[x, y][self.A_TYPE_IDX_OH:self.A_TYPE_IDX_MAX_OH] = self.itb(board[x, y, A_TYPE_IDX], self.A_TYPE_IDX_INC_OH)
                b[x, y][self.HEALTH_IDX_OH:self.HEALTH_IDX_MAX_OH] = self.itb(board[x, y, HEALTH_IDX], self.HEALTH_IDX_INC_OH)
                b[x, y][self.CARRY_IDX_OH:self.CARRY_IDX_MAX_OH] = self.itb(board[x, y, CARRY_IDX], self.CARRY_IDX_INC_OH)
                b[x, y][self.MONEY_IDX_OH:self.MONEY_IDX_MAX_OH] = self.itb(board[x, y, MONEY_IDX], self.MONEY_IDX_INC_OH)
                b[x, y][self.REMAIN_IDX_OH:self.REMAIN_IDX_MAX_OH] = self.itb(board[x, y, TIME_IDX], self.REMAIN_IDX_INC_OH)
        return b
