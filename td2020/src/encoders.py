from typing import List

import numpy as np


class Encoder:

    def __init__(self):
        self.NUM_ENCODERS = None

    def encode(self, board) -> np.ndarray:
        pass

    @property
    def num_encoders(self):
        return self.NUM_ENCODERS


class NumericEncoder(Encoder):

    def __init__(self) -> None:
        super().__init__()
        self.NUM_ENCODERS = 6  # player_name, act_type, health, carrying, money, remaining_time

    def encode(self, board) -> np.ndarray:
        return board


class OneHotEncoder(Encoder):
    def __init__(self) -> None:
        super().__init__()
        self._build_indexes()

    def _build_indexes(self):

        self.P_NAME_IDX_INC_OH = 2  # playerName 2 bit - 00(neutral), 01(1) or 10(-1),
        self.A_TYPE_IDX_INC_OH = 4  # actor type -> 4 bit,
        self.HEALTH_IDX_INC_OH = 2  # health-> 2 bit,
        self.CARRY_IDX_INC_OH = 1  # carrying-> 1 bit,
        self.MONEY_IDX_INC_OH = 5  # money-> 5 bits (32 aka 4 town halls or 32 workers) [every unit has the same for player]
        self.REMAIN_IDX_INC_OH = 13  # 2^13 8192(za total annihilation)

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
        if length == 1:
            return [int(i) for i in '{0:01b}'.format(num)]
        if length == 2:
            return [int(i) for i in '{0:02b}'.format(num)]
        if length == 4:
            return [int(i) for i in '{0:04b}'.format(num)]
        if length == 5:
            return [int(i) for i in '{0:05b}'.format(num)]
        if length == 13:
            return [int(i) for i in '{0:013b}'.format(num)]
        raise TypeError("Length not supported:", length)

    def encode(self, board) -> np.ndarray:
        from td2020.src.dicts import P_NAME_IDX, A_TYPE_IDX, HEALTH_IDX, CARRY_IDX, MONEY_IDX, REMAIN_IDX

        n = board.shape[0]
        b = np.zeros((n,n, self.NUM_ENCODERS))
        for y in range(n):
            for x in range(n):
                b[x, y][self.P_NAME_IDX_OH:self.P_NAME_IDX_MAX_OH] = self.itb(board[x, y, P_NAME_IDX], self.P_NAME_IDX_INC_OH)
                b[x, y][self.A_TYPE_IDX_OH:self.A_TYPE_IDX_MAX_OH] = self.itb(board[x, y, A_TYPE_IDX], self.A_TYPE_IDX_INC_OH)
                b[x, y][self.HEALTH_IDX_OH:self.HEALTH_IDX_MAX_OH] = self.itb(board[x, y, HEALTH_IDX], self.HEALTH_IDX_INC_OH)
                b[x, y][self.CARRY_IDX_OH:self.CARRY_IDX_MAX_OH] = self.itb(board[x, y, CARRY_IDX], self.CARRY_IDX_INC_OH)
                b[x, y][self.MONEY_IDX_OH:self.MONEY_IDX_MAX_OH] = self.itb(board[x, y, MONEY_IDX], self.MONEY_IDX_INC_OH)
                b[x, y][self.REMAIN_IDX_OH:self.REMAIN_IDX_MAX_OH] = self.itb(board[x, y, REMAIN_IDX], self.REMAIN_IDX_INC_OH)
        return b
