import numpy as np
import random


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def find_indices(self, list_to_check, item_to_find):
        indices = []
        for idx, value in enumerate(list_to_check):
            if value == item_to_find:
                indices.append(idx)
        return indices

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        indices = self.find_indices(valids, 1)
        a = random.choice(indices)
        return a
