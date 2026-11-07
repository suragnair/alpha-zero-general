"""
Board class for tic-tac-toe in four dimensions.

The board is n×n×n×n: squares are addressed as pieces[w][z][y][x], 1 for white, -1 for black, 0 for empty.
A line is n squares in a row along any of the 40 directions of the hypercube, so it can lie inside
one cube or run through all of them. There are ((n + 2)^4 - n^4) / 2 lines: 520 on the 4×4×4×4 board.
"""
import functools
import itertools

import numpy as np


@functools.lru_cache(maxsize=None)
def winning_lines(n):
    """All winning lines of the n×n×n×n board as an array of flat square indices, one line per row."""
    lines = set()
    for start in itertools.product(range(n), repeat=4):
        for step in itertools.product((-1, 0, 1), repeat=4):
            if not any(step):
                continue
            end = [s + d * (n - 1) for s, d in zip(start, step)]
            if all(0 <= c < n for c in end):
                squares = [[s + d * t for s, d in zip(start, step)] for t in range(n)]
                lines.add(tuple(sorted(int(np.ravel_multi_index(square, (n,) * 4)) for square in squares)))
    return np.array(sorted(lines))


def forced_moves(pieces):
    """Moves that need no judgement for the player on turn, who is 1 in pieces: the squares that
    complete his own line, else the squares that block a line of the opponent.
    Returns the squares and whose lines they finish: 1, -1, or 0 with no squares at all."""
    n = pieces.shape[0]
    lines = winning_lines(n)
    flat = pieces.ravel()
    values = flat[lines]
    one_left = (values == 0).sum(axis=1) == 1
    for color in (1, -1):
        almost = lines[one_left & ((values == color).sum(axis=1) == n - 1)]
        if len(almost):
            return np.unique(almost[flat[almost] == 0]), color
    return np.array([], dtype=int), 0


class Board():

    def __init__(self, n=4):
        """Set up an empty board."""
        self.n = n
        self.pieces = np.zeros((n,) * 4, dtype=int)

    def get_legal_moves(self):
        """Returns all the empty squares as flat indices."""
        return np.flatnonzero(self.pieces == 0)

    def has_legal_moves(self):
        return bool((self.pieces == 0).any())

    def is_win(self, color):
        """Check whether the given player has collected a whole line in any direction."""
        return bool((self.pieces.ravel()[winning_lines(self.n)] == color).all(axis=1).any())

    def execute_move(self, action, color):
        """Perform the given move on the board, action being a flat square index."""
        square = np.unravel_index(action, self.pieces.shape)
        assert self.pieces[square] == 0
        self.pieces[square] = color
