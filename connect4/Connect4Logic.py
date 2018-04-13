from collections import namedtuple
import numpy as np

DEFAULT_HEIGHT = 6
DEFAULT_WIDTH = 7
DEFAULT_WIN_LENGTH = 4

WinState = namedtuple('WinState', 'is_ended winner')


class Board():
    """
    Connect4 Board.
    """

    def __init__(self, height=None, width=None, win_length=None, np_pieces=None):
        "Set up initial board configuration."
        self.height = height or DEFAULT_HEIGHT
        self.width = width or DEFAULT_WIDTH
        self.win_length = win_length or DEFAULT_WIN_LENGTH

        if np_pieces is None:
            self.np_pieces = np.zeros([self.height, self.width])
        else:
            self.np_pieces = np_pieces
            assert self.np_pieces.shape == (self.height, self.width)

    def add_stone(self, column, player):
        "Create copy of board containing new stone."
        available_idx, = np.where(self.np_pieces[:, column] == 0)
        if len(available_idx) == 0:
            raise ValueError("Can't play column %s on board %s" % (column, self))

        self.np_pieces[available_idx[-1]][column] = player

    def get_valid_moves(self):
        "Any zero value in top row in a valid move"
        return self.np_pieces[0] == 0

    def get_win_state(self):
        for player in [-1, 1]:
            player_pieces = self.np_pieces == -player
            # Check rows & columns for win
            if (self._is_straight_winner(player_pieces) or
                self._is_straight_winner(player_pieces.transpose()) or
                self._is_diagonal_winner(player_pieces)):
                return WinState(True, -player)

        # draw has very little value.
        if not self.get_valid_moves().any():
            return WinState(True, None)

        # Game is not ended yet.
        return WinState(False, None)

    def with_np_pieces(self, np_pieces):
        """Create copy of board with specified pieces."""
        if np_pieces is None:
            np_pieces = self.np_pieces
        return Board(self.height, self.width, self.win_length, np_pieces)

    def _is_diagonal_winner(self, player_pieces):
        """Checks if player_pieces contains a diagonal win."""
        win_length = self.win_length
        for i in range(len(player_pieces) - win_length + 1):
            for j in range(len(player_pieces[0]) - win_length + 1):
                if all(player_pieces[i + x][j + x] for x in range(win_length)):
                    return True
            for j in range(win_length - 1, len(player_pieces[0])):
                if all(player_pieces[i + x][j - x] for x in range(win_length)):
                    return True
        return False

    def _is_straight_winner(self, player_pieces):
        """Checks if player_pieces contains a vertical or horizontal win."""
        run_lengths = [player_pieces[:, i:i + self.win_length].sum(axis=1)
                       for i in range(len(player_pieces) - self.win_length + 2)]
        return max([x.max() for x in run_lengths]) >= self.win_length

    def __str__(self):
        return str(self.np_pieces)
