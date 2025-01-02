'''
Author: Bo Song
Date: Nov, 2024.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the stone in column 1 row 7,
Stones are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
(0, 0) is the top-left corner.
'''
import logging
import numpy as np

log = logging.getLogger(__name__)

class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    # order matters. The first 4 directions must not be on a same line.
    __directions = [(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,-1),(-1,0),(-1,1)]

    def __init__(self, n):
        """Set up initial board configuration.
          Arg: n - board side length 
        """

        self.n = n
        # Create the empty board array.
        self.pieces = [None] * self.n
        for i in range(self.n):
            self.pieces[i] = [0] * self.n

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def is_initial_board(self):
        """Returns True if the current board is the initial board (no stones on the board)
        Otherwise returns False.
        """
        return np.count_nonzero(self.pieces) == 0

    def is_board_full(self):
        """Returns True if the board is full of stones
        """
        return np.count_nonzero(self.pieces) == self.n * self.n

    def has_five_in_a_row(self, color):
        """Returns if there is a 5-in-a-row pattern for the given color.
        This indicates the given color wins the game.
        """
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]!=color:
                    continue
                if len(self._get_n_in_a_row(x, y, 5)) > 0:
                    return True
        return False

    def _get_n_in_a_row(self, x, y, n):
        """Returns a list of directions if the board has n same color stones starting from (x,y) in a row
        Starting means (x,y) is the leftmost, bottommost,left-top or left-bottom most stone in the row.
        """
        if self[x][y] == 0:
            return []
        directions = []
        color = self[x][y]
        # we only need to count 4 directions. Thanks to the symmetrics. 
        for i in range(0, 4):
            direction = Board.__directions[i]
            count = 0
            nx, ny = x, y
            while self.is_within_board(nx, ny) and self[nx][ny] == color:
                count += 1                        
                nx += direction[0]
                ny += direction[1]
            if count == n:
                directions.append(direction)
        return directions

    def is_within_board(self, x, y):
        return x >= 0 and x < self.n and y >= 0 and y < self.n

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        moves = set()  # stores the legal moves.
        if self.is_initial_board():
            # always place the stone in the middle in the begining.
            moves.add((self.n // 2, self.n // 2))
            return moves

        # We add following two heuristics to reduce search space.
        # If there is 4 in a row with open end for the current color, the only leagal moves are to extend it to 5 to win the game.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] != color:
                    continue
                directions = self._get_n_in_a_row(x, y, 4)
                for d in directions:
                    nx = x - d[0]
                    ny = x - d[1]
                    if self.is_within_board(nx, ny) and self[nx][ny] == 0:
                        moves.add((nx, ny))
                    nx = x + 4 * d[0]
                    ny = y + 4 * d[1]
                    if self.is_within_board(nx, ny) and self[nx][ny] == 0:
                        moves.add((nx, ny))
                    if len(moves) > 0:
                        return moves        
        # If there is a 4 in a row with open end for the opposite color, the only legal moves are to block it.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] != -color:
                    continue
                directions = self._get_n_in_a_row(x, y, 4)
                for d in directions:
                    nx = x - d[0]
                    ny = x - d[1]
                    if self.is_within_board(nx, ny) and self[nx][ny] == 0:
                        moves.add((nx, ny))
                    nx = x + 4 * d[0]
                    ny = y + 4 * d[1]
                    if self.is_within_board(nx, ny) and self[nx][ny] == 0:
                        moves.add((nx, ny))
                    if len(moves) > 0:
                        return moves
        
        # Otherwise, the legal moves are all empty positions (x,y) on the board that satisfies following conditions:
        # 1. there is at least one stone on the board (a, b) (either white or black) that satisfies abs(a - x) + abs (b-y) <= 2
        # This is to reduce the search space.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]== 0:
                    continue
                newmoves = self._get_all_available_moves_next_to(x, y, 2)
                moves.update(newmoves)
        return list(moves)

    def _get_all_available_moves_next_to(self, x, y, distance):
        """Get all avaiable moves next to (x, y), within distance.
        Return: list of tuples [(x,y), ...]
        """
        available_moves = []
        for nx in range(x - distance, x + distance + 1):
            for ny in range(y - distance, y + distance + 1):
                if self.is_within_board(nx, ny) and self[nx][ny] == 0:
                    available_moves.append((nx, ny))
        return available_moves



    def execute_move(self, move, color):
        """Perform the given move on the board: put the stone on the board
        """
        assert self[move[0]][move[1]] == 0, "invalid move. The position is occupied. Bug in the code?"
        self[move[0]][move[1]] = color


