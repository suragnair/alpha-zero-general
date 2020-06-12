'''
Author: Taylor P. Santos
Date: May 28, 2020.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     first 9 columns are board state, 10th column is sub-grid wins, 11th column is playable grids
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.

Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''

class Board():

    # list of all 8 winning combinations
    __wins = [
        ((0, 0), (1, 0), (2, 0)),
        ((0, 1), (1, 1), (2, 1)),
        ((0, 2), (1, 2), (2, 2)),
        ((0, 0), (0, 1), (0, 2)),
        ((1, 0), (1, 1), (1, 2)),
        ((2, 0), (2, 1), (2, 2)),
        ((0, 0), (1, 1), (2, 2)),
        ((2, 0), (1, 1), (0, 2))
    ]

    def __init__(self):
        "Set up initial board configuration."
        # Create the empty board array.
        self.pieces = [[0]*9 for _ in range(11)]



    # add [][] indexer syntax to the Board
    def __getitem__(self, i):
        (x, y) = i
        return self.pieces[y][x]

    def __setitem__(self, i, val):
        (x, y) = i
        self.pieces[y][x] = val

    def wins(self):
        return self.pieces[9]

    def nextMoves(self):
        return self.pieces[10]

    def get_legal_moves(self):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        moves = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for (X, Y) in [(i, j) for i in range(3) for j in range(3)]:
            if self.nextMoves()[3*Y + X] == 0:
                for (x, y) in [(i, j) for i in range(3) for j in range(3)]:
                    if self[3*X + x, 3*Y + y] == 0:
                        moves.add((3*X + x, 3*Y + y))
        return list(moves)

    def has_legal_moves(self):
        for (X, Y) in [(i, j) for i in range(3) for j in range(3)]:
            if self.nextMoves()[3 * Y + X] == 0:
                for (x, y) in [(i, j) for i in range(3) for j in range(3)]:
                    if self[3 * X + x, 3 * Y + y] == 0:
                        return True
        return False

    def is_win(self):
        # global wins
        # i = 0
        # for j in range(9):
        #     i *= 3
        #     i += {0: 0, -1: 1, 1: 2, 2:0}[self.wins()[j]]
        # w = wins[i]
        # if w != 0:
        #     return w
        # for i in range(9):
        #     if self.wins()[i] == 0:
        #         return 0
        # return 2

        for (ax, ay), (bx, by), (cx, cy) in self.__wins:
            for player in (-1, 1):
                if self.wins()[3*ay + ax] == player and self.wins()[3*by + bx] == player and self.wins()[3*cy + cx] == player:
                    return player
        for i in range(9):
            if self.wins()[i] == 0:
                return 0
        return 2  # Tie

    def execute_move(self, move, color):
        """Perform the given move on the board
        """
        (x, y) = move
        assert self[x, y] == 0
        self[x, y] = color
        s_x = int(x / 3)
        s_y = int(y / 3)

        for (ax, ay), (bx, by), (cx, cy) in self.__wins:
            if self[3*s_x + ax, 3*s_y + ay] == color and \
                    self[3*s_x + bx, 3*s_y + by] == color and \
                    self[3*s_x + cx, 3*s_y + cy] == color:
                self.wins()[3*s_y + s_x] = color
                break
        if self.wins()[3*s_y + s_x] == 0:
            tied = True
            for (c_x, c_y) in [(i, j) for i in range(3) for j in range(3)]:
                if self[3*s_x + c_x, 3*s_y + c_y] == 0:
                    tied = False
                    break
            if tied:
                self.wins()[3*s_y + s_x] = 2
        # i = 0
        # for y in range(3):
        #     for x in range(3):
        #         i *= 3
        #         i += {0: 0, -1: 1, 1: 2}[self[3*s_x + x, 3*s_y + y]]
        # global wins
        # self.wins()[3*s_y + s_x] = wins[i]
        c_x = x % 3
        c_y = y % 3
        for i in range(9):
            if self.wins()[3 * c_y + c_x] == 0:
                if 3*c_y + c_x == i:
                    self.nextMoves()[i] = 0
                else:
                    self.nextMoves()[i] = 1
            else:
                if self.wins()[i] == 0:
                    self.nextMoves()[i] = 0
                else:
                    self.nextMoves()[i] = 1
