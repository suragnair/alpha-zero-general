import numpy as np


class Board:

    def __init__(self, n=3):
        self.n = n
        self.N = n ** 2
        self.last_move = None
        self.pieces = np.zeros((self.N,self.N)).astype(int)
        self.win_status = np.zeros((n,n)).astype(int)


    def copy(self, other):
        self.n = other.n
        self.N = other.N
        self.last_move = other.last_move
        self.pieces = other.pieces
        self.win_status = other.win_status

    def __getitem__(self, index):
        return self.pieces[index]

    def get_legal_moves(self):

        moves = set()
        legal_coord = self.get_legal_area()

        if legal_coord and not self.is_locked(legal_coord[0], legal_coord[1]):
            for x in range(legal_coord[0] * self.n, (legal_coord[0] + 1) * self.n):
                for y in range(legal_coord[1] * self.n, (legal_coord[1] + 1) * self.n):
                    if self[x][y] == 0:
                        legal_move = (x, y)
                        moves.add(legal_move)
        else:
            for x in range(self.N):
                for y in range(self.N):
                    x_area, y_area = self.get_area(x, y)
                    if legal_coord:
                        if x_area != legal_coord[0] or y_area != legal_coord[1]:
                            legal_move = (x, y)
                            moves.add(legal_move)
                    else:
                        legal_move = (x, y)
                        moves.add(legal_move)
                        


        return list(moves)

    def get_area(self, x, y):
        area_x = x // self.n
        area_y = y // self.n

        return area_x, area_y

    def get_legal_area(self):
        return self.get_area(self.last_move[0], self.last_move[1]) if self.last_move else None

    def is_locked(self, x, y):
        return self.win_status[x][y] != 0

    def has_legal_moves(self):
        for y in range(self.n):
            for x in range(self.n):
                if self.win_status[x][y] == 0:
                    return True
        return False

    def is_win(self, player):
        win = self.n

        # check y-strips
        for y in range(self.n):
            count = 0
            for x in range(self.n):
                if self.win_status[x][y] == player:
                    count += 1
            if count == win:
                return True

        # check x-strips
        for x in range(self.n):
            count = 0
            for y in range(self.n):
                if self.win_status[x][y] == player:
                    count += 1
            if count == win:
                return True

        # check two diagonal strips
        count = 0
        for d in range(self.n):
            if self.win_status[d][d] == player:
                count += 1
        if count == win:
            return True

        count = 0
        for d in range(self.n):
            if self.win_status[d][self.n - d - 1] == player:
                count += 1
        if count == win:
            return True

        return False

    def is_local_win(self, area, player):
        win = self.n

        # check y-strips
        for y in range(area[1] * self.n, (area[1] + 1) * self.n):
            count = 0
            for x in range(area[0] * self.n, (area[0] + 1) * self.n):
                if self[x][y] == player:
                    count += 1
                if count == win:
                    return True

        # check x-strips
        for x in range(area[0] * self.n, (area[0] + 1) * self.n):
            count = 0
            for y in range(area[1] * self.n, (area[1] + 1) * self.n):
                if self[x][y] == player:
                    count += 1
                if count == win:
                    return True

        # check two diagonal strips
        count = 0
        for x, y in \
                zip(range(area[0] * self.n, (area[0] + 1) * self.n), range(area[1] * self.n, (area[1] + 1) * self.n)):
            if self[x][y] == player:
                count += 1
        if count == win:
            return True

        count = 0
        for x, y in \
                zip(range(area[0] * self.n, (area[0] + 1) * self.n), range(area[1] * self.n, (area[1] + 1) * self.n)):
            if self[x][self.n - y - 1] == player:
                count += 1
        if count == win:
            return True

        return False

    def execute_move(self, move, player):

        (x, y) = move

        assert self[x][y] == 0
        self[x][y] = player
        self.last_move = move

        area_x, area_y = self.get_area(x,y)
        if self.is_local_win((area_x, area_y), player):
            self.win_status[area_x][area_y] = player

    def get_canonical_form(self, player):
        self.pieces = player * self.pieces
        self.win_status = player * self.win_status

    def rot90(self, i, copy=False):
        if copy:
            board = Board(self.n)
            board.copy(self)

            board.pieces = np.rot90(board.pieces, i)
            board.win_status = np.rot90(board.win_status, i)

            return board
        else:
            self.pieces = np.rot90(self.pieces, i)
            self.win_status = np.rot90(self.win_status, i)

            return True

    def fliplr(self, copy=False):
        if copy:
            board = Board(self.n)
            board.copy(self)

            board.pieces = np.fliplr(board.pieces)
            board.win_status = np.fliplr(board.win_status)

            return board
        else:
            self.pieces = np.fliplr(self.pieces)
            self.win_status = np.fliplr(self.win_status)

            return True

    def tostring(self):
        return np.array(self.pieces).tostring()

