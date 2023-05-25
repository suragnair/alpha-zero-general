import numpy as np


class Board():

    def __init__(self, n=5):
        "Set up initial board configuration."
        self.n = n
        self.pieces = np.zeros((2*n+1, n+1))

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def increase_score(self, score, player):
        if player == 1:
            self.pieces[0, -1] += score
        else:
            self.pieces[1, -1] += score

    def is_pass_on(self):
        return self.pieces[2, -1]

    def toggle_pass(self, state=False):
        self.pieces[2, -1] = state

    def get_legal_moves(self, color=1):
        """Returns all the legal moves
        @param color not used and came from previous version.
        """
        legal_moves = np.logical_not(self.pieces)
        legal_moves = np.hstack((legal_moves[:self.n+1, :-1].flatten(), legal_moves[-self.n:, :].flatten(), False))
        if self.is_pass_on():
            legal_moves[:] = False
            legal_moves[-1] = True
        return legal_moves

    def has_legal_moves(self):
        is_board_full = np.all(self.pieces[:self.n+1, :-1]) and np.all(self.pieces[-self.n:, :])
        return not is_board_full

    def execute_move(self, action, color=1):
        """Perform the given move on the board; 
        color gives the color pf the piece to play (1=white,-1=black)
        """
        assert self.is_pass_on() == 0

        is_horizontal = action < self.n*(self.n+1)
        if is_horizontal:
            move = (int(action / self.n), action % self.n)
        else:
            action -= self.n*(self.n+1)
            move = (int(action / (self.n+1)) + self.n + 1, action % (self.n+1))

        (x, y) = move

        # Add the piece to the empty square.
        assert self[x][y] == 0
        self[x][y] = 1  # The color doesn't matter

        # Need to check if we have closed a square
        # If so, increase score and mark pass
        horizontal = np.zeros((self.n+3, self.n+2))
        horizontal[1:-1, 1:-1] = self.pieces[:self.n+1, :self.n]

        vertical = np.zeros((self.n+2, self.n+3))
        vertical[1:-1, 1:-1] = self.pieces[-self.n:, :]

        score = 0
        if is_horizontal:
            x += 1
            y += 1
            if horizontal[x+1][y]:
                score += (vertical[x][y] and vertical[x][y+1])
            if horizontal[x-1][y]:
                score += (vertical[x-1][y] and vertical[x-1][y+1])
        else:
            x = x - self.n
            y += 1
            if vertical[x, y+1]:
                score += (horizontal[x][y] and horizontal[x+1][y])
            if vertical[x, y-1]:
                score += (horizontal[x][y-1] and horizontal[x+1][y-1])

        self.increase_score(score, color)
        self.toggle_pass(score > 0)
