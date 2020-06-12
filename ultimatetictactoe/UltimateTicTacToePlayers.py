import numpy as np


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanUltimateTicTacToePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, proc, verbose=False):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        # for i in range(len(valid)):
        #     if valid[i]:
        #         print("[", int(i/9), int(i%9), end="] ")
        while True:
            input_move = proc.stdout.readline().decode()
            input_a = input_move.split(" ")
            if len(input_a) == 2:
                try:
                    x,y = [int(i) for i in input_a]
                    if ((0 <= x) and (x < 9) and (0 <= y) and (y < 9)) or \
                            ((x == 9) and (y == 0)):
                        a = 9 * x + y if x != -1 else 9 ** 2
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')
        return a
