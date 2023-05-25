import numpy as np

"""
Random and Human-ineracting players for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloPlayers by Surag Nair.

"""
class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanTicTacToePlayer():
    def __init__(self, game, n):
        self.game = game
        self.n = n

    def play(self, board):
        boardvalues = np.arange(self.n*self.n*self.n).reshape(self.n,self.n,self.n)
        validvalue = np.arange(self.n*self.n*self.n)
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i] == 1:
                action = validvalue[i]
                print(np.argwhere(boardvalues == action))

        while True: 
            # Python 3.x
            a = input()
            # Python 2.x 
            # a = raw_input()

            z,x,y = [int(x) for x in a.split(' ')]
            boardvalues = np.arange(self.n*self.n*self.n).reshape(self.n,self.n,self.n)
            a = boardvalues[z][x][y]
            if valid[a]:
                break
            else:
                print('Invalid')

        return a
