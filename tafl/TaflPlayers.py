import numpy as np
from .Digits import int2base

class RandomTaflPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanTaflPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        m=[]
        for i in range(len(valid)):
            if valid[i]:
                m.extend([int2base(i,self.game.n,4)])
        print(m)    
        while True:
            a = input()

            x1,y1,x2,y2 = [int(x) for x in a.strip().split(' ')]
            a = x1 + y1 * self.game.n + x2 * self.game.n**2 + y2 * self.game.n**3 
            if valid[a]:
                break
            else:
                print('Invalid')

        return a


class GreedyTaflPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]


    
