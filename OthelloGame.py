import Game
from OthelloLogic import Board
import numpy as np

class OthelloGame():
    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n*self.n:
        	return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action/self.n), action%self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player)
        if len(legalMoves)==0:
        	valids[-1]=1
        	return np.array(valids)
        for x, y in legalMoves:
        	valids[self.n*x+y]=1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves1 =  b.get_legal_moves(player)
        legalMoves2 =  b.get_legal_moves(-player)
        if len(legalMoves1)>0 or len(legalMoves2)>0:
        	return 0
        if b.count(player) > b.count(-player):
        	return 1
        return -1

    def getCanonicalForm(self, board, player):
        # return state if player==0, else return -state ?
        return player*board

    def getSymmetries(self, board):
        # mirror, rotational
        l = []
        for i in range(1, 5):
            for j in [True, False]:
                for k in [True, False]:
                    newB = np.rot90(board, i)
                    if j:
                        newB = np.fliplr(newB)
                    if k:
                        newB = np.flipud(newB)
                    l += [newB]
        return l

    def stringRepresentation(self, board):#, player):
    	# 64 + 1 digits; 0 for player2, 1 for empty, 2 for player1
    	# followed by 2 or 1 for whose turn it is
    	l = [x+1 for x in np.ravel(board)]# + [player+1]
    	return ''.join([str(x) for x in l])

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.count(player)-b.count(-player)
