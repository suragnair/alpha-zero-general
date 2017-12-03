import numpy as np

def display(board):
    n = board.shape[0]
    print "   -----------------------"
    for y in range(n-1,-1,-1):
        print y+1, "|",    # print the row #
        for x in range(n):
            piece = board[x][y]    # get the piece to print
            if piece == -1: print "b ",
            elif piece == 1: print "W ",
            else:
                if x==n:
                    print "-",
                else:
                    print "- ",
        print "|"

    print "   -----------------------"
    
class Arena():
    def __init__(self, player1, player2, game):
    	# player1 and player2 are functions which take in board, return action
    	self.player1 = player1
    	self.player2 = player2
    	self.game = game

    def playGame(self, verbose=False):
        # execute one game and return winner
        players = [self.player1, None, self.player2]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer)==0:
        	it+=1
        	if verbose:
        		print "Turn ", str(it), "Player ", str(curPlayer)
        		display(board)
        	action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))
        	board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
        	print "Turn ", str(it), "Player ", str(curPlayer)
        	display(board)
        return self.game.getScore(board, 1), it

class RandomPlayer():
	def __init__(self, game):
		self.game = game

	def play(self, board):
		a = np.random.randint(self.game.getActionSize())
		valids = self.game.getValidMoves(board, 1)
		while valids[a]!=1:
			a = np.random.randint(self.game.getActionSize())
		return a

class GreedyOthelloPlayer():
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

from OthelloGame import OthelloGame
curGame = OthelloGame(8)
p1 = RandomPlayer(curGame)
p2 = GreedyOthelloPlayer(curGame)

p1,p2 = p1, p2

arena = Arena(p1.play, p2.play, curGame)
l = []
for _ in range(100):
	score, length =  arena.playGame(verbose=False)
	print score, length
	raw_input()
	l += [score]

print len([x for x in l if x>0])
print len([x for x in l if x==0])
print len([x for x in l if x<0])