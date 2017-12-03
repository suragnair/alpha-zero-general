import numpy as np

def display(board):
    n = board.shape[0]
    
    for y in range(n):
    	print (y,"|",)
    print("")
    print(" -----------------------")
    for y in range(n-1,-1,-1):
        print(y, "|",)    # print the row #
        for x in range(n):
            piece = board[x][y]    # get the piece to print
            if piece == -1: print("b "),
            elif piece == 1: print("W "),
            else:
                if x==n:
                    print("-"),
                else:
                    print("- "),
        print("|")

    print("   -----------------------")
    
class Arena():
    def __init__(self, player1, player2, game):
    	# player1 and player2 are functions which take in board, return action
    	self.player1 = player1
    	self.player2 = player2
    	self.game = game

    def playGame(self, verbose=False):
        # execute one game and return winner
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer)==0:
        	it+=1
        	if verbose:
        		print("Turn ", str(it), "Player ", str(curPlayer))
        		display(board)
        	action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))
        	board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
        	print("Turn ", str(it), "Player ", str(curPlayer))
        	display(board)
        return self.game.getGameEnded(board, 1)

    def playGames(self, num):
    	num = int(num/2)
    	oneWon = 0
    	twoWon = 0
    	for _ in range(num):
    		if self.playGame()==1:
    			oneWon+=1
    		else:
    			twoWon+=1
    	self.player1, self.player2 = self.player2, self.player1
    	for _ in range(num):
    		if self.playGame()==-1:
    			oneWon+=1
    		else:
    			twoWon+=1
    	return oneWon, twoWon

class RandomPlayer():
	def __init__(self, game):
		self.game = game

	def play(self, board):
		a = np.random.randint(self.game.getActionSize())
		valids = self.game.getValidMoves(board, 1)
		while valids[a]!=1:
			a = np.random.randint(self.game.getActionSize())
		return a

class HumanOthelloPlayer():
	def __init__(self, game):
		self.game = game

	def play(self, board):
		display(board)
		valid = self.game.getValidMoves(board, 1)
		for i in range(len(valid)):
			if valid[i]:
				print(i%self.game.n, i/self.game.n)
		a = raw_input().strip().split(',')
		x,y = int(a[0]),int(a[1])
		a = self.game.n * y + x if x!= -1 else self.game.n ** 2

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

if __name__ == "__main__":
	from OthelloGame import OthelloGame
	curGame = OthelloGame(6)
	p1 = RandomPlayer(curGame)
	# p2 = GreedyOthelloPlayer(curGame)
	p2 = HumanOthelloPlayer(curGame)

	p1,p2 = p1, p2

	arena = Arena(p1.play, p2.play, curGame)
	l = []
	for _ in range(1):
		score, length =  arena.playGame(verbose=False)
		print(score, length)
		raw_input()
		l += [score]

	print(len([x for x in l if x>0]))
	print(len([x for x in l if x==0]))
	print(len([x for x in l if x<0]))

