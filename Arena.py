class Arena():
    def __init__(self, player1, player2, game):
    	# player1 and player2 are functions which take in board, return action
    	self.player1 = player1
    	self.player2 = player2
    	self.game = game

    def playGame(self):
        # execute one game and return winner
        players = [self.player1, None, self.player2]
        curPlayer = 1
        board = game.getInitBoard()
        while self.game.getGameEnded()!=0:
        	action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))
        	board, curPlayer = self.game.getNextState(board, curPlayer, action)
        return self.game.getGameEnded(board, curPlayer)

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