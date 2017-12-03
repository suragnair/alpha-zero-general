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