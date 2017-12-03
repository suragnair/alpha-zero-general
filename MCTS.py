Class MCTS():
	def __init__(self, game, nnet):
		self.game = game
		self.nnet = nnet

	def GetActionProb(self, canonicalBoard):
		# return pi 
		pass

	def GetBestAction(self, canonicalBoard):
		return np.argmax(self.GetActionProb(canonicalBoard))


