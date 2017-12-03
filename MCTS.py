import math
import numpy as np
import NNet

Class MCTS():
	def __init__(self, game, nnet, cpuct):
		self.game = game
		self.nnet = nnet
		self.cpuct = cpuct
		self.Qsa = {}
		self.Nsa = {}
		self.Ns = {}
		self.Ps = {}

	def GetActionCounts(self, canonicalBoard):
		# return pi
		for i in range(len(self.numSimulations)):
			self.search(canonicalBoard)

		s = self.game.stringRepresentation(canonicalBoard)
		return [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(len(actions))]


	def search(self, canonicalBoard):
		# v' for the newest leaf node

		s = self.game.stringRepresentation(canonicalBoard)
		valids = self.game.getValidMoves(canonicalBoard, 1)

		if s not in Ps:
			self.Ps[s], v = self.nnet.predict(canonicalBoard)
			self.Ps[s] = self.Ps[s]*valids
			self.Ps[s] /= np.sum(self.Ps[s])	# renormalize

			self.Ns[s] = 0
			return -v

		cur_best = -float('inf')
		best_act = -1
		for a in range(len(game.getActionSize)):
			if valids[a]:
				if (s,a) in self.Qsa:
					u = self.Qsa[(s,a)] + self.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
				else:
					u = self.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])		# Q = 0 ?

				if u > cur_best:
					cur_best = u
					best_act = a

		next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
		next_s = self.game.getCanonicalForm(next_s, next_player)

		v = self.search(next_s)
		a = best_act
		
		if (s,a) in self.Qsa:
			self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
			self.Nsa[(s,a)] += 1

		else:
			self.Qsa[(s,a)] = v
			self.Nsa[(s,a)] = 1

		self.Ns[s] += 1
		return -v

	def GetBestAction(self, canonicalBoard, temp=0):
		counts = self.GetActionCounts(canonicalBoard)

		if temp==0:
			return np.argmax(counts)

		counts = [x**(1./temp) for x in counts]
		counts = [x/sum(counts) for x in counts]
		return np.random.choice(list(range(self.game.getActionSize)), p=counts)
