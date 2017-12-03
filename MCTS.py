import math
import numpy as np
import NNet

Class MCTS():
	def __init__(self, game, nnet, args):
		self.game = game
		self.nnet = nnet
		self.cpuct = args.cpuct
		self.Qsa = {}
		self.Nsa = {}
		self.Ns = {}
		self.Ps = {}

	def getActionProb(self, canonicalBoard, temp=1):
		# return pi
		for i in range(len(args.numMCTSSims)):
			self.search(canonicalBoard)

		s = self.game.stringRepresentation(canonicalBoard)
		counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(len(actions))]

		if temp==0:
			bestA = np.argmax(counts)
			probs = [0]*len(counts)
			probs[bestA]=1probs
			return probs

		counts = [x**(1./temp) for x in counts]
		probs = [x/sum(counts) for x in counts]
		return probs


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
		for a in range(len(game.getActionSize())):
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
