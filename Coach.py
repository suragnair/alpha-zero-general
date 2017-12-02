class Coach():
    def __init__(self, game, nnet):
        # maintain self.board, self.curPlayer
        # other hyperparams (numIters, numEps, MCTSParams etc)
        pass

    def executeEpisode(self):
        # performs one full game
        # returns a list of training examples from this episode [ < s,a_vec,r > ]
        pass    

    def learn(self):
        # performs numIters x numEps games
        # after every Iter, retrains nnet and only updates if it wins > cutoff% games
        pass
