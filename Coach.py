from collections import deque

class Coach():
    def __init__(self, game, nnet, mcts):
        # maintain self.board, self.curPlayer
        self.game = game
        self.board = game.getInitBoard()
        self.nnet = nnet
        self.curPlayer = 1
        self.numIters = 1000
        self.numEps = 100
        self.mcts = mcts
        self.maxlenOfQueue = 5000
        # other hyperparams (numIters, numEps, MCTSParams etc)

    def executeEpisode(self):
        # performs one full game
        # returns a list of training examples from this episode [ < s,a_vec,r > ]
        trainExamples = []
        self.board = game.getInitBoard()

        while True:
            pi = self.mcts.getActionProb(self.board, self.player)
            canonicalBoard = self.game.getCanonicalForm(self.board,self.curPlayer)
            trainExamples.append((self.canonicalBoard, self.curPlayer, actionProb, 0))

            action = np.argmax(pi)
            (self.board, self.curPlayer) = self.game.getNextState(self.board, self.curPlayer, action)
            
            r = getGameEnded(self.board, self.curPlayer)
            if r!=0:
                for i in xrange(len(trainExamples)):
                    e = trainExamples[i]
                    e[3] = r if self.curPlayer == e[1] else -r
                    trainExamples[i] = (e[0],e[2],e[3])
                break
        return train_examples    


    def learn(self):
        # performs numIters x numEps games
        # after every Iter, retrains nnet and only updates if it wins > cutoff% games
        trainExamples = deque([], maxlen=self.maxlenOfQueue)
        for Iter in xrange(numIters):
            for eps in xrange(numEps):
                trainExamples.append(executeEpisode())
            self.nnet.trainNNet(trainExamples)
        

