from collections import deque
from NNet import NNetWrapper as NNet
from Arena import Arena

class Coach():
    def __init__(self, game, nnet, mcts):
        # maintain self.board, self.curPlayer
        self.game = game
        self.board = game.getInitBoard()
        self.nnet = nnet
        self.curPlayer = 1
        self.numIters = 1000
        self.numEps = 100
        self.mcts = mcts(self.game, self.nnet)
        self.maxlenOfQueue = 5000
        # other hyperparams (numIters, numEps, MCTSParams etc)

    def executeEpisode(self):
        # performs one full game
        # returns a list of training examples from this episode [ < s,a_vec,r > ]
        trainExamples = []
        self.board = game.getInitBoard()

        while True:

            canonicalBoard = self.game.getCanonicalForm(self.board,self.curPlayer)
            pi = self.mcts.getActionProb(canonicalBoard)
            sym = self.game.getSymmetries(canonicalBoard)
            for b in sym:
                trainExamples.append((b, self.curPlayer, pi, 0))
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
            self.
            for eps in xrange(numEps):
                trainExamples.append(executeEpisode())


            self.nnet.save_checkpoint(folder='checkpoint', filename='checkpoint.pth.tar')
            pnet = NNet(self.game)
            pnet.load_checkpoint(folder='checkpoint', filename='checkpoint.pth.tar')
            pmcts = MCTS(self.game, pnet)
            self.nnet.trainNNet(trainExamples)
            nmcts = MCTS(self.game, self.nnet)
            arena = Arena(pmcts.GetBestAction,  nmcts.GetBestAction)
