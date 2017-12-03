from collections import deque
from NNet import NNetWrapper as NNet
from Arena import Arena

class Coach():
    def __init__(self, game, nnet):
        # maintain self.board, self.curPlayer
        self.game = game
        self.board = game.getInitBoard()
        self.nnet = nnet
        self.numIters = 1000
        self.numEps = 100
        self.tempThreshold = 15
        self.updateThreshold = 0.6
        self.maxlenOfQueue = 5000
        self.arenaCompare = 100
        # other hyperparams (numIters, numEps, MCTSParams etc)

    def executeEpisode(self):
        # performs one full game
        # returns a list of training examples from this episode [ < s,a_vec,r > ]
        trainExamples = []
        self.board = game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(self.board,self.curPlayer)
            temp = int(episodeStep < self.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard)
            for b in sym:
                trainExamples.append((b, self.curPlayer, pi, None))
            action = np.argmax(pi)
            self.board, self.curPlayer = self.game.getNextState(self.board, self.curPlayer, action)

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
            self.mcts = MCTS(self.game, self.nnet)
            for eps in xrange(numEps):
                trainExamples.append(executeEpisode())


            self.nnet.save_checkpoint(folder='checkpoints', filename='checkpoint_' + str(iter) +  '.pth.tar')
            pnet = NNet(self.game)
            pnet.load_checkpoint(folder='checkpoints', filename='checkpoint_' + str(iter) + '.pth.tar')
            pmcts = MCTS(self.game, pnet)
            self.nnet.trainNNet(trainExamples)
            nmcts = MCTS(self.game, self.nnet)
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)))
            pwins, nwins = arena.playGames(self.arenaCompare)

            if float(nwins)/(pwins+nwins) < self.updateThreshold:
                print('NEW MODEL SUCKS')
                self.nnet = pnet

            else:
                print('NEW MODEL AWESOME')
