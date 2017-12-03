from collections import deque
from NNet import NNetWrapper as NNet
from Arena import Arena
from MCTS import MCTS
import numpy as np

class Coach():
    def __init__(self, game, nnet, args):
        # maintain self.board, self.curPlayer
        self.game = game
        self.board = game.getInitBoard()
        self.nnet = nnet
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        # other hyperparams (numIters, numEps, MCTSParams etc)

    def executeEpisode(self):
        # performs one full game
        # returns a list of training examples from this episode [ < s,a_vec,r > ]
        trainExamples = []
        self.board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(self.board,self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard)
            for b in sym:
                trainExamples.append([b, self.curPlayer, pi, None])

            action = np.argmax(pi)
            self.board, self.curPlayer = self.game.getNextState(self.board, self.curPlayer, action)

            r = self.game.getGameEnded(self.board, self.curPlayer)
            if r!=0:
                for i in range(len(trainExamples)):
                    e = trainExamples[i]
                    e[3] = r if self.curPlayer == e[1] else -r
                    trainExamples[i] = (e[0],e[2],e[3])
                break

        return trainExamples

    def learn(self):
        # performs numIters x numEps games
        # after every Iter, retrains nnet and only updates if it wins > cutoff% games
        trainExamples = deque([], maxlen=self.args.maxlenOfQueue)
        for i in range(self.args.numIters): 
            for eps in range(self.args.numEps):
                print('EPISODE # ' + str(eps+1))
                trainExamples += self.executeEpisode()

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='checkpoint_' + str(i+1) +  '.pth.tar')
            pnet = NNet(self.game)
            pnet.load_checkpoint(folder=self.args.checkpoint, filename='checkpoint_' + str(i+1) + '.pth.tar')
            pmcts = MCTS(self.game, pnet, self.args)
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins = arena.playGames(self.args.arenaCompare)

            print('ENDING ITER ' + str(i+1))
            print('NEW/PREV WINS : ' + str(nwins) + '/' + str(pwins))
            if float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('NEW MODEL SUCKS')
                self.nnet = pnet

            else:
                print('NEW MODEL AWESOME')
                self.mcts = MCTS(self.game, self.nnet, self.args)
