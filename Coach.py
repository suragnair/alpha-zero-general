from collections import deque
from othello.NNet import NNetWrapper as NNet
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, Logger, AverageMeter
import time

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
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b,p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.argmax(pi)
            self.board, self.curPlayer = self.game.getNextState(self.board, self.curPlayer, action)

            r = self.game.getGameEnded(self.board, self.curPlayer)

            if r!=0:
                return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples]

    def learn(self):
        # performs numIters x numEps games
        # after every Iter, retrains nnet and only updates if it wins > cutoff% games
        trainExamples = deque([], maxlen=self.args.maxlenOfQueue)
        for i in range(self.args.numIters):
            # bookkeeping
            print('------ITER ' + str(i+1) + '------')
            eps_time = AverageMeter()
            bar = Bar('Self Play', max=self.args.numEps)
            end = time.time()

            for eps in range(self.args.numEps):
                trainExamples += self.executeEpisode()

                eps_time.update(time.time() - end)
                end = time.time()
                # plot progress
                bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                           total=bar.elapsed_td, eta=bar.eta_td)
                bar.next()
            bar.finish()


            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pnet = NNet(self.game)
            pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, pnet, self.args)
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)

            print('PITTING AGAINST PREVIOUS VERSION')
            pwins, nwins = arena.playGames(self.args.arenaCompare)

            print('ENDING ITER ' + str(i+1))
            print('NEW/PREV WINS : ' + str(nwins) + '/' + str(pwins))
            if float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('NEW MODEL SUCKS')
                self.nnet = pnet

            else:
                print('NEW MODEL AWESOME')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='checkpoint_' + str(i) + '.pth.tar')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
