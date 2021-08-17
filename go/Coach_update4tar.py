import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS

log = logging.getLogger(__name__)


class Coach_update4tar():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.firstIter = args.firstIter #set true if it produce first chechpoint to save
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.n2net = self.nnet.__class__(self.game)  # the competitor network
        self.n3net = self.nnet.__class__(self.game)  # the competitor network
        self.n4net = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.

        Go edit:
        load best.pth.tar to compare and self-iteration
        temp.pth.tar only for training. it may be overwrite shortly
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            self.trainExamplesHistory = [] # empty the history 
            # examples of the iteration
            self.loadExamples() #Load the best.pth.tar.examples
            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='250games.tar') 
#            self.n2net.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar') 
            self.n3net.load_checkpoint(folder=self.args.checkpoint, filename='250games.tar') 

            shuffle(trainExamples)              
            self.nnet.train(trainExamples)
            shuffle(trainExamples)              
            self.n3net.train(trainExamples)
            self.n4net.load_checkpoint(folder=self.args.checkpoint, filename='250games.tar') #Load n3 result, see if it perform better
            shuffle(trainExamples)              
            self.n4net.train(trainExamples)

            nmcts = MCTS(self.game, self.nnet, self.args)
        #    n2mcts = MCTS(self.game, self.n2net, self.args)
            n3mcts = MCTS(self.game, self.n3net, self.args)
            n4mcts = MCTS(self.game, self.n4net, self.args)

            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar') #Load the best after training to maximize efficenty
            pmcts = MCTS(self.game, self.pnet, self.args)
            log.info('Arena Start! A tournament between PREVIOUS VERSION and 4 new ones!')

            player_prev = lambda x: np.argmax(pmcts.getActionProb(x, arena=1, temp=0,instinctPlay=self.args.instinctArena, levelBased=self.args.levelBased)[0])
            player_n = lambda x: np.argmax(nmcts.getActionProb(x, arena=1, temp=0,instinctPlay=self.args.instinctArena, levelBased=self.args.levelBased)[0])
         #   player_n2 = lambda x: np.argmax(n2mcts.getActionProb(x, temp=0)[0])
            player_n3 = lambda x: np.argmax(n3mcts.getActionProb(x, arena=1, temp=0,instinctPlay=self.args.instinctArena, levelBased=self.args.levelBased)[0])
            player_n4 = lambda x: np.argmax(n4mcts.getActionProb(x, arena=1, temp=0,instinctPlay=self.args.instinctArena, levelBased=self.args.levelBased)[0])

            playerList = [player_prev, player_n, player_n3, player_n4]
            mctsList = [pmcts, nmcts, n3mcts, n4mcts]
            nnList = [self.pnet, self.nnet, self.n3net, self.n4net]
            result = self.tournament(playerList)
            winner = nnList[result.index(max(result))]
            winner.save_checkpoint(folder=self.args.checkpoint, filename='250games.tar')
            log.info('UPDATING BEST')
            log.info('ARENA RESULT: ', result)
            winner.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))

                
    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    #load examples only
    def loadExamples(self):
        folder = self.args.checkpoint
        filename = os.path.join(folder, "best.pth.tar.examples")
        log.info("File with trainExamples found. Loading it...")
        with open(filename, "rb") as f:
            count = 0
            while True:
                try:
                    if count == 0:
                        self.trainExamplesHistory = Unpickler(f).load()
                        count += 1
                    else:
                        self.trainExamplesHistory += Unpickler(f).load()
                except EOFError:
                    break 
        while len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
            self.trainExamplesHistory.pop(0)
        log.info('Loading done!')

    def playGame(self, player1, str1, player2, str2):
        arena = Arena(player1, player2, self.game)
        x, y, z, xb = arena.playGames(self.args.arenaCompare, verbose=False)
        print(str1, " win: ", x)
        print(str2, " win: ", y)
        print(str1, " win black: ", xb)
        return x, y

    def tournament(self, playList):
        tournamentResult = dict.fromkeys(playList, 0)
        repeat = [] #each player only play 2 games with each other
        for a in playList:
            for b in playList:
                #this 
                if (playList.index(a)+1)*(playList.index(b)+1) not in repeat:
                    if a is not b:
                        aWin, bWin = self.playGame(a, 'p1', b, 'p2')           
                        tournamentResult[a] += (aWin - bWin + self.args.arenaCompare)/2
                        tournamentResult[b] += (bWin - aWin + self.args.arenaCompare)/2
                        repeat.append((playList.index(a)+1)*(playList.index(b)+1))
        return list(tournamentResult.values())
