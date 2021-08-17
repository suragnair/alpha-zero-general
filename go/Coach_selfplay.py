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


class Coach_selfplay():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.firstIter = args.firstIter #set true if it produce first chechpoint to save
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        expectedWinner = np.random.choice([-1, 1], p=[0.5, 0.5])
        log.info(f'Expected Winner #{expectedWinner}')
        
        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)
            #fastDecision data should not be collected (ketaGo Paper)
            pi, fastDecision, resign = self.mcts.getActionProb(canonicalBoard, temp=temp, training=1, ew=expectedWinner)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                if not fastDecision:  #only add example of slow decisions
                    trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            #log.info(f'After action board: #/n\n{board}')
            r = self.game.getGameEnded(board, self.curPlayer)           
            if resign:
                log.info(f'Resigned')
                r = 1 # previous player resigned          
            if r != 0:
                log.info(f'End Game Board')
                log.info(f'#/n\n{board}')
                log.info(f'Ending turn #{board.turns}')
                winner = r*self.curPlayer
                log.info(f'Winner: #{winner}')
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

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
            self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            self.trainExamplesHistory = [] # empty the history 
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()
                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples()
           # self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
          

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    #add the way to save best example
    def saveTrainExamples(self):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, "best.pth.tar.examples")
        with open(filename, "ab+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed






