import logging
import math
import sys

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)

        counts = np.array(
            [self.Nsa.get((s, a), 0) for a in range(self.game.getActionSize())],
            dtype=np.float32
        )

        if 'verbose' in self.args and self.args.verbose == 1:
            total_counts = counts.sum()
            probs = counts.reshape(canonicalBoard.shape)
            MCTS.display(probs)
            MCTS.display(probs / (total_counts + EPS))
            s = self.game.stringRepresentation(canonicalBoard)
            probs = np.array(self.Ps[s]).reshape(canonicalBoard.shape)
            MCTS.display(probs)

        if temp == 0:
            bestA = np.random.choice(np.flatnonzero(counts == counts.max()))
            probs = np.zeros_like(counts, dtype=np.float32)
            probs[bestA] = 1
            return probs
        else:
            counts = counts ** (1. / temp)
            probs = counts / (counts.sum() + EPS)
            
            return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            if self.Es[s] == 2:
                # draw
                return 0
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] *=  valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = valids / valids.sum()

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        sqrt_Ns = math.sqrt(self.Ns[s] + EPS)

        # Vectorized UCB calculation        
        ucb_values = np.array([
            self.Qsa.get((s, a), 0) +
            self.args.cpuct * self.Ps[s][a] * sqrt_Ns / (1 + self.Nsa.get((s, a), 0))
            if valids[a] else -float('inf')
            for a in range(self.game.getActionSize())
        ])

        # pick the action with the highest upper confidence bound
        best_act = np.argmax(ucb_values)
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, best_act)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, best_act) in self.Qsa:
            self.Qsa[(s, best_act)] = (self.Nsa[(s, best_act)] * self.Qsa[(s, best_act)] + v) / (self.Nsa[(s, best_act)] + 1)
            self.Nsa[(s, best_act)] += 1

        else:
            self.Qsa[(s, best_act)] = v
            self.Nsa[(s, best_act)] = 1

        self.Ns[s] += 1
        return -v


    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[x][y]    # get the piece to print
                print(f"{piece:.2f}", end=" ")
            print("|")

        print("-----------------------")

