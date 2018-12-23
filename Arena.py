import time

import numpy as np

from pytorch_classification.utils import Bar, AverageMeter
from td2020.src.config import NUM_ACTS, ACTS_REV,  CONFIG


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False, game_iteration=-1, game_episode=-1):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert (self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                print(action)
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

            # #####################################################################################################################
            # #################################################  STATS  ###########################################################
            # #####################################################################################################################

            score = self.game.getScore(board, curPlayer)
            y, x, action_index = np.unravel_index(action, [self.game.n, self.game.n, NUM_ACTS])

            # get direction
            my_str = ACTS_REV[action_index]
            dir = None
            if 'up' in my_str:
                dir = 'up'
            if 'down' in my_str:
                dir = 'down'
            if 'left' in my_str:
                dir = 'left'
            if 'right' in my_str:
                dir = 'right'

            stat = "iteration:" + str(game_iteration) + "$game_ep:" + str(game_episode) + "$player:" + str(curPlayer) + "$x:" + str(x) + "$y:" + str(y) + "$action_index:" + str(action_index) + "$act_rev:" + ACTS_REV[action_index] + '$direction:' + str(dir) + "$score:" + str(
                score) + "it:" + str(it)
            # print(stat)

            # csv format
            stat = str(game_iteration) + "," + str(game_episode) + "," + str(curPlayer) + "," + str(x) + "," + str(y) + "," + str(action_index) + "," + ACTS_REV[action_index] + "," + str(dir) + "," + str(score) + "," + str(it)

            CONFIG.append_item(stat)
            # #####################################################################################################################
            # ################################################## END STATS ########################################################
            # #####################################################################################################################

        if verbose:
            assert (self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return self.game.getGameEnded(board, 1)

    def playGames(self, num, verbose=False, game_iteration=-1):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for i in range(num):
            gameResult = self.playGame(verbose=verbose, game_iteration=game_iteration, game_episode=i + 1)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps + 1, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        self.player1, self.player2 = self.player2, self.player1

        for i in range(num):
            gameResult = self.playGame(verbose=verbose, game_iteration=game_iteration, game_episode=i + 1 + num)

            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps + 1, maxeps=num, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        bar.finish()

        return oneWon, twoWon, draws
