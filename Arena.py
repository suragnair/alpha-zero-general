import logging
import random
from subprocess import Popen, PIPE

from tqdm import tqdm

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, AlphaPlayer, MinimaxPlayer, game, display=None):
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
        self.AlphaPlayer = AlphaPlayer
        self.MinimaxPlayer = MinimaxPlayer
        self.game = game
        self.display = display

    def playGameVsMinimax(self, startPlayer, proc, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: 1 if startPlayer won, -1 if startPlayer lost
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.MinimaxPlayer, None, self.AlphaPlayer]
        curPlayer = startPlayer
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            # assert self.display
            # print("Turn ", str(it), "Player ", str(curPlayer))
            # self.display(board)
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer), proc,  verbose=verbose)

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            if verbose:
                print('(', int(action/9), ', ', action%9, ')', sep='')
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        # assert self.display
        # print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
        # self.display(board)
        return self.game.getGameEnded(board, startPlayer)

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.MinimaxPlayer, None, self.AlphaPlayer]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            if verbose:
                print('(', int(action/9), ', ', action%9, ')', sep='')
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGamesVsMinimax(self, num, depth, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = [0, 0]
        twoWon = [0, 0]
        draws = [0, 0]
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            # AlphaZero starts, playing as O
            proc = Popen(['ultimatettt.exe', 'O', str(depth), str(random.randint(1, 1000000))], stdin=PIPE, stdout=PIPE, stderr=PIPE)
            gameResult = self.playGameVsMinimax(1, proc, verbose=verbose)
            if gameResult == 1:
                oneWon[0] += 1
            elif gameResult == -1:
                twoWon[0] += 1
            else:
                draws[0] += 1

        # self.AlphaPlayer, self.MinimaxPlayer = self.MinimaxPlayer, self.AlphaPlayer

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            # Minimax starts, playing as X
            proc = Popen(['ultimatettt.exe', 'X', str(depth), str(random.randint(1, 1000000))], stdin=PIPE, stdout=PIPE, stderr=PIPE)
            gameResult = self.playGameVsMinimax(-1, proc, verbose=verbose)
            if gameResult == -1:
                oneWon[1] += 1
            elif gameResult == 1:
                twoWon[1] += 1
            else:
                draws[1] += 1

        return oneWon, twoWon, draws

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.AlphaPlayer, self.MinimaxPlayer = self.MinimaxPlayer, self.AlphaPlayer

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws
