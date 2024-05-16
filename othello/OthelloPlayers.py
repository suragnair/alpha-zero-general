import numpy as np
import subprocess

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print("[", int(i/self.game.n), int(i%self.game.n), end="] ")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 2:
                try:
                    x,y = [int(i) for i in input_a]
                    if ((0 <= x) and (x < self.game.n) and (0 <= y) and (y < self.game.n)) or \
                            ((x == self.game.n) and (y == 0)):
                        a = self.game.n * x + y if x != -1 else self.game.n ** 2
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')
        return a


class GreedyOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]

class GTPOthelloPlayer():
    """
    Player that plays with Othello programs using the Go Text Protocol.
    """

    # The colours are reversed as the Othello programs seems to have the board setup with the opposite colours
    player_colors = {
        -1: "white",
         1: "black",
    }

    def __init__(self, game, gtpClient):
        """
        Input:
            game: the game instance
            gtpClient: list with the command line arguments to start the GTP client with.
                       The first argument should be the absolute path to the executable.
        """
        self.game = game
        self.gtpClient = gtpClient

    def startGame(self):
        """
        Should be called before the game starts in order to setup the board.
        """
        self._currentPlayer = 1 # Arena does not notify players about their colour so we need to keep track here
        self._process = subprocess.Popen(self.gtpClient, bufsize = 0, stdin = subprocess.PIPE, stdout = subprocess.PIPE)
        self._sendCommand("boardsize " + str(self.game.n))
        self._sendCommand("clear_board")

    def endGame(self):
        """
        Should be called after the game ends in order to clean-up the used resources.
        """
        if hasattr(self, "_process") and self._process is not None:
            self._sendCommand("quit")
            # Waits for the client to terminate gracefully for 10 seconds. If it does not - kills it.
            try:
                self._process.wait(10)
            except (subprocessTimeoutExpired):
                self._process.kill()
            self._process = None

    def notify(self, board, action):
        """
        Should be called after the opponent turn. This way we can update the GTP client with the opponent move.
        """
        color = GTPOthelloPlayer.player_colors[self._currentPlayer]
        move = self._convertActionToMove(action)
        self._sendCommand("play {} {}".format(color, move))
        self._switchPlayers()

    def play(self, board):
        color = GTPOthelloPlayer.player_colors[self._currentPlayer]
        move = self._sendCommand("genmove {}".format(color))
        action = self._convertMoveToAction(move)
        self._switchPlayers()
        return action

    def _switchPlayers(self):
        self._currentPlayer = -self._currentPlayer

    def _convertActionToMove(self, action):
        if action < self.game.n ** 2:
            row, col = int(action / self.game.n), int(action % self.game.n)
            return "{}{}".format(chr(ord("A") + col), row + 1)
        else:
            return "PASS"

    def _convertMoveToAction(self, move):
        if move != "PASS":
            col, row = ord(move[0]) - ord('A'), int(move[1:])
            return (row - 1) * self.game.n + col
        else:
            return self.game.n ** 2

    def _sendCommand(self, cmd):
        self._process.stdin.write(cmd.encode() + b"\n")

        response = ""
        while True:
            line = self._process.stdout.readline().decode()
            if line == "\n":
                if response:
                    break  # Empty line means end of the response is reached
                else:
                    continue  # Ignoring leading empty lines
            response += line

        # If the first character of the response is '=', then is success. '?' is error.
        if response.startswith("="):
            # Some clients return uppercase other lower case.
            # Normalizing to uppercase in order to simplify handling.
            return response[1:].strip().upper()
        else:
            raise Exception("Error calling GTP client: {}".format(response[1:].strip()))

    def __call__(self, game):
        return self.play(game)
