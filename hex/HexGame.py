import sys
sys.path.append('..')
from Game import Game
from .HexBoard import HexBoard
import numpy as np

class HexGame(Game):
    def __init__(self, size):
        self.size = size

    def getInitBoard(self):
        board = HexBoard(self.size)
        return np.array(board.positions)

    def getBoardSize(self):
        return (self.size, self.size)

    def getActionSize(self):
        return self.size * self.size

    def getNextState(self, positions, player, action):
        # if action == self.size * self.size:
        #     return (positions, -player)
        board = HexBoard(self.size)
        board.positions = np.copy(positions)
        move = (int(action / self.size), action % self.size)
        board.makeMove(move, player)
        return (board.positions, -player)

    def getValidMoves(self, positions, player):
        board = HexBoard(self.size)
        board.positions = np.copy(positions)

        valids = [0] * self.getActionSize()
        validMoves = board.getValidMoves(recodeBlackAsWhite=(player == -1))
        # if len(validMoves) == 0:
        #     valids[-1] = 1
        #     return np.array(valids)
        for x, y in validMoves:
            valids[self.size * x + y] = 1
        return np.array(valids)

    def getGameEnded(self, positions, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        board = HexBoard(self.size)
        board.positions = np.copy(positions)
        if board.hasWhiteWon():
            return 1 if player == 1 else -1
        if board.hasBlackWon():
            return -1 if player == 1 else 1
        # if not board.hasValidMoves():
        #     return -1
        return 0

    def getCanonicalForm(self, positions, player):
        board = HexBoard(self.size)
        board.positions = np.copy(positions)
        return board.positions
        # if player == 1:
        #     return board.positions

        # if player == -1:
        #     """
        #     Returns a board where black is recoded as white and wants to connect horizontally. This corresponds to flipping the board along the south-west to north-east diagonal and swapping colors.
        #     """
        #     flipped_board = self.getInitBoard()
        #     #flipping and color change
        #     for i in range(self.size):
        #         for j in range(self.size):
        #             if board[self.size-1-j][self.size-1-i] == 1:
        #                 flipped_board[i][j] = -1
        #             if board[self.size-1-j][self.size-1-i] == -1:
        #                 flipped_board[i][j] = 1
        #     return flipped_board

        # return player * board.positions

    def getSymmetries(self, board, pi):
        return [(board, pi)]
        # mirror, rotational
        # pi_board = np.reshape(pi[:], (self.size, self.size))
        # l = []
        # for i in range(1, 5):
        #     for j in [True, False]:
        #         newB = np.rot90(board, i)
        #         newPi = np.rot90(pi_board, i)
        #         if j:
        #             newB = np.fliplr(newB)
        #             newPi = np.fliplr(newPi)
        #         l += [(newB, list(newPi.ravel()))]
        # return l

    def stringRepresentation(self, positions):
        return hash(positions.tostring())

    def getScore(self, positions, player):
        return self.getGameEnded(positions, player)

    @staticmethod
    def display(positions):
        """
        This method prints a visualization of the hex board to the standard output. If the standard output prints black text on a white background, one must set invert_colors=False.
        """
        size = positions.shape[0]
        names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        indent = 0
        headings = " "*5+(" "*3).join(names[:size])
        print(headings)
        tops = " "*5+(" "*3).join("_"*size)
        print(tops)
        roof = " "*4+"/ \\"+"_/ \\"*(size-1)
        print(roof)
        
        #Attention: Color mapping inverted by default for display in terminal.
        color_mapping = lambda i: " " if i==0 else ("\u25CB" if i==-1 else "\u25CF")
        
        for r in range(size):
            row_mid = " "*indent
            row_mid += "   | "
            row_mid += " | ".join(map(color_mapping,positions[r]))
            row_mid += " | {} ".format(r+1)
            print(row_mid)
            row_bottom = " "*indent
            row_bottom += " "*3+" \\_/"*size
            if r<size-1:
                row_bottom += " \\"
            print(row_bottom)
            indent += 2
        headings = " "*(indent-2)+headings
        print(headings)
