import sys
sys.path.append('..')
from Game import Game
from .YoxiiLogic import Board, Isometry
import numpy as np

class YoxiiGame(Game):

    def __init__(self):
        pass

    def getInitBoard(self):
        "Standard Board setup"
        return Board().fields

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions

        Yoxii has a board with 37 squares. Additionally, we can save the remaining stacks of coins in 8 squares.
        In sum 45 squares so all the board data can be sqeezed into 45 squares.
        """
        return (7,7) 

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions

        In Yoxi we have two moves each turn. First, move the totem (we have 37 possibilities for that since there are 37 squares).
        Secondly, place a coin on one of the remaining squares (now 36 since the totem has been placed)
        There are 4 options which coin we place, consequentially 37*36*4 possible actions
        Let us make the mapping a bit easier for the coins with also 37 squares. 
        """
        
        return 37*37*4 

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        b = Board(board)
        b.conduct_action(action,player)
        return (b.fields, -player)

    def getValidMoves(self, board, player, DEBUG=False):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vecr of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        valids = [0]*self.getActionSize()
        b = Board(board)
        legalMoves =  b.get_possible_actions(player)
        if DEBUG:
            print("Legal moves: ",sorted(legalMoves))
        for action in legalMoves:
            valids[action] = 1
        return np.array(valids)

    def getValidMovesAsActions(self, board, player):
        b = Board()
        b.fields = np.copy(board)
        return b.get_possible_actions(player)

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
                small non-zero value for draw.
        
        Keep in mind that the game ends when the current player has no valid moves.

        """
        b = Board(board)
        if len(b.get_totem_moves(player))>0: # if the totem can still be moved by the current player
            return 0
        # else: game ended
        end_sum = b.evaluate_position(player)
        if end_sum > 0:
            return 1
        elif end_sum == 0:
            return -0.00001*player
        return -1

    def getCanonicalForm(self, board, player, DEBUG=False):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: if player == 1, return the board as is
                            if player == -1, toggle the perspective of the board 
                            as if player -1 would be player 1. 
        
        We cannot just use -board since the places where is stored how many coins who has must swap, too. 
        """
        if player == 1:
            return board
        #else player == -1 -> we need to toggle the board
        return Board(board).toggle_perspective()

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                        form of the board and the corresponding pi vector. This
                        is used when training the neural network from examples.
        """
        all_sym = []

        for iso in Isometry.get_isometry_list():
            b = Board(board)
            b.board_isometries(iso)
            pi_new = b.action_vector_isometries(pi,iso)
            all_sym += [(b.fields,pi_new)]

        return all_sym

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                        Required by MCTS for hashing.

        Uses np.tostring() to convert the board to a string format.
        """
        return str(board)
    
    def stringRepToNPboard(self, boardString):
        editedStr = boardString.replace("[","").replace("]","")
        rows = editedStr.split("\n")
        rows = [row.strip() for row in rows]
        editedStr = []
        for row in rows:
            editedStr += [[int(x) for x in [e for e in row.split(" ") if e != ""]]]
        return np.array(editedStr)
    
    def printBoard(self,board):
        b = Board()
        b.fields = board
        b.print_board()

    @staticmethod
    def display(board):
        print("-"*20)
        b = Board()
        b.fields = np.copy(board)
        b.print_board()
        print("-"*20)
