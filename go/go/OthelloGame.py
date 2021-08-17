from __future__ import print_function
import sys
import copy
sys.path.append('..')
from Game import Game
import numpy as np
from go.goGame import GameUI 
from go.group import Group, GroupManager
'''
from go.board import Board
from go.utils import Stone, make_2d_array
from go.group import Group, GroupManager
from go.exceptions import (
    SelfDestructException, KoException, InvalidInputException)
'''
class Game(Game):
    square_content = {
        -1: "w",
        +0: "-",
        +1: "b"
    }

    @staticmethod
    def getSquarePiece(piece):
        return Game.square_content[piece]

    def __init__(self, args):
        self.n = args.size
        self.args = args
        self.goGame = None

    def getInitBoard(self):
        # return initial board (numpy board)
        self.goGame = GameUI(self.n)
        #goGame.game.board = np.array(goGame.game.board).append([False, False])
        #b = Board(self.n)
        #return np.array(b.pieces)
        return self.goGame.game.board

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n+1 #9*9+1
    
    def getMaxMoves(self):
        # return the max number of moves
        return self.n*self.n*3/2

    def getNextState(self, board, player, action):
        #print(action)
        # if player takes action on board, return next (board,player)
        # ###action must be a valid move

        #if action is pass, record it
        if action == self.n*self.n: #81
            b = copy.deepcopy(board)
            if b.previous_is_pass == True: 
                b.pre_previous_is_pass = True
            b.previous_is_pass = True
            b.turns += 1
            return (b, -player)

        move = (int(action/self.n), action%self.n) #Interpret the Action

        # initialize the board and group manager for simulating
        self.goGame.game.board = copy.deepcopy(board)
        self.goGame.game.gm = GroupManager(self.goGame.game.board, enable_self_destruct=False)
        self.goGame.game.gm._group_map = copy.deepcopy(board._group_map)
        self.goGame.game.gm._captured_groups = copy.deepcopy(board._captured_groups)
        self.goGame.game.gm._num_captured_stones = copy.deepcopy(board._num_captured_stones)
        self.goGame.game.gm._ko = copy.deepcopy(board._ko)

        #make the move
        self.goGame._place_stone(move, player) #^

        #load necessary fields back to the board
        board = copy.deepcopy(self.goGame.game.board)
        board.previous_is_pass = False
#        board.pre_previous_is_pass = False
        board.turns += 1
        board._group_map = copy.deepcopy(self.goGame.game.gm._group_map)
        board._captured_groups = copy.deepcopy(self.goGame.game.gm._captured_groups)
        board._num_captured_stones = copy.deepcopy(self.goGame.game.gm._num_captured_stones)
        board._ko = copy.deepcopy(self.goGame.game.gm._ko)

        #bug log
        #print("turns", board.turns)
        #print(self.goGame.game.board)
        #print(board._group_map)
        return (copy.deepcopy(board), -player) #^

    def getValidMoves(self, board, player):
        #print('here')
        #print('turn for '+ str(player))
        #initialize a valids list contains all actions
        valids = [0]*self.getActionSize() #keep

        #load board properities into tempGame
        self.goGame.game.board = copy.deepcopy(board)
        self.goGame.game.gm = GroupManager(self.goGame.game.board, enable_self_destruct=False)
        self.goGame.game.gm._group_map = copy.deepcopy(board._group_map)
        self.goGame.game.gm._captured_groups = copy.deepcopy(board._captured_groups)
        self.goGame.game.gm._num_captured_stones = copy.deepcopy(board._num_captured_stones)
        self.goGame.game.gm._ko = copy.deepcopy(board._ko)

        # Construct all possible tuples, then filter away ilegals
        x = list(range(0,self.n))
        y = list(range(0,self.n))
        legalMoves = [ (a,b) for a in x for b in y]
        ilegalMoves = [] 
        #print(board)
        #print(tempGame.game.gm.board)
        for x, y in legalMoves:
            #print((x,y))
            legal = self.goGame._place_stone((x,y), player)
            if legal == False:
                #print('ilegal')
                ilegalMoves.append((x,y))
                self.goGame.game.board = copy.deepcopy(board)
                self.goGame.game.gm = GroupManager(self.goGame.game.board, enable_self_destruct=False)
                self.goGame.game.gm._group_map = copy.deepcopy(board._group_map)
                self.goGame.game.gm._captured_groups = copy.deepcopy(board._captured_groups)
                self.goGame.game.gm._num_captured_stones = copy.deepcopy(board._num_captured_stones)
                self.goGame.game.gm._ko = copy.deepcopy(board._ko)
            else:
                #print('legal')
                self.goGame.game.board = copy.deepcopy(board)
                self.goGame.game.gm = GroupManager(self.goGame.game.board, enable_self_destruct=False)
                self.goGame.game.gm._group_map = copy.deepcopy(board._group_map)
                self.goGame.game.gm._captured_groups = copy.deepcopy(board._captured_groups)
                self.goGame.game.gm._num_captured_stones = copy.deepcopy(board._num_captured_stones)
                self.goGame.game.gm._ko = copy.deepcopy(board._ko)
        legalMoves = self.Diff(legalMoves, ilegalMoves) 
        if board.turns > self.n*self.n/2: #only alow pass when reach certain turns
            valids[-1]=1 
        if len(legalMoves)==0:
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        #print(np.array(valids))
        #print('here, hopefully')
        return np.array(valids)

    def getGameEnded(self, board, player):
        
        
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        #b = Board(self.n)
        #b.pieces = np.copy(board)
        ##goGame = GameUI() #^
        self.goGame.game.board = board#^
        #print(goGame.game.board)       debug
        #end with 2 consective passes
        # 
        #Black should win when 43:38 (5 points higher) 
        #for simplicity, whoever get 41 will win
        #print('self.goGame.game.board.previous_is_pass')
        #print(self.goGame.game.board.previous_is_pass)
        #print(self.goGame.game.board.pre_previous_is_pass)
        #print('self.goGame.game.board.pre_previous_is_pass')
        if self.args.balancedGame is True:
            if ((self.goGame.game.board.previous_is_pass and self.goGame.game.board.pre_previous_is_pass) or board.turns > self.n*self.n*3/2):
                #print('enddd')
                #print(board)
                diff = self.goGame.game.get_scores().get(player) - self.goGame.game.get_scores().get(-player)
                #print(self.goGame.game.board)
                if board.turns%2 and diff > -6:
                    #print("current player win")
                    return 1
                elif diff > 6 and not board.turns%2:
                    return 1
                else:
                    #print("oppo player win")
                    return -1
            else:
            #  print('not end')
                return 0
        else:
            if ((self.goGame.game.board.previous_is_pass and self.goGame.game.board.pre_previous_is_pass) or board.turns > self.n*self.n*3/2):
                diff = self.goGame.game.get_scores().get(player) - self.goGame.game.get_scores().get(-player)
                if diff > 0:
                    return 1
                else:
                    return -1
            else:
                return 0


    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        #change of board
        board = player*board
        #change of _group_map:  reverse  _group_map[0][1].stone
        row = self.n
        col = self.n
        for j in range(col):
            for i in range(row):
                if board._group_map[i][j] is not None:
                    board._group_map[i][j].stone = board._group_map[i][j].stone*(-1)

        #change of _num_captured_stones: reverse it
        temp = board._num_captured_stones.get(-1)
        board._num_captured_stones[-1] = board._num_captured_stones[1]
        board._num_captured_stones[1] = temp


        return board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        return board.tostring() 

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        #b = Board(self.n)
        #b.pieces = np.copy(board)
        ##goGame = GameUI() #^
        self.goGame.game.board = board.copy() #^
        #return b.countDiff(player)
        return self.goGame.game.get_scores().get(player)

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
                piece = board[y][x]    # get the piece to print
                print(Game.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
    
    def Diff(self, li1, li2):
        return list(set(li1) - set(li2)) + list(set(li2) - set(li1))
