'''
Author: Conrad Schweiker
Date: 23.05.2024
Board class.
Board data:
    self.fields[x][y] yields the item on that position
    1,2,3,4 for white (1); -1,-2,-3,-4 for red (-1); 0 for nothing
    (even the totem) and None if not part of the board

    self.getTotem() is a (x,y) position of the totem

    self.coins are the not yet placed coins as list in order
    [3,4,2,1] means white has 3*O, 4*II, 2*Y, 1*X
Squares are stored and manipulated as (x,y) tuples.
x...row, y...column
'''

print(
"""
    __   _______  _____ ___       
    \ \ / / _ \ \/ /_ _|_ _|   
     \ V / (_) >  < | | | |  
      |_| \___/_/\_\___|___|                            
"""
)

"""
Set up initial board configuration.
The self.fields variable is important because it is used to represent an entire state.
The Board class only delivers tools to manipulate this variable.

Since it is well advised to have a numpy representation of the game state,
we need to come up with an idea, how to represent the Yoxii board as a numpy array.

Solution: the board is stored as 7x7 array:

A B X X X C D
E X X X X X F
X X X X X X X
X X X X X X X
X X X X X X X
0 X X X X X 0
I J X X X K L

where X is a number in {-4,-3,-2,-1,0,1,2,3,4} representing the placed coins
|A| the nr of available 1-coins for player 1, |B| for player 2
|C| the nr of available 2-coins for player 1, |D| for player 2
|I| the nr of available 3-coins for player 1, |J| for player 2
|K| the nr of available 4-coins for player 1, |L| for player 2

Note that we are using the abs() for this positive number, so that we are able to flip the array
easily later with -array.
To get the corresponding number of coins, we can use the formula array[x][y+(Player==-1)]
where x,y are the coordinates for A,C,I,K  

The totems position is stored as a (x,y) tuple in (|E|,|F|), not in any X since it is neither +,-,0

Since the Board-class is called often, let us configure a couple of constants in the global scope:
"""
# Initialise colorama for colorful representation of the board when printed
import colorama
import numpy as np
from colorama import Fore, Style


colorama.init(autoreset=True)     # Replace these with ansi esc codes if you dont want to use colorama
C1 = Fore.CYAN
C2 = Fore.MAGENTA
C3 = Fore.YELLOW
CR = Style.RESET_ALL

# list of all 8 directions on the board, as (x,y) offsets
_DIRECTIONS = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

# visuals are used later for printing the board colorized to the terminal
_VISUALS = {
    1:C1+"1"+CR, 2:C1+"2"+CR, 3:C1+"3"+CR, 4:C1+"4"+CR,     # Player white in cyan  (1)
    0:"-",
    -1:C2+"1"+CR, -2:C2+"2"+CR, -3:C2+"3"+CR, -4:C2+"4"+CR, # Player red in magenta (-1)
}

# Maps the coin type to the corners where the number of available coins is stored
_COIN_POS = {        1:(0,0), 2:(0,5), 3:(6,0), 4:(6,5)       }

_INIT_BOARD = np.zeros((7,7),dtype=int)
# Set board start position
for (x,y,nr) in [(0,0,5),(0,1,5),(0,5,5),(0,6,5),(1,0,3),(1,6,3),(6,0,5),(6,1,5),(6,5,3),(6,6,3)]:  
    _INIT_BOARD[x][y] = nr  

_ALL_SQUARES = set([(x,y) for x in range(7) for y in range(7)])
# Remove corners so to exclude those squares
for (x,y) in [  (0,0),(0,1),(1,0),  (6,0),(6,1),(5,0),\
                (0,6),(0,5),(1,6),  (6,6),(5,6),(6,5) ]:
    _ALL_SQUARES.remove((x,y))

_ACTION_TO_COMBOS = dict() # Shall store: action number <--> combo string
_COMBOS_TO_ACTION = dict() # Shall store: combo string  --> action number
def _fill_ACTION_TO_COMBOS():
    global _ACTION_TO_COMBOS, _COMBOS_TO_ACTION, _ALL_SQUARES
    p2n = {} # position to nr map: (x,y) <--> nr in {0,...,48}
    counter = 0
    for (x,y) in [(x,y) for x in range(7) for y in range(7)]:
        if (x,y) in _ALL_SQUARES:
            p2n[(x,y)] = counter
            p2n[counter] = (x,y)
            counter += 1

    strfy = lambda nr,coord: str(p2n[nr][coord])

    for tot in range(37):         # tot = token position as nr
        for c in range(37):     # c = coin position as nr
            for typ in range(4):  # typ = type of coin
                combostr = strfy(tot,0) + strfy(tot,1) + strfy(c,0) + strfy(c,1) + str(typ)
                _ACTION_TO_COMBOS[tot*37*4 + c*4 + typ] = combostr
                _COMBOS_TO_ACTION[combostr] = tot*37*4 + c*4 + typ
_fill_ACTION_TO_COMBOS()

class Board:

    def __init__(self,testing=False):
        global _INIT_BOARD, _ALL_SQUARES
        self.fields = np.copy(_INIT_BOARD)

        if testing: # To try out visuals without playing, set up a board with pieces laying around already. 
            self.fields[2][2], self.fields[2][3], self.fields[3][4], self.fields[3][5] = 1 , -2 , 4 , 3
            self.fields[4][2], self.fields[4][3], self.fields[5][3], self.fields[6][3] = -4, 3 , 2 , -2

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.fields[index]
    
    def getTotem(self):
        return (abs(self.fields[1][0]), abs(self.fields[1][6]))

    def setTotem(self, pos: tuple[int,int]):
        self.fields[1][0], self.fields[1][6] = pos

    
    # returns a set of absolute (x,y) positions for placing coins in relation to the totem position
    def get_allowed_squares(self,totem_pos=None):
        global _ALL_SQUARES
        totem_pos = self.getTotem() if totem_pos is None else totem_pos
        
        "First we will try to find free spots adjacent to the totem. "
        free_squares = self.get_totem_moves(player=1,Range=1,totem_pos=totem_pos)

        "If there is no adjacent free square, place it on any free spot on the board"
        if not free_squares:
            for x in range(7):
                for y in range(7):
                    if (x,y) in _ALL_SQUARES:
                        if self.fields[x][y] == 0 and (x,y) != totem_pos:
                            free_squares |= {(x,y)}

        return free_squares # a set of (x,y) positions
    
    def analyse_move_combo(self,tpos,cpos,coin,player):
        "Needed only for debugging to find out what is the wrong part of the combo."

        totem_moves = self.get_totem_moves(player)
        print("Totem move available: ",tpos in totem_moves)
        
        coin_positions = self.get_allowed_squares(tpos)
        print("Square allowed: ",cpos in coin_positions)

        remaining_coins = self.get_remaining_coins(player,coin)
        print("Enough coins: ",remaining_coins > 0)

    def get_possible_move_combos(self,player):
        """
        There are 37*37*4 possible move combos (some of which are illegal by default)
        See section in YoxiiGame.getActionSize()
        """
        all_possible_combos = []
        totem_moves = self.get_totem_moves(player)
        for tpos in totem_moves:
            possible_coin_places = self.get_allowed_squares(tpos)
            for cpos in possible_coin_places:
                for coin in range(1,5):
                    if self.get_remaining_coins(player,coin) > 0:

                        all_possible_combos +=[(tpos,cpos,coin)]
        
        "all_possible_combos is a list of (totem_pos,coin_pos,coin_type) tuples"
        return all_possible_combos
        
    def get_possible_actions(self,player):
        all_possible_combos = self.get_possible_move_combos(player)
        return [self.map_combo_to_action(tpos,cpos,coin) for (tpos,cpos,coin) in all_possible_combos]

    def conduct_action(self,action,player):
        #print("Local:",sum(self.get_possible_actions(player)))
        #print("Local belongs to state\n",str(self.fields))
        #self.print_board()
        "Maybe for performance it would be wise to delete this extra iteration through all possible actions:"
        """
        if not action in self.get_possible_actions(player):
            self.print_board()
            print(self.fields)
            self.analyse_move_combo(*self.map_action_to_combo(action),player)
            print("Action not allowed:",self.map_action_to_combo(action),player)
            raise ValueError("Action cannot be performed.")
        """
        
        totem_pos,coin_pos,coin_type = self.map_action_to_combo(action)

        self.setTotem(totem_pos)
        self.place_coin(coin_pos,player,coin_type)
    
    def map_combo_to_action(self,tot:tuple,c:tuple,typ:int) -> int:
        global _COMBOS_TO_ACTION
        typ -= 1 # Since coins 1,2,3,4 are mathematically stored as 0,1,2,3
        strfy = lambda tpl: str(tpl[0]) + str(tpl[1])
        return _COMBOS_TO_ACTION[strfy(tot) + strfy(c) + str(typ)]

    def map_action_to_combo(self,action) -> tuple[tuple[int,int],tuple[int,int],int]:
        global _ACTION_TO_COMBOS
        c = _ACTION_TO_COMBOS[action]
        c = [int(x) for x in c]
        c[4] += 1 # Since the as 0,1,2,3 stored coins have to be returned as 1,2,3,4
        return (c[0],c[1]), (c[2],c[3]), c[4]

    def set_remaining_coins(self,player,typ,amount, mod=1):
        """ Sets the remaining amount of coins of type typ for player player to amount.
        Mod is used for proper swapping in self.getCanonicalBoard()"""
        global _COIN_POS
        self.fields[_COIN_POS[typ][0]][_COIN_POS[typ][1]+(-1==player)] = amount * mod

    def get_remaining_coins(self,player,typ):
        # Returns the amount of coins of type typ for player player
        global _COIN_POS
        return abs(self.fields[_COIN_POS[typ][0]][_COIN_POS[typ][1]+(-1==player)])

    # Finds the surrounding free spots and jumping spots
    def get_totem_moves(self, player, Range=6, free_only=True, totem_pos=None):
        """
        Evaluates the squares around the totem position.
        Returns: dictionary with (x,y) positions

        Jumping rules are also taken into account as long as Range>1.
        By default only free squares are considered.
        Set free_only=False to get a list of _all_ on-board adjacent squares of Range=1.
        (Player will be ignored in this case)
        """
        global _ALL_SQUARES
        totem_pos = self.getTotem() if totem_pos is None else totem_pos
        # (so that a fictional totem position can override the actual totem position)

        possible_moves = set()

        global _DIRECTIONS
        for dir in _DIRECTIONS:
            for i in range(Range): 
                pos = self.add(totem_pos,self.scale(i+1,dir)) # vector addition and multiplication
                
                if pos in _ALL_SQUARES: # If possible position still on the board

                    coin = self.fields[pos[0]][pos[1]]
                    if coin == 0 or not free_only:
                        possible_moves |= {pos}
                        break

                    # As soon as a coin of the other player is met, this direction is invalid
                    elif (coin < 0 and player > 0) or (coin > 0 and player < 0):
                        break

        return possible_moves
    
    def evaluate_position(self,player):
        return sum([self.fields[x][y] for (x,y) in self.get_totem_moves(player,free_only=False)]) * player


    
    def place_coin(self,pos: tuple, player: int, typ: int): # Typ is 1,2,3,4 and always positive. Player is +-1
        if self.get_remaining_coins(player,typ) > 0:   # Check first, if this coin is still available
            self.fields[pos[0]][pos[1]] = player*typ
            self.set_remaining_coins(player,typ,self.get_remaining_coins(player,typ)-1) # Remove one coin since it has been used. 
        else:
            raise IndexError



    # Method for adding multiple n-tuple vectors together. (1,2,4) + (-2,3,-1) = (-1,5,3)
    @staticmethod
    def add(*vectors):
        sum = 0
        length = 0

        if vectors:
            length = len(vectors[0])
            sum = [0]*length
            for v in vectors:
                if len(v) != length:
                    return 0 # If vectors do not have the same length return 0
                
                for i in range(len(v)):
                    sum[i] += v[i]
            
            return tuple(sum) # Return resulting vector as tuple
        else:
            return 0 # If vectors empty return 0
    
    # Method for multiplying a vector with a scalar
    @staticmethod
    def scale(scalar,vector):
        result = [0]*len(vector)
        for i in range(len(vector)):
            result[i] = vector[i] * scalar
    
        return tuple(result)

    def board_isometries(self,isometry_type="rotate"):
        # Save corner information that is rotation sensitive
        coins = {(player,typ):self.get_remaining_coins(player,typ) for player in [1,-1] for typ in range(1,5)}
        totem = self.getTotem()
        self.fields[totem[0]][totem[1]] = 100 # So that this can be also rotated
        

        # Which manipulation shall be done

        match isometry_type:
            case "rotate":
                self.fields = np.rot90(self.fields)
            case "flipX":
                self.fields = np.flip(self.fields,axis=0)
            case "flipY":
                self.fields = np.flip(self.fields,axis=1)
            case "flipXY":
                self.fields = np.rot90(np.fliplr(self.fields))
            case "flipYX":
                self.fields = np.rot90(np.flipud(self.fields))


        # Restore
        # Find 100 as totem representation
        global _ALL_SQUARES
        for x in range(7):
            for y in range(7):
                if (x,y) in _ALL_SQUARES and self.fields[x][y] == 100:
                    self.fields[x][y] = 0
                    self.setTotem((x,y))
                    break
        for key in coins.keys():
            self.set_remaining_coins(*key,coins[key])
        self.fields[5][0], self.fields[5][6] = 0,0 

    def getCanonicalVersion(self,player,DEBUG=False):
        """Returns the version of the board where player=1 is forced.
        Consequently, if the game is align to present for the other player,
        coin save positions and turn around all values   """

        totem_sum = self.fields[1][0] + self.fields[1][6]  # !=0 since totem is not allowed on (0,0)
        current_perspective = int(totem_sum/abs(totem_sum)) # 1 or -1
        #print(totem_sum)
        if DEBUG:
            print("Current Perspective: ",current_perspective)
            print("Requested perspective: ",player)
            print("Original Version:")
            self.print_board()
            print("Canonical Version:")

        if player == current_perspective: 
            if DEBUG: self.print_board()
            return self.fields
        #print("Needs swapping.")
        #else swap
        for typ in range(1,5):
            left = self.get_remaining_coins(1,typ)
            right = self.get_remaining_coins(-1,typ)
            self.set_remaining_coins(1,typ,right,mod=(-1 if player == 1 else 1))
            self.set_remaining_coins(-1,typ,left,mod=(-1 if player == 1 else 1))

        self.fields = -self.fields
        if DEBUG: self.print_board()
        return self.fields

    # For quickly printing a visual representation of the self.fields variable
    def print_board(self):
        global _VISUALS, _ALL_SQUARES
        print()
        #print("\tF"+7*"--"+"1")

        styling = {1:C1,-1:C2}
        maximas = {}
        for typ in range(1,5):
            maximas[typ] = max(self.get_remaining_coins(1,typ),self.get_remaining_coins(-1,typ))

        for x in range(7):
            line = "\t" #"\t|"
            for y in range(7):
                if (x,y) in _ALL_SQUARES:
                    if (x,y) == self.getTotem():
                        line += C3 + "●" + CR + " " 
                    else:
                        # Instead of printing -1 / 4 / 0 ... print a visual equivalent
                        line += str(_VISUALS[self.fields[x][y]]) + " "
                else:
                    line += "  "

            Player = None
            if x == 2: Player = 1
            elif x == 4: Player = -1
            
            if Player:
                line += "\t\t"+ styling[Player] + self.get_remaining_coins(Player,1) * "1," + " " + "  "*(maximas[1]-self.get_remaining_coins(Player,1)) \
                        + self.get_remaining_coins(Player,2)*"2," + " " + "  "*(maximas[2]-self.get_remaining_coins(Player,2)) \
                        + self.get_remaining_coins(Player,3)*"3," + " " + "  "*(maximas[3]-self.get_remaining_coins(Player,3)) \
                        + self.get_remaining_coins(Player,4)*"4," + CR

            print(line) #+"|")
        print()
        #print("\tL"+7*"--"+"J")


"""
Test this class:

"""
print("--TESTS--\n")
board = Board()
print(board.fields)

print(board.map_action_to_combo(2331))
print(board.map_combo_to_action((0,2),(0,3),2))
print(board.get_possible_actions(1))
board.place_coin((1,2),1,1)
board.place_coin((1,3),-1,2)
board.conduct_action(2555,1)
board.print_board()
print(board.fields,end="\n\n")
board.fields = board.getCanonicalVersion(1)
board.print_board()
board.fields = board.getCanonicalVersion(-1)
board.print_board()
board.fields = board.getCanonicalVersion(-1)
board.print_board()
board.fields = board.getCanonicalVersion(1)
board.print_board()
for action in [740, 780, 788, 1248]:
    print(board.map_action_to_combo(action))

"""
    ((1, 3), (0, 2), 1)
((1, 3), (2, 2), 1)
((1, 3), (2, 4), 1)
((2, 0), (3, 1), 1)
"""