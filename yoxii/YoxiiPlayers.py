import numpy as np
from .YoxiiLogic import Board

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        print("Chose a position for your totem: x,y")
        # display(board)
        b = Board()
        b.fields = np.copy(board)
        posActions = self.game.getValidMovesAsActions(board, 1)
        totem_moves = set()
        totem_to_action = {}
        for a in posActions:
            new_move = b.map_action_to_combo(a)[0]
            totem_moves |= {new_move}

            if new_move in totem_to_action:
                totem_to_action[new_move] += [a]
            else: 
                totem_to_action[new_move] = [a]
        
        print(totem_moves)
        while True:
            inp = input()
            try: 
                x,y = inp.split(",")
                x,y = int(x),int(y)
                if (x,y) in totem_moves:
                    chosen_move = (x,y)
                    break
                raise TypeError

            except:
                print("Not a valid position.")

        b.move_totem(chosen_move)
        b.print_board()

        print("Chose a position and coin type: x,y,t")

        all_possible = totem_to_action[chosen_move]

        all_coin_placements = set()
        all_coins = set()
        for a in all_possible:
            all_coin_placements |= {b.map_action_to_combo(a)[1]}
            all_coins |= {b.map_action_to_combo(a)[2]}
        
        print("Possible placements:",all_coin_placements)
        print("Possible coins:",all_coins)
        
        while True:
            inp = input()
            try: 
                x,y,t = inp.split(",")
                x,y,t = int(x),int(y),int(t)
                if (x,y) in all_coin_placements and t in all_coins:
                    chosen_coin_pos, chosen_coin_type = (x,y),t
                    break
                raise TypeError

            except:
                print("Not a valid x,y,t tuple.")

        return b.map_combo_to_action(chosen_move,chosen_coin_pos,chosen_coin_type)


        


