import random


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = random.randint(0, self.game.getActionSize() - 1)
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = random.randint(0, self.game.getActionSize() - 1)
        print("Random player played: " + str(a))
        return a


class HumanMazeBattlePlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valids = self.game.getValidMoves(board, 1)
        print(valids)
        while True:
            # Python 3.x
            a = input()
            # Python 2.x 
            # a = raw_input()

            move = int(a)
            if 0 <= move <= len(valids) and valids[move] == 1:
                return move
            else:
                print('Invalid')