class Game():
    def __init__(self):
        pass

    def getInitBoard(self):
        # return initial board (numpy board)
        pass

    def getBoardSize(self):
        # (a,b) tuple
        pass

    def getNextBoard(self, board, player, action):
        # if player takes action on board, return next board
        pass

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        pass

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player won, -1 if player lost
        pass

    def getCanonicalForm(self, board, player):
        # return state if player==0, else return -state ?
        pass

    def gameSymmetries(self):
        # mirror, rotational
        pass
