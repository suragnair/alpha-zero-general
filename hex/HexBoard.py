class HexBoard():
    def __init__(self, size):
        self.size = size
        self.positions = [[0 for x in range(self.size)] for y in range(self.size)]

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.positions[index]
    
    def getValidMoves(self, recodeBlackAsWhite=False):
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.positions[x][y] == 0:
                    moves.append((x, y))
        return moves
        # if recodeBlackAsWhite:
        #     return [self.recodeCoordinates(move) for move in moves]
        # else:
        #     return moves

    def recodeCoordinates (self, coordinates):
        """
        Transforms a coordinate tuple (with respect to the board) analogously to the method recodeBlackAsWhite.
        """
        return (self.size-1-coordinates[1], self.size-1-coordinates[0])

    def hasValidMoves(self):
        remainingMoves = self.getValidMoves()
        if len (remainingMoves) == 0:
            return False
        return True

    def makeMove(self, move, player):
        #  assert (self.winner == 0), "The game is already won."
         self.positions[move[0]][move[1]] = player

    def hasWhiteWon (self, verbose=False):
        """
        Evaluate whether the board position is a win for 'white'. Uses breadth first search. If verbose=True a winning path will be printed to the standard output (if one exists). This method may be time-consuming, especially for larger board sizes.
        """
        paths = []
        visited = []
        for i in range(self.size):
            if self.positions[i][0] == 1:
                paths.append([(i,0)])
                visited.append([(i,0)])
        while True:
            if len(paths) == 0:
                return False             
            for path in paths:
                prolongations = self._prolongPath(path)
                paths.remove(path)
                for new in prolongations:
                    if new[-1][1] == self.size-1:
                        if verbose:
                            print("A winning path for White:\n",new)
                        self.winner = 1
                        return True
                    if new[-1] not in visited:
                        paths.append(new)
                        visited.append(new[-1])
    
    def hasBlackWon (self, verbose=False):
        """
        Evaluate whether the board position is a win for 'black'. Uses breadth first search. If verbose=True a winning path will be printed to the standard output (if one exists). This method may be time-consuming, especially for larger board sizes.
        """
        paths = []
        visited = []
        for i in range(self.size):
            if self.positions[0][i] == -1:
                paths.append([(0,i)])
                visited.append([(0,i)])
        while True:
            if len(paths) == 0:
                return False
            for path in paths:
                prolongations = self._prolongPath(path)
                paths.remove(path)
                for new in prolongations:
                    if new[-1][0] == self.size-1:
                        if verbose:
                            print("A winning path for Black:\n",new)
                        self.winner = -1
                        return True
                    if new[-1] not in visited:
                        paths.append(new)
                        visited.append(new[-1])

    def _prolongPath (self, path):
        """
        A helper function used for board evaluation.
        """
        player = self.positions[path[-1][0]][path[-1][1]]
        candidates = self._getAdjacent(path[-1])
        candidates = [cand for cand in candidates if cand not in path]
        candidates = [cand for cand in candidates if self.positions[cand[0]][cand[1]] == player]
        return [path+[cand] for cand in candidates]

    def _getAdjacent (self, position):
        """
        Helper function to obtain adjacent cells in the board array.
        """
        u = (position[0]-1, position[1])
        d = (position[0]+1, position[1])
        r = (position[0], position[1]-1)
        l = (position[0], position[1]+1)
        ur = (position[0]-1, position[1]+1)
        dl = (position[0]+1, position[1]-1)
        return [pair for pair in [u,d,r,l,ur,dl] if max(pair[0], pair[1]) <= self.size-1 and min(pair[0], pair[1]) >= 0]
