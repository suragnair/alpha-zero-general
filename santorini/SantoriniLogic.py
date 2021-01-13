import numpy as np

class Board():
    """
    A Santorini Board of default shape: (2,5,5)
    
    Board logic:
        board shape: (2, self.n, self.n)
           [[[ 0,  0,  0,  0,  0],
             [ 0,  0,  1,  0,  0],
             [ 0, -1,  0, -2,  0],
             [ 0,  0,  2,  0,  0],
             [ 0,  0,  0,  0,  0]]
            
            [[ 0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0]]]
    
    BOARD[0]: character locations
    board[0] shape = (self.n,self.n) here this is (5,5)
    Cannonical version of this board shows 
    a player with their pieces as +1, +2 and opponents as -1, -2
        
    LOCATIONS: 
        Locations are given as (x,y) (ROW, COLUMN) coordinates,
        e.g. the 1 in board[0] is at location (1,2), and the 2 at (3,2), whereas
        the -1 is at location (2,1), and the -2 at (2,3)
    
    ACTIONS: 
        Actions are stored as list of tuples of the form:
            action = [piece_location, move_location, build_location]
                     [(x1,y1),        (x2, y2),      (x3, y3)]

    
    BOARD 1: Location heights
        board shape: (self.n,self.n)
        Cannonical board shows player height of each board space.
        The height of each space ranges from 0,...,4 (this is independent of self.n)

    
    
    """
    # NOTE THESE ARE NEITHER CCW NOR CW!
    __directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    #                  Nw,     N,     Ne,   W      E,    Sw,    S,    Se,    
    
    def __init__(self, board_length, true_random_placement=False):
        """
        Initializes an empty board of shape (2, board_length, board_length)
                                          =  (dimension, row, column)
        Unless true_random_placement, both player's pieces are placed in the center of the board.
        Currently there is no way to directly place one's own pieces at the game start.
        """
        self.n = board_length
        self.pieces = np.zeros((2, self.n, self.n), dtype='int')
        self.true_random_placement = true_random_placement
        
        chars_placed = 0
        char_list = [-1,-2, +1, +2]
        if self.true_random_placement:
            while chars_placed < 4:
                char = char_list[chars_placed]
                
                a = np.random.randint(0, self.n)
                b = np.random.randint(0, self.n)
                if self.pieces[0][a][b] == 0:
                    self.pieces[0][a][b] = char
                    chars_placed += 1   
        elif (self.n % 2 == 0):
            offset = int(self.n/2)
            if np.random.randint(0, 2) % 2 == 0:
                self.pieces[0][offset][offset]       = -1
                self.pieces[0][offset -1][offset -1] = -2
                self.pieces[0][offset][offset -1]    = +1
                self.pieces[0][offset -1][offset]    = +2
            else:
                self.pieces[0][offset][offset]       = +1
                self.pieces[0][offset -1][offset -1] = +2
                self.pieces[0][offset][offset -1]    = -1
                self.pieces[0][offset -1][offset]    = -2
        else: # self.n is odd
            boardCenter = int((self.n -1)/2)
            if np.random.randint(0, 2) % 2 == 0:
                self.pieces[0][boardCenter -1][boardCenter]    = -1
                self.pieces[0][boardCenter +1][boardCenter]    = -2
                self.pieces[0][boardCenter][boardCenter -1]    = +1
                self.pieces[0][boardCenter][boardCenter +1]    = +2
            else:
                self.pieces[0][boardCenter -1][boardCenter]    = +1
                self.pieces[0][boardCenter +1][boardCenter]    = +2
                self.pieces[0][boardCenter][boardCenter -1]    = -1
                self.pieces[0][boardCenter][boardCenter +1]    = -2


    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        """
        Currently unused.
        """
        return self.pieces[index]
                
    
    def getCharacterLocations(self, player):  
        """
        Returns a list of both character's locations as tuples for the player
        """
        
        color = player
    
        # Get all the squares with pieces of the given color.
        char1_location = np.where(self.pieces[0] == 1*color)
        char1_location = (char1_location[0][0], char1_location[1][0])

        char2_location = np.where(self.pieces[0] == 2*color)
        char2_location = (char2_location[0][0], char2_location[1][0])
        
        return [char1_location, char2_location]
    
    
    
    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        moves = []

        # Get all the squares with pieces of the given color.
        piece_locations = self.getCharacterLocations(color)
        for piece_location in piece_locations:
                    moves.extend(self.get_moves_for_location(piece_location)[0])
        return moves

    def get_all_moves(self, color):
        """Returns 3 np arrays:
            all the legal moves for the given color.
            all moves for a given color
            binary vector indicating which moves are legal for given color
        (1 for white, -1 for black
        """
        legal_moves = []
        all_moves = []
        all_moves_binary = []

        # Get all the squares with pieces of the given color.
        piece_locations = self.getCharacterLocations(color)

        for piece_location in piece_locations:
            moves, a_moves, a_moves_binary = self.get_moves_for_location(piece_location)
            legal_moves.extend(moves)
            all_moves.extend(a_moves)
            all_moves_binary.extend(a_moves_binary)

        return (legal_moves, all_moves, all_moves_binary)

    def get_legal_moves_binary(self, color):
        """Returns a binary vector of legal moves for the given color.
        (1 for white, -1 for black
        """    
        moves = []

        # Get all the squares with pieces of the given color.
        piece_locations = self.getCharacterLocations(color)
        for piece_location in piece_locations:
                    moves.extend(self.get_moves_for_location(piece_location)[2])

        return moves
    
    def get_moves_for_location(self, location):
        """
        Given a board location as an (x,y) tuple, this returns a tripple containing:
          1. an np.array of all legal actions in a vaugly human-readable format. 
              This is used by the game to perform actions.
          2. an np.array of all 128 possible actions, include those that are illegal.
          3. a binary np.array of which of the 128 actions are legal.
        
        Technically, we should be careful to discriminate between 'actions,' and 'moves,'
        as in Santorini, an action is comprised of a (move, build) pair. Players make
        actions, not moves. We are, unfortunately, not always that careful in description,
        and typically refer to actions as "moves" since the framework calls them moves,
        and in many games, where moves are atomic, there is no distinction.
        """
            
        x,y = location
        
        actions = []
        actions_all = []
        actions_all_bool = []

        # creating an intersection of possible move locations and the board.
        move_locations = np.copy(self.pieces[0][max(x-1,0):min(x+2,self.n), max(y-1,0):min(y+2,self.n)])
        unoccupied_locations = move_locations == 0

        location_height = self.pieces[1][location]
        height_diff = self.pieces[1][max(x-1,0):min(x+2,self.n), max(y-1,0):min(y+2,self.n)] - location_height

        # piecs can only change height by -2,-1,0 or 1 
        valid_height_diff = (height_diff <= 1)


        valid_move_locations = unoccupied_locations * valid_height_diff
        #print("valid_move_locations: ", move_locations)

        
        # Neither of these should be possible:
        assert(valid_move_locations.shape[0] != 1) #x is -1 or self.n
        assert(valid_move_locations.shape[1] != 1) #y is -1 or self.n
        
        if valid_move_locations.shape[0] == 2:
            assert((x == 0) or (x == self.n-1))
            valid_move_locations = np.insert(valid_move_locations, 2*(x != 0), False, axis=0) #insert at index 0 if x ==0, else at index 2
        if valid_move_locations.shape[1] == 2:
            assert((y == 0) or (y == self.n-1))
            valid_move_locations = np.insert(valid_move_locations, 2*(y != 0), False, axis=1) #insert at index 0 if y ==0, else at index 2

        assert(valid_move_locations.shape == (3,3))
        valid_moves = valid_move_locations.flatten()
        valid_moves = np.delete(valid_moves, 4) # remove False value corresponding to not moving anywhere
        
        directions = np.array(self.__directions)
        directions += [x,y]
        
        all_moves = directions.tolist()
        moves = directions[valid_moves]
        moves = moves.tolist()
        
#TODO: Combine the following two functions into one more efficient function.
        
        for move in moves:
            actions.extend(self.get_builds_for_location(move, [x-move[0],y-move[1]], location))

        for move in all_moves:
            all_builds, all_builds_bool = self.get_all_builds_for_location(move, [x-move[0],y-move[1]], location)
            all_builds_bool *= (move in moves)
            actions_all.extend(all_builds)
            actions_all_bool.extend(all_builds_bool)

        return np.array(actions), np.array(actions_all), np.array(actions_all_bool).astype(int)



    def get_builds_for_location(self, move, offset, original_location):
        """
        Function gets legal builds for a given move location.
        Input: move, the location that was moved TO. Assume we are already here
        for the purpose of this function. 
               offset, the (x,y) offset that returns us the to the original location.
               original_location: the location where we came from.
        Returns a list of actions of the form [original_location, move_location, build_location]
        """
        x,y = move
        x_offset,y_offset = offset

        build_locations = np.copy(self.pieces[0][max(x-1,0):min(x+2,self.n), max(y-1,0):min(y+2,self.n)])

        if(self.pieces[1][tuple(move)] == 3):
            # Piece moved to height 3, so the game ends. Since there will be 
            # no build afterwards, we set all build locations on the board as valid
            # to hopefully give the network an easier chance of picking one of 
            # the correct moves.
            valid_build_locations = build_locations
        else:
            
            unoccupied_locations = build_locations == 0
            
            board_heights = self.pieces[1][max(x-1,0):min(x+2,self.n), max(y-1,0):min(y+2,self.n)]
            valid_height_build = (board_heights <= 3)
    
            valid_build_locations = unoccupied_locations * valid_height_build

        assert(valid_build_locations.shape[0] != 1) #x is -1 or self.n
        assert(valid_build_locations.shape[1] != 1) #y is -1 or self.n

        if valid_build_locations.shape[0] == 2:
            assert((x == 0) or (x == self.n -1))
            valid_build_locations = np.insert(valid_build_locations, 2*(x != 0), False, axis=0)
        if valid_build_locations.shape[1] == 2:
            assert((y == 0) or (y == self.n -1))
            valid_build_locations = np.insert(valid_build_locations, 2*(y != 0), False, axis=1)


        # Where the piece just moved from. Must be empty
        valid_build_locations[1+x_offset, 1+y_offset] = True

        valid_builds = valid_build_locations.flatten()

        # remove False value corresponding to building where piece is standing
        valid_builds = np.delete(valid_builds, 4) 

        
        directions = np.array(self.__directions)
        directions += [x,y]                       # This gives us board locations
        
        builds = directions[valid_builds]
        builds = builds.tolist()
        builds = list(map(lambda x: [tuple(original_location), tuple(move), tuple(x)], builds))

        return builds

    def get_all_builds_for_location(self, move, offset, original_location):
        """
        Function gets ALL builds for a given move location, as well as a binary rep
        of whether they are valid.

        Input: move, the location that was moved TO. Assume we are already here
        for the purpose of this function. 
               offset, the (x,y) offset that returns us the to the original location.
               original_location: the location where we came from.
        Returns a list of actions of the form [original_location, move_location, build_location]
        and a binary vector of valid actions.
        """
        x,y = move #Location we moved to. Consider this our current location
        x_offset,y_offset = offset # offset to get back to original location

        # intersection of the board, and 1 space in every direction from current loc.
        build_locations = np.copy(self.pieces[0][max(x-1,0):min(x+2,self.n), max(y-1,0):min(y+2,self.n)])

        if(((0<=x<=self.n - 1) and (0<=y<=self.n - 1)) and self.pieces[1][tuple(move)] == 3):
            # Piece moved to height 3, so the game ends. Since there will be 
            # no build afterwards, we set all build locations on the board as valid
            # to hopefully give the network an easier chance of picking one of 
            # the correct moves.
            valid_build_locations = build_locations
        else:

            # Wherever there is an empty space:
            unoccupied_locations = (build_locations == 0)
            
            # Create build_loc analog, but for the board heights
            board_heights = self.pieces[1][max(x-1,0):min(x+2,self.n), max(y-1,0):min(y+2,self.n)]
            
            # Can only build on spaces with height 3 or fewer:
            valid_height_build = (board_heights <= 3)
    
            #Location must be unoccupied and have height <= 3
            valid_build_locations = unoccupied_locations * valid_height_build
        
        # Later we expect valid_build_locations to have shape (3,3) so we pad 
        #  if it is neccesary.
        # Here we consider the potentially that we may have moved off the board since
        # we are looking at every action, not just valid ones.
        if valid_build_locations.shape[0] == 1: #x is -1 or self.n
            if x < 0:
                valid_build_locations = np.insert(valid_build_locations, 0, False, axis=0)
                valid_build_locations = np.insert(valid_build_locations, 0, False, axis=0)
            else: # x is self.n
                valid_build_locations = np.insert(valid_build_locations, 1, False, axis=0)
                valid_build_locations = np.insert(valid_build_locations, 1, False, axis=0)
        
        if valid_build_locations.shape[1] == 1: #y is -1 or self.n
            if y < 0:
                valid_build_locations = np.insert(valid_build_locations, 0, False, axis=1)
                valid_build_locations = np.insert(valid_build_locations, 0, False, axis=1)
            else: #x is self.n
                valid_build_locations = np.insert(valid_build_locations, 1, False, axis=1)
                valid_build_locations = np.insert(valid_build_locations, 1, False, axis=1)

        if valid_build_locations.shape[0] == 2:
            valid_build_locations = np.insert(valid_build_locations, 2*(x != 0), False, axis=0)
        if valid_build_locations.shape[1] == 2:
            valid_build_locations = np.insert(valid_build_locations, 2*(y != 0), False, axis=1)

        # can always build where we were just standing:
        valid_build_locations[1+x_offset, 1+y_offset] = True

        valid_builds = valid_build_locations.flatten()

        valid_builds = np.delete(valid_builds, 4) # remove False value corresponding to not moving anywhere

        directions = np.array(self.__directions)
        directions += [x,y]
        
        all_builds = directions.tolist()
        #builds = directions[valid_builds]
        all_builds = list(map(lambda x: [tuple(original_location), tuple(move), tuple(x)], all_builds))

        # all builds is an (8,3,2) array of 8 lists (one for each direction),
        #each containing tuples: [orig_loc, move_loc, build_loc]
        
        # valid_builds contains a boolean (8,) array that corresponds
        #   to whether the builds in all_builds are legal
        return all_builds, valid_builds 





        
    def has_legal_moves(self, color):
        """
        Returns a boolean (whether player of given color has legal actions)
        """
        return (len(self.get_legal_moves(color)) > 0)

                
    def execute_move(self, move, color):
        """Perform the given move on the board; color gives the color of 
        the piece to play (1=white,-1=black). Assumes move is legal
        """
        assert(len(move) == 3)  # Move is list of 3 tuples:
        current_location = tuple(move[0])
        move_location = tuple(move[1])
        build_location = tuple(move[2])
        piece = self.pieces[0][current_location]
        
        try:
            self.pieces[0][current_location] = 0  # Remove piece from current square
        except IndexError as e:
            
            #self.pieces[0][current_location] = color
            #self.pieces[0][move_location] = 0
            print(e)
            print(self.pieces)
            print(current_location)
            print(move_location)
            print(build_location)
            print("IGNORING MOVE:")

        try:
            #self.pieces[0][current_location] = 0  # Remove piece from current square
            self.pieces[0][move_location] = piece # Add piece to new square
            #self.pieces[1][build_location] += 1   # Build one block on build location
        except IndexError as e:
            
            self.pieces[0][current_location] = piece
            #self.pieces[0][move_location] = 0
            print(e)
            print(self.pieces)
            print(current_location)
            print(move_location)
            print(build_location)
            print("IGNORING MOVE:")
            
        try:
            self.pieces[1][build_location] += 1   # Build one block on build location
        except IndexError as e:
            
            self.pieces[0][current_location] = piece
            self.pieces[0][move_location] = 0
            print(e)
            print(self.pieces)
            print(current_location)
            print(move_location)
            print(build_location)
            print("IGNORING MOVE:")    
            
