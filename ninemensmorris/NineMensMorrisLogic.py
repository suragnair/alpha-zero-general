'''
Author: Jonas Jakob
Created: May 31, 2023

Implementation of the NineMensMorris Game Logic
'''
import numpy as np

class Board():

    """
    A Ninemensmorris Board is represented as a array of (25)
    The item on board[24] represents the placing phase. "0" if
    the phase is not over yet, "1" if it is.

    Board logic:

    The pieces are represented as
    - 1 for player one (black), 1 for player 2 (white) and 0 if there is no
    piece on the position (for the canonical Board the
    current players pieces are always shown as 1 and the
    opponents as -1). The initial board:

        board shape:
        [0,0,0,0,0,0,0,0,    -> outer ring
        0,0,0,0,0,0,0,0,     -> middle ring
        0,0,0,0,0,0,0,0]     -> inner ring



    Locations:

    Locations are given as the index in the board array.

    Actions:

    Actions are stored in a list of tuples of the form:
        action = [piece_location, move_location, remove_piece]
    """

    """
    6x6 configuration
    24 spots for pieces
    1 spot to count the placed pieces
    1 spot to count the current moves without mills

    -> need to be in the board itself, since only the board is
    """
    def __init__(self):
        "Set up initial board configuration."
        self.n = 6
        self.pieces = np.zeros((6,6), dtype=int)

    """
    currently not used
    """
    def __getitem__(self, index):
      return self.pieces[index]


    """
    returns a vector of ones and zeros, marking all the legal moves for the
    current board state
    """
    def get_legal_move_vector(self, player, all_moves):
        """
        Input:
            player: current player (1 or -1)
            all_moves: list with all possible moves

        Returns:
            legal_move_vector: vector of length = all_moves with ones and zeros
        """
        legal_moves = self.get_legal_moves(player)
        legal_move_vector = [0] * len(all_moves)

        for move in legal_moves:
          index = all_moves.index(move)
          legal_move_vector[index] = 1
        return legal_move_vector

    """
    Transforms the array form of the NineMensMorris board into a Image, that
    can be used as Input for the Neural Network
    """
    def arrayToImage(self, array, placements_and_moves):
        """
        Input:
            array: list with all 24 board positions
            placements_and_moves: Tuple containing the placed pieces in phase
            zero and the current number of moves without a mill

        Returns:
            legal_move_vector: vector of length = all_moves with ones and zeros
        """
        board_image = np.zeros((6,6), dtype=int)
        boardx = 0
        boardy = 0
        count_placements, current_moves = placements_and_moves
        assert(len(array) == 24)
        assert(0 <= count_placements <= 18)
        index = 0
        while index < 24:

          board_image[boardx][boardy] = np.copy(array[index])
          if boardy == 5:
            boardx += 1
            boardy = 0
          else:
            boardy += 1
          index += 1


        board_image[4][0] = count_placements
        board_image[4][1] = current_moves
        assert(0 <= board_image[4][0] <= 18)

        return board_image

    """
    Transforms the Image form used in the training of the Neural Network into an
    Array of the board and a Tuple containing the placed pieces in phase zero
    and the current number of moves without a mill.
    """
    def piecesToArray(self):
        """
        Returns:
            re_board: list with all 24 board positions
            placements_and_moves: Tuple containing the placed pieces in phase
            zero and the current number of moves without a mill
        """
        re_board = []
        re_board.extend(self.pieces[0])
        re_board.extend(self.pieces[1])
        re_board.extend(self.pieces[2])
        re_board.extend(self.pieces[3])


        assert(0 <= self.pieces[4][0] <= 18)
        assert(len(re_board) == 24)
        placements_and_moves = (self.pieces[4][0], self.pieces[4][1])

        return (re_board, placements_and_moves)

    """
    Gets the current game phase for the current player, then calls the
    right method to retrieve the legal moves for the specific game phase, board
    and player. Returns a list
    """
    def get_legal_moves(self, player):
        """
        Input:
            player: current player (1 or -1)

        Returns:
            legal_move_vector: list with all the move Tuples that are legal for
            the current board state
        """
        game_phase = self.get_game_phase(player)
        assert(0 <= game_phase <= 2)
        if game_phase == 0:
            return list(self.get_legal_moves_0(player))

        elif game_phase == 1:
            return list(self.get_legal_moves_1(player))
        elif game_phase == 2:
            return list(self.get_legal_moves_2(player))

    """
    Gets the current game phase for the current player and board
    """
    def get_game_phase(self, player):
        """
        Input:
            player: current player (1 or -1)

        Returns:
            number: number representing the game phase
        """

        array, placements_and_moves = self.piecesToArray()
        assert(0 <= placements_and_moves[0] <= 18)

        if placements_and_moves[0] < 18:
            return 0
        elif len(self.get_player_pieces(player)) <= 3:
            return 2
        else:
            return 1

    """
    Gets all positions for the given players pieces in the array form of
    the board
    """
    def get_player_pieces(self, player):
        """
        Input:
            player: current player (1 or -1)

        Returns:
            locations: list of the locations for all the pieces of the given player
        """
        board, placements = self.piecesToArray()
        locations = []

        index = 0
        while index < len(board):
            if board[index] == player:
                locations.append(index)
            index += 1
        if locations == []:
          return []
        else:
          return list(locations)

    """
    Gets all the positions on the board that are empty
    """
    def get_empty_positions(self):
        """
        Returns:
            locations: list of all empty positions
        """
        board, placements = self.piecesToArray()
        assert(0 <= placements[0] <= 18)
        assert(len(board) == 24)

        locations = []

        index = 0
        while index < len(board):
            if board[index] == 0:
                locations.append(index)
            index += 1

        return list(locations)

    """
    Checks for each possible move, if a new mill is formed.
    Each check makes sure, that the origin of the move, isnt one of the pieces in the
    potentially new mill.
    Returns a list of all move Tuples that form a new mill.
    """
    def get_possible_mills(self, move_locations, player):
        """
        Input:
            move_locations: list of Tuples with (origin, destination)
            player: current player (1 or -1)

        Returns:
            number: list of all moves that form a mill on the board
        """
        board, placements = self.piecesToArray()
        assert(0 <= placements[0] <= 18)
        assert(len(board) == 24)
        move_forms_mill = []

        for move in move_locations:
            if (move != None) and (move[1] < 24) and (move[1] >= 0) :
                if (move[1] % 2) == 0: #move is in a corner
                    if (move[1] % 8) == 0: # move is in the top left corner of a ring
                        if (([move[1] + 7] == player) and (board[move[1] + 6] == player) and
                           (move[1] + 7 != move[0]) and (move[1] + 6 != move[0])): #check down
                            move_forms_mill.append(move)
                        if ((board[move[1] + 1] == player) and (board[move[1] + 2] == player) and
                           (move[1] + 1 != move[0]) and (move[1] + 2 != move[0])): #check right
                            move_forms_mill.append(move)
                    elif move in [6,14,22]: #move is in the bottom left corner of a ring
                        if ((board[move[1] + 1] == player) and (board[move[1] - 6] == player) and
                           (move[1] + 1 != move[0])and (move[1] - 6 != move[0])): #check up
                            move_forms_mill.append(move)
                        if ((board[move[1] - 1] == player) and (board[move[1] - 2] == player) and
                           (move[1] - 1 != move[0]) and (move[1] - 2 != move[0])): #check right
                            move_forms_mill.append(move)
                    elif move in [2,10,18,4,12,20]: #move is in the bottom or top right corner of a ring
                        if ((board[move[1] + 1] == player) and (board[move[1] + 2] == player) and
                           (move[1] + 1 != move[0]) and (move[1] + 2 != move[0])): #check down/ left
                            move_forms_mill.append(move)
                        if ((board[move[1] - 1] == player) and (board[move[1] - 2] == player) and
                           (move[1] - 1 != move[0]) and (move[1] - 2 != move[0])): #check left/ up
                            move_forms_mill.append(move)

                else: #move is in the middle of a row
                    if move[1] in [1,3,5,7]: #outer ring
                        if move[1] == 7:
                            if ((board[move[1] - 7] == player) and (board[move[1] - 1] == player) and
                               (move[1] - 7 != move[0]) and (move[1] - 1 != move[0])): #check ring
                                move_forms_mill.append(move)
                        else:
                            if ((board[move[1] - 1] == player) and (board[move[1] + 1] == player) and
                               (move[1] - 1 != move[0]) and (move[1] + 1 != move[0])): #check ring
                                move_forms_mill.append(move)
                        if ((board[move[1] + 8] == player) and (board[move[1] + 16] == player) and
                           (move[1] + 8 != move[0]) and (move[1] + 16 != move[0])): #check intersections
                                move_forms_mill.append(move)

                    elif move[1] in [9,11,13,15]: #middle ring
                        if move[1] == 15:
                            if ((board[move[1] - 7] == player) and (board[move[1] - 1] == player) and
                               (move[1] - 7 != move[0]) and (move[1] - 1 != move[0])): #check ring
                                move_forms_mill.append(move)
                        else:
                            if ((board[move[1] - 1] == player) and (board[move[1] + 1] == player) and
                               (move[1] - 1 != move[0]) and (move[1] + 1 != move[0])): #check ring
                                move_forms_mill.append(move)
                        if ((board[move[1] + 8] == player) and (board[move[1] - 8] == player) and
                           (move[1] + 8 != move[0]) and (move[1] - 8 != move[0])): #check intersections
                                move_forms_mill.append(move)

                    elif move[1] in [17,19,21,23]: #inner ring
                        if move[1] == 23:
                            if ((board[move[1] - 7] == player) and (board[move[1] - 1] == player) and
                               (move[1] - 7 != move[0]) and (move[1] - 1 != move[0])): #check ring
                                move_forms_mill.append(move)
                        else:
                            if ((board[move[1] - 1] == player) and (board[move[1] + 1] == player) and
                               (move[1] - 1 != move[0]) and (move[1] + 1 != move[0])): #check ring
                                move_forms_mill.append(move)
                        if ((board[move[1] - 8] == player) and (board[move[1] - 16] == player) and
                           (move[1] - 8 != move[0]) and (move[1] - 16 != move[0])): #check intersections
                                move_forms_mill.append(move)

        return list(move_forms_mill)

    """
    Looks at the board and returns all current mills for a given player,
    in tuples of their coordinates
    """
    def check_for_mills(self, player):
        """
        Input:
            player: current player (1 or -1)

        Returns:
            current_mills: all mills for the current player
        """

        current_mills = []
        board, placements = self.piecesToArray()
        assert(0 <= placements[0] <= 18)
        assert(len(board) == 24)

        index = 0

        while index < 23: #check rings
            if (index in [6,14,22]):
              if (board[index] == board[index + 1] == board[index - 6] == player):
                current_mills.append((index, index + 1, index - 6))
            elif (board[index] == board[index + 1] == board[index + 2] == player):
              current_mills.append((index, index + 1, index + 2))

            index += 2

        index = 1

        while index < 8: #check intersections
            if (board[index] == board[index + 8] == board[index + 16] == player):
              current_mills.append((index, index + 8, index + 16))

            index += 2

        return list(current_mills)

    """
    Gets all neighbour postions for a position on the board
    """
    def get_neighbours(self, position):
        """
        Input:
            position: postion index on the board

        Returns:
            neighbours: Tuple of all neighbours
        """
        assert(0 <= position <= 23)
        if position != None:
                if (position % 2) == 0: #position is in a corner

                    if (position % 8) == 0: # position is in the top left corner of a ring
                        return (position + 1, position + 7)

                    else: #position is in top right, or bottom corners
                        return (position - 1, position + 1)

                else: #position is in a intersection
                    if position in [1,3,5,7]: #outer ring
                        if position == 7:
                            return (0, 6, 15)
                        else:
                            return (position - 1, position + 1, position + 8)


                    elif position in [9,11,13,15]: #middle ring
                        if position == 15:
                            return (7, 8, 14, 23)
                        else:
                            return (position - 8, position - 1, position + 1, position + 8)

                    elif position in [17,19,21,23]: #outer ring
                        if position == 23:
                            return (15, 16, 22)
                        else:
                            return (position - 8, position - 1, position + 1)


        return

    """
    Gets all pieces that are outside of mills for the given player and the
    current board
    """
    def get_pieces_outside_mills(self, player):
        """
        Input:
            player: current player (1 or -1)

        Returns:
            pieces: all pieces for the given player outside of mills
        """
        all_pieces = self.get_player_pieces(player)

        mills = self.check_for_mills(player)

        remaining_pieces = self.get_player_pieces(player)

        for piece in all_pieces:
            if len(mills) != 0:
                for mill in mills:
                    if piece in mill and piece in remaining_pieces:
                        remaining_pieces.remove(piece)


        return list(remaining_pieces)

    """
    Looks at the board, given the current player and identifies all
    legal moves for the current gamestate, given that the player is
    in Phase 0
    """
    def get_legal_moves_0(self, player):
        """
        Input:
            player: current player (1 or -1)

        Returns:
            moves: list of move tuples that are legal for the given player,
            the players game phase and the current board
        """
        #get enemy pieces that can be taken if a mill is formed
        enemies_outside_mills = self.get_pieces_outside_mills(-player)
        if len(enemies_outside_mills) > 0:
            enemies_to_take = enemies_outside_mills
        else:
            enemies_to_take = self.get_player_pieces(-player)


        #get empty positions, they represent all possible move locations for phase zero
        empty_locations = []
        for position in self.get_empty_positions():
            empty_locations.append(('none',position))

        #get moves -> for each move_location, check if a mill is formed (check row(s))
        mill_moves = self.get_possible_mills(empty_locations, player)


        #generate action tuples
        moves = []

        for move in empty_locations:
            if move in mill_moves:
                for enemy in enemies_to_take:
                    moves.append(('none',move[1],enemy))
            else:
                moves.append(('none',move[1],'none'))


        return list(moves)


    """
    Looks at the board, given the current player and identifies all
    legal moves for the current gamestate, given that the player is
    in Phase 1
    """
    def get_legal_moves_1(self, player):
        """
        Input:
            player: current player (1 or -1)

        Returns:
            moves: list of move tuples that are legal for the given player,
            the players game phase and the current board
        """
        moves = []
        board, placements = self.piecesToArray()
        assert(placements[0] == 18)
        assert(len(board) == 24)

        #get enemy pieces that can be taken if a mill is formed
        enemies_outside_mills = self.get_pieces_outside_mills(-player)
        if len(enemies_outside_mills) > 0:
            enemies_to_take = enemies_outside_mills
        else:
            enemies_to_take = self.get_player_pieces(-player)

        #get the current players pieces that will be moved
        current_positions = self.get_player_pieces(player)

        #creating the first part of the moves
        part_moves = []

        for position in current_positions:
            neighbours = self.get_neighbours(position)
            index = 0
            while index < len(neighbours):
                if board[neighbours[index]] == 0:
                    part_moves.append((position, neighbours[index]))
                index += 1

        #finding the part moves that create mills, then pairing them accordingly with enemy pieces to beat
        #get moves -> for each move_location, check if a mill is formed (check row(s))
        mill_moves = self.get_possible_mills(part_moves, player)

        for move in part_moves:
            if move in mill_moves:
                for enemy in enemies_to_take:
                    moves.append((move[0],move[1],enemy))
            else:
                moves.append((move[0],move[1],'none'))



        return list(moves)


    """
    Looks at the board, given the current player and identifies all
    legal moves for the current gamestate, given that the player is
    in Phase 2
    """
    def get_legal_moves_2(self, player):
        """
        Input:
            player: current player (1 or -1)

        Returns:
            moves: list of move tuples that are legal for the given player,
            the players game phase and the current board
        """
        moves = []

        #get enemy pieces that can be taken if a mill is formed
        enemies_outside_mills = self.get_pieces_outside_mills(-player)
        if len(enemies_outside_mills) > 0:
            enemies_to_take = enemies_outside_mills
        else:
            enemies_to_take = self.get_player_pieces(-player)

        #get the current players pieces that will be moved
        current_positions = self.get_player_pieces(player)

        #creating the first part of the moves
        part_moves = []

        empty_locations = self.get_empty_positions()

        #pair the locations of current positions with all empty locations on the board
        for position in current_positions:
            for location in empty_locations:
                part_moves.append((position, location))

        #finding the part moves that create mills, then pairing them accordingly with enemy pieces to beat
        #get moves -> for each move_location, check if a mill is formed (check row(s))
        mill_moves = self.get_possible_mills(part_moves, player)

        for move in part_moves:
            if move in mill_moves:
                for enemy in enemies_to_take:
                    moves.append((move[0],move[1],enemy))
            else:
                moves.append((move[0],move[1],'none'))

        return list(moves)

    """
    checks if the given player has any legal moves on the current board
    """
    def has_legal_moves(self, player):
        """
        Returns:
            Boolean: has legal moves
        """
        if (len(self.get_legal_moves(player)) > 0):
            return True
        return False

    '''
    Rotates the board three times, each time creating a pair of the rotated
    board and the rotated vector of legal moves.
    Uses a shift vector for the board to calculate the new position for each
    index in the array and a lookup list for the vector of legal moves.
    '''
    def get_board_rotations(self, pi, all_moves, policy_rotation_vector):
        """
        Input:
            pi: the legal move vector
            all_moves: list with all legal moves
            policy_rotation_vector: lookup list for the vector of legal moves

        Returns:
            rotated_results: list of Tuples (image, legal_moves)
        """
        #vector to rotate the board 90 degrees -> move each ring by two positions
        rot90_vector = [2,2,2,2,2,2,-6,-6,2,2,2,2,2,2,-6,-6,2,2,2,2,2,2,-6,-6]

        old_board, placements = self.piecesToArray()
        new_board = np.zeros((24), dtype = int)
        new_pi = np.zeros((len(all_moves)), dtype = int)

        rotated_results = []

        #rotates the board 3 times
        for i in range(3):
            index = 0
            while index < 24:
                new_board[index+rot90_vector[index]]= np.copy(old_board[index])
                index+=1

            index = 0
            while index < len(all_moves):
                new_pi[policy_rotation_vector[index]] = np.copy(pi[index])
                index += 1

            rotated_results += [(self.arrayToImage(new_board, placements),new_pi)]
            #print("rotating")
            #print(old_board)
            old_board = np.copy(new_board)
            #print(new_board)
            pi = np.copy(new_pi)

            i+=1

        return rotated_results


    """
    Exectues a move on the current board for the given player
    """
    def execute_move(self, player, move_index, all_moves):
        """
        Input:
            player: the legal move vector
            move_index: index for the move in the all_moves list
            all_moves: list with all legal moves
        """
        move = all_moves[move_index]
        assert(len(move)==3) #move is a tuple of length 3
        board, placements = self.piecesToArray()
        assert(0 <= placements[0] <= 18)
        assert(len(board) == 24)

        count_placements, current_moves = placements
        if self.get_game_phase(player) == 0:
          count_placements += 1
        if move[0] != 'none':
          board[move[0]] = 0
        if move[2] != 'none':
          board[move[2]] = 0
          current_moves = 0
        elif move[2] == 'none':
          current_moves += 1
        board[move[1]] = player
        if current_moves > 50:
          print(current_moves)

        placements = (count_placements, current_moves)

        image = self.arrayToImage(board, placements)
        self.pieces = np.copy(image)





