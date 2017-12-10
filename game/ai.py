from __future__ import print_function
import datetime
import sys

__author__ = 'bengt'

from game.settings import *


class AlphaBetaPruner(object):
    """Alpha-Beta Pruning algorithm."""

    def __init__(self, mutex, duration, pieces, first_player, second_player):
        self.mutex = mutex
        self.board = 0
        self.move = 1
        self.white = 2
        self.black = 3
        self.duration = duration
        self.lifetime = None
        self.infinity = 1.0e400
        self.first_player, self.second_player = (self.white, self.black) \
            if first_player == WHITE else (self.black, self.white)
        self.state = self.make_state(pieces)

    def make_state(self, pieces):
        """ Returns a tuple in the form of "current_state", that is: (current_player, state).
        """
        results = {BOARD: self.board, MOVE: self.board, WHITE: self.white, BLACK: self.black}
        return self.first_player, [results[p.get_state()] for p in pieces]

    def alpha_beta_search(self):
        """ Returns a valid action for the AI.
        """
        self.lifetime = datetime.datetime.now() + datetime.timedelta(seconds=self.duration)
        depth = 0
        fn = lambda action: self.min_value(depth, self.next_state(self.state, action), -self.infinity,
                                           self.infinity)
        maxfn = lambda value: value[0]
        actions = self.actions(self.state)
        moves = [(fn(action), action) for action in actions]

        if len(moves) == 0:
            return 0,WIDTH

        return max(moves, key=maxfn)[1]

    def max_value(self, depth, current_state, alpha, beta):
        """ Calculates the best possible move for the AI.
        """
        if self.cutoff_test(current_state, depth):
            return self.evaluation(current_state, self.first_player)

        value = -self.infinity

        actions = self.actions(current_state)
        for action in actions:
            value = max([value, self.min_value(depth + 1, self.next_state(current_state, action), alpha, beta)])
            if value >= beta:
                return value
            alpha = max(alpha, value)

        return value

    def min_value(self, depth, state, alpha, beta):
        """ Calculates the best possible move for the player.
        """
        if self.cutoff_test(state, depth):
            return self.evaluation(state, self.second_player)

        value = self.infinity

        actions = self.actions(state)
        for action in actions:
            value = min([value, self.max_value(depth + 1, self.next_state(state, action), alpha, beta)])
            if value <= alpha:
                return value
            beta = min([beta, value])

        return value

    def evaluation(self, current_state, player_to_check):
        """ Returns a positive value when the player wins.
            Returns zero when there is a draw.
            Returns a negative value when the opponent wins."""

        player_state, state = current_state
        player = player_to_check
        opponent = self.opponent(player)

        # count_eval stands for the player with the most pieces next turn
        moves = self.get_moves(player, opponent, state)
        player_pieces = len([p for p in state if p == player])
        opponent_pieces = len([p for p in state if p == opponent])
        count_eval = 1 if player_pieces > opponent_pieces else \
            0 if player_pieces == opponent_pieces else \
                -1

        # moves_player    = moves
        # moves_oppponent = self.get_moves(opponent, player, state)
        # move_eval       = 1 if moves_player > moves_oppponent else \
        #                   0 if moves_player == moves_oppponent else \
        #                  -1

        corners_player = (state[0] == player) + \
                         (state[7] == player) + \
                         (state[56] == player) + \
                         (state[63] == player)
        corners_opponent = -1 * (state[0] == opponent) + \
                           (state[7] == opponent) + \
                           (state[56] == opponent) + \
                           (state[63] == opponent)
        corners_eval = corners_player + corners_opponent

        edges_player = len([x for x in state if state == player and (state % WIDTH == 0 or state % HEIGHT == WIDTH)]) / (
            WIDTH * HEIGHT)
        edges_opponent = -1 * len([x for x in state if state == opponent and (state % WIDTH == 0 or state % WIDTH == WIDTH)]) / (
            WIDTH * HEIGHT)
        edges_eval = edges_player + edges_opponent

        eval = count_eval * 2 + corners_eval * 1.5 + edges_eval * 1.2

        return eval

    def actions(self, current_state):
        """ Returns a list of tuples as coordinates for the valid moves for the current player.
        """
        player, state = current_state
        return self.get_moves(player, self.opponent(player), state)

    def opponent(self, player):
        """ Returns the opponent of the specified player.
        """
        return self.second_player if player is self.first_player else self.first_player

    def next_state(self, current_state, action):
        """ Returns the next state in the form of a "current_state" tuple, (current_player, state).
        """
        player, state = current_state
        opponent = self.opponent(player)

        xx, yy = action
        state[xx + (yy * WIDTH)] = player
        for d in DIRECTIONS:
            tile = xx + (yy * WIDTH) + d
            if tile < 0 or tile >= WIDTH*WIDTH:
                continue

            while state[tile] != self.board:
                state[tile] = player
                tile += d
                if tile < 0 or tile >= WIDTH * HEIGHT:
                    tile -= d
                    break

        return opponent, state

    def get_moves(self, player, opponent, state):
        """ Returns a generator of (x,y) coordinates.
        """
        moves = [self.mark_move(player, opponent, tile, state, d)
                 for tile in range(WIDTH * HEIGHT)
                 for d in DIRECTIONS
                 if not outside_board(tile, d) and state[tile] == player]

        return [(x, y) for found, x, y, tile in moves if found]


    def mark_move(self, player, opponent, tile, pieces, direction):
        """ Returns True whether the current tile piece is a move for the current player,
            otherwise it returns False.
        """
        if not self.valid(tile, direction):
            tile += direction
        else:
            return False, int(tile % WIDTH), int(tile / HEIGHT), tile

        if pieces[tile] == opponent:
            while pieces[tile] == opponent:
                if self.valid(tile, direction):
                    break
                else:
                    tile += direction
                    #print(tile)

            if pieces[tile] == self.board:
                return True, int(tile % WIDTH), int(tile / HEIGHT), tile

        return False, int(tile % WIDTH), int(tile / HEIGHT), tile

    def cutoff_test(self, state, depth):
        """ Returns True when the cutoff limit has been reached.
        """
        return depth > 1000 or datetime.datetime.now() > self.lifetime
    def valid(self, tile, direction):
        #print(tile, direction)
        return (direction in (NORTHWEST, NORTH, NORTHEAST) and 0 <= tile <= WIDTH-1) or \
           (direction in (SOUTHWEST, SOUTH, SOUTHEAST) and WIDTH*(WIDTH-1) <= tile <= WIDTH*WIDTH-1) or \
           (direction in (NORTHEAST, EAST, SOUTHEAST) and tile % WIDTH == WIDTH-1) or \
           (direction in (NORTHWEST, WEST, SOUTHWEST) and tile % WIDTH == 0)


