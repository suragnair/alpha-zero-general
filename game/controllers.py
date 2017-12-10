from __future__ import print_function
import datetime
import os
import queue
import threading
import sys
from game.ai import AlphaBetaPruner
from game.brain import Brain
from game.settings import *
from game.board import Board

__author__ = 'bengt'


class Controller(object):
    """ Interface for different types of controllers of the board
    """

    def next_move(self, pieces):
        """ Will return a single valid move as an (x, y) tuple.
        """
        pass

    def get_colour(self):
        """ Returns the colour of the controller.
        """
        pass


class PlayerController(Controller):
    """ Controller for a real, alive and kicking player.
    """

    def __init__(self, colour):
        self.colour = colour

    def next_move(self, board):
        """ Will return a single valid move as an (x, y) tuple.

            Processes input from the user, parses it, and then returns the
            chosen move if it is valid, otherwise the user can retry sending
            new input until successful.
        """


        result = None
        while result is None:
            event = input('Enter a coordinate, ex: c3, or Ctrl+D to quit: ')
            # if event[0] == '/':
            #     if event[1:] == 'quit' or event[1:] == 'q':
            #         print('Quitting. Thank you for playing.')
            #         exit()
            # else:
            try:
                if len(event) != 2:
                    raise ValueError
                x, y = event[0], event[1]
                result = self._parse_coordinates(x, y)
                found_moves = [p.get_position() for p in board.get_move_pieces(self.get_colour())]

                if not found_moves:
                    raise NoMovesError

                if result not in found_moves:
                    raise TypeError

            except (TypeError, ValueError):
                result = None
                print("Invalid coordinates, retry.")

        return result

    def get_colour(self):
        """ Returns the colour of the controller.
        """
        return self.colour

    def __str__(self):
        return "Player"

    def __repr__(self):
        return "PlayerController"

    @staticmethod
    def _parse_coordinates(x, y):
        """ Parses board coordinates into (x, y) coordinates.
        """
        return ord(x) - ord('a'), ord(y) - ord('0') - 1


stdoutmutex = threading.Lock()
workQueue = queue.Queue(1)
threads = []


class AiController(Controller):
    """ Artificial Intelligence Controller.
    """

    def __init__(self, id, colour, duration):
        self.id = id
        self.colour = colour
        self.duration = duration

    def next_move(self, board):
        """ Will return a single valid move as an (x, y) tuple.

            Will create a new Brain to start a Minimax calculation with
            the Alpha-Beta Pruning optimization to find optimal moves based
            on an evaluation function, in another thread.

            Meanwhile the AiController will output to stdout to show
            that it hasn't crashed.
        """
        size = board.shape[0]
        # print(size)
        new_board = Board(size,'RED')
        pieces = new_board.getBoardPieces()
        for i in range(size):
            for j in range(size):
                # print(board[i][j])
                if board[i][j] == -1:
                    new_board.set_black(j,i)
                elif board[i][j] == +1:
                    new_board.set_white(j,i)

        # raw_input()
        board = new_board
        # print(board.draw())
        # raw_input()
        brain = Brain(self.duration, stdoutmutex, workQueue, board.pieces, self.colour,
                      BLACK if self.colour is WHITE else WHITE)
        brain.start()

        threads.append(brain)

        # print('Brain is thinking ', end='')
        update_step_duration = datetime.timedelta(microseconds=10000)
        goal_time = datetime.datetime.now() + update_step_duration
        accumulated_time = datetime.datetime.now()

        while workQueue.empty():
            if accumulated_time >= goal_time:
                # print('.', end='')
                goal_time = datetime.datetime.now() + update_step_duration
                sys.stdout.flush()

            accumulated_time = datetime.datetime.now()

        # print()

        for thread in threads:
            thread.join()

        try:
            (i,j) = workQueue.get()
            # print(i,j)
            return i+size*j
        except:
            return size**2
       
        # raw_input()
        

    def get_colour(self):
        """ Returns the colour of the controller.
        """
        return self.colour

    def __str__(self):
        return "Ai"

    def __repr__(self):
        return "AiController[" + self.id + "]"
