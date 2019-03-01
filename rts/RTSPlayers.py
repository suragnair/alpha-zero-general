import ctypes
import os
import sys
from math import sqrt
from typing import List

import numpy as np
import pygame
from pygame.rect import Rect

sys.path.append('..')
from rts.src.config import NUM_ACTS, P_NAME_IDX, A_TYPE_IDX, d_user_shortcuts, FPS, ACTS, d_a_type, ACTS_REV, d_user_shortcuts_rev
from rts.visualization.rts_pygame import init_visuals, update_graphics, message_display
from utils import dotdict

"""
RTSPlayers.py

Contains 3 players (human player, random player, greedy player (if searching for nnet player, it is defined by pre-learnt model)
Human player has defined input controls for Pygame and console
"""


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanRTSPlayer:
    def __init__(self, game) -> None:
        self.game = game
        self.USER_PLAYER = 1  # used by Human Player - this does not change if human pit player is 1 or -1

    def play(self, board: np.ndarray) -> List:
        """
        Manages input using PyGame canvas/ console input
        :param board: current board
        :return: action to execute on current board
        """
        from rts.src.config_class import CONFIG

        n = board.shape[0]
        valid = self.game.getValidMoves(board, 1)
        self.display_valid_moves(board, valid)
        while True:

            if CONFIG.visibility > 3:
                a = self._manage_input(board)
                x, y, action_index = a

            else:
                a = (input('type one of above actions in "x y action_index" format\n')).split(" ")
                x, y, action = a
                action_index = ACTS[action]
            # convert to action index in valids array

            try:

                tup = (int(y), int(x), int(action_index))
                a = np.ravel_multi_index(tup, (n, n, NUM_ACTS))
            except Exception as e:
                print("Could not parse action")
            if valid[a]:
                break
            else:
                print('This action is invalid!')
                self.display_valid_moves(board, valid)

        return a

    def display_valid_moves(self, board, valid) -> None:
        """
        Displays all valid moves in console for specific board
        :param board: board to display moves upon
        :param valid: vector of valid moves
        """
        if valid is None:
            valid = self.game.getValidMoves(board, 1)
        n = board.shape[0]
        print("----------")
        for i in range(len(valid)):
            if valid[i]:
                y, x, action_index = np.unravel_index(i, [n, n, NUM_ACTS])
                print(x, y, ACTS_REV[action_index])
                print("----------")

    @staticmethod
    def select_object(board: np.ndarray, click_location: tuple) -> dotdict:
        """
        Selects object on PyGame canvas using mouse click
        :param board: game state board
        :param click_location: tuple (x,y) that represents canvas click location
        :return: game tile coordinate (x,y)
        """
        n = board.shape[0]
        canvas_scale = int(ctypes.windll.user32.GetSystemMetrics(1) * (16 / 30) / n)  # for drawing - it takes 2 thirds of screen height

        # select object by clicking on it - you can select only your objects

        for y in range(n):
            for x in range(n):
                actor_location = (int(x * canvas_scale + canvas_scale / 2 + canvas_scale), int(y * canvas_scale + canvas_scale / 2) + canvas_scale)
                actor_x, actor_y = actor_location
                actor_size = int(canvas_scale / 3)

                click_x, click_y = click_location

                dist = sqrt((actor_x - click_x) ** 2 + (actor_y - click_y) ** 2)
                if dist <= actor_size:
                    return dotdict({"x": x, "y": y})
        return dotdict({"x": -1, "y": -1})

    def _manage_input(self, board: np.ndarray) -> list:
        """
        Manages click and keyboard selections on PyGame canvas
        :param board: game state
        :return: /
        """
        from rts.src.Board import Board
        from rts.src.config_class import CONFIG

        n = board.shape[0]

        game_display, clock = init_visuals(n, n, CONFIG.visibility)
        update_graphics(board, game_display, clock, FPS)

        canvas_scale: int = int(ctypes.windll.user32.GetSystemMetrics(1) * (16 / 30) / n)
        clicked_actor = None
        clicked_actor_index_arr = []
        while True:
            for event in pygame.event.get():
                # print(event)
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit(0)
                if event.type == pygame.KEYDOWN:

                    if clicked_actor and (board[clicked_actor.x][clicked_actor.y][P_NAME_IDX] == self.USER_PLAYER):
                        try:

                            shortcut_pressed = d_user_shortcuts[event.unicode]
                            action_to_execute = shortcut_pressed
                            clicked_actor_index_arr.append(action_to_execute)
                            return clicked_actor_index_arr
                        except Exception as e:
                            print("shortcut '" + event.unicode + "' not supported.")

                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        os._exit(0)

                # handle mouse
                if event.type == pygame.MOUSEBUTTONUP:
                    lmb, rmb = 1, 3
                    pos = pygame.mouse.get_pos()

                    if event.button == lmb:
                        clicked_actor = self.select_object(board, pos)
                        if clicked_actor and board[clicked_actor.x][clicked_actor.y][P_NAME_IDX] == self.USER_PLAYER and board[clicked_actor.x][clicked_actor.y][A_TYPE_IDX] != d_a_type['Gold']:
                            clicked_actor_index_arr = [clicked_actor.x, clicked_actor.y]

                            # draw selected bounding box
                            game_display, clock = init_visuals(n, n, CONFIG.visibility)
                            update_graphics(board, game_display, clock, FPS)

                            actor_size = int(canvas_scale / 3)
                            actor_location = (int(clicked_actor.x * canvas_scale + canvas_scale / 2 + canvas_scale - actor_size), int(clicked_actor.y * canvas_scale + canvas_scale / 2 + canvas_scale - actor_size))
                            rect = Rect(actor_location, (2 * actor_size, 2 * actor_size))

                            blue = (0, 0, 255)
                            pygame.draw.rect(game_display, blue, rect, int(canvas_scale / 20))

                            # display valid actions on canvas
                            b = Board(n)
                            b.pieces = np.copy(board)

                            if self.USER_PLAYER == 1:
                                config = CONFIG.player1_config
                            else:
                                config = CONFIG.player2_config
                            valids_square = b.get_moves_for_square(clicked_actor.x, clicked_actor.y, config=config)

                            printed_actions = 0
                            for i in range(len(valids_square)):
                                if valids_square[i]:
                                    text_scale = int(actor_size * 0.5)
                                    message_display(game_display, u"" + ACTS_REV[i] + " s: '" + d_user_shortcuts_rev[i] + "'", (3 * canvas_scale + int(printed_actions % 3) * canvas_scale * 2, (n + 1) * canvas_scale + text_scale / 2 + int(printed_actions / 3) * text_scale + int(text_scale / 4)),
                                                    text_scale)
                                    printed_actions += 1
                            # update display
                            pygame.display.update()

                        else:
                            print("You can select only your actors!")
                    if event.button == rmb:
                        if clicked_actor:

                            l_x = clicked_actor.x
                            l_y = clicked_actor.y
                            l_type = board[l_x][l_y][A_TYPE_IDX]

                            right_clicked_actor = self.select_object(board, pos)

                            # right clicked actor exists and (if player 1 or player -1) and not clicked self
                            if right_clicked_actor and board[right_clicked_actor.x][right_clicked_actor.y][P_NAME_IDX] != 0 and right_clicked_actor != clicked_actor:
                                r_x = right_clicked_actor.x
                                r_y = right_clicked_actor.y
                                r_type = board[r_x][r_y][A_TYPE_IDX]
                                r_player = board[r_x][r_y][P_NAME_IDX]

                                # this is actor of type MyActor
                                if l_type == d_a_type['Work']:
                                    if r_player == self.USER_PLAYER:
                                        if r_type == d_a_type['Gold']:
                                            clicked_actor_index_arr.append(ACTS["mine_resources"])
                                        if r_type == d_a_type['Hall']:
                                            clicked_actor_index_arr.append(ACTS["return_resources"])

                                if l_type == d_a_type['Rifl']:
                                    if r_player != self.USER_PLAYER and r_type != d_a_type['Gold']:
                                        clicked_actor_index_arr.append(ACTS["attack"])
                            else:

                                actor_size = int(canvas_scale / 3)

                                clicked_x, clicked_y = pos

                                clicked_actors_world_x = int(l_x * canvas_scale + canvas_scale / 2 + canvas_scale - actor_size)
                                clicked_actors_world_y = int(l_y * canvas_scale + canvas_scale / 2 + canvas_scale - actor_size)
                                if abs(clicked_y - clicked_actors_world_y) > abs(clicked_x - clicked_actors_world_x):
                                    # we moved mouse more in y direction than x, so its vertical movement
                                    if clicked_y < clicked_actors_world_y:
                                        print("clicked up...")
                                        clicked_actor_index_arr.append(ACTS["up"])
                                    if clicked_y > clicked_actors_world_y:
                                        print("clicked down...")
                                        clicked_actor_index_arr.append(ACTS["down"])
                                else:
                                    # we moved mouse more in x direction than y, so its horizontal movement
                                    if clicked_x < clicked_actors_world_x:
                                        print("clicked left...")
                                        clicked_actor_index_arr.append(ACTS["left"])

                                    if clicked_x > clicked_actors_world_x:
                                        print("clicked right...")
                                        clicked_actor_index_arr.append(ACTS["right"])
                            if len(clicked_actor_index_arr) == 3:
                                return clicked_actor_index_arr
                            else:
                                print("invalid")
                                self.display_valid_moves(board, None)
                        else:
                            print("First left click on actor to select it")


class GreedyRTSPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)

        print("sum valids", sum(valids))
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a] == 0:
                continue
            next_board, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(next_board, 1)
            candidates += [(-score, a)]
        candidates.sort()

        n = board.shape[0]
        y, x, action_index = np.unravel_index(candidates[0][1], [n, n, NUM_ACTS])

        print("returned act", x, y, ACTS_REV[action_index])

        return candidates[0][1]
