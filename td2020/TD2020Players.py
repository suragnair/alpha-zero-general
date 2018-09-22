import ctypes
from math import sqrt

import numpy as np
from pygame.rect import Rect

from td2020.src.Board import Board
from td2020.src.Graphics import init_visuals, update_graphics, message_display
from td2020.src.dicts import NUM_ACTS, VERBOSE, P_NAME_IDX, A_TYPE_IDX, d_user_shortcuts, USER_PLAYER, FPS, ACTS, d_a_type, ACTS_REV, d_user_shortcuts_rev, SHOW_PYGAME_WELCOME
from utils import dotdict

if SHOW_PYGAME_WELCOME:
    import pygame
else:
    import os
    import sys

    with open(os.devnull, 'w') as f:
        # disable stdout
        old_std_out = sys.stdout
        sys.stdout = f
        import pygame

        # enable stdout
        sys.stdout = old_std_out


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanTD2020Player:
    def __init__(self, game) -> None:
        self.game = game

    def play(self, board: np.ndarray) -> int:
        n = board.shape[0]
        valid = self.game.getValidMoves(board, 1)
        self.display_valid_moves(board, valid)
        while True:

            if VERBOSE > 3:
                a = self._manage_input(board)

            else:
                a = (input('type one of above actions in "x y action_index" format\n')).split(" ")
            # convert to action index in valids array

            try:
                x, y, action_index = a
                x = int(x)
                y = int(y)
                action_index = int(action_index)

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
        if valid is None:
            valid = self.game.getValidMoves(board, 1)
        n = board.shape[0]
        print("----------")
        for i in range(len(valid)):
            if valid[i]:
                # print("printing i", i)
                y, x, action_index = np.unravel_index(i, [n, n, NUM_ACTS])

                # print("numpy action index", np.ravel_multi_index((y, x, action_index), (n, n, NUM_ACTS)))

                print(x, y, ACTS_REV[action_index])
                # action_into_array_print(board, i)
                print("----------")

    @staticmethod
    def select_object(board: np.ndarray, click_location: tuple) -> dotdict:

        n = board.shape[0]
        CANVAS_SCALE: int = int(ctypes.windll.user32.GetSystemMetrics(1) * (16 / 30) / n)  # for drawing - it takes 2 thirds of screen height

        # select object by clicking on it - you can select only your objects

        for y in range(n):

            for x in range(n):

                actor_location = (int(x * CANVAS_SCALE + CANVAS_SCALE / 2 + CANVAS_SCALE), int(y * CANVAS_SCALE + CANVAS_SCALE / 2) + CANVAS_SCALE)
                actor_x, actor_y = actor_location
                actor_size = int(CANVAS_SCALE / 3)

                click_x, click_y = click_location

                # check if actor is within click bounds

                dist = sqrt((actor_x - click_x) ** 2 + (actor_y - click_y) ** 2)
                if dist <= actor_size:
                    return dotdict({
                        "x": x,
                        "y": y
                    })
        return None
        # return [actor, actor_location, actor_size], [1, x, y, actor_index]  # has to have prefix number 1

    def _manage_input(self, board: np.ndarray) -> list:
        # returns array like this [1, 7, 7, 0, "idle")
        n = board.shape[0]

        game_display, clock = init_visuals(n, n, VERBOSE)
        update_graphics(board, game_display, clock, FPS)

        CANVAS_SCALE: int = int(ctypes.windll.user32.GetSystemMetrics(1) * (16 / 30) / n)  # for drawing - it takes 2 thirds of screen height

        # from td2020.src.Actors import MyActor
        clicked_actor = None
        clicked_actor_index_arr = []  # index on which this actor is located - x,y,actor_index
        while True:
            for event in pygame.event.get():
                # print(event)
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit(0)
                if event.type == pygame.KEYDOWN:

                    if clicked_actor and (board[clicked_actor.x][clicked_actor.y][P_NAME_IDX] == USER_PLAYER):
                        try:

                            shortcut_pressed = d_user_shortcuts[event.unicode]
                            action_to_execute = shortcut_pressed
                            clicked_actor_index_arr.append(action_to_execute)
                            return clicked_actor_index_arr
                        except Exception as e:
                            print("shortcut '" + event.unicode + "' not supported.")

                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        raise SystemExit(0)

                # handle mouse
                if event.type == pygame.MOUSEBUTTONUP:
                    lmb, rmb = 1, 3
                    pos = pygame.mouse.get_pos()

                    if event.button == lmb:

                        clicked_actor = self.select_object(board, pos)
                        if clicked_actor and board[clicked_actor.x][clicked_actor.y][P_NAME_IDX] == USER_PLAYER and board[clicked_actor.x][clicked_actor.y][A_TYPE_IDX] != d_a_type['Gold']:
                            clicked_actor_index_arr = [clicked_actor.x, clicked_actor.y]

                            # draw selected bounding box
                            game_display, clock = init_visuals(n, n, VERBOSE)
                            update_graphics(board, game_display, clock, FPS)

                            actor_size = int(CANVAS_SCALE / 3)
                            actor_location = (int(clicked_actor.x * CANVAS_SCALE + CANVAS_SCALE / 2 + CANVAS_SCALE - actor_size), int(clicked_actor.y * CANVAS_SCALE + CANVAS_SCALE / 2 + CANVAS_SCALE - actor_size))
                            rect = Rect(actor_location, (2 * actor_size, 2 * actor_size))

                            blue = (0, 0, 255)
                            pygame.draw.rect(game_display, blue, rect, int(CANVAS_SCALE / 20))

                            # display valid actions on canvas

                            b = Board(n)
                            b.pieces = np.copy(board)

                            valids_square = b.get_moves_for_square((clicked_actor.x, clicked_actor.y))

                            printed_actions = 0
                            for i in range(len(valids_square)):
                                if valids_square[i]:
                                    text_scale = int(actor_size * 0.5)
                                    message_display(game_display, u"" + ACTS_REV[i] + " s: '" + d_user_shortcuts_rev[i] + "'", (n * CANVAS_SCALE + CANVAS_SCALE / 2, CANVAS_SCALE / 4 + printed_actions * text_scale), text_scale)
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
                                    if r_player == USER_PLAYER:
                                        if r_type == d_a_type['Gold']:
                                            clicked_actor_index_arr.append(ACTS["mine_resources"])
                                        if r_type == d_a_type['Hall']:
                                            clicked_actor_index_arr.append(ACTS["return_resources"])

                                if l_type == d_a_type['Rifl']:
                                    if r_player != USER_PLAYER and r_type != d_a_type['Gold']:
                                        clicked_actor_index_arr.append(ACTS["attack"])
                            else:

                                actor_size = int(CANVAS_SCALE / 3)

                                clicked_x, clicked_y = pos

                                clicked_actors_world_x = int(l_x * CANVAS_SCALE + CANVAS_SCALE / 2 + CANVAS_SCALE - actor_size)
                                clicked_actors_world_y = int(l_y * CANVAS_SCALE + CANVAS_SCALE / 2 + CANVAS_SCALE - actor_size)
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


class GreedyTD2020Player:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)

        print("sum valids", sum(valids))
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a] == 0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()

        n = board.shape[0]
        y, x, action_index = np.unravel_index(candidates[0][1], [n, n, NUM_ACTS])

        print("returned act", x, y, ACTS_REV[action_index])

        return candidates[0][1]
