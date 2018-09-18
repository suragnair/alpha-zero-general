from math import sqrt, ceil

import numpy as np
import pandas as pd
import pygame
from numpy import size
from pygame.rect import Rect

from td2020.src.config_file import CANVAS_SCALE, BORDER

from td2020.src import Board
from td2020.src.FunctionLibrary import arr_into_action, action_into_array_print


class RandomTD2020Player:
    def __init__(self, game) -> None:
        self.game = game

    def play(self, board: 'Board.Board', player: int) -> int:
        action_size = self.game.getActionSize
        a = np.random.randint(action_size)
        valids = self.game.getValidMoves(board, player)
        while valids[a] != 1:
            a = np.random.randint(action_size)
        return a


class HumanTD2020Player:
    def __init__(self, game) -> None:
        self.game= game

    def play(self, board: 'Board.Board', player: int) -> int:
        # board.display()
        print("----------")
        valid = self.game.getValidMoves(board, player)
        for i in range(len(valid)):
            if valid[i]:
                action_into_array_print(board, i)
                print("----------")
        while True:

            if board.verbose > 3:
                print("select action on canvas")
                a = self._manage_input(board, player)
                print("action from pygame returned:", a)
            else:
                a = ("1 " + input('type one of above actions\n')).split(" ")
            # convert to action index in valids array
            a = arr_into_action(board, a)
            print("action parsed in player.py", a)
            if valid[a]:
                break
            else:
                print('This action is invalid')

        return a

    @staticmethod
    def select_object(board, click_location: tuple) -> tuple:
        # select object by clicking on it - you can select only your objects

        # draw objects
        border = CANVAS_SCALE / 20

        for y in range(board.height):

            for x in range(board.width):
                tile = board[x][y]
                num_actors = size(tile.actors)
                if num_actors == 0:
                    continue

                num_in_row_column = ceil(sqrt(num_actors))
                actor_size = int((CANVAS_SCALE / 2 - 2 * border) / num_in_row_column)

                for actor_index, actor in enumerate(tile.actors):
                    # print("DRAWING ACTOR " + str(type(_actor)))

                    # offset if multiple actors are on same tile
                    multiple_offset = int((CANVAS_SCALE / num_in_row_column) * actor_index)

                    actor_x = int(actor.x * CANVAS_SCALE + int(multiple_offset % CANVAS_SCALE) + actor_size + border)
                    actor_y = int(actor.y * CANVAS_SCALE + int(multiple_offset / CANVAS_SCALE) * actor_size * 2 + actor_size + border)
                    actor_location = (actor_x, actor_y)
                    click_x, click_y = click_location

                    # check if actor is within click bounds

                    dist = sqrt((actor_x - click_x) ** 2 + (actor_y - click_y) ** 2)
                    if dist <= actor_size:
                        print("actor selected", str(type(actor)))
                        return [actor, actor_location, actor_size], [1, x, y, actor_index]  # has to have prefix number 1
        return [None] * 3, [None] * 4

    def _manage_input(self, board, player: int) -> list:
        # returns array like this [1, 7, 7, 0, "idle")

        from td2020.src.Graphics import init_visuals, update_graphics
        game_display, clock = init_visuals(board.width, board.height, board.verbose)
        update_graphics(board, game_display, clock, board.fps)

        from td2020.src.Actors import MyActor
        clicked_actor = None
        clicked_actor_index_arr = []  # index on which this actor is located - x,y,actor_index
        while True:
            for event in pygame.event.get():
                # print(event)
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit(0)
                if event.type == pygame.KEYDOWN:
                    if clicked_actor and issubclass(type(clicked_actor), MyActor) and clicked_actor.player == player:
                        # now this is our actor
                        if event.key == pygame.K_UP:
                            print("pressed up...")
                            clicked_actor_index_arr.append("up")
                            return clicked_actor_index_arr
                        if event.key == pygame.K_DOWN:
                            print("pressed down...")
                            clicked_actor_index_arr.append("down")
                            return clicked_actor_index_arr
                        if event.key == pygame.K_LEFT:
                            print("pressed left...")
                            clicked_actor_index_arr.append("left")
                            return clicked_actor_index_arr
                        if event.key == pygame.K_RIGHT:
                            print("pressed right...")
                            clicked_actor_index_arr.append("right")
                            return clicked_actor_index_arr
                        from td2020.src.FunctionLibrary import retrieve_json
                        print("retrieving shortcut for", str(type(clicked_actor).__name__))
                        td_my_actor: pd.DataFrame = retrieve_json('td_myactor', str(type(clicked_actor).__name__))
                        actor_shortcut = td_my_actor["Shortcut"].values[0]
                        print("printing actor shortcut", actor_shortcut)
                        from td2020.src.Actors import TownHall, Barracks, NPC
                        if type(clicked_actor) == TownHall:  # these if statements are hardcoded - (optional) - use variable actor_shortcut
                            if event.key == pygame.K_q:
                                clicked_actor_index_arr.append("npc")
                                return clicked_actor_index_arr
                        if type(clicked_actor) == Barracks:
                            if event.key == pygame.K_q:
                                print("mele infantry unsupported")
                            if event.key == pygame.K_w:
                                clicked_actor_index_arr.append("rifle_infantry")
                                return clicked_actor_index_arr
                            if event.key == pygame.K_e:
                                print("bow infantry unsupported")
                        if type(clicked_actor) == NPC:
                            if event.key == pygame.K_q:
                                clicked_actor_index_arr.append("town_hall")
                                return clicked_actor_index_arr
                            if event.key == pygame.K_w:
                                print("shack unsupported")
                            if event.key == pygame.K_e:
                                print("constr office unsupported")
                            if event.key == pygame.K_a:
                                clicked_actor_index_arr.append("mining_shack")
                                return clicked_actor_index_arr
                            if event.key == pygame.K_s:
                                print("apartment unsupported")
                            if event.key == pygame.K_d:
                                clicked_actor_index_arr.append("barracks")
                                return clicked_actor_index_arr
                            if event.key == pygame.K_y:
                                clicked_actor_index_arr.append("sentry")
                                return clicked_actor_index_arr
                        # idle
                        if event.key == pygame.K_SPACE:
                            clicked_actor_index_arr.append("idle")
                            return clicked_actor_index_arr
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        raise SystemExit(0)

                # handle mouse
                if event.type == pygame.MOUSEBUTTONUP:
                    lmb, rmb = 1, 3
                    pos = pygame.mouse.get_pos()

                    if event.button == lmb:
                        actor_arr, index_arr = self.select_object(board, pos)
                        temp_clicked_actor, actor_location, clicked_actor_size = actor_arr[0], actor_arr[1], actor_arr[2]

                        if issubclass(type(temp_clicked_actor), MyActor) and temp_clicked_actor.player == player:
                            clicked_actor = temp_clicked_actor
                            clicked_actor_index_arr = index_arr

                            # draw selected bounding box
                            from td2020.src.Graphics import init_visuals, update_graphics
                            game_display, clock = init_visuals(board.width, board.height, board.verbose)
                            update_graphics(board, game_display, clock, board.fps)
                            actor_x, actor_y = actor_location
                            rect = Rect((actor_x - clicked_actor_size, actor_y - clicked_actor_size), (2 * clicked_actor_size, 2 * clicked_actor_size))
                            blue = (0, 0, 255)
                            pygame.draw.rect(game_display, blue, rect, BORDER)
                            pygame.display.update()
                        else:
                            print("left clicked invalid actor - it may not be our or its not of type MyActor")
                    if event.button == rmb:
                        if clicked_actor:

                            actor_arr, _ = self.select_object(board, pos)
                            right_clicked_actor = actor_arr[0]
                            if right_clicked_actor:
                                if issubclass(type(right_clicked_actor), MyActor):
                                    # this is actor of type MyActor
                                    if right_clicked_actor.player == player:
                                        print("right clicked our actor")

                                        if right_clicked_actor.current_production_time < right_clicked_actor.production_time:
                                            print("right clicked on construction proxy")

                                            clicked_actor_index_arr.append("continue_building")
                                        else:
                                            print("return resources")
                                            clicked_actor_index_arr.append("return_resources")
                                    else:
                                        clicked_actor.action_manager.enemy_actor = right_clicked_actor
                                        clicked_actor_index_arr.append("attack")
                                else:
                                    print("right clicked minerals")
                                    clicked_actor_index_arr.append("mine_resources")
                            else:
                                clicked_x, clicked_y = pos

                                if abs(clicked_y - clicked_actor.y * CANVAS_SCALE) > abs(clicked_x - clicked_actor.x * CANVAS_SCALE):
                                    # we moved mouse more in y direction than x, so its vertical movement
                                    if clicked_y < clicked_actor.y * CANVAS_SCALE:
                                        print("clicked up...")
                                        clicked_actor_index_arr.append("up")
                                    if clicked_y > clicked_actor.y * CANVAS_SCALE:
                                        print("clicked down...")
                                        clicked_actor_index_arr.append("down")
                                else:
                                    # we moved mouse more in x direction than y, so its horizontal movement
                                    if clicked_x < clicked_actor.x * CANVAS_SCALE:
                                        print("clicked left...")
                                        clicked_actor_index_arr.append("left")

                                    if clicked_x > clicked_actor.x * CANVAS_SCALE:
                                        print("clicked right...")
                                        clicked_actor_index_arr.append("right")
                            return clicked_actor_index_arr
                        else:
                            print("first left click on actor to select it")


class GreedyTD2020Player:
    def __init__(self, game) -> None:
        self.game = game

    def play(self, board: Board, player: int) -> int:
        valids = self.game.getValidMoves(board, player)
        candidates = []
        action_size = self.game.getActionSize
        for a in range(action_size):
            if valids[a] == 0:
                continue
            next_board, _ = self.game.getNextState(board, player, a)
            score = self.game.getScore(next_board, player)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]
