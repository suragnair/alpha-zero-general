import ctypes
from typing import Any, Tuple, Optional

import numpy as np

from td2020.src.dicts import P_NAME_IDX, A_TYPE_IDX, d_a_color, d_type_rev, PATH, SHOW_PYGAME_WELCOME, MONEY_IDX, TIME_IDX, CARRY_IDX, HEALTH_IDX

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


def message_display(game_display, text, position, text_size, color=(0, 0, 0)) -> None:
    large_text = pygame.font.Font(PATH + '\\Cyberbit.ttf', text_size)
    text_surf = large_text.render(text, True, color)
    text_rect = text_surf.get_rect()
    text_rect.center = position
    game_display.blit(text_surf, text_rect)


def init_visuals(world_width: int, world_height: int, verbose=True) -> Optional[Tuple[Any, Any]]:
    if verbose:
        pygame.init()
        CANVAS_SCALE: int = int(ctypes.windll.user32.GetSystemMetrics(1) * (2 / 3) / world_height)  # for drawing - it takes 2 thirds of screen height

        # square

        display_width, display_height = world_width * CANVAS_SCALE, world_height * CANVAS_SCALE  # for example 800

        game_display = pygame.display.set_mode((display_width, display_height))
        pygame.display.set_caption('TD2020 Python game')
        clock = pygame.time.Clock()

        return game_display, clock


def update_graphics(board: np.ndarray, game_display, clock, fps: int = 1) -> None:
    n = board.shape[0]

    CANVAS_SCALE: int = int(ctypes.windll.user32.GetSystemMetrics(1) * (16 / 30) / n)  # for drawing - it takes 2 thirds of screen height

    # clear display
    game_display.fill((255, 255, 255))
    # self.display_img(game_display, x,y)

    # draw grid:
    for y in range(CANVAS_SCALE, (n + 2) * CANVAS_SCALE, CANVAS_SCALE):

        pygame.draw.line(game_display, (0, 0, 0), [y, CANVAS_SCALE], [y, (n + 1) * CANVAS_SCALE])
        for x in range(CANVAS_SCALE, (n + 2) * CANVAS_SCALE, CANVAS_SCALE):
            pygame.draw.line(game_display, (0, 0, 0), [CANVAS_SCALE, x], [(n + 1) * CANVAS_SCALE, x])
            if x < (n + 1) * CANVAS_SCALE and y < (n + 1) * CANVAS_SCALE:
                message_display(game_display, u"" + str(x / CANVAS_SCALE - 1) + ", " + str(y / CANVAS_SCALE - 1), ((x + CANVAS_SCALE / 4), (y + CANVAS_SCALE / 10)), int(CANVAS_SCALE / 8))

    # gold for each player:
    gold_p1 = board[int(n / 2) - 1][int(n / 2)][MONEY_IDX]
    gold_p2 = board[int(n / 2)][int(n / 2) - 1][MONEY_IDX]

    message_display(game_display, u"" + 'Gold Player +1: ' + str(gold_p1), (int((n / 4) * CANVAS_SCALE), int(0 + CANVAS_SCALE / 6)), int(CANVAS_SCALE / 3))
    message_display(game_display, u"" + 'Gold Player -1: ' + str(gold_p2), (int((n / 4) * CANVAS_SCALE), int(0 + CANVAS_SCALE * (2 / 3))), int(CANVAS_SCALE / 3))

    time_remaining = board[0][0][TIME_IDX]

    message_display(game_display, u"" + 'Remaining ' + str(time_remaining), (int((n * (3 / 4)) * CANVAS_SCALE), int(0 + CANVAS_SCALE / 6)), int(CANVAS_SCALE / 3))

    for y in range(n):
        for x in range(n):
            a_player = board[x][y][P_NAME_IDX]

            if a_player == 1 or a_player == -1:

                a_type = board[x][y][A_TYPE_IDX]
                actor_color = d_a_color[a_type]

                actor_location = (int(x * CANVAS_SCALE + CANVAS_SCALE / 2 + CANVAS_SCALE), int(y * CANVAS_SCALE + CANVAS_SCALE / 2) + CANVAS_SCALE)
                actor_x, actor_y = actor_location

                actor_size = int(CANVAS_SCALE / 3)
                actor_short_name = d_type_rev[a_type]

                actor_carry = board[x][y][CARRY_IDX]
                actor_health = board[x][y][HEALTH_IDX]

                pygame.draw.circle(game_display, actor_color, actor_location, actor_size)

                player_color = (0, 0, 0)
                if a_player == 1:
                    player_color = (0, 255, 0)
                if a_player == -1:
                    player_color = (255, 0, 0)

                pygame.draw.circle(game_display, player_color, actor_location, actor_size, int(actor_size / 10))
                message_display(game_display, u"" + actor_short_name, actor_location, int(actor_size * 0.7))

                if a_type != 1:  # if not gold
                    message_display(game_display, u"hp: " + str(actor_health), (actor_x, actor_y + CANVAS_SCALE * (2 / 10)), int(actor_size * 0.5))

                if a_type == 2:  # if npc
                    message_display(game_display, u"carry: " + str(actor_carry), (actor_x, actor_y + CANVAS_SCALE * (4 / 10)), int(actor_size * 0.5))

    pygame.display.update()

    clock.tick(fps)
