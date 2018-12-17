import ctypes
from typing import Any, Tuple, Optional

import numpy as np

from td2020.src.config import P_NAME_IDX, A_TYPE_IDX, d_a_color, d_type_rev, MONEY_IDX, TIME_IDX, CARRY_IDX, HEALTH_IDX

"""
Graphics.py

Used for displaying Gama visualization using Pygame

"""


def message_display(game_display, text, position, text_size, color=(0, 0, 0)) -> None:
    import pygame

    large_text = pygame.font.SysFont('arial', text_size)
    text_surf = large_text.render(text, True, color)
    text_rect = text_surf.get_rect()
    text_rect.center = position
    game_display.blit(text_surf, text_rect)


def init_visuals(world_width: int, world_height: int, verbose=True) -> Optional[Tuple[Any, Any]]:
    if verbose:
        import pygame

        pygame.init()
        canvas_scale = int(ctypes.windll.user32.GetSystemMetrics(1) * (2 / 3) / world_height)  # for drawing - it takes 2 thirds of screen height

        # square

        display_width, display_height = world_width * canvas_scale, world_height * canvas_scale  # for example 800

        game_display = pygame.display.set_mode((display_width, display_height))
        pygame.display.set_caption('RTS visualization Python game')

        clock = pygame.time.Clock()

        return game_display, clock


def update_graphics(board: np.ndarray, game_display, clock, fps: int = 1) -> None:
    import pygame

    n = board.shape[0]

    canvas_scale = int(ctypes.windll.user32.GetSystemMetrics(1) * (16 / 30) / n)  # for drawing - it takes 2 thirds of screen height

    # clear display
    game_display.fill((255, 255, 255))
    # self.display_img(game_display, x,y)

    # title
    # message_display(game_display, u"" + ' ' + str(gold_p1), (int((n / 8) * canvas_scale), (n+1) * canvas_scale + int(int(canvas_scale / 12) + canvas_scale * (0 / 4) + int(canvas_scale * (1 / 8)))), int(canvas_scale / 6))

    # draw grid:
    for y in range(canvas_scale, (n + 2) * canvas_scale, canvas_scale):
        pygame.draw.line(game_display, (0, 0, 0), [y, canvas_scale], [y, (n + 1) * canvas_scale])
        for x in range(canvas_scale, (n + 2) * canvas_scale, canvas_scale):
            pygame.draw.line(game_display, (0, 0, 0), [canvas_scale, x], [(n + 1) * canvas_scale, x])
            if x < (n + 1) * canvas_scale and y < (n + 1) * canvas_scale:
                message_display(game_display, u"" + str(x / canvas_scale - 1) + ", " + str(y / canvas_scale - 1), ((x + canvas_scale / 4), (y + canvas_scale / 10)), int(canvas_scale / 8))

    # gold for each player:
    gold_p1 = board[int(n / 2) - 1][int(n / 2)][MONEY_IDX]
    gold_p2 = board[int(n / 2)][int(n / 2) - 1][MONEY_IDX]

    message_display(game_display, u"" + 'Gold Player +1: ' + str(gold_p1), (int((n / 8) * canvas_scale), (n + 1) * canvas_scale + int(int(canvas_scale / 12) + canvas_scale * (0 / 4) + int(canvas_scale * (1 / 8)))), int(canvas_scale / 6))
    message_display(game_display, u"" + 'Gold Player -1: ' + str(gold_p2), (int((n / 8) * canvas_scale), (n + 1) * canvas_scale + int(int(canvas_scale / 12) + canvas_scale * (1 / 4) + int(canvas_scale * (1 / 8)))), int(canvas_scale / 6))

    time_remaining = board[0][0][TIME_IDX]
    message_display(game_display, u"" + 'Remaining ' + str(time_remaining), (int((n / 8) * canvas_scale), (n + 1) * canvas_scale + int(int(canvas_scale / 12) + canvas_scale * (2 / 4) + int(canvas_scale * (1 / 8)))), int(canvas_scale / 6))

    for y in range(n):
        for x in range(n):
            a_player = board[x][y][P_NAME_IDX]

            if a_player == 1 or a_player == -1:

                a_type = board[x][y][A_TYPE_IDX]
                actor_color = d_a_color[a_type]

                actor_location = (int(x * canvas_scale + canvas_scale / 2 + canvas_scale), int(y * canvas_scale + canvas_scale / 2) + canvas_scale)
                actor_x, actor_y = actor_location

                actor_size = int(canvas_scale / 3)
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
                    message_display(game_display, u"hp: " + str(actor_health), (actor_x, actor_y + canvas_scale * (2 / 10)), int(actor_size * 0.5))

                if a_type == 2:  # if npc
                    message_display(game_display, u"carry: " + str(actor_carry), (actor_x, actor_y + canvas_scale * (4 / 10)), int(actor_size * 0.5))

    pygame.display.update()

    clock.tick(fps)
