from math import sqrt, ceil
from typing import Any, Tuple, Optional

"""
from config_file import CANVAS_SCALE, BORDER, SHOW_PYGAME_WELCOME, PATH

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
from numpy import size, clip

"""
def _message_display(game_display, text, position, text_size, color=(0, 0, 0)) -> None:
    pass
    """
    large_text = pygame.font.Font(PATH + '\\games\\td2020\\src\\Cyberbit.ttf', text_size)
    text_surf = large_text.render(text, True, color)
    text_rect = text_surf.get_rect()
    text_rect.center = position
    game_display.blit(text_surf, text_rect)
    """

def init_visuals(world_width: int, world_height: int, verbose=True) -> Optional[Tuple[Any, Any]]:
    pass
    """
    if verbose:
        pygame.init()
        # square
        display_width, display_height = world_width * CANVAS_SCALE, world_height * CANVAS_SCALE  # for example 800

        game_display = pygame.display.set_mode((display_width, display_height))
        pygame.display.set_caption('TD2020 Python game')
        clock = pygame.time.Clock()

        return game_display, clock
    """

def update_graphics(board, game_display, clock, fps: int = 1) -> None:



    # clear display
    game_display.fill((255, 255, 255))
    # self.display_img(game_display, x,y)

    """
    # draw grid:
    for y in range(CANVAS_SCALE, board.height * CANVAS_SCALE, CANVAS_SCALE):
        pygame.draw.line(game_display, (0, 0, 0), [y, 0], [y, board.height * CANVAS_SCALE])
        for x in range(CANVAS_SCALE, board.width * CANVAS_SCALE, CANVAS_SCALE):
            pygame.draw.line(game_display, (0, 0, 0), [0, x], [board.width * CANVAS_SCALE, x])

    # draw objects
    for tiles in board.tiles:
        for tile in tiles:
            num_actors = size(tile.actors)
            if num_actors == 0:
                continue

            num_in_row_column = ceil(sqrt(num_actors))
            actor_size = int((CANVAS_SCALE / 2 - 2 * BORDER) / num_in_row_column)

            for index, _actor in enumerate(tile.actors):
                # print("DRAWING ACTOR " + str(type(_actor)))

                # offset if multiple actors are on same tile
                multiple_offset = int((CANVAS_SCALE / num_in_row_column) * index)

                actor: GeneralActor = _actor

                x = int(actor.x * CANVAS_SCALE + int(multiple_offset % CANVAS_SCALE) + actor_size + BORDER)
                y = int(actor.y * CANVAS_SCALE + int(multiple_offset / CANVAS_SCALE) * actor_size * 2 + actor_size + BORDER)

                actor_location = (x, y)

                actor_color = (actor.color["R"], actor.color["G"], actor.color["B"])

                from games.td2020.src.Actors import MyActor
                if isinstance(actor, MyActor):
                    production_percent = float(actor.current_production_time) / float(actor.production_time)  # value between 0 and 1
                    if production_percent != 1:
                        actor_color = (int(float(actor_color[0]) * production_percent + 25.5), int(float(actor_color[1]) * production_percent + 25.5), int(float(actor_color[2]) * production_percent + 25.5))

                pygame.draw.circle(game_display, actor_color, actor_location, actor_size)

                if isinstance(actor, MyActor):
                    if actor.player == -1:
                        player_color = (0, 255, 0)
                    else:
                        player_color = (255, 0, 0)
                    pygame.draw.circle(game_display, player_color, actor_location, actor_size, int(actor_size / 10))
                    production_percent = clip(float(actor.current_production_time) / float(actor.production_time), 0, 1)  # value between 0 and 1

                    if production_percent == 1:
                        _message_display(game_display, u"" + actor.current_action, (x, int(y - actor_size / 2)), int(actor_size / 3))
                    else:

                        _message_display(game_display, u"" + str(production_percent * 100) + "%", (x, int(y - actor_size / 2)), int(actor_size / 3), (int(255 - 255 * production_percent), int(255 - 255 * production_percent), int(255 - 255 * production_percent)))

                _message_display(game_display, u"" + actor.short_name, actor_location, actor_size)
    """
    import pygame

    pygame.display.update()

    clock.tick(fps)
