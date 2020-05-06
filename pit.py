"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
import argparse
import importlib
import logging
import sys

import coloredlogs

import Arena

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


class HelperParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help()

        sys.stderr.write('\nerror: %s\n' % message)
        sys.exit(2)


parser = HelperParser(description='Pit two players against each other.')
parser.add_argument('--game', '-g', type=str, help='game directory (i.e. othello)', required=True)


def main(args):
    full_module_name = args.game + '.' + 'pit_config'
    try:
        game_module = importlib.import_module(full_module_name)
    except Exception as e:
        log.error('Could not load game "%s": %s' % (args.game, e))
        log.error('Make sure "%s" directory exists and it has "pit_config.py" file in it' % args.game)
        sys.exit(2)
    player1, player2, game = game_module.get_config()
    arena = Arena.Arena(player1, player2, game, game.display)
    print(arena.playGames(2, verbose=True))


if __name__ == '__main__':
    main(parser.parse_args())
