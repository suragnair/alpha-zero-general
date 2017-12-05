from __future__ import print_function
"""
Borrowed from: https://gist.github.com/christian-oudard/220521

Utilities for 256 color support in terminals.
 
Adapted from:
http://stackoverflow.com/questions/1403353/256-color-terminal-library-for-ruby
 
The color palette is indexed as follows:
 
0-15: System colors
    0  black         8  dark gray
    1  red           9  bright red
    2  green         10 bright green
    3  yellow        11 bright yellow
    4  blue          12 bright blue
    5  magenta       13 bright magenta
    6  cyan          14 bright cyan
    7  light gray    15 white
 
16-231: 6x6x6 Color Cube
    All combinations of red, green, and blue from 0 to 5.
 
232-255: Grayscale Ramp
    24 shades of gray, not including black and white.
"""
 
# System color name constants.
(
    BLACK,
    RED,
    GREEN,
    YELLOW,
    BLUE,
    MAGENTA,
    CYAN,
    LIGHT_GRAY,
    DARK_GRAY,
    BRIGHT_RED,
    BRIGHT_GREEN,
    BRIGHT_YELLOW,
    BRIGHT_BLUE,
    BRIGHT_MAGENTA,
    BRIGHT_CYAN,
    WHITE,
) = range(16)
 
def rgb(red, green, blue):
    """
    Calculate the palette index of a color in the 6x6x6 color cube.
 
    The red, green and blue arguments may range from 0 to 5.
    """
    return 16 + (red * 36) + (green * 6) + blue
 
def gray(value):
    """
    Calculate the palette index of a color in the grayscale ramp.
 
    The value argument may range from 0 to 23.
    """
    return 232 + value
 
def set_color(fg=None, bg=None):
    """
    Print escape codes to set the terminal color.
 
    fg and bg are indices into the color palette for the foreground and
    background colors.
    """
    sys.stdout.write(_set_color(fg, bg))
 
def _set_color(fg=None, bg=None):
    result = ''
    if fg:
        result += '\x1b[38;5;%dm' % fg
    if bg:
        result += '\x1b[48;5;%dm' % bg
    return result
 
def reset_color():
    """
    Reset terminal color to default.
    """
    print(_reset_color(), end='')
 
def _reset_color():
    return '\x1b[0m'
 
def print_color(*args, **kwargs):
    """
    Print function, with extra arguments fg and bg to set colors.
    """
    fg = kwargs.pop('fg', None)
    bg = kwargs.pop('bg', None)
    set_color(fg, bg)
    print(*args, **kwargs)
    reset_color()
 
def format_color(string, fg=None, bg=None):
    return _set_color(fg, bg) + string + _reset_color()
