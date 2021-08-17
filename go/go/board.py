import numpy as np
from go.utils import Stone, make_2d_array, get_opposite_stone

class Board(np.ndarray):
    '''
    Instance of a 2D grid board extended from np.ndarray
    '''
    def __new__(cls, n):
        '''
        Standard procedure for subclassing np.ndarray
        '''
        # dimension of the board
        board_size = n
        shape = (board_size, board_size)
        obj = super(Board, cls).__new__(cls, shape, dtype=np.int)

        obj.board_size = board_size

        obj.previous_is_pass = False
        obj.pre_previous_is_pass = False
        obj._group_map = make_2d_array(board_size, board_size)
        obj._captured_groups = set()
        obj._num_captured_stones = {
            Stone.WHITE: 0,
            Stone.BLACK: 0
        }
        obj.turns = 0
        obj._ko = None


        # string to display as a black stone
        obj.black_stone_render = 'b'

        # string to display as a white stone
        obj.white_stone_render = 'w'

        # fill board with empty slots
        obj.fill(Stone.EMPTY)

        return obj

    def __array_finalize__(self, obj):
        '''
        Standard procedure for subclassing np.ndarray
        '''
        if obj is None:
            return
        self.board_size = getattr(obj, 'board_size')
        self.black_stone_render = getattr(obj, 'black_stone_render')
        self.white_stone_render = getattr(obj, 'white_stone_render')
        self.previous_is_pass = getattr(obj, 'previous_is_pass')
        self.pre_previous_is_pass = getattr(obj, 'pre_previous_is_pass')
        self._group_map = getattr(obj, '_group_map')
        self._captured_groups = getattr(obj, '_captured_groups')
        self._num_captured_stones = getattr(obj, '_num_captured_stones')
        self.turns = getattr(obj, 'turns')
        self._ko = getattr(obj, '_ko')
    def get_liberty_coords(self, y, x):
        '''
        Return the liberty coordinates for (y, x). This constitutes
        "up", "down", "left", "right" if possible.
        '''
        coords = []
        if y > 0:
            coords.append((y-1, x))
        if y < self.board_size-1:
            coords.append((y+1, x))
        if x > 0:
            coords.append((y, x-1))
        if x < self.board_size-1:
            coords.append((y, x+1))
        return coords

    def place_stone(self, stone, y, x):
        '''
        Place a stone at the specified coordinate
        '''
        self[y][x] = stone

    def remove_stone(self, y, x):
        '''
        Remove the stone at the specified coordinate
        '''
        self[y][x] = Stone.EMPTY

    def is_within_bounds(self, y, x):
        '''
        Check if the given coordinate is within bounds of the board
        '''
        return 0 <= y <= self.board_size and 0 <= x <= self.board_size

    def _value_to_render(self, stone):
        '''
        Map from the stone to the displayed string for that stone
        '''
        s = None
        if stone == Stone.EMPTY:
            s = ' '
        elif stone == Stone.BLACK:
            s = self.black_stone_render
        elif stone == Stone.WHITE:
            s = self.white_stone_render
        return f'[{s}]'

    def _render(self):
        '''
        Render the board, with axes labelled from 0, 1, 2, ..., 9, A, B, ...
        and so on
        '''
        # horizontal axis
        print('\n   ' + '  '.join([self._index_to_label(x) \
                            for x in range(self.board_size)]))

        # vertical axis is printed with each row
        for row in range(self.board_size):
            label = self._index_to_label(row)
            board_row = map(self._value_to_render, self[row])
            print(f'{label} ' + ''.join(board_row))

        print('')

    def _index_to_label(self, idx):
        '''
        Map the index to displayed axis coordinate
        Eg. _index_to_label(3) --> '3'
            _index_to_label(13) --> 'D'
        '''
        if idx < 10:
            return str(idx)
        return chr(idx - 10 + ord('A'))

