class Stone:
    EMPTY = 0
    BLACK = 1
    WHITE = -1

def get_opposite_stone(stone):
    #assert(stone != Stone.EMPTY)
    if stone == Stone.BLACK:
        return Stone.WHITE
    return Stone.BLACK

def make_2d_array(h, w, default=lambda: None):
    return [[default() for i in range(w)] for j in range(h)]
