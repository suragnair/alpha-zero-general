#test getInitBoard
import go.Game as a
g = a.Game(9)
g.getInitBoard()
from go.GoLogic import Board
b = Board(9)
import numpy as np
print(np.array(b.pieces))
#same result, test pass

import go.goGame as x
a = x.Game()
a.place_black(1,1)
a.place_white(2,1)
a._place_stone(-1, 5, 4)
a._place_stone(1, 5, 4)
print(a.board)
