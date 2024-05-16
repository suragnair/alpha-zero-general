import unittest
import numpy as np
from dotsandboxes.DotsAndBoxesGame import DotsAndBoxesGame


class DotsAndBoxesTests(unittest.TestCase):
    def testGetBoardSize(self):
        self.assertEqual((5,3), DotsAndBoxesGame(n=2).getBoardSize())
        self.assertEqual((7,4), DotsAndBoxesGame(n=3).getBoardSize())

    def testGetActionSize(self):
        self.assertEqual(13, DotsAndBoxesGame(n=2).getActionSize())
        self.assertEqual(25, DotsAndBoxesGame(n=3).getActionSize())

    def testGetNextState(self):
        g = DotsAndBoxesGame(n=2)

        board = np.array([
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0]
        ])
        action = 0
        player = 1
        next_state = g.getNextState(board, player, action)
        self.assertTrue(np.array_equal(next_state[0], np.array([
            [1,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0]
        ])))
        board = next_state[0]

        action = 1
        player = -1
        next_state = g.getNextState(board, player, action)
        self.assertTrue(np.array_equal(next_state[0], np.array([
            [1,1,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0]
        ])))
        board = next_state[0]

        action = 11
        player = 1
        next_state = g.getNextState(board, player, action)
        self.assertTrue(np.array_equal(next_state[0], np.array([
            [1,1,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,1]
        ])))

    def testGetNextStateScoreN2Player2(self):
        g = DotsAndBoxesGame(n=2)

        board = np.array([
            [1,0,0],
            [1,0,0],
            [0,0,0],
            [1,0,0],
            [0,0,0]
        ])
        action = 7
        player = -1
        next_state = g.getNextState(board, player, action)
        self.assertTrue(np.array_equal(next_state[0], np.array([
            [1,0,0],
            [1,0,1],
            [0,0,1],
            [1,1,0],
            [0,0,0]
        ])))

    def testGetNextStateScoreN2(self):
        g = DotsAndBoxesGame(n=2)

        board = np.array([
            [1,0,0],
            [1,0,0],
            [0,0,0],
            [1,0,0],
            [0,0,0]
        ])
        action = 7
        player = 1
        next_state = g.getNextState(board, player, action)
        self.assertTrue(np.array_equal(next_state[0], np.array([
            [1,0,1],
            [1,0,0],
            [0,0,1],
            [1,1,0],
            [0,0,0]
        ])))

        board = np.array([
            [0,1,0],
            [0,1,0],
            [0,0,0],
            [0,1,0],
            [0,0,0]
        ])
        action = 8
        player = 1
        next_state = g.getNextState(board, player, action)
        self.assertTrue(np.array_equal(next_state[0], np.array([
            [0,1,1],
            [0,1,0],
            [0,0,1],
            [0,1,1],
            [0,0,0]
        ])))

    def testGetNextStateScoreN3(self):
        g = DotsAndBoxesGame(n=3)

        board = np.array([
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        action = 13
        player = 1
        next_state = g.getNextState(board, player, action)
        self.assertTrue(np.array_equal(next_state[0], np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])))

        board = np.array([
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        action = 15
        player = 1
        next_state = g.getNextState(board, player, action)
        self.assertTrue(np.array_equal(next_state[0], np.array([
            [1, 0, 1, 1],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [1, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])))

    def testGetValidMoves(self):
        g = DotsAndBoxesGame(n=2)

        # All valid
        board = np.array([
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0]
        ])
        valid = g.getValidMoves(board, 1)
        self.assertTrue(np.array_equal(valid, np.array([True, True, True, True, True, True, True, True, True, True, True, True, False])))

        # Test Pass
        board = np.array([
            [0,0,0],
            [0,0,0],
            [0,0,1],
            [0,0,0],
            [0,0,0]
        ])
        valid = g.getValidMoves(board, 1)
        self.assertTrue(np.array_equal(valid, np.array([False, False, False, False, False, False, False, False, False, False, False, False,  True])))

        board = np.array([
            [1,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0]
        ])
        valid = g.getValidMoves(board, 1)
        self.assertTrue(np.array_equal(valid, np.array([False, True, True, True, True, True, True, True, True, True, True, True, False])))

        board = np.array([
            [1,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,1]
        ])
        valid = g.getValidMoves(board, 1)
        self.assertTrue(np.array_equal(valid, np.array([False, True, True, True, True, True, True, True, True, True, True, False, False])))

        board = np.array([
            [1,1,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,1]
        ])
        valid = g.getValidMoves(board, 1)
        self.assertTrue(np.array_equal(valid, np.array([False, False, True, True, True, True, True, True, True, True, True, False, False])))

        board = np.array([
            [1,1,0],
            [0,0,0],
            [0,0,0],
            [1,0,0],
            [0,0,1]
        ])
        valid = g.getValidMoves(board, 1)
        self.assertTrue(np.array_equal(valid, np.array([False, False, True, True, True, True, False, True, True, True, True, False, False])))

    def testGetGameEnded(self):
        g = DotsAndBoxesGame(n=2)

        board = np.array([
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0]
        ])
        outcome = g.getGameEnded(board, 1)
        self.assertEquals(0, outcome)

        board = np.array([
            [1,1,5],
            [1,1,4],
            [1,1,0],
            [1,1,1],
            [1,1,1]
        ])
        outcome = g.getGameEnded(board, 1)
        self.assertEquals(1, outcome)
        outcome = g.getGameEnded(board, -1)
        self.assertEquals(-1, outcome)

        board = np.array([
            [1,1,4],
            [1,1,5],
            [1,1,0],
            [1,1,1],
            [1,1,1]
        ])
        outcome = g.getGameEnded(board, 1)
        self.assertEquals(-1, outcome)
        outcome = g.getGameEnded(board, -1)
        self.assertEquals(1, outcome)

        board = np.array([
            [1,1,5],
            [1,1,5],
            [1,1,0],
            [1,1,1],
            [1,1,1]
        ])
        outcome = g.getGameEnded(board, 1)
        self.assertEquals(-1, outcome)
        outcome = g.getGameEnded(board, -1)
        self.assertEquals(1, outcome)

    def testSymmetriesN2(self):
        g = DotsAndBoxesGame(n=2)
        board = np.array([
            [0,1,0],
            [2,3,0],
            [4,5,0],
            [6,7,8],
            [9,10,11]
        ])
        pi = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])
        sym = g.getSymmetries(board, pi)

        self.assertTrue(np.array_equal(sym[0][0], np.array([
            [11,  8,  0],
            [10,  7,  0],
            [9,  6,   0],
            [5,  3,   1],
            [4,  2,   0]
        ])))
        self.assertTrue(np.array_equal(sym[0][1], np.array([11, 8, 10, 7, 9, 6, 5, 3, 1, 4, 2, 0, 12])))

        self.assertTrue(np.array_equal(sym[1][0], np.array([
            [8,  11,  0],
            [7,  10,  0],
            [6,  9,   0],
            [1,  3,   5],
            [0,  2,   4]
        ])))
        self.assertTrue(np.array_equal(sym[1][1], np.array([8, 11, 7, 10, 6, 9, 1, 3, 5, 0, 2, 4, 12])))

        self.assertTrue(np.array_equal(sym[2][0], np.array([
            [4,  5,  0],
            [2,  3,  0],
            [0,  1,  0],
            [9,  10, 11],
            [6,  7,  8]
        ])))
        self.assertTrue(np.array_equal(sym[2][1], np.array([4, 5, 2, 3, 0, 1, 9, 10, 11, 6, 7, 8, 12])))

        self.assertTrue(np.array_equal(sym[3][0], np.array([
            [5,  4,  0],
            [3,  2,  0],
            [1,  0,  0],
            [11, 10, 9],
            [8,  7,  6]
        ])))
        self.assertTrue(np.array_equal(sym[3][1], np.array([5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12])))

        self.assertTrue(np.array_equal(sym[4][0], np.array([
            [6,  9,  0],
            [7,  10, 0],
            [8,  11, 0],
            [0,  2,  4],
            [1,  3,  5]
        ])))
        self.assertTrue(np.array_equal(sym[4][1], np.array([6, 9, 7, 10, 8, 11, 0, 2, 4, 1, 3, 5, 12])))

        self.assertTrue(np.array_equal(sym[5][0], np.array([
            [9,  6,  0],
            [10, 7,  0],
            [11, 8,  0],
            [4,  2,  0],
            [5,  3,  1]
        ])))
        self.assertTrue(np.array_equal(sym[5][1], np.array([9, 6, 10, 7, 11, 8, 4, 2, 0, 5, 3, 1, 12])))

        self.assertTrue(np.array_equal(sym[6][0], np.array([
            [1,  0,  0],
            [3,  2,  0],
            [5,  4,  0],
            [8,  7,  6],
            [11, 10, 9]
        ])))
        self.assertTrue(np.array_equal(sym[6][1], np.array([1, 0, 3, 2, 5, 4, 8, 7, 6, 11, 10, 9, 12])))

        self.assertTrue(np.array_equal(sym[7][0], np.array([
            [0,  1,  0],
            [2,  3,  0],
            [4,  5,  0],
            [6,  7,  8],
            [9,  10, 11]
        ])))
        self.assertTrue(np.array_equal(sym[7][1], np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])))

    def testSymmetriesN2_score(self):
        g = DotsAndBoxesGame(n=2)
        board = np.array([
            [0,1,1],
            [2,3,2],
            [4,5,1],
            [6,7,8],
            [9,10,11]
        ])
        pi = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])
        sym = g.getSymmetries(board, pi)

        self.assertTrue(np.array_equal(sym[0][0], np.array([
            [11,  8,  1],
            [10,  7,  2],
            [9,  6,   1],
            [5,  3,   1],
            [4,  2,   0]
        ])))
        self.assertTrue(np.array_equal(sym[0][1], np.array([11, 8, 10, 7, 9, 6, 5, 3, 1, 4, 2, 0, 12])))

        self.assertTrue(np.array_equal(sym[1][0], np.array([
            [8,  11,  1],
            [7,  10,  2],
            [6,  9,   1],
            [1,  3,   5],
            [0,  2,   4]
        ])))
        self.assertTrue(np.array_equal(sym[1][1], np.array([8, 11, 7, 10, 6, 9, 1, 3, 5, 0, 2, 4, 12])))

        self.assertTrue(np.array_equal(sym[2][0], np.array([
            [4,  5,  1],
            [2,  3,  2],
            [0,  1,  1],
            [9,  10, 11],
            [6,  7,  8]
        ])))
        self.assertTrue(np.array_equal(sym[2][1], np.array([4, 5, 2, 3, 0, 1, 9, 10, 11, 6, 7, 8, 12])))

        self.assertTrue(np.array_equal(sym[3][0], np.array([
            [5,  4,  1],
            [3,  2,  2],
            [1,  0,  1],
            [11, 10, 9],
            [8,  7,  6]
        ])))
        self.assertTrue(np.array_equal(sym[3][1], np.array([5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12])))

        self.assertTrue(np.array_equal(sym[4][0], np.array([
            [6,  9,  1],
            [7,  10, 2],
            [8,  11, 1],
            [0,  2,  4],
            [1,  3,  5]
        ])))
        self.assertTrue(np.array_equal(sym[4][1], np.array([6, 9, 7, 10, 8, 11, 0, 2, 4, 1, 3, 5, 12])))

        self.assertTrue(np.array_equal(sym[5][0], np.array([
            [9,  6,  1],
            [10, 7,  2],
            [11, 8,  1],
            [4,  2,  0],
            [5,  3,  1]
        ])))
        self.assertTrue(np.array_equal(sym[5][1], np.array([9, 6, 10, 7, 11, 8, 4, 2, 0, 5, 3, 1, 12])))

        self.assertTrue(np.array_equal(sym[6][0], np.array([
            [1,  0,  1],
            [3,  2,  2],
            [5,  4,  1],
            [8,  7,  6],
            [11, 10, 9]
        ])))
        self.assertTrue(np.array_equal(sym[6][1], np.array([1, 0, 3, 2, 5, 4, 8, 7, 6, 11, 10, 9, 12])))

        self.assertTrue(np.array_equal(sym[7][0], np.array([
            [0,  1,  1],
            [2,  3,  2],
            [4,  5,  1],
            [6,  7,  8],
            [9,  10, 11]
        ])))
        self.assertTrue(np.array_equal(sym[7][1], np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])))

    def testSymmetriesN3(self):
        g = DotsAndBoxesGame(n=3)
        board = np.array([
            [0,  1,  2,  0],
            [3,  4,  5,  0],
            [6,  7,  8,  0],
            [9,  10, 11,  0],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]
        ])
        pi = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
        sym = g.getSymmetries(board, pi)

        self.assertTrue(np.array_equal(sym[0][0], np.array([
            [23, 19, 15,  0],
            [22, 18, 14,  0],
            [21, 17, 13,  0],
            [20, 16, 12,  0],
            [11, 8, 5, 2],
            [10, 7, 4, 1],
            [9, 6, 3, 0]
        ])))
        self.assertTrue(np.array_equal(sym[0][1], np.array([23,19,15,22,18,14,21,17,13,20,16,12,11,8,5,2,10,7,4,1,9,6,3,0,24])))
        self.assertTrue(np.array_equal(sym[1][0], np.array([
            [15, 19, 23,  0],
            [14, 18, 22,  0],
            [13, 17, 21,  0],
            [12, 16, 20,  0],
            [2,  5,  8,  11],
            [1,  4,  7,  10],
            [0,  3,  6,   9]
        ])))
        self.assertTrue(np.array_equal(sym[1][1], np.array([15,19,23,14,18,22,13,17,21,12,16,20,2,5,8,11,1,4,7,10,0,3,6,9,24])))

        self.assertTrue(np.array_equal(sym[2][0], np.array([
            [9,  10, 11, 0],
            [6,  7,  8,  0],
            [3,  4,  5,  0],
            [0,  1,  2,  0],
            [20, 21, 22, 23],
            [16, 17, 18, 19],
            [12, 13, 14, 15]
        ])))
        self.assertTrue(np.array_equal(sym[2][1], np.array([9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2, 20, 21, 22, 23, 16, 17, 18, 19, 12, 13, 14, 15, 24])))
        self.assertTrue(np.array_equal(sym[3][0], np.array([
            [11, 10, 9,  0],
            [8,  7,  6,  0],
            [5,  4,  3,  0],
            [2,  1,  0,  0],
            [23, 22, 21, 20],
            [19, 18, 17, 16],
            [15, 14, 13, 12]
        ])))
        self.assertTrue(np.array_equal(sym[3][1], np.array([11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 24])))


if __name__ == '__main__':
    unittest.main()
