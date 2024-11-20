import unittest
import numpy as np
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append('../../')
from gomoku.GomokuLogic import Board 

class TestBoard(unittest.TestCase):

    def setUp(self):
        self.n = 8
        self.board = Board(self.n)

    def test_initial_board_empty_return_true(self):
        # Ensure the initial board is empty
        self.assertTrue(self.board.is_initial_board())
        self.assertEqual(np.count_nonzero(self.board.pieces), 0)


    def test_initial_board_empty_return_false(self):
        # Ensure the initial board is empty
        self.assertTrue(self.board.is_initial_board())
        self.assertEqual(np.count_nonzero(self.board.pieces), 0)
        self.board.execute_move((3,3),1)

        self.assertFalse(self.board.is_initial_board())
        self.assertEqual(np.count_nonzero(self.board.pieces), 1)
        
    def test_is_board_full(self):
        """Test that the board can detect when it is full."""
        for i in range(self.n):
            for j in range(self.n):
                self.board.execute_move((i, j), 1)
        self.assertTrue(self.board.is_board_full())
        
    def test_execute_move(self):
        # Test placing a piece on the board
        self.board.execute_move((3, 3), 1)
        self.assertEqual(self.board[3][3], 1)
        self.board.execute_move((4, 4), -1)
        self.assertEqual(self.board[4][4], -1)

    def test_has_five_in_a_row_player_1_horizontal(self):
        # Test detecting a five-in-a-row condition
        for i in range(5):
            self.board.execute_move((i, 0), 1)
        self.assertTrue(self.board.has_five_in_a_row(1))
        self.assertFalse(self.board.has_five_in_a_row(-1))

    def test_has_five_in_a_row_player_1_vertical(self):
        # Test detecting a five-in-a-row condition
        for i in range(5):
            self.board.execute_move((0, i), 1)
        self.assertTrue(self.board.has_five_in_a_row(1))
        self.assertFalse(self.board.has_five_in_a_row(-1))

    def test_has_five_in_a_row_player_2_diagnol_1(self):
        # Test detecting a five-in-a-row condition
        for i in range(5):
            self.board.execute_move((i, i), -1)
        self.assertTrue(self.board.has_five_in_a_row(-1))
        self.assertFalse(self.board.has_five_in_a_row(1))


    def test_has_five_in_a_row_player_2_diagnol_2(self):
        # Test detecting a five-in-a-row condition
        for i in range(5):
            self.board.execute_move((self.n - i - 1, self.n - i - 1), -1)
        self.assertTrue(self.board.has_five_in_a_row(-1))
        self.assertFalse(self.board.has_five_in_a_row(1))

    def test_no_five_in_a_row(self):
        # Test when there is no five-in-a-row pattern
        for i in range(4):
            self.board.execute_move((i, 0), 1)
        self.assertFalse(self.board.has_five_in_a_row(1))

    def test_get_legal_moves_initial_board(self):
        # Test legal moves on an initial board (should start in the center)
        legal_moves = self.board.get_legal_moves(1)
        expected_move = {(self.n // 2, self.n // 2)}
        self.assertEqual(set(legal_moves), expected_move)

    def test_get_legal_moves_with_one_stone(self):
        # Place some stones and test legal move generation
        self.board.execute_move((3, 3), 1)
        legal_moves = self.board.get_legal_moves(-1)

        expected_moves = {(1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                          (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
                          (3, 1), (3, 2), (3, 4), (3, 5), 
                          (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
                          (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)}

        self.assertEqual(set(legal_moves), expected_moves)
        
    def test_get_legal_moves_with_current_player_has_4_in_a_row(self):
        # Place some stones and test legal move generation
        self.board.execute_move((1, 1), 1)
        self.board.execute_move((2, 2), 1)
        self.board.execute_move((3, 3), 1)
        self.board.execute_move((4, 4), 1)

        legal_moves = self.board.get_legal_moves(1)

        logger.info(legal_moves)
        self.assertEqual(set(legal_moves), set([(0, 0), (5, 5)]))

    def test_get_legal_moves_with_opponent_player_has_4_in_a_row(self):
        # Place some stones and test legal move generation
        self.board.execute_move((1, 1), 1)
        self.board.execute_move((2, 2), 1)
        self.board.execute_move((3, 3), 1)
        self.board.execute_move((4, 4), 1)

        legal_moves = self.board.get_legal_moves(-1)

        self.assertEqual(set(legal_moves), set([(0, 0), (5, 5)]))

    def test_get_legal_moves_with_opponent_player_has_4_in_a_row(self):
        # Place some stones and test legal move generation
        self.board.execute_move((1, 1), 1)
        self.board.execute_move((2, 2), 1)
        self.board.execute_move((3, 3), 1)
        self.board.execute_move((4, 4), 1)
        self.board.execute_move((5, 5), -1)

        legal_moves = self.board.get_legal_moves(-1)

        self.assertEqual(set(legal_moves), set([(0, 0)]))

    def test_is_within_board(self):
        # Test the is_within_board function
        self.assertTrue(self.board.is_within_board(0, 0))
        self.assertTrue(self.board.is_within_board(self.n - 1, self.n - 1))
        self.assertFalse(self.board.is_within_board(-1, 0))
        self.assertFalse(self.board.is_within_board(0, self.n))

    def test_get_all_available_moves_next_to(self):
        # Test available moves within a specified distance
        self.board.execute_move((3, 3), 1)
        moves = self.board._get_all_available_moves_next_to(3, 3, 1)
        expected_moves = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 4), (4, 2), (4, 3), (4, 4)]
        self.assertCountEqual(moves, expected_moves)

if __name__ == "__main__":
    unittest.main()
