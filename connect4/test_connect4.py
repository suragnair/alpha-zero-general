"""
To run tests:
pytest-3 connect4
"""

from collections import namedtuple
import textwrap
import numpy as np

from .Connect4Game import Connect4Game

# Tuple of (Board, Player, Game) to simplify testing.
BPGTuple = namedtuple('BPGTuple', 'board player game')


def init_board_from_moves(moves, height=None, width=None):
    """Returns a BPGTuple based on series of specified moved."""
    game = Connect4Game(height=height, width=width)
    board, player = game.getInitBoard(), 1
    for move in moves:
        board, player = game.getNextState(board, player, move)
    return BPGTuple(board, player, game)


def init_board_from_array(board, player):
    """Returns a BPGTuple based on series of specified moved."""
    game = Connect4Game(height=len(board), width=len(board[0]))
    return BPGTuple(board, player, game)


def test_simple_moves():
    board, player, game = init_board_from_moves([4, 5, 4, 3, 0, 6])
    expected = textwrap.dedent("""\
        [[ 0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  1.  0.  0.]
         [ 1.  0.  0. -1.  1. -1. -1.]]""")
    assert expected == game.stringRepresentation(board)


def test_overfull_column():
    for height in range(1, 10):
        # Fill to max height is ok
        init_board_from_moves([4] * height, height=height)

        # Check overfilling causes an error.
        try:
            init_board_from_moves([4] * (height + 1), height=height)
            assert False, "Expected error when overfilling column"
        except ValueError:
            pass  # Expected.


def test_get_valid_moves():
    """Tests vector of valid moved is correct."""
    move_valid_pairs = [
        ([], [True] * 7),
        ([0, 1, 2, 3, 4, 5, 6], [True] * 7),
        ([0, 1, 2, 3, 4, 5, 6] * 5, [True] * 7),
        ([0, 1, 2, 3, 4, 5, 6] * 6, [False] * 7),
        ([0, 1, 2] * 3 + [3, 4, 5, 6] * 6, [True] * 3 + [False] * 4),
    ]

    for moves, expected_valid in move_valid_pairs:
        board, player, game = init_board_from_moves(moves)
        assert (np.array(expected_valid) == game.getValidMoves(board, player)).all()


def test_symmetries():
    """Tests symetric board are produced."""
    board, player, game = init_board_from_moves([0, 0, 1, 0, 6])
    pi = 0.8
    (board1, pi1), (board2, pi2) = game.getSymmetries(board, pi)
    assert pi == pi1 and pi == pi2

    expected_board1 = textwrap.dedent("""\
        [[ 0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.]
         [-1.  0.  0.  0.  0.  0.  0.]
         [-1.  0.  0.  0.  0.  0.  0.]
         [ 1.  1.  0.  0.  0.  0.  1.]]""")
    assert expected_board1 == game.stringRepresentation(board1)

    expected_board2 = textwrap.dedent("""\
        [[ 0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0. -1.]
         [ 0.  0.  0.  0.  0.  0. -1.]
         [ 1.  0.  0.  0.  0.  1.  1.]]""")
    assert expected_board2 == game.stringRepresentation(board2)


def test_game_ended():
    """Tests game end detection logic based on fixed boards."""
    array_end_state_pairs = [
        (np.array([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]]), 1, 0),
        (np.array([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]]), 1, 1),
        (np.array([[0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]]), -1, -1),
        (np.array([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0]]), -1, -1),
        (np.array([[0, 0, 0, -1],
                   [0, 0, -1, 0],
                   [0, -1, 0, 0],
                   [-1, 0, 0, 0]]), 1, -1),
        (np.array([[0, 0, 0, 0, 1],
                   [0, 0, 0, 1, 0],
                   [0, 0, 1, 0, 0],
                   [0, 1, 0, 0, 0]]), -1, -1),
        (np.array([[1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0]]), -1, -1),
        (np.array([[ 0,  0,  0,  0,  0,  0,  0],
                   [ 0,  0,  0, -1,  0,  0,  0],
                   [ 0,  0,  0, -1,  0,  0,  1],
                   [ 0,  0,  0,  1,  1, -1, -1],
                   [ 0,  0,  0, -1,  1,  1,  1],
                   [ 0, -1,  0, -1,  1, -1,  1]]), -1, 0),
        (np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0., -1.,  0.,  0.,  0.],
                   [ 1.,  0.,  1., -1.,  0.,  0.,  0.],
                   [-1., -1.,  1.,  1.,  0.,  0.,  0.],
                   [ 1.,  1.,  1., -1.,  0.,  0.,  0.],
                   [ 1., -1.,  1., -1.,  0., -1.,  0.]]), -1, -1),
        (np.array([[ 0.,  0.,  0.,  1.,  0.,  0.,  0.,],
                   [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,],
                   [ 0.,  0.,  0., -1.,  0.,  0.,  0.,],
                   [ 0.,  0.,  1.,  1., -1.,  0., -1.,],
                   [ 0.,  0., -1.,  1.,  1.,  1.,  1.,],
                   [-1.,  0., -1.,  1., -1., -1., -1.,],]), 1, 1),
        ]

    for np_pieces, player, expected_end_state in array_end_state_pairs:
        board, player, game = init_board_from_array(np_pieces, player)
        end_state = game.getGameEnded(board, player)
        assert expected_end_state == end_state, ("expected=%s, actual=%s, board=\n%s" % (expected_end_state, end_state, board))


def test_immutable_move():
    """Test original board is not mutated whtn getNextState() called."""
    board, player, game = init_board_from_moves([1, 2, 3, 3, 4])
    original_board_string = game.stringRepresentation(board)

    new_np_pieces, new_player = game.getNextState(board, 3, -1)

    assert original_board_string == game.stringRepresentation(board)
    assert original_board_string != game.stringRepresentation(new_np_pieces)
