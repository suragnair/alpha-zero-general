"""
To run tests:
python -m unittest tictactoe_4d.test_tictactoe_4d
"""
import itertools
import unittest

import numpy as np
import torch
import torch.nn as nn

from utils import dotdict
from .TicTacToe4DGame import TicTacToe4DGame
from .TicTacToe4DLogic import Board, forced_moves, winning_lines
from .TicTacToe4DPlayers import OneStepPlayer, RandomPlayer
from .pytorch.TicTacToe4DNNet import Conv4d, TicTacToe4DNNet


class TestRules(unittest.TestCase):
    def test_number_of_lines_matches_the_formula(self):
        for n in (2, 3, 4, 5):
            self.assertEqual(len(winning_lines(n)), ((n + 2) ** 4 - n ** 4) // 2)

    def test_every_line_wins(self):
        for n in (3, 4):
            for line in winning_lines(n):
                board = Board(n)
                board.pieces.ravel()[line] = 1
                self.assertTrue(board.is_win(1), line)
                self.assertFalse(board.is_win(-1), line)

    def test_no_win_without_a_whole_line(self):
        n = 4
        lines = [set(line.tolist()) for line in winning_lines(n)]
        rng = np.random.default_rng(1)
        for _ in range(300):
            stones = set(rng.choice(n ** 4, size=int(rng.integers(1, 40)), replace=False).tolist())
            board = Board(n)
            board.pieces.ravel()[list(stones)] = 1
            self.assertEqual(board.is_win(1), any(line <= stones for line in lines))

    def test_game_result_follows_the_player(self):
        game = TicTacToe4DGame(3)
        board = game.getInitBoard()
        player = 1
        for action in (0, 1, 40, 2, 80):  # white takes the main diagonal of the hypercube
            board, player = game.getNextState(board, player, action)
        self.assertEqual(game.getGameEnded(board, 1), 1)
        self.assertEqual(game.getGameEnded(board, -1), -1)
        self.assertEqual(game.getGameEnded(game.getCanonicalForm(board, -1), 1), -1)

    def test_valid_moves(self):
        game = TicTacToe4DGame(3)
        board, _ = game.getNextState(game.getInitBoard(), 1, 5)
        valids = game.getValidMoves(board, -1)
        self.assertEqual(len(valids), game.getActionSize())
        self.assertEqual(valids[5], 0)
        self.assertEqual(valids.sum(), 80)


class TestForcedMoves(unittest.TestCase):
    def test_nothing_forced_on_a_quiet_board(self):
        board = np.zeros((4,) * 4, dtype=int)
        board.ravel()[[0, 1]] = 1
        board.ravel()[[85, 170]] = -1
        squares, color = forced_moves(board)
        self.assertEqual((len(squares), color), (0, 0))

    def test_own_line_comes_before_a_block(self):
        board = np.zeros((4,) * 4, dtype=int)
        board.ravel()[[0, 1, 2]] = 1
        board.ravel()[[252, 253, 254]] = -1
        squares, color = forced_moves(board)
        self.assertEqual(squares.tolist(), [3])
        self.assertEqual(color, 1)

    def test_every_block_is_listed(self):
        board = np.zeros((4,) * 4, dtype=int)
        board.ravel()[[0, 1, 2, 252, 253, 254]] = -1
        squares, color = forced_moves(board)
        self.assertEqual(squares.tolist(), [3, 255])
        self.assertEqual(color, -1)

    def test_a_stone_of_the_other_side_kills_the_line(self):
        board = np.zeros((4,) * 4, dtype=int)
        board.ravel()[[0, 1, 2]] = 1
        board.ravel()[3] = -1
        self.assertEqual(len(forced_moves(board)[0]), 0)


class TestSymmetries(unittest.TestCase):
    def test_board_and_policy_move_together(self):
        game = TicTacToe4DGame(4)
        marks = np.arange(256.0).reshape((4,) * 4)
        pi = list(np.arange(256.0))
        symmetries = game.getSymmetries(marks, pi)
        self.assertEqual(len(symmetries), 16)
        self.assertTrue(any(np.array_equal(b, marks) for b, _ in symmetries), 'the position itself')
        self.assertEqual(len({b.tobytes() for b, _ in symmetries}), 16)
        for b, p in symmetries:
            self.assertTrue(np.array_equal(b.ravel(), np.array(p)))

    def test_symmetries_keep_the_lines(self):
        game = TicTacToe4DGame(4)
        lines = {frozenset(line.tolist()) for line in winning_lines(4)}
        cells = np.arange(256).reshape((4,) * 4)
        for b, _ in game.getSymmetries(cells, list(range(256))):
            where = np.argsort(b.ravel())
            self.assertEqual({frozenset(where[list(line)].tolist()) for line in lines}, lines)


class TestPlayers(unittest.TestCase):
    def test_one_step_player_wins_and_blocks(self):
        game = TicTacToe4DGame(4)
        board = game.getInitBoard()
        for action in (0, 85, 170):
            board.ravel()[action] = 1
        self.assertEqual(OneStepPlayer(game).play(board), 255)
        self.assertEqual(OneStepPlayer(game).play(-board), 255)

    def test_random_player_makes_legal_moves(self):
        game = TicTacToe4DGame(3)
        board, _ = game.getNextState(game.getInitBoard(), 1, 0)
        for _ in range(50):
            self.assertNotEqual(RandomPlayer(game).play(board), 0)


class TestNetwork(unittest.TestCase):
    def test_conv4d_matches_the_definition(self):
        torch.manual_seed(1)
        conv = Conv4d(2, 3)
        x = torch.randn(2, 2, 3, 3, 3, 3)
        fast = conv(x)
        kernel = conv.conv.weight.detach().reshape(3, 3, 2, 3, 3, 3)  # step along w, output, input, 3×3×3
        slow = torch.zeros_like(fast)
        for out, channel in itertools.product(range(3), range(2)):
            for point in itertools.product(range(3), repeat=4):
                for shift in itertools.product((-1, 0, 1), repeat=4):
                    source = [p + s for p, s in zip(point, shift)]
                    if all(0 <= c < 3 for c in source):
                        weight = kernel[shift[0] + 1, out, channel, shift[1] + 1, shift[2] + 1, shift[3] + 1]
                        slow[(slice(None), out) + point] += weight * x[(slice(None), channel) + tuple(source)]
        self.assertLess((fast - slow).abs().max().item(), 1e-4)

    def test_outputs(self):
        game = TicTacToe4DGame(4)
        net = TicTacToe4DNNet(game, dotdict({'num_channels': 8, 'num_blocks': 1})).eval()
        pi, v = net(torch.zeros(2, 4, 4, 4, 4))
        self.assertEqual(pi.shape, (2, 256))
        self.assertEqual(v.shape, (2,))
        self.assertAlmostEqual(pi.exp().sum(dim=1)[0].item(), 1, places=4)

    def test_hints_count_lines(self):
        game = TicTacToe4DGame(4)
        net = TicTacToe4DNNet(game, dotdict({'num_channels': 8, 'num_blocks': 1}))
        board = torch.zeros(1, 4, 4, 4, 4)
        board.view(-1)[0] = 1  # one stone in a corner: 15 lines pass through it
        planes = net.planes(board)
        self.assertEqual(planes.shape, (1, 9, 4, 4, 4, 4))
        self.assertAlmostEqual((planes[0, 2].flatten()[0] * net.scale).item(), 15)


if __name__ == '__main__':
    unittest.main()
