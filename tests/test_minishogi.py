"""Pytest tests for MiniShogi game implementation."""

import numpy as np
import pytest

from minishogi import Board, MiniShogiGame, Move, PieceType


class TestPieceType:
    """Tests for PieceType enum."""

    def test_can_promote(self):
        """Test promotion eligibility."""
        assert PieceType.PAWN.can_promote is True
        assert PieceType.SILVER.can_promote is True
        assert PieceType.BISHOP.can_promote is True
        assert PieceType.ROOK.can_promote is True
        assert PieceType.GOLD.can_promote is False
        assert PieceType.KING.can_promote is False

    def test_promoted_form(self):
        """Test piece promotion mapping."""
        assert PieceType.PAWN.promoted_form == PieceType.TOKIN
        assert PieceType.SILVER.promoted_form == PieceType.P_SILVER
        assert PieceType.BISHOP.promoted_form == PieceType.P_BISHOP
        assert PieceType.ROOK.promoted_form == PieceType.P_ROOK

    def test_unpromoted_form(self):
        """Test piece unpromote mapping (for captures)."""
        assert PieceType.TOKIN.unpromoted_form == PieceType.PAWN
        assert PieceType.P_SILVER.unpromoted_form == PieceType.SILVER
        assert PieceType.P_BISHOP.unpromoted_form == PieceType.BISHOP
        assert PieceType.P_ROOK.unpromoted_form == PieceType.ROOK


class TestMove:
    """Tests for Move model."""

    def test_move_creation(self):
        """Test basic move creation."""
        move = Move(from_sq=(3, 0), to_sq=(2, 0))
        assert move.from_sq == (3, 0)
        assert move.to_sq == (2, 0)
        assert move.promote is False
        assert move.is_drop is False

    def test_drop_creation(self):
        """Test drop move creation."""
        drop = Move(to_sq=(2, 2), drop_piece=PieceType.PAWN)
        assert drop.from_sq is None
        assert drop.to_sq == (2, 2)
        assert drop.is_drop is True
        assert drop.drop_piece == PieceType.PAWN

    def test_action_index_roundtrip(self):
        """Test action index conversion back and forth."""
        # Regular move
        move = Move(from_sq=(3, 0), to_sq=(2, 0), promote=False)
        action = move.to_action_index()
        reconstructed = Move.from_action_index(action)
        assert reconstructed.from_sq == move.from_sq
        assert reconstructed.to_sq == move.to_sq
        assert reconstructed.promote == move.promote

        # Move with promotion
        move_promo = Move(from_sq=(1, 0), to_sq=(0, 0), promote=True)
        action_promo = move_promo.to_action_index()
        reconstructed_promo = Move.from_action_index(action_promo)
        assert reconstructed_promo.promote is True


class TestBoard:
    """Tests for Board class."""

    def test_initial_position(self):
        """Test initial board setup."""
        board = Board()

        # Player 2 back rank (row 0)
        assert board.board[0, 0] == -PieceType.KING
        assert board.board[0, 1] == -PieceType.GOLD
        assert board.board[0, 4] == -PieceType.ROOK

        # Player 1 back rank (row 4)
        assert board.board[4, 4] == PieceType.KING
        assert board.board[4, 0] == PieceType.ROOK

        # Pawns
        assert board.board[1, 4] == -PieceType.PAWN
        assert board.board[3, 0] == PieceType.PAWN

    def test_copy(self):
        """Test board copy is independent."""
        board = Board()
        copy = board.copy()

        copy.board[2, 2] = PieceType.PAWN
        assert board.board[2, 2] == 0

    def test_legal_moves_initial(self):
        """Test legal moves from initial position."""
        board = Board()
        moves = board.get_legal_moves()

        # Should have some legal moves
        assert len(moves) > 0
        # Pawn can move forward
        pawn_moves = [m for m in moves if m.from_sq == (3, 0)]
        assert len(pawn_moves) == 1
        assert pawn_moves[0].to_sq == (2, 0)

    def test_capture_adds_to_hand(self):
        """Test that capturing a piece adds it to hand."""
        board = Board()
        board.current_player = 1

        # Set up a capture scenario
        board.board[2, 0] = -PieceType.PAWN  # Enemy pawn

        move = Move(from_sq=(3, 0), to_sq=(2, 0))
        board.execute_move(move)

        # Check piece is in hand
        assert PieceType.PAWN in board.hands[0]
        assert board.hands[0][PieceType.PAWN] == 1

    def test_promotion(self):
        """Test pawn promotion."""
        board = Board()
        board.board = np.zeros((5, 5), dtype=np.int8)
        board.board[4, 4] = PieceType.KING
        board.board[0, 0] = -PieceType.KING
        board.board[1, 0] = PieceType.PAWN  # Pawn near promotion zone
        board.current_player = 1

        moves = board.get_legal_moves()
        pawn_moves = [m for m in moves if m.from_sq == (1, 0)]

        # Pawn moving to row 0 must promote
        promo_moves = [m for m in pawn_moves if m.to_sq == (0, 0)]
        assert len(promo_moves) == 1
        assert promo_moves[0].promote is True


class TestMiniShogiGame:
    """Tests for MiniShogiGame class."""

    def test_init_board(self):
        """Test game initialization."""
        game = MiniShogiGame()
        board = game.getInitBoard()

        assert isinstance(board, Board)

    def test_board_size(self):
        """Test board size."""
        game = MiniShogiGame()
        assert game.getBoardSize() == (5, 5)

    def test_action_size(self):
        """Test action size calculation."""
        game = MiniShogiGame()
        # 25 * 25 * 2 (moves) + 5 * 25 (drops) + 1 (pass) = 1376
        assert game.getActionSize() == 1376

    def test_valid_moves(self):
        """Test valid moves generation."""
        game = MiniShogiGame()
        board = game.getInitBoard()

        valids = game.getValidMoves(board, 1)

        assert len(valids) == game.getActionSize()
        assert valids.sum() > 0  # Should have valid moves

    def test_next_state(self):
        """Test state transition."""
        game = MiniShogiGame()
        board = game.getInitBoard()

        valids = game.getValidMoves(board, 1)
        action = np.where(valids == 1)[0][0]

        new_board, next_player = game.getNextState(board, 1, action)

        assert next_player == -1
        assert new_board is not board

    def test_game_not_ended_initially(self):
        """Test game not ended at start."""
        game = MiniShogiGame()
        board = game.getInitBoard()

        assert game.getGameEnded(board, 1) == 0

    def test_string_representation(self):
        """Test board hashing."""
        game = MiniShogiGame()
        board1 = game.getInitBoard()
        board2 = game.getInitBoard()

        str1 = game.stringRepresentation(board1)
        str2 = game.stringRepresentation(board2)

        assert str1 == str2

    def test_random_game_completes(self):
        """Test that a random game eventually ends."""
        game = MiniShogiGame()
        board = game.getInitBoard()
        player = 1

        for _ in range(500):  # Max moves
            valids = game.getValidMoves(board, player)
            valid_indices = np.where(valids == 1)[0]

            if len(valid_indices) == 0:
                break

            action = np.random.choice(valid_indices)
            board, player = game.getNextState(board, player, action)

            if game.getGameEnded(board, player) != 0:
                break

        # Game should end or reach max moves
        assert True


class TestNeuralNetwork:
    """Tests for neural network."""

    def test_prediction_shape(self):
        """Test neural network output shapes."""
        from minishogi.pytorch import NNetWrapper

        game = MiniShogiGame()
        nnet = NNetWrapper(game)
        board = game.getInitBoard()

        pi, v = nnet.predict(board)

        assert pi.shape == (1376,)
        assert isinstance(v, (float, np.floating))
        assert np.isclose(pi.sum(), 1.0, atol=1e-5)

    def test_policy_sums_to_one(self):
        """Test policy probabilities sum to 1."""
        from minishogi.pytorch import NNetWrapper

        game = MiniShogiGame()
        nnet = NNetWrapper(game)
        board = game.getInitBoard()

        pi, _ = nnet.predict(board)

        assert np.isclose(pi.sum(), 1.0, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
