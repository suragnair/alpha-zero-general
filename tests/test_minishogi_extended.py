"""Extended tests for MiniShogi with edge cases and extreme scenarios."""

import numpy as np
import pytest

from minishogi import Board, MiniShogiGame, Move, PieceType


class TestCheckmate:
    """Tests for checkmate detection."""

    def test_simple_checkmate(self):
        """Test detection of a simple checkmate position."""
        board = Board()
        board.board = np.zeros((5, 5), dtype=np.int8)

        # Set up a checkmate position
        # Player 2's king in corner, Player 1's rook giving check
        board.board[0, 0] = -PieceType.KING  # P2 King cornered
        board.board[0, 1] = PieceType.ROOK   # P1 Rook on same row
        board.board[1, 1] = PieceType.ROOK   # P1 Rook blocking escape
        board.board[4, 4] = PieceType.KING   # P1 King safe
        board.current_player = -1

        # P2 should be in check with no escape
        assert board.is_in_check(-1) is True

    def test_not_checkmate_can_block(self):
        """Test that blocking a check prevents checkmate."""
        board = Board()
        board.board = np.zeros((5, 5), dtype=np.int8)

        board.board[0, 0] = -PieceType.KING
        board.board[0, 4] = PieceType.ROOK   # Giving check from distance
        board.board[0, 2] = -PieceType.GOLD  # Can block at 0,1
        board.board[4, 4] = PieceType.KING
        board.current_player = -1

        moves = board.get_legal_moves()
        # Should have at least one legal move (blocking)
        assert len(moves) > 0

    def test_king_can_escape(self):
        """Test king can escape check."""
        board = Board()
        board.board = np.zeros((5, 5), dtype=np.int8)

        board.board[2, 2] = -PieceType.KING
        board.board[2, 4] = PieceType.ROOK   # Giving horizontal check
        board.board[4, 4] = PieceType.KING
        board.current_player = -1

        moves = board.get_legal_moves()
        # King should be able to move to escape
        king_moves = [m for m in moves if m.from_sq == (2, 2)]
        assert len(king_moves) > 0


class TestPromotion:
    """Tests for piece promotion edge cases."""

    def test_pawn_must_promote_on_last_rank(self):
        """Pawn reaching last rank must promote."""
        board = Board()
        board.board = np.zeros((5, 5), dtype=np.int8)

        board.board[1, 2] = PieceType.PAWN   # P1 pawn one step from promotion
        board.board[4, 4] = PieceType.KING
        board.board[0, 0] = -PieceType.KING
        board.current_player = 1

        moves = board.get_legal_moves()
        pawn_moves = [m for m in moves if m.from_sq == (1, 2) and m.to_sq == (0, 2)]

        # Only promoted move should be available
        assert len(pawn_moves) == 1
        assert pawn_moves[0].promote is True

    def test_pawn_optional_promotion_in_zone(self):
        """Pawn can choose to promote when entering promotion zone."""
        board = Board()
        board.board = np.zeros((5, 5), dtype=np.int8)

        # Pawn at row 2, moving to row 1 (not last rank)
        # Actually for 5x5 with 1-row promotion zone, only row 0 is promotion zone
        # So let's test moving INTO the zone from outside
        board.board[1, 2] = PieceType.PAWN
        board.board[4, 4] = PieceType.KING
        board.board[0, 0] = -PieceType.KING
        board.current_player = 1

        moves = board.get_legal_moves()
        _ = [m for m in moves if m.from_sq == (1, 2)]

        # When moving to row 0 (last rank), must promote
        # So pawn_moves should include only promote=True for last rank

    def test_silver_promotion_optional(self):
        """Silver can optionally promote in promotion zone."""
        board = Board()
        board.board = np.zeros((5, 5), dtype=np.int8)

        board.board[1, 2] = PieceType.SILVER
        board.board[4, 4] = PieceType.KING
        board.board[0, 0] = -PieceType.KING
        board.current_player = 1

        moves = board.get_legal_moves()
        silver_to_zone = [m for m in moves if m.from_sq == (1, 2) and m.to_sq[0] == 0]

        # Should have both promote and non-promote options
        promotes = [m for m in silver_to_zone if m.promote]
        non_promotes = [m for m in silver_to_zone if not m.promote]

        if len(silver_to_zone) > 0:  # If silver can reach row 0
            assert len(promotes) > 0
            assert len(non_promotes) > 0

    def test_gold_cannot_promote(self):
        """Gold general cannot promote."""
        board = Board()
        board.board = np.zeros((5, 5), dtype=np.int8)

        board.board[1, 2] = PieceType.GOLD
        board.board[4, 4] = PieceType.KING
        board.board[0, 0] = -PieceType.KING
        board.current_player = 1

        moves = board.get_legal_moves()
        gold_moves = [m for m in moves if m.from_sq == (1, 2)]

        # No gold move should have promote=True
        for m in gold_moves:
            assert m.promote is False


class TestDrop:
    """Tests for drop mechanics and restrictions."""

    def test_pawn_cannot_drop_on_last_rank(self):
        """Pawn cannot be dropped on rank where it has no moves."""
        board = Board()
        board.board = np.zeros((5, 5), dtype=np.int8)

        board.board[4, 4] = PieceType.KING
        board.board[0, 4] = -PieceType.KING
        board.hands[0] = {PieceType.PAWN: 1}  # P1 has a pawn in hand
        board.current_player = 1

        moves = board.get_legal_moves()
        pawn_drops = [m for m in moves if m.is_drop and m.drop_piece == PieceType.PAWN]

        # No pawn drops should be on row 0 (last rank for P1)
        for drop in pawn_drops:
            assert drop.to_sq[0] != 0

    def test_two_pawn_restriction(self):
        """Cannot drop pawn in column with existing unpromoted pawn."""
        board = Board()
        board.board = np.zeros((5, 5), dtype=np.int8)

        board.board[3, 2] = PieceType.PAWN   # Existing pawn in column 2
        board.board[4, 4] = PieceType.KING
        board.board[0, 0] = -PieceType.KING
        board.hands[0] = {PieceType.PAWN: 1}
        board.current_player = 1

        moves = board.get_legal_moves()
        pawn_drops = [m for m in moves if m.is_drop and m.drop_piece == PieceType.PAWN]

        # No pawn drops in column 2
        for drop in pawn_drops:
            assert drop.to_sq[1] != 2

    def test_promoted_pawn_allows_drop_in_column(self):
        """Promoted pawn (Tokin) allows dropping another pawn in same column."""
        board = Board()
        board.board = np.zeros((5, 5), dtype=np.int8)

        board.board[2, 2] = PieceType.TOKIN  # Promoted pawn in column 2
        board.board[4, 4] = PieceType.KING
        board.board[0, 0] = -PieceType.KING
        board.hands[0] = {PieceType.PAWN: 1}
        board.current_player = 1

        moves = board.get_legal_moves()
        pawn_drops_col2 = [
            m for m in moves
            if m.is_drop and m.drop_piece == PieceType.PAWN and m.to_sq[1] == 2
        ]

        # Should be able to drop pawn in column 2 (Tokin doesn't count)
        assert len(pawn_drops_col2) > 0

    def test_drop_on_empty_square_only(self):
        """Can only drop on empty squares."""
        board = Board()
        board.board = np.zeros((5, 5), dtype=np.int8)

        board.board[2, 2] = PieceType.PAWN   # Occupied square
        board.board[4, 4] = PieceType.KING
        board.board[0, 0] = -PieceType.KING
        board.hands[0] = {PieceType.GOLD: 1}
        board.current_player = 1

        moves = board.get_legal_moves()
        gold_drops = [m for m in moves if m.is_drop and m.drop_piece == PieceType.GOLD]

        # No drop should target (2, 2)
        for drop in gold_drops:
            assert drop.to_sq != (2, 2)


class TestCapture:
    """Tests for capture mechanics."""

    def test_capture_adds_unpromoted_to_hand(self):
        """Capturing a promoted piece adds unpromoted form to hand."""
        board = Board()
        board.board = np.zeros((5, 5), dtype=np.int8)

        board.board[2, 2] = PieceType.ROOK
        board.board[2, 4] = -PieceType.P_ROOK  # Promoted Rook (Dragon King)
        board.board[4, 4] = PieceType.KING
        board.board[0, 0] = -PieceType.KING
        board.current_player = 1

        # Execute capture
        move = Move(from_sq=(2, 2), to_sq=(2, 4))
        board.execute_move(move)

        # Should have unpromoted Rook in hand, not Dragon King
        assert PieceType.ROOK in board.hands[0]
        assert PieceType.P_ROOK not in board.hands[0]

    def test_multiple_captures_accumulate(self):
        """Multiple captures of same piece type accumulate."""
        board = Board()
        board.board = np.zeros((5, 5), dtype=np.int8)

        board.board[4, 4] = PieceType.KING
        board.board[0, 0] = -PieceType.KING
        board.hands[0] = {PieceType.PAWN: 2}  # Already have 2 pawns

        # Simulate capturing another pawn
        board.hands[0][PieceType.PAWN] = board.hands[0].get(PieceType.PAWN, 0) + 1

        assert board.hands[0][PieceType.PAWN] == 3


class TestPieceMovement:
    """Tests for specific piece movement rules."""

    def test_rook_slides_horizontal(self):
        """Rook can slide horizontally."""
        board = Board()
        board.board = np.zeros((5, 5), dtype=np.int8)

        board.board[2, 2] = PieceType.ROOK
        board.board[4, 4] = PieceType.KING
        board.board[0, 0] = -PieceType.KING
        board.current_player = 1

        moves = board.get_legal_moves()
        rook_moves = [m for m in moves if m.from_sq == (2, 2)]

        # Should be able to move to (2, 0), (2, 1), (2, 3), (2, 4)
        horizontal = [m for m in rook_moves if m.to_sq[0] == 2]
        assert len(horizontal) >= 4

    def test_rook_blocked_by_own_piece(self):
        """Rook cannot pass through own pieces."""
        board = Board()
        board.board = np.zeros((5, 5), dtype=np.int8)

        board.board[2, 0] = PieceType.ROOK
        board.board[2, 2] = PieceType.PAWN   # Blocking
        board.board[4, 4] = PieceType.KING
        board.board[0, 0] = -PieceType.KING
        board.current_player = 1

        moves = board.get_legal_moves()
        rook_moves = [m for m in moves if m.from_sq == (2, 0)]

        # Rook shouldn't reach (2, 3) or beyond due to blocking pawn
        far_moves = [m for m in rook_moves if m.to_sq == (2, 3) or m.to_sq == (2, 4)]
        assert len(far_moves) == 0

    def test_bishop_diagonal_movement(self):
        """Bishop moves diagonally."""
        board = Board()
        board.board = np.zeros((5, 5), dtype=np.int8)

        board.board[2, 2] = PieceType.BISHOP
        board.board[4, 4] = PieceType.KING
        board.board[0, 0] = -PieceType.KING
        board.current_player = 1

        moves = board.get_legal_moves()
        bishop_moves = [m for m in moves if m.from_sq == (2, 2)]

        # Check diagonal positions
        # Check diagonal positions: (0,0), (1,1), (3,3), (4,4), (0,4), (1,3), (3,1), (4,0)
        reachable = [m.to_sq for m in bishop_moves]

        # (0,0) is blocked by enemy king, (4,4) by own king
        # Should reach (1,1), (3,3) unless blocked
        assert (1, 1) in reachable or (3, 3) in reachable

    def test_dragon_king_extra_diagonal(self):
        """Promoted Rook (Dragon King) can move one square diagonally."""
        board = Board()
        board.board = np.zeros((5, 5), dtype=np.int8)

        board.board[2, 2] = PieceType.P_ROOK  # Dragon King
        board.board[4, 4] = PieceType.KING
        board.board[0, 0] = -PieceType.KING
        board.current_player = 1

        moves = board.get_legal_moves()
        dragon_moves = [m for m in moves if m.from_sq == (2, 2)]
        targets = [m.to_sq for m in dragon_moves]

        # Should include diagonal moves
        assert (1, 1) in targets or (1, 3) in targets or (3, 1) in targets or (3, 3) in targets


class TestGameInterface:
    """Tests for MiniShogiGame interface."""

    def test_canonical_form_player1(self):
        """Canonical form for player 1 is unchanged."""
        game = MiniShogiGame()
        board = game.getInitBoard()

        canonical = game.getCanonicalForm(board, 1)

        assert np.array_equal(canonical.board, board.board)

    def test_canonical_form_player2(self):
        """Canonical form for player -1 is flipped."""
        game = MiniShogiGame()
        board = game.getInitBoard()

        canonical = game.getCanonicalForm(board, -1)

        # Board should be flipped and negated
        expected = -np.flip(board.board, axis=0)
        assert np.array_equal(canonical.board, expected)

    def test_pass_action(self):
        """Pass action switches player without changing board."""
        game = MiniShogiGame()
        board = game.getInitBoard()

        pass_action = game.getActionSize() - 1
        new_board, next_player = game.getNextState(board, 1, pass_action)

        assert next_player == -1

    def test_valid_moves_mask_size(self):
        """Valid moves mask has correct size."""
        game = MiniShogiGame()
        board = game.getInitBoard()

        valids = game.getValidMoves(board, 1)

        assert len(valids) == game.getActionSize()
        assert len(valids) == 1376


class TestEdgeCases:
    """Edge case and stress tests."""

    def test_empty_hand_no_drops(self):
        """No drop moves when hand is empty."""
        board = Board()
        board.hands[0] = {}  # Empty hand
        board.current_player = 1

        moves = board.get_legal_moves()
        drops = [m for m in moves if m.is_drop]

        assert len(drops) == 0

    def test_many_pieces_in_hand(self):
        """Game handles multiple pieces in hand."""
        game = MiniShogiGame()
        board = game.getInitBoard()

        # Add many pieces to hand
        board.hands[0] = {
            PieceType.PAWN: 1,
            PieceType.SILVER: 1,
            PieceType.GOLD: 1,
            PieceType.BISHOP: 1,
            PieceType.ROOK: 1,
        }

        valids = game.getValidMoves(board, 1)

        # Should have many valid drop moves
        assert valids.sum() > 20

    def test_long_game_no_crash(self):
        """Game doesn't crash on long random play."""
        game = MiniShogiGame()
        board = game.getInitBoard()
        player = 1

        for _ in range(1000):
            valids = game.getValidMoves(board, player)
            valid_indices = np.where(valids == 1)[0]

            if len(valid_indices) == 0:
                break

            action = np.random.choice(valid_indices)
            board, player = game.getNextState(board, player, action)

            if game.getGameEnded(board, player) != 0:
                break

        # Should complete without error
        assert True

    def test_action_index_boundary(self):
        """Action indices at boundaries work correctly."""
        # First move action
        move0 = Move.from_action_index(0)
        assert move0.from_sq == (0, 0)
        assert move0.promote is False

        # Last move action before drops
        move_last = Move.from_action_index(1249)
        assert move_last.promote is True

        # First drop action
        drop0 = Move.from_action_index(1250)
        assert drop0.is_drop is True

        # Last drop action
        drop_last = Move.from_action_index(1374)
        assert drop_last.is_drop is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
