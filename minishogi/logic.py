"""Core game logic for MiniShogi."""

from __future__ import annotations

import numpy as np

from .models import GameConfig, Move, PieceType


class Board:
    """MiniShogi board representation and game logic.

    Board is represented as a 5x5 numpy array.
    Positive values = Player 1 (sente, moves first)
    Negative values = Player 2 (gote)

    Coordinate system:
    - Row 0 is Player 2's back rank (promotion zone for Player 1)
    - Row 4 is Player 1's back rank (promotion zone for Player 2)
    - (row, col) format used throughout
    """

    # Movement directions for each piece type
    # Directions are (row_delta, col_delta) from the moving player's perspective
    GOLD_MOVES = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)]
    SILVER_MOVES = [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 1)]
    KING_MOVES = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Sliding piece directions
    ROOK_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    BISHOP_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def __init__(self, config: GameConfig | None = None):
        """Initialize the board with starting position."""
        self.config = config or GameConfig()
        self.n = self.config.board_size

        # Board state: 5x5 array
        self.board = np.zeros((self.n, self.n), dtype=np.int8)

        # Hands for captured pieces: [player1_hand, player2_hand]
        # Each hand is a dict mapping PieceType -> count
        self.hands: list[dict[PieceType, int]] = [{}, {}]

        # Current player: 1 = Player 1 (sente), -1 = Player 2 (gote)
        self.current_player = 1

        self._setup_initial_position()

    def _setup_initial_position(self) -> None:
        """Set up the initial piece positions for MiniShogi.

        Initial layout:
        Row 0 (P2 back): K  G  S  B  R  (Player 2's pieces, negative)
        Row 1 (P2 pawn): .  .  .  .  P  (Player 2's pawn)
        Row 2 (middle):  .  .  .  .  .
        Row 3 (P1 pawn): P  .  .  .  .  (Player 1's pawn)
        Row 4 (P1 back): R  B  S  G  K  (Player 1's pieces, positive)
        """
        # Player 2 (negative values) - back rank (row 0)
        self.board[0, 0] = -PieceType.KING
        self.board[0, 1] = -PieceType.GOLD
        self.board[0, 2] = -PieceType.SILVER
        self.board[0, 3] = -PieceType.BISHOP
        self.board[0, 4] = -PieceType.ROOK
        self.board[1, 4] = -PieceType.PAWN  # P2's pawn

        # Player 1 (positive values) - back rank (row 4)
        self.board[4, 4] = PieceType.KING
        self.board[4, 3] = PieceType.GOLD
        self.board[4, 2] = PieceType.SILVER
        self.board[4, 1] = PieceType.BISHOP
        self.board[4, 0] = PieceType.ROOK
        self.board[3, 0] = PieceType.PAWN  # P1's pawn

    def copy(self) -> Board:
        """Create a deep copy of the board."""
        new_board = Board.__new__(Board)
        new_board.config = self.config
        new_board.n = self.n
        new_board.board = self.board.copy()
        new_board.hands = [h.copy() for h in self.hands]
        new_board.current_player = self.current_player
        return new_board

    def _get_hand_index(self, player: int) -> int:
        """Get the hand index for a player (0 for player 1, 1 for player -1)."""
        return 0 if player == 1 else 1

    def _is_valid_square(self, row: int, col: int) -> bool:
        """Check if coordinates are within the board."""
        return 0 <= row < self.n and 0 <= col < self.n

    def _get_piece_at(self, row: int, col: int) -> tuple[int, PieceType]:
        """Get piece at a square. Returns (owner, piece_type).

        owner: 1 = player 1, -1 = player 2, 0 = empty
        """
        val = self.board[row, col]
        if val == 0:
            return 0, PieceType.EMPTY
        owner = 1 if val > 0 else -1
        piece_type = PieceType(abs(val))
        return owner, piece_type

    def _is_in_promotion_zone(self, row: int, player: int) -> bool:
        """Check if a row is in the promotion zone for a player."""
        if player == 1:
            return row < self.config.promotion_zone_size
        else:
            return row >= self.n - self.config.promotion_zone_size

    def _get_piece_moves(
        self, piece_type: PieceType, from_sq: tuple[int, int], player: int
    ) -> list[tuple[int, int]]:
        """Get all squares a piece can move to (ignoring check)."""
        row, col = from_sq
        moves = []

        # Direction multiplier: P1 moves "up" (negative row), P2 moves "down" (positive row)
        dir_mult = -1 if player == 1 else 1

        if piece_type == PieceType.PAWN:
            # Pawn moves one square forward
            new_row = row + dir_mult
            if self._is_valid_square(new_row, col):
                owner, _ = self._get_piece_at(new_row, col)
                if owner != player:
                    moves.append((new_row, col))

        elif piece_type == PieceType.KING:
            for dr, dc in self.KING_MOVES:
                new_row, new_col = row + dr, col + dc
                if self._is_valid_square(new_row, new_col):
                    owner, _ = self._get_piece_at(new_row, new_col)
                    if owner != player:
                        moves.append((new_row, new_col))

        elif piece_type == PieceType.GOLD or piece_type == PieceType.TOKIN or piece_type == PieceType.P_SILVER:
            # Gold and promoted pieces move like Gold
            for dr, dc in self.GOLD_MOVES:
                # Adjust direction based on player
                actual_dr = dr * dir_mult
                new_row, new_col = row + actual_dr, col + dc
                if self._is_valid_square(new_row, new_col):
                    owner, _ = self._get_piece_at(new_row, new_col)
                    if owner != player:
                        moves.append((new_row, new_col))

        elif piece_type == PieceType.SILVER:
            for dr, dc in self.SILVER_MOVES:
                actual_dr = dr * dir_mult
                new_row, new_col = row + actual_dr, col + dc
                if self._is_valid_square(new_row, new_col):
                    owner, _ = self._get_piece_at(new_row, new_col)
                    if owner != player:
                        moves.append((new_row, new_col))

        elif piece_type == PieceType.ROOK or piece_type == PieceType.P_ROOK:
            # Rook slides horizontally/vertically
            for dr, dc in self.ROOK_DIRS:
                for dist in range(1, self.n):
                    new_row, new_col = row + dr * dist, col + dc * dist
                    if not self._is_valid_square(new_row, new_col):
                        break
                    owner, _ = self._get_piece_at(new_row, new_col)
                    if owner == player:
                        break
                    moves.append((new_row, new_col))
                    if owner != 0:  # Captured enemy piece
                        break

            # Dragon King (promoted Rook) also moves one square diagonally
            if piece_type == PieceType.P_ROOK:
                for dr, dc in self.BISHOP_DIRS:
                    new_row, new_col = row + dr, col + dc
                    if self._is_valid_square(new_row, new_col):
                        owner, _ = self._get_piece_at(new_row, new_col)
                        if owner != player:
                            moves.append((new_row, new_col))

        elif piece_type == PieceType.BISHOP or piece_type == PieceType.P_BISHOP:
            # Bishop slides diagonally
            for dr, dc in self.BISHOP_DIRS:
                for dist in range(1, self.n):
                    new_row, new_col = row + dr * dist, col + dc * dist
                    if not self._is_valid_square(new_row, new_col):
                        break
                    owner, _ = self._get_piece_at(new_row, new_col)
                    if owner == player:
                        break
                    moves.append((new_row, new_col))
                    if owner != 0:
                        break

            # Dragon Horse (promoted Bishop) also moves one square orthogonally
            if piece_type == PieceType.P_BISHOP:
                for dr, dc in self.ROOK_DIRS:
                    new_row, new_col = row + dr, col + dc
                    if self._is_valid_square(new_row, new_col):
                        owner, _ = self._get_piece_at(new_row, new_col)
                        if owner != player:
                            moves.append((new_row, new_col))

        return moves

    def _find_king(self, player: int) -> tuple[int, int] | None:
        """Find the king's position for a player."""
        king_val = PieceType.KING * player
        positions = np.argwhere(self.board == king_val)
        if len(positions) > 0:
            return tuple(positions[0])
        return None

    def is_in_check(self, player: int) -> bool:
        """Check if the given player's king is in check."""
        king_pos = self._find_king(player)
        if king_pos is None:
            return True  # King captured = in check

        opponent = -player

        # Check if any opponent piece can attack the king
        for row in range(self.n):
            for col in range(self.n):
                owner, piece_type = self._get_piece_at(row, col)
                if owner == opponent:
                    moves = self._get_piece_moves(piece_type, (row, col), opponent)
                    if king_pos in moves:
                        return True
        return False

    def _can_drop_pawn_checkmate(self, col: int, player: int) -> bool:
        """Check if dropping a pawn on this column would give immediate checkmate.

        This is illegal in Shogi (打ち歩詰め / uchifuzume).
        """
        # Find valid drop row (can't drop on last rank)
        opponent = -player
        if player == 1:
            # Player 1 can't drop on row 0
            for row in range(1, self.n):
                if self.board[row, col] == 0:
                    # Try the drop
                    test_board = self.copy()
                    test_board.board[row, col] = PieceType.PAWN * player
                    # Check if opponent is in checkmate
                    if test_board.is_in_check(opponent):
                        # Check if opponent has any legal moves
                        test_board.current_player = opponent
                        if len(test_board.get_legal_moves()) == 0:
                            return True
        else:
            # Player -1 can't drop on row 4
            for row in range(self.n - 1):
                if self.board[row, col] == 0:
                    test_board = self.copy()
                    test_board.board[row, col] = PieceType.PAWN * player
                    if test_board.is_in_check(opponent):
                        test_board.current_player = opponent
                        if len(test_board.get_legal_moves()) == 0:
                            return True
        return False

    def _has_pawn_in_column(self, col: int, player: int) -> bool:
        """Check if player has an unpromoted pawn in the given column."""
        pawn_val = PieceType.PAWN * player
        return pawn_val in self.board[:, col]

    def get_legal_moves(self) -> list[Move]:
        """Get all legal moves for the current player."""
        player = self.current_player
        moves = []

        # 1. Generate all piece moves
        for row in range(self.n):
            for col in range(self.n):
                owner, piece_type = self._get_piece_at(row, col)
                if owner == player:
                    piece_moves = self._get_piece_moves(piece_type, (row, col), player)
                    for to_sq in piece_moves:
                        from_sq = (row, col)
                        to_row, _ = to_sq

                        # Check promotion possibilities
                        can_promote = (
                            piece_type.can_promote
                            and (
                                self._is_in_promotion_zone(row, player)
                                or self._is_in_promotion_zone(to_row, player)
                            )
                        )

                        # Must promote if pawn reaches last rank
                        must_promote = (
                            piece_type == PieceType.PAWN
                            and self._is_in_promotion_zone(to_row, player)
                        )

                        if must_promote:
                            moves.append(Move(from_sq=from_sq, to_sq=to_sq, promote=True))
                        elif can_promote:
                            moves.append(Move(from_sq=from_sq, to_sq=to_sq, promote=False))
                            moves.append(Move(from_sq=from_sq, to_sq=to_sq, promote=True))
                        else:
                            moves.append(Move(from_sq=from_sq, to_sq=to_sq, promote=False))

        # 2. Generate drop moves
        hand_idx = self._get_hand_index(player)
        hand = self.hands[hand_idx]

        for piece_type, count in hand.items():
            if count <= 0:
                continue

            # Special restrictions for pawn drops
            is_pawn = piece_type == PieceType.PAWN

            for row in range(self.n):
                # Pawn can't be dropped on last rank (would have no moves)
                if is_pawn:
                    if player == 1 and row == 0:
                        continue
                    if player == -1 and row == self.n - 1:
                        continue

                for col in range(self.n):
                    if self.board[row, col] != 0:
                        continue  # Square not empty

                    # Two-pawn restriction: can't have two unpromoted pawns in same column
                    if is_pawn and self._has_pawn_in_column(col, player):
                        continue

                    moves.append(Move(to_sq=(row, col), drop_piece=piece_type))

        # 3. Filter out moves that leave king in check
        legal_moves = []
        for move in moves:
            test_board = self.copy()
            test_board._execute_move_unchecked(move)
            if not test_board.is_in_check(player):
                # Additional check: pawn drop can't give immediate checkmate
                if move.is_drop and move.drop_piece == PieceType.PAWN:
                    test_board.current_player = -player
                    if test_board.is_in_check(-player) and len(test_board.get_legal_moves()) == 0:
                        continue  # Illegal: drop pawn checkmate
                legal_moves.append(move)

        return legal_moves

    def _execute_move_unchecked(self, move: Move) -> None:
        """Execute a move without checking legality."""
        player = self.current_player

        if move.is_drop:
            # Drop a piece from hand
            hand_idx = self._get_hand_index(player)
            piece_type = move.drop_piece
            assert piece_type is not None
            self.hands[hand_idx][piece_type] -= 1
            if self.hands[hand_idx][piece_type] == 0:
                del self.hands[hand_idx][piece_type]
            self.board[move.to_sq[0], move.to_sq[1]] = piece_type * player
        else:
            # Regular move
            assert move.from_sq is not None
            from_row, from_col = move.from_sq
            to_row, to_col = move.to_sq

            # Get the moving piece
            piece_val = self.board[from_row, from_col]
            piece_type = PieceType(abs(piece_val))

            # Check for capture
            target_owner, target_type = self._get_piece_at(to_row, to_col)
            if target_owner != 0:
                # Add captured piece to hand (unpromoted form)
                captured_type = target_type.unpromoted_form
                hand_idx = self._get_hand_index(player)
                self.hands[hand_idx][captured_type] = self.hands[hand_idx].get(captured_type, 0) + 1

            # Move the piece
            self.board[from_row, from_col] = 0
            if move.promote:
                piece_type = piece_type.promoted_form
            self.board[to_row, to_col] = piece_type * player

        # Switch player
        self.current_player = -player

    def execute_move(self, move: Move) -> None:
        """Execute a move (with basic validation)."""
        self._execute_move_unchecked(move)

    def is_game_over(self) -> int:
        """Check if the game is over.

        Returns:
            0 if game is ongoing
            1 if player 1 wins
            -1 if player 2 wins
        """
        # Check if current player has any legal moves
        legal_moves = self.get_legal_moves()

        if len(legal_moves) == 0:
            # No legal moves = current player loses (checkmate or stalemate)
            return -self.current_player

        return 0

    def get_canonical_form(self, player: int) -> np.ndarray:
        """Get canonical board representation from player's perspective.

        For player 1: return board as-is
        For player -1: flip board vertically and negate values
        """
        if player == 1:
            return self.board.copy()
        else:
            # Flip board and negate
            return -np.flip(self.board, axis=0)

    def get_state_tensor(self) -> np.ndarray:
        """Get board state as a tensor for neural network input.

        Returns a 5x5 array with piece values.
        """
        return self.board.astype(np.float32)

    def __str__(self) -> str:
        """String representation of the board."""
        piece_chars = {
            0: ".",
            PieceType.PAWN: "P",
            PieceType.SILVER: "S",
            PieceType.GOLD: "G",
            PieceType.BISHOP: "B",
            PieceType.ROOK: "R",
            PieceType.KING: "K",
            PieceType.TOKIN: "+P",
            PieceType.P_SILVER: "+S",
            PieceType.P_BISHOP: "+B",
            PieceType.P_ROOK: "+R",
        }

        lines = []
        lines.append("  " + " ".join(str(c) for c in range(self.n)))
        lines.append("  " + "-" * (self.n * 2 - 1))

        for row in range(self.n):
            row_str = f"{row}|"
            for col in range(self.n):
                val = self.board[row, col]
                if val == 0:
                    row_str += ". "
                else:
                    piece_type = PieceType(abs(val))
                    char = piece_chars.get(piece_type, "?")
                    if val < 0:
                        char = char.lower()
                    row_str += f"{char:2}"
            lines.append(row_str)

        # Show hands
        lines.append("")
        for i, player in enumerate([1, -1]):
            hand = self.hands[i]
            hand_str = f"P{1 if player == 1 else 2} hand: "
            if hand:
                hand_str += ", ".join(
                    f"{piece_chars[pt]}{cnt}" for pt, cnt in hand.items()
                )
            else:
                hand_str += "(empty)"
            lines.append(hand_str)

        return "\n".join(lines)
