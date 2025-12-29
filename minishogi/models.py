"""Pydantic models for MiniShogi game."""

from enum import IntEnum

from pydantic import BaseModel, Field


class PieceType(IntEnum):
    """Piece types in MiniShogi.

    Positive values are used for player 1 (sente/black, moves first).
    Negative values are used for player 2 (gote/white).
    """

    EMPTY = 0

    # Basic pieces
    PAWN = 1
    SILVER = 2
    GOLD = 3
    BISHOP = 4
    ROOK = 5
    KING = 6

    # Promoted pieces
    TOKIN = 7  # Promoted Pawn (moves like Gold)
    P_SILVER = 8  # Promoted Silver (moves like Gold)
    P_BISHOP = 9  # Promoted Bishop (Dragon Horse)
    P_ROOK = 10  # Promoted Rook (Dragon King)

    @property
    def can_promote(self) -> bool:
        """Check if this piece type can promote."""
        return self in (
            PieceType.PAWN,
            PieceType.SILVER,
            PieceType.BISHOP,
            PieceType.ROOK,
        )

    @property
    def promoted_form(self) -> "PieceType":
        """Get the promoted form of this piece."""
        promotion_map = {
            PieceType.PAWN: PieceType.TOKIN,
            PieceType.SILVER: PieceType.P_SILVER,
            PieceType.BISHOP: PieceType.P_BISHOP,
            PieceType.ROOK: PieceType.P_ROOK,
        }
        return promotion_map.get(self, self)

    @property
    def unpromoted_form(self) -> "PieceType":
        """Get the unpromoted form of this piece (for capturing)."""
        unpromote_map = {
            PieceType.TOKIN: PieceType.PAWN,
            PieceType.P_SILVER: PieceType.SILVER,
            PieceType.P_BISHOP: PieceType.BISHOP,
            PieceType.P_ROOK: PieceType.ROOK,
        }
        return unpromote_map.get(self, self)

    @property
    def is_promoted(self) -> bool:
        """Check if this is a promoted piece."""
        return self in (
            PieceType.TOKIN,
            PieceType.P_SILVER,
            PieceType.P_BISHOP,
            PieceType.P_ROOK,
        )


class Move(BaseModel):
    """Represents a move in MiniShogi.

    For normal moves: from_sq is set, drop_piece is None
    For drops: from_sq is None, drop_piece is set
    """

    from_sq: tuple[int, int] | None = Field(
        default=None, description="Source square (row, col), None for drops"
    )
    to_sq: tuple[int, int] = Field(description="Destination square (row, col)")
    promote: bool = Field(default=False, description="Whether to promote after the move")
    drop_piece: PieceType | None = Field(
        default=None, description="Piece type to drop, None for normal moves"
    )

    @property
    def is_drop(self) -> bool:
        """Check if this is a drop move."""
        return self.drop_piece is not None

    def to_action_index(self, board_size: int = 5) -> int:
        """Convert move to action index for neural network.

        Action space:
        - Moves: from_sq * 25 * 2 + to_sq * 2 + promote_flag (0-1249)
        - Drops: 1250 + piece_type * 25 + to_sq (1250-1374)
        - Pass: 1375
        """
        if self.is_drop:
            # Drop action: 1250 + (piece_type - 1) * 25 + to_sq
            assert self.drop_piece is not None
            piece_idx = self.drop_piece.value - 1  # PAWN=1 -> idx=0
            to_idx = self.to_sq[0] * board_size + self.to_sq[1]
            return 1250 + piece_idx * board_size * board_size + to_idx
        else:
            # Move action: from_sq * 50 + to_sq * 2 + promote
            assert self.from_sq is not None
            from_idx = self.from_sq[0] * board_size + self.from_sq[1]
            to_idx = self.to_sq[0] * board_size + self.to_sq[1]
            return from_idx * board_size * board_size * 2 + to_idx * 2 + int(self.promote)

    @classmethod
    def from_action_index(cls, action: int, board_size: int = 5) -> "Move":
        """Convert action index back to Move object."""
        if action == 1375:
            # Pass action (represented as a null move)
            return cls(to_sq=(0, 0))
        elif action >= 1250:
            # Drop action
            drop_idx = action - 1250
            piece_idx = drop_idx // (board_size * board_size)
            to_idx = drop_idx % (board_size * board_size)
            to_sq = (to_idx // board_size, to_idx % board_size)
            return cls(to_sq=to_sq, drop_piece=PieceType(piece_idx + 1))
        else:
            # Move action
            promote = action % 2
            action //= 2
            to_idx = action % (board_size * board_size)
            from_idx = action // (board_size * board_size)
            from_sq = (from_idx // board_size, from_idx % board_size)
            to_sq = (to_idx // board_size, to_idx % board_size)
            return cls(from_sq=from_sq, to_sq=to_sq, promote=bool(promote))


class GameConfig(BaseModel):
    """Configuration for MiniShogi game."""

    board_size: int = Field(default=5, description="Board size (5x5 for MiniShogi)")
    promotion_zone_size: int = Field(default=1, description="Rows in promotion zone")

    @property
    def action_size(self) -> int:
        """Total number of possible actions.

        Moves: 25 * 25 * 2 = 1250 (from * to * promote)
        Drops: 5 * 25 = 125 (5 droppable piece types * 25 squares)
        Pass: 1
        Total: 1376
        """
        n = self.board_size
        move_actions = n * n * n * n * 2  # from * to * promote
        drop_actions = 5 * n * n  # 5 piece types (P, S, G, B, R) * squares
        return move_actions + drop_actions + 1  # +1 for pass


class NNetConfig(BaseModel):
    """Configuration for neural network training."""

    lr: float = Field(default=0.001, description="Learning rate")
    dropout: float = Field(default=0.3, description="Dropout rate")
    epochs: int = Field(default=10, description="Training epochs per iteration")
    batch_size: int = Field(default=64, description="Batch size for training")
    num_channels: int = Field(default=256, description="Number of CNN channels")
    cuda: bool = Field(default=True, description="Use CUDA if available")


class ParallelConfig(BaseModel):
    """Configuration for parallel self-play."""

    num_workers: int = Field(default=8, description="Number of CPU workers for self-play")
    games_per_worker: int = Field(default=25, description="Games per worker per iteration")
