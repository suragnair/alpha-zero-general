"""MiniShogi game interface for alpha-zero-general framework."""

from __future__ import annotations

import numpy as np

from Game import Game

from .logic import Board
from .models import GameConfig, Move


class MiniShogiGame(Game):
    """MiniShogi game implementation for alpha-zero-general.

    This class provides the interface required by the alpha-zero-general
    framework for training and playing MiniShogi.
    """

    def __init__(self, config: GameConfig | None = None):
        """Initialize the MiniShogi game.

        Args:
            config: Game configuration. Uses defaults if not provided.
        """
        super().__init__()
        self.config = config or GameConfig()
        self.n = self.config.board_size

    def getInitBoard(self) -> Board:
        """Return the initial board state.

        Returns:
            Board object with pieces in starting positions.
        """
        return Board(self.config)

    def getBoardSize(self) -> tuple[int, int]:
        """Return the board dimensions.

        Returns:
            Tuple of (rows, cols) = (5, 5) for MiniShogi.
        """
        return (self.n, self.n)

    def getActionSize(self) -> int:
        """Return the total number of possible actions.

        Action space:
        - Move actions: 25 * 25 * 2 = 1250 (from * to * promote)
        - Drop actions: 5 * 25 = 125 (5 piece types * 25 squares)
        - Pass action: 1
        Total: 1376

        Returns:
            The action space size (1376).
        """
        return self.config.action_size

    def getNextState(self, board: Board, player: int, action: int) -> tuple[Board, int]:
        """Execute an action and return the resulting state.

        Args:
            board: Current board state.
            player: Current player (1 or -1).
            action: Action index to execute.

        Returns:
            Tuple of (new_board, next_player).
        """
        # Create a copy of the board
        b = board.copy()
        b.current_player = player

        # Handle pass action
        if action == self.getActionSize() - 1:
            b.current_player = -player
            return (b, -player)

        # Convert action index to Move object
        move = Move.from_action_index(action, self.n)

        # Execute the move
        b.execute_move(move)

        return (b, -player)

    def getValidMoves(self, board: Board, player: int) -> np.ndarray:
        """Return a binary vector of valid moves.

        Args:
            board: Current board state.
            player: Current player (1 or -1).

        Returns:
            Binary numpy array of size getActionSize() where 1 = valid move.
        """
        valids = np.zeros(self.getActionSize(), dtype=np.int8)

        # Ensure we're checking moves for the correct player
        b = board.copy()
        b.current_player = player

        legal_moves = b.get_legal_moves()

        if len(legal_moves) == 0:
            # No legal moves - only pass is valid
            valids[-1] = 1
            return valids

        for move in legal_moves:
            action_idx = move.to_action_index(self.n)
            if 0 <= action_idx < self.getActionSize() - 1:
                valids[action_idx] = 1

        return valids

    def getGameEnded(self, board: Board, player: int) -> float:
        """Check if the game has ended.

        Args:
            board: Current board state.
            player: Player to check for (1 or -1).

        Returns:
            0 if game is ongoing.
            1 if player has won.
            -1 if player has lost.
            Small value (1e-4) for draw (rare in Shogi).
        """
        b = board.copy()
        b.current_player = player

        result = b.is_game_over()

        if result == 0:
            return 0
        elif result == player:
            return 1
        else:
            return -1

    def getCanonicalForm(self, board: Board, player: int) -> Board:
        """Return the canonical (player-independent) board form.

        For player 1: returns board as-is.
        For player -1: flips board vertically and swaps piece ownership.

        Args:
            board: Current board state.
            player: Current player perspective (1 or -1).

        Returns:
            Canonical board representation.
        """
        if player == 1:
            return board

        # Create a new board with flipped perspective
        canonical = board.copy()
        canonical.board = -np.flip(board.board, axis=0)
        # Swap hands
        canonical.hands = [board.hands[1].copy(), board.hands[0].copy()]
        canonical.current_player = 1

        return canonical

    def getSymmetries(self, board: Board, pi: list[float]) -> list[tuple[Board, list[float]]]:
        """Return board symmetries for data augmentation.

        MiniShogi has left-right mirror symmetry.

        Args:
            board: Board state.
            pi: Policy vector of size getActionSize().

        Returns:
            List of (board, pi) tuples representing symmetrical positions.
        """
        # For now, return just the original (symmetries are complex for Shogi due to drops)
        # TODO: Implement left-right mirror symmetry with proper action remapping
        return [(board, pi)]

    def stringRepresentation(self, board: Board) -> str:
        """Return a string representation for hashing.

        Args:
            board: Board state.

        Returns:
            A unique string representation of the board state.
        """
        # Include board, hands, and current player
        board_str = board.board.tobytes()
        hand1 = tuple(sorted(board.hands[0].items()))
        hand2 = tuple(sorted(board.hands[1].items()))
        return str((board_str, hand1, hand2, board.current_player))

    def getScore(self, board: Board, player: int) -> float:
        """Get a heuristic score for the board position.

        Used by greedy players. Higher is better for player.

        Args:
            board: Board state.
            player: Player to evaluate for.

        Returns:
            Score value (piece count differential).
        """
        # Simple material count
        score = 0

        # Piece values
        piece_values = {
            1: 1,  # PAWN
            2: 3,  # SILVER
            3: 4,  # GOLD
            4: 5,  # BISHOP
            5: 6,  # ROOK
            6: 100,  # KING
            7: 4,  # TOKIN
            8: 4,  # P_SILVER
            9: 7,  # P_BISHOP
            10: 8,  # P_ROOK
        }

        # Count pieces on board
        for row in range(self.n):
            for col in range(self.n):
                val = board.board[row, col]
                if val != 0:
                    piece_type = abs(val)
                    owner = 1 if val > 0 else -1
                    value = piece_values.get(piece_type, 0)
                    score += value * owner

        # Count pieces in hand
        for i, hand_player in enumerate([1, -1]):
            for piece_type, count in board.hands[i].items():
                value = piece_values.get(piece_type.value, 0)
                score += value * count * hand_player

        return score * player

    @staticmethod
    def display(board: Board) -> None:
        """Display the board in a human-readable format.

        Args:
            board: Board state to display.
        """
        print(board)


# Alias for compatibility
display = MiniShogiGame.display
