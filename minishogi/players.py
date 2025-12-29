"""Player implementations for MiniShogi."""

from __future__ import annotations

import numpy as np

from .game import MiniShogiGame
from .logic import Board


class RandomPlayer:
    """A player that makes random legal moves."""

    def __init__(self, game: MiniShogiGame):
        """Initialize the random player.

        Args:
            game: The MiniShogi game instance.
        """
        self.game = game

    def play(self, board: Board) -> int:
        """Select a random valid action.

        Args:
            board: Current board state.

        Returns:
            A random valid action index.
        """
        valids = self.game.getValidMoves(board, 1)
        valid_indices = np.where(valids == 1)[0]

        if len(valid_indices) == 0:
            return self.game.getActionSize() - 1  # Pass

        return np.random.choice(valid_indices)


class HumanMiniShogiPlayer:
    """A human player that accepts input from the console."""

    def __init__(self, game: MiniShogiGame):
        """Initialize the human player.

        Args:
            game: The MiniShogi game instance.
        """
        self.game = game

    def play(self, board: Board) -> int:
        """Get a move from human input.

        Args:
            board: Current board state.

        Returns:
            The action index selected by the human.
        """
        valids = self.game.getValidMoves(board, 1)

        # Display current board
        print("\nCurrent board:")
        print(board)
        print()

        while True:
            try:
                print("Enter move as: 'from_row from_col to_row to_col [p]' for moves")
                print("or: 'd piece_type to_row to_col' for drops")
                print("Piece types: P=1, S=2, G=3, B=4, R=5")
                print("Add 'p' at the end to promote")

                user_input = input("Your move: ").strip().lower()

                if user_input.startswith("d"):
                    # Drop move: d piece_type to_row to_col
                    parts = user_input.split()
                    if len(parts) != 4:
                        print("Invalid drop format. Use: d piece_type to_row to_col")
                        continue

                    piece_type = int(parts[1])
                    to_row = int(parts[2])
                    to_col = int(parts[3])

                    # Calculate action index for drop
                    # Drop action: 1250 + (piece_type - 1) * 25 + to_sq
                    to_idx = to_row * 5 + to_col
                    action = 1250 + (piece_type - 1) * 25 + to_idx

                else:
                    # Regular move: from_row from_col to_row to_col [p]
                    parts = user_input.split()
                    if len(parts) < 4:
                        print("Invalid move format. Use: from_row from_col to_row to_col [p]")
                        continue

                    from_row = int(parts[0])
                    from_col = int(parts[1])
                    to_row = int(parts[2])
                    to_col = int(parts[3])
                    promote = len(parts) > 4 and parts[4] == "p"

                    # Calculate action index for move
                    from_idx = from_row * 5 + from_col
                    to_idx = to_row * 5 + to_col
                    action = from_idx * 50 + to_idx * 2 + int(promote)

                if 0 <= action < len(valids) and valids[action] == 1:
                    return action
                else:
                    print("Invalid move! Try again.")

            except (ValueError, IndexError) as e:
                print(f"Error parsing input: {e}. Try again.")


class GreedyMiniShogiPlayer:
    """A player that selects moves based on material advantage."""

    def __init__(self, game: MiniShogiGame):
        """Initialize the greedy player.

        Args:
            game: The MiniShogi game instance.
        """
        self.game = game

    def play(self, board: Board) -> int:
        """Select the move that maximizes material score.

        Args:
            board: Current board state.

        Returns:
            The action index with the best immediate score.
        """
        valids = self.game.getValidMoves(board, 1)
        valid_indices = np.where(valids == 1)[0]

        if len(valid_indices) == 0:
            return self.game.getActionSize() - 1  # Pass

        best_action = valid_indices[0]
        best_score = float("-inf")

        for action in valid_indices:
            next_board, _ = self.game.getNextState(board, 1, action)
            score = self.game.getScore(next_board, 1)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action
