"""Neural network wrapper for MiniShogi using PyTorch."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from NeuralNet import NeuralNet
from utils import AverageMeter

from ..game import MiniShogiGame
from ..models import NNetConfig
from .model import MiniShogiNNet


class NNetWrapper(NeuralNet):
    """Neural network wrapper for training and inference.

    This class wraps the MiniShogiNNet model and provides the interface
    required by the alpha-zero-general framework.
    """

    def __init__(self, game: MiniShogiGame, config: NNetConfig | None = None):
        """Initialize the neural network wrapper.

        Args:
            game: MiniShogiGame instance.
            config: Network configuration.
        """
        self.config = config or NNetConfig()
        self.nnet = MiniShogiNNet(game, self.config)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        # Move to GPU if available and configured
        self.device = torch.device("cuda" if self.config.cuda and torch.cuda.is_available() else "cpu")
        self.nnet.to(self.device)

    def train(self, examples: list[tuple]) -> None:
        """Train the neural network on examples.

        Args:
            examples: List of (board, pi, v) tuples where:
                - board: numpy array of board state
                - pi: policy vector of action probabilities
                - v: value (-1 to 1) indicating outcome
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.config.lr)

        for epoch in range(self.config.epochs):
            print(f"EPOCH ::: {epoch + 1}")
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.config.batch_size)

            t = tqdm(range(batch_count), desc="Training Net")
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.config.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # Convert board objects to numpy arrays if needed
                board_arrays = []
                for board in boards:
                    if hasattr(board, "get_state_tensor"):
                        board_arrays.append(board.get_state_tensor())
                    elif hasattr(board, "board"):
                        board_arrays.append(board.board.astype(np.float32))
                    else:
                        board_arrays.append(np.array(board).astype(np.float32))

                boards_tensor = torch.FloatTensor(np.array(board_arrays)).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs)).to(self.device)

                # Forward pass
                out_pi, out_v = self.nnet(boards_tensor)

                # Compute losses
                l_pi = self._loss_pi(target_pis, out_pi)
                l_v = self._loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # Record losses
                pi_losses.update(l_pi.item(), boards_tensor.size(0))
                v_losses.update(l_v.item(), boards_tensor.size(0))
                t.set_postfix(Loss_pi=pi_losses.avg, Loss_v=v_losses.avg)

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board) -> tuple[np.ndarray, float]:
        """Predict action probabilities and value for a board position.

        Args:
            board: Board state (Board object or numpy array).

        Returns:
            Tuple of (pi, v) where:
            - pi: numpy array of action probabilities
            - v: position value (-1 to 1)
        """
        # Prepare input
        if hasattr(board, "get_state_tensor"):
            board_array = board.get_state_tensor()
        elif hasattr(board, "board"):
            board_array = board.board.astype(np.float32)
        else:
            board_array = np.array(board).astype(np.float32)

        board_tensor = torch.FloatTensor(board_array).to(self.device)
        board_tensor = board_tensor.view(1, self.board_x, self.board_y)

        self.nnet.eval()
        with torch.no_grad():
            log_pi, v = self.nnet(board_tensor)

        # Convert log probabilities to probabilities
        pi = torch.exp(log_pi).cpu().numpy()[0]
        value = v.cpu().numpy()[0][0]

        return pi, value

    def _loss_pi(self, targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Policy loss (cross-entropy)."""
        return -torch.sum(targets * outputs) / targets.size(0)

    def _loss_v(self, targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Value loss (MSE)."""
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size(0)

    def save_checkpoint(self, folder: str = "checkpoint", filename: str = "checkpoint.pth.tar") -> None:
        """Save model checkpoint.

        Args:
            folder: Directory to save checkpoint.
            filename: Checkpoint filename.
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.makedirs(folder)
        else:
            print("Checkpoint Directory exists!")

        torch.save({"state_dict": self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder: str = "checkpoint", filename: str = "checkpoint.pth.tar") -> None:
        """Load model checkpoint.

        Args:
            folder: Directory containing checkpoint.
            filename: Checkpoint filename.
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path '{filepath}'")

        map_location = self.device
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint["state_dict"])
