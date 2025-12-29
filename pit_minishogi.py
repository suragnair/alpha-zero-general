"""Play MiniShogi interactively.

Usage:
    uv run python pit_minishogi.py              # Human vs Random
    uv run python pit_minishogi.py --mode random  # Random vs Random
    uv run python pit_minishogi.py --mode greedy  # Human vs Greedy
    uv run python pit_minishogi.py --mode ai      # Human vs AI (requires trained model)
"""

import argparse

import numpy as np

import Arena
from MCTS import MCTS
from minishogi import MiniShogiGame
from minishogi.players import GreedyMiniShogiPlayer, HumanMiniShogiPlayer, RandomPlayer
from minishogi.pytorch import NNetWrapper as NNet
from utils import dotdict


def main():
    parser = argparse.ArgumentParser(description="Play MiniShogi")
    parser.add_argument(
        "--mode",
        choices=["random", "greedy", "ai", "human"],
        default="random",
        help="Opponent type: random, greedy, ai, or human (human vs human)",
    )
    parser.add_argument(
        "--model",
        default="./minishogi_checkpoints/best.pth.tar",
        help="Path to trained model (for ai mode)",
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=50,
        help="MCTS simulations for AI player",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=2,
        help="Number of games to play",
    )
    args = parser.parse_args()

    # Create game
    game = MiniShogiGame()
    print("=== MiniShogi ===")
    print(f"Board size: {game.getBoardSize()}")
    print(f"Opponent: {args.mode}")
    print()

    # Create players
    human = HumanMiniShogiPlayer(game).play
    random_player = RandomPlayer(game).play
    greedy = GreedyMiniShogiPlayer(game).play

    # Player 1 is human by default
    player1 = human

    # Player 2 depends on mode
    if args.mode == "random":
        player2 = random_player
    elif args.mode == "greedy":
        player2 = greedy
    elif args.mode == "human":
        player2 = human
    elif args.mode == "ai":
        # Load trained neural network
        try:
            nnet = NNet(game)
            nnet.load_checkpoint(*args.model.rsplit("/", 1))
            mcts_args = dotdict({"numMCTSSims": args.mcts_sims, "cpuct": 1.5})
            mcts = MCTS(game, nnet, mcts_args)

            def ai_player(x):
                return np.argmax(mcts.getActionProb(x, temp=0))

            player2 = ai_player
            print(f"Loaded AI model from: {args.model}")
        except FileNotFoundError:
            print(f"Model not found: {args.model}")
            print("Falling back to greedy player.")
            print("Train a model first with: uv run python main_minishogi.py")
            player2 = greedy
    else:
        player2 = random_player

    # Create arena and play
    arena = Arena.Arena(player1, player2, game, display=MiniShogiGame.display)

    print("\n--- Starting Games ---\n")
    result = arena.playGames(args.games, verbose=True)
    print(f"\nResults: Player 1 wins={result[0]}, Player 2 wins={result[1]}, Draws={result[2]}")


if __name__ == "__main__":
    main()
