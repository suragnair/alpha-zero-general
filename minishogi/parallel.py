"""Parallel self-play for MiniShogi training."""

from __future__ import annotations

import logging
import pickle
from collections import deque
from multiprocessing import Pool
from random import shuffle

import numpy as np

from Arena import Arena
from MCTS import MCTS

from .game import MiniShogiGame
from .models import ParallelConfig

log = logging.getLogger(__name__)


def _execute_episode_worker(args: tuple) -> list:
    """Worker function for parallel self-play.

    This function runs a single episode of self-play in a separate process.

    Args:
        args: Tuple of (game_pickle, nnet_state_dict, mcts_args, temp_threshold)

    Returns:
        List of training examples from the episode.
    """
    # Note: imports inside worker are necessary for multiprocessing
    # as each worker is a separate process
    import numpy as np

    from MCTS import MCTS
    from minishogi.game import MiniShogiGame
    from minishogi.pytorch.nnet import NNetWrapper

    game_config, nnet_state_bytes, mcts_args, temp_threshold = args

    # Recreate game and neural network
    game = MiniShogiGame(game_config)

    # Create neural network and load weights
    nnet = NNetWrapper(game)
    nnet_state = pickle.loads(nnet_state_bytes)
    nnet.nnet.load_state_dict(nnet_state)

    # Create MCTS
    mcts = MCTS(game, nnet, mcts_args)

    # Execute episode
    train_examples = []
    board = game.getInitBoard()
    cur_player = 1
    episode_step = 0

    while True:
        episode_step += 1
        canonical_board = game.getCanonicalForm(board, cur_player)
        temp = int(episode_step < temp_threshold)

        pi = mcts.getActionProb(canonical_board, temp=temp)
        sym = game.getSymmetries(canonical_board, pi)

        for b, p in sym:
            train_examples.append([b, cur_player, p, None])

        action = np.random.choice(len(pi), p=pi)
        board, cur_player = game.getNextState(board, cur_player, action)

        r = game.getGameEnded(board, cur_player)

        if r != 0:
            return [(x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in train_examples]


def parallel_self_play(
    game: MiniShogiGame,
    nnet,
    mcts_args,
    config: ParallelConfig | None = None,
    temp_threshold: int = 15,
) -> list:
    """Execute parallel self-play using multiple processes.

    Args:
        game: MiniShogi game instance.
        nnet: Neural network wrapper.
        mcts_args: MCTS configuration arguments.
        config: Parallel execution configuration.
        temp_threshold: Episode step below which temperature=1 is used.

    Returns:
        List of training examples from all episodes.
    """
    config = config or ParallelConfig()

    # Serialize neural network state for workers
    nnet_state_bytes = pickle.dumps(nnet.nnet.state_dict())

    # Prepare worker arguments
    total_games = config.num_workers * config.games_per_worker
    worker_args = [
        (game.config, nnet_state_bytes, mcts_args, temp_threshold)
        for _ in range(total_games)
    ]

    # Execute in parallel
    all_examples = []
    with Pool(config.num_workers) as pool:
        results = pool.map(_execute_episode_worker, worker_args)
        for episode_examples in results:
            all_examples.extend(episode_examples)

    return all_examples


class ParallelCoach:
    """Coach with parallel self-play support.

    This is a drop-in replacement for Coach that uses parallel self-play
    to speed up training.
    """

    def __init__(self, game: MiniShogiGame, nnet, args, parallel_config: ParallelConfig | None = None):
        """Initialize the parallel coach.

        Args:
            game: MiniShogi game instance.
            nnet: Neural network wrapper.
            args: Training arguments (from main.py).
            parallel_config: Parallel execution configuration.
        """
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)
        self.args = args
        self.parallel_config = parallel_config or ParallelConfig()
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.train_examples_history: list[deque] = []
        self.skip_first_self_play = False

    def execute_parallel_episodes(self) -> list:
        """Execute parallel self-play episodes.

        Returns:
            List of training examples.
        """
        return parallel_self_play(
            self.game,
            self.nnet,
            self.args,
            self.parallel_config,
            self.args.tempThreshold,
        )

    def learn(self) -> None:
        """Main training loop with parallel self-play."""
        for i in range(1, self.args.numIters + 1):
            log.info(f"Starting Iter #{i} ...")

            if not self.skip_first_self_play or i > 1:
                iteration_train_examples: deque = deque([], maxlen=self.args.maxlenOfQueue)

                # Parallel self-play
                log.info(f"Running parallel self-play with {self.parallel_config.num_workers} workers...")
                examples = self.execute_parallel_episodes()
                iteration_train_examples.extend(examples)
                log.info(f"Collected {len(examples)} training examples")

                self.train_examples_history.append(iteration_train_examples)

            if len(self.train_examples_history) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing oldest entry. len(trainExamplesHistory)={len(self.train_examples_history)}"
                )
                self.train_examples_history.pop(0)

            # Train
            train_examples: list = []
            for e in self.train_examples_history:
                train_examples.extend(e)
            shuffle(train_examples)

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(train_examples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            # Arena comparison
            log.info("PITTING AGAINST PREVIOUS VERSION")
            arena = Arena(
                lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                self.game,
            )
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info(f"NEW/PREV WINS : {nwins} / {pwins} ; DRAWS : {draws}")
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info("REJECTING NEW MODEL")
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")
            else:
                log.info("ACCEPTING NEW MODEL")
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=f"checkpoint_{i}.pth.tar")
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename="best.pth.tar")
