# 4D TicTacToe for Alpha Zero General

TicTacToe on a 4×4×4×4 board: 256 squares, and 4 in a row wins. A line can stay inside one cube or run through all four, along any of the 40 directions of the hypercube, which makes 520 winning lines. For comparison, the 4×4×4 cube has 76.

`Game(n)` works for any `n`, and a game has `((n + 2)^4 - n^4) / 2` lines. The pretrained model is for `n = 4`.

A square is `pieces[w][z][y][x]`, and an action is its flat index `w·n³ + z·n² + y·n + x`. Passing is never possible, so the action size is `n⁴`.

## Network

PyTorch has no 4D convolution, so `pytorch/TicTacToe4DNNet.py` builds one from `Conv3d`: a 3×3×3×3 kernel is three 3×3×3 kernels, one per step along the fourth axis. `test_tictactoe_4d.py` checks it against the definition cell by cell.

Long diagonals are hard to find with small kernels, so the input also has hint planes: for each `k`, on how many lines through a square one side has `k` stones and the other side has none.

`NNetWrapper.predict` plays forced moves by itself: it completes a line if it can, otherwise it blocks one. With 256 squares and a few simulations the search often misses them, and the pretrained network relies on this rule (see below).

`getSymmetries` returns the 16 reflections. The hypercube has 384 symmetries in all, which is too many to store for each position.

## Training

Change the imports in `main.py` to:

```python
from tictactoe_4d.TicTacToe4DGame import TicTacToe4DGame as Game
from tictactoe_4d.pytorch.NNet import NNetWrapper as nn
```

and create the game with `g = Game(4)`. Make similar changes to `pit.py`.

## Playing against the pretrained model

Run this from the root of the repository:

```python
import numpy as np
import Arena
from MCTS import MCTS
from utils import dotdict
from tictactoe_4d.TicTacToe4DGame import TicTacToe4DGame as Game
from tictactoe_4d.TicTacToe4DPlayers import HumanPlayer
from tictactoe_4d.pytorch.NNet import NNetWrapper as nn

game = Game(4)
net = nn(game)
net.load_checkpoint('pretrained_models/tictactoe_4d/pytorch', '4x4x4x4_best.pth.tar')
mcts = MCTS(game, net, dotdict({'numMCTSSims': 100, 'cpuct': 1.5}))
ai = lambda board: np.argmax(mcts.getActionProb(board, temp=0))
human = HumanPlayer(game).play
print(Arena.Arena(ai, human, game, display=Game.display).playGames(2, verbose=True))
```

A board in a terminal is hard to read in four dimensions. The [browser version](https://ilyakz.github.io/qubic-zero/) shows it as a hypercube you can rotate, and it plays with the same weights.

## The pretrained model

`pretrained_models/tictactoe_4d/pytorch/4x4x4x4_best.pth.tar` has 32 channels, 4 residual blocks and 690,722 parameters, 2.8 MB in all.

It was not trained with `Coach.py`. It comes from [qubic-zero](https://github.com/ilyakz/qubic-zero), a separate engine with a parallel self-play core. Its training mixed self-play with games against a threat-search bot and against an earlier version of the network, using the root search of Gumbel AlphaZero. That engine plays forced moves outside the network and looks for winning threat sequences during the search, so the network was never taught those moves. This is why `predict` plays them here.

## Strength

Measured with this repository's `Arena` and `MCTS`, `cpuct` 1.5, a new search tree for each game, and half of the games moving first:

| Opponent | Simulations | Games | Network wins |
|---|---|---|---|
| `RandomPlayer` | 25 | 40 | 100% |
| `OneStepPlayer` | 100 | 40 | 100% |
| Threat-search bot from qubic-zero | 25 | 100 | 20% |
| Threat-search bot from qubic-zero | 100 | 60 | 30% |

The last two rows show the limit of this setup. The bot plays long sequences of forcing moves, and plain MCTS with so few simulations does not see them coming: at 100 simulations the network won 14 games of 30 moving first and 4 of 30 moving second. In qubic-zero's own engine, which adds threat-sequence search and a safety check at the root, the same weights score 97% against this bot over 400 games at 100 simulations.
