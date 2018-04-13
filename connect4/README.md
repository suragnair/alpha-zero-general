# Connect4 implementation for Alpha Zero General

Alpha-zero general implementation of connect4.
Neural network architecture was copy-pasted from the game of Othello, so could likely be improved.

To train a model for Connect4, update the imports in ```main.py``` to:
```python
from Coach import Coach
from connect4.Connect4Game import Connect4Game
from connect4.tensorflow.NNet import NNetWrapper as nn
from utils import dotdict
```

and the first line of ```__main__``` to
```python
g = Connect4Game()
```

Make similar changes to ```pit.py```.

To start training a model for Connect4:
```bash
python main.py
```
To start a tournament of 100 episodes with the model-based player against a random player:
```bash
python pit.py
```
You can play againt the model by switching to HumanPlayer in ```pit.py```

### Experiments

Experiments so far have been CPU constrained. I trained a tensorflow model for 6x7 Connect4 board (100 episodes,
10 epochs per iteration and 25 MCTS simulations per turn) for 6 iterations which took about ~5 hours on an old
i7-2670QM laptop without CUDA.  Running a further 6 iterations (20 hours?) has not yielded a noticable improvment.

After 6 iterations, the model is wins 50% of the time vs the 'OneStepLookaheadConnect4Player' player.
