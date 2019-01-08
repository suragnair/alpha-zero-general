# TicTacToe implementation for Alpha Zero General

An implementation of a simple game provided to check extendability of the framework. Main difference of this game comparing to Othello is that it allows draws, i.e. the cases when nobody won after the game ended. To support such outcomes ```Arena.py``` and ```Coach.py``` classes were modified. Neural network architecture was copy-pasted from the game of Othello, so possibly it can be simplified. 

To train a model for TicTacToe, change the imports in ```main.py``` to:
```python
from Coach import Coach
from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.keras.NNet import NNetWrapper as nn
from utils import *
```

and the first line of ```__main__``` to
```python
g = TicTacToeGame()
```
 Make similar changes to ```pit.py```.

To start training a model for TicTacToe:
```bash
python main.py
```
To start a tournament of 100 episodes with the model-based player against a random player:
```bash
python pit.py
```
You can play againt the model by switching to HumanPlayer in ```pit.py```

### Experiments
I trained a Keras model for 3x3 TicTacToe (3 iterations, 25 episodes, 10 epochs per iteration and 25 MCTS simulations per turn). This took about 30 minutes on an i5-4570 without CUDA. The pretrained model (Keras) can be found in ```pretrained_models/tictactoe/keras/```. You can play a game against it using ```pit.py```. 

### Contributors and Credits
* [Evgeny Tyurin](https://github.com/evg-tyurin)

The implementation is based on the game of Othello (https://github.com/suragnair/alpha-zero-general/tree/master/othello).

### AlphaGo / AlphaZero Talks
* February 8, 2018 - [Advanced Spark/Tensorflow Meetup at Thumbtack](https://www.meetup.com/Advanced-Spark-and-TensorFlow-Meetup/events/245308722/): [Youtube](https://youtu.be/dhmBrTouCKk?t=1017) / [Slides](http://static.brettkoonce.com/presentations/go_v1.pdf)
* March 6, 2018 - [Advanced Spark/Tensorflow Meetup at Strata San Jose](https://www.meetup.com/Advanced-Spark-and-TensorFlow-Meetup/events/246530339/): [Youtube](https://www.youtube.com/watch?time_continue=1257&v=hw9VccUyXdY) / [Slides](http://static.brettkoonce.com/presentations/go.pdf)
