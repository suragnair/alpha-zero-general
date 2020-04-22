# TicTacToe implementation for Alpha Zero General

An implementation of a simple game provided to check extendability of the framework. Main difference of this game comparing to Othello is that it allows draws, i.e. the cases when nobody won after the game ended. To support such outcomes ```Arena.py``` and ```Coach.py``` classes were modified. Neural network architecture was copy-pasted from the game of Othello, so possibly it can be simplified. 

To train a model for TicTacToe, change the imports in ```main.py``` to:
```python
from Coach import Coach
from tictactoe_3d.TicTacToeGame import TicTacToeGame as Game
from tictactoe_3d.keras.NNet import NNetWrapper as nn
from utils import *
```

and the first line of ```__main__``` to
```python
g = Game(3)
```
or
```python
g = Game(4)
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
The input is in the form z x y

### Contributors and Credits
* [Adam Lawson](https://github.com/goshawk22)

The implementation is based on the game of Othello (https://github.com/suragnair/alpha-zero-general/tree/master/othello).

### AlphaGo / AlphaZero Talks
* February 8, 2018 - [Advanced Spark/Tensorflow Meetup at Thumbtack](https://www.meetup.com/Advanced-Spark-and-TensorFlow-Meetup/events/245308722/): [Youtube](https://youtu.be/dhmBrTouCKk?t=1017) / [Slides](http://static.brettkoonce.com/presentations/go_v1.pdf)
* March 6, 2018 - [Advanced Spark/Tensorflow Meetup at Strata San Jose](https://www.meetup.com/Advanced-Spark-and-TensorFlow-Meetup/events/246530339/): [Youtube](https://www.youtube.com/watch?time_continue=1257&v=hw9VccUyXdY) / [Slides](http://static.brettkoonce.com/presentations/go.pdf)
