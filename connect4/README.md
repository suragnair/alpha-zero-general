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


### Experiments 2 
I trained a tensorflow model for 6x7 Connect4 board (100 episodes,
10 epochs per iteration and 25 MCTS simulations per turn) for 32 iterations which took about ~5 hours on a i7-6700HQ laptop

After 32 iterations, the model get a pretty good result (83 win,7 lose,10 draw)  vs the 'OneStepLookaheadConnect4Player' player.

##front-end
###by chatgpt

Sure! Here is a high-level overview of how you could implement a Connect Four game using React:

Set up a new React project using a tool like create-react-app.
Create a Board component that represents the game board. This component should render a 6x7 grid of cells.
Create a Cell component that represents an individual cell on the board. This component should be able to display a colored circle if it is occupied by a player's piece.
Create a Game component that keeps track of the state of the game, including which player's turn it is, which cells are occupied, and whether the game has been won or drawn. This component should render the Board component and handle user interaction (e.g. clicking on a cell to place a piece).
Implement the game logic for determining when a player has won or when the game is a draw. This can be done by checking for horizontal, vertical, and diagonal lines of four connected pieces.
Add some styling to make the game look nice and add some UI elements, such as buttons for starting a new game or displaying the current player's turn.
Here is some sample code that demonstrates how the Cell and Board components could be implemented:

Copy code
import React from 'react';

const Cell = ({ color }) => {
  return (
    <div
      className="cell"
      style={{
        backgroundColor: color,
        border: '1px solid black',
        width: '100%',
        height: '100%',
        borderRadius: '50%',
      }}
    />
  );
};

const Board = ({ board }) => {
  return (
    <div className="board">
      {board.map((row, i) => (
        <div className="row" key={i}>
          {row.map((color, j) => (
            <Cell color={color} key={j} />
          ))}
        </div>
      ))}
    </div>
  );
};
The board prop for the Board component should be a 2D array of strings representing the colors of each cell on the board. For example:

Copy code
const board = [
  ['red', 'yellow', 'red', 'yellow', 'red', 'yellow', 'red'],
  ['yellow', 'red', 'yellow', 'red', 'yellow', 'red', 'yellow'],
  ['red', 'yellow', 'red', 'yellow', 'red', 'yellow', 'red'],
  ['yellow', 'red', 'yellow', 'red', 'yellow', 'red', 'yellow'],
  ['red', 'yellow', 'red', 'yellow', 'red', 'yellow', 'red'],
  ['yellow', 'red', 'yellow', 'red', 'yellow', 'red', 'yellow'],
];
I hope this gives you a good starting point for building your Connect Four game! Let me know if you have any questions or need further guidance.
