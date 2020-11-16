# Ultimate TicTacToe implementation for Alpha Zero General
*Carlos Sosa, Eduardo Cuya, Ivonne Heredia, David Aguilar 2020*

This is part of a undergraduate course final project in which different reinforcement learning algorithms are tested on Ultimate TicTacToe.
This part includes an implementation of that game in Alpha Zero General wrapper created by Surag Nair in [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general).

## Game Description

An Ultimate TicTacToe board consists on a 9x9 grid which represents a TicTacToe board filled with TicTacToe 3x3 local boards.


After the first move, each move on a game is restricted to the relative board that correspond to the spot in which the last move was made. For example, if X plays in the bottom right space in their local board, O can only choose for their next move a square on the bottom right local board unless that local board was already won by someone or ended as a draw. In that case, O can choose any free local board.

If a player wins a local board, it is considered as making a move on that position on the global board. The game ends when a player wins the global board or there are no moves left.

## State Representation

During a game, the board is represented by an object that has three attributes:

- **pieces:** a (9,9) Numpy array that represents the board state
- **win_status:** a (3,3) Numpy array that represents the global board
- **last_move:** a tuple containing the position of the last move in the game

## Implementation

 Game, Logic and Players implementations ared based on TicTacToe implementation by Evgeny Tyurin. UltimateTicTacToeNNet is based on the Keras implementation of OthelloNNet by Shantanu Kumar.
 
## Test Scripts

To train a model for Ultimate Tic Tac Toe:
````bash
python ultimate_tic_tac_toe/utt_main.py
````
To test a model against a random player or a human player:
````bash
python ultimate_tic_tac_toe/utt_pit.py
````

## Experiments 

We trained a Keras model for Ultimate TicTacToe (15 iterations, 100 episodes, 20 epochs per iteration and 25 MCTS simulations per turn) for about 50 hours on an AMD Radeon Pro 560 4GB with OpenCL and PlaidML.
