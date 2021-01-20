# This Fork
The Santorini branch of this fork add the logic for the board game [Santorini](https://en.wikipedia.org/wiki/Santorini_(game)) to this repository. 

All of the added code in contained within the Santorini folder with the exception of the changes to the README, and to testallgames.py, where minor changes to add Santorini were made.

## Santorini Rules and Structures
Santorini is a two-player competative board game that takes place on a 5x5 board. Each player has two character pieces, (an infinite number of) cubic building pieces as well as dome pieces. 
Players begin the game by alternatingly placing their character pieces on an empty board space. At this point the board is empty except for the 4 player pieces (2 for each player). 

__The gameplay__ involves a player moving one of their pieces one space in any direction (including diagonal) and *then* building a block on a space adjacent to the location they end up at. 

When a block is placed on top of a space, the height of the space is increased by one. 

Multiple blocks can be stacked on top of one another, so if the board itself has a height of zero, a space with two blocks on it has a height of 2. 

A piece can move down any amount of height, but cannot move up by more than 1 in a turn. So a piece cannot move from the board onto the space with two blocks on it, without first moving onto a space next to that one with one block on it, and then waiting for the opponent to perform their turn.

If there are already three blocks on a space, instead of another block, a dome can be built on that space. Pieces cannot stand on, nor build on spaces which have domes. See [this photo](https://www.1843magazine.com/sites/default/files/styles/il_manual_crop_16_9/public/Santorini-header-V3.jpg) for an image of the domed roofs in the actual town of Santorini, Italy.  

A player **wins** when they move a piece onto a space of height 3 (having three blocks on it, and no dome), or when their opponent is unable to make a legal move.


The following must be true about a player's move for it to be legal:
* The player cannot move onto, or build on, any space with another player's piece in it. 
* The player cannot move onto, or build on, any space with a dome on it, regardless of who placed the dome. 
* The block must be built in a space adjacent to where the piece that was moved ended up. The player cannot move one piece and build next to another piece. The player cannot move one piece and build on a space that was adjacent to where the piece started, but not adjacent to where the piece ended.
* The player must move and then must build. If the player either cannot move a piece, or cannot build after moving, that player loses. It is not possible for a player to move a piece onto a winner square, and then be unable to build, because the player will always have the option to build on space they moved from.
* The player cannot move a piece onto a square that is more than 1 block higher than the square the piece moved from.
* The player cannot pass their turn. 
* The player cannot place a build a block on a space that already has three blocks on it. The player can only build a dome on that space, or build elsewhere. 

### Implementation specifics:
Instead of treating the board as 5x5 grid, the board is treated as an array of shape (2,5,5). The first 5x5 subarray contains the locations of each player's characters. The second contains the heights of each space on the board. A player's characters are referred to internally as 1 and 2 (with -1 and -2 being the opponents), and visually represented as O and U for player 1, and X and Y for player 2 (player -1). The visualizations can be easily changed in SantorniGame.py.

Arbitrary nxn sized boards are supported. The default is 5x5. Calling SantoriniGame(n) (see Main.py) will create games with an nxn shaped board. Internally this will be (2,n,n).

By default, Each player's pieces are positioned around the center of the board at the start of the game, rather than having them place their own pieces. This is done to simplify the game for the neural net, since the game complexity is rather high, and the net struggles considerably to learn how to play (well). As an alternative, players can have their pieces placed randomly by setting true_random_placement=True when calling SantoriniGame: SantoriniGame(board_size, true_random_placement=True). At present, no way for players to place their pieces at the start of the game has been implemented.  


    """
    A Santorini Board of default shape: (2,5,5)
    
    Board logic:
        board shape: (2, self.n, self.n)
           [[[ 0,  0,  0,  0,  0],
             [ 0,  0,  1,  0,  0],
             [ 0, -1,  0, -2,  0],
             [ 0,  0,  2,  0,  0],
             [ 0,  0,  0,  0,  0]]
            
            [[ 0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0]]]
    
    BOARD[0]: character locations
    board[0] shape = (self.n,self.n) here this is (5,5)
    Cannonical version of this board shows 
    a player with their pieces as +1, +2 and opponents as -1, -2
        
    LOCATIONS: 
        Locations are given as (x,y) (ROW, COLUMN) coordinates,
        e.g. the 1 in board[0] is at location (1,2), and the 2 at (3,2), whereas
        the -1 is at location (2,1), and the -2 at (2,3)
    
    ACTIONS: 
        Actions are stored as list of tuples of the form:
            action = [piece_location, move_location, build_location]
                     [(x1,y1),        (x2, y2),      (x3, y3)]
    
    BOARD 1: Location heights
        board shape: (self.n,self.n)
        Cannonical board shows player height of each board space.
        The height of each space ranges from 0,...,4 (this is independent of self.n)
    
Original README follows:
# Alpha Zero General (any game, any framework!)
A simplified, highly flexible, commented and (hopefully) easy to understand implementation of self-play based reinforcement learning based on the AlphaGo Zero paper (Silver et al). It is designed to be easy to adopt for any two-player turn-based adversarial game and any deep learning framework of your choice. A sample implementation has been provided for the game of Othello in PyTorch, Keras, TensorFlow and Chainer. An accompanying tutorial can be found [here](http://web.stanford.edu/~surag/posts/alphazero.html). We also have implementations for GoBang and TicTacToe.

To use a game of your choice, subclass the classes in ```Game.py``` and ```NeuralNet.py``` and implement their functions. Example implementations for Othello can be found in ```othello/OthelloGame.py``` and ```othello/{pytorch,keras,tensorflow,chainer}/NNet.py```. 

```Coach.py``` contains the core training loop and ```MCTS.py``` performs the Monte Carlo Tree Search. The parameters for the self-play can be specified in ```main.py```. Additional neural network parameters are in ```othello/{pytorch,keras,tensorflow,chainer}/NNet.py``` (cuda flag, batch size, epochs, learning rate etc.). 

To start training a model for Othello:
```bash
python main.py
```
Choose your framework and game in ```main.py```.

### Docker Installation
For easy environment setup, we can use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Once you have nvidia-docker set up, we can then simply run:
```
./setup_env.sh
```
to set up a (default: pyTorch) Jupyter docker container. We can now open a new terminal and enter:
```
docker exec -ti pytorch_notebook python main.py
```

### Experiments
We trained a PyTorch model for 6x6 Othello (~80 iterations, 100 episodes per iteration and 25 MCTS simulations per turn). This took about 3 days on an NVIDIA Tesla K80. The pretrained model (PyTorch) can be found in ```pretrained_models/othello/pytorch/```. You can play a game against it using ```pit.py```. Below is the performance of the model against a random and a greedy baseline with the number of iterations.
![alt tag](https://github.com/suragnair/alpha-zero-general/raw/master/pretrained_models/6x6.png)

A concise description of our algorithm can be found [here](https://github.com/suragnair/alpha-zero-general/raw/master/pretrained_models/writeup.pdf).

### Contributing
While the current code is fairly functional, we could benefit from the following contributions:
* Game logic files for more games that follow the specifications in ```Game.py```, along with their neural networks
* Neural networks in other frameworks
* Pre-trained models for different game configurations
* An asynchronous version of the code- parallel processes for self-play, neural net training and model comparison. 
* Asynchronous MCTS as described in the paper

### Contributors and Credits
* [Shantanu Thakoor](https://github.com/ShantanuThakoor) and [Megha Jhunjhunwala](https://github.com/jjw-megha) helped with core design and implementation.
* [Shantanu Kumar](https://github.com/SourKream) contributed TensorFlow and Keras models for Othello.
* [Evgeny Tyurin](https://github.com/evg-tyurin) contributed rules and a trained model for TicTacToe.
* [MBoss](https://github.com/1424667164) contributed rules and a model for GoBang.
* [Jernej Habjan](https://github.com/JernejHabjan) contributed RTS game.
* [Adam Lawson](https://github.com/goshawk22) contributed rules and a trained model for 3D TicTacToe.
* [Carlos Aguayo](https://github.com/carlos-aguayo) contributed rules and a trained model for Dots and Boxes along with a [JavaScript implementation](https://github.com/carlos-aguayo/carlos-aguayo.github.io/tree/master/alphazero).
* [Robert Ronan](https://github.com/rlronan) contributed rules for Santorini.

