# Alpha Zero General
### for any game with any deep learning framework

A simplified, highly flexible, commented and (hopefully) easy to understand implementation of self-play based reinforcement learning based on the AlphaGo Zero paper (Silver et al). It is designed to be easy to adopt for any two-player turn-based adversarial game and any deep learning framework of your choice. A sample implementation has been provided for the game of Othello in PyTorch.

To use a game of your choice, subclass ```Game.py``` and ```NeuralNet.py``` implement their functions. Example implementations for Othello can be found in ```othello/OthelloGame.py``` and ```othello/NNet.py```. 

```Coach.py``` contains the core training loop and ```MCTS.py``` performs the Monte Carlo Tree Search. The parameters for the self-play can be specified in ```main.py```. Additional neural network parameters are in ```othello/NNet.py```. 


### Contributors
* [Shantanu Thakoor](https://github.com/ShantanuThakoor)
* [Megha Jhunjhunwala](https://github.com/jjw-megha)
