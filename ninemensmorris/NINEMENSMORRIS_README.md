# Nine Mens Morris Rules and Structures

Nine Mens Morris, is a two player board game, the rules of the game can be found here:
http://www.move-this.com/spielregeln/
I used additional tournament rules, that determine, that after 50 moves without a mill, the game ends in a draw

### Using this implementation
This Implementation works like the other games in the repository. Pick the game in main.py to start training a model or use the pretrained model to test it out. I trained the pretrained model for 32 Iterations, 100 Episodes and 15 Epochs, all other Parameters were left on the default values. The model wins about 79% of the games against a random player, I think thats fairly impressive, regarding the number of possible moves of the game. The game has 13272 possible moves, so it should be trained with a higher number of episodes and for more iterations. 

I also created a Colab Notebook, where you can run the code and save the models to your google drive. Before doing that, you got to create the folders, where you want the models and checkpoints to be saved.

To start the training, run the main.py file with the right game selected, and if you want to test it out, there is a random player available and the player that was trained for 32 iterations.
The easiest way to try it out, is to use the notebook.

Both the Keras NNet and the PyTorch NNet are copied from othello, it can surely be optimized. I worked with the PyTorch version.

### Understanding this implementation
The important files for the Game Logic are NineMensMorrisGame.py and NineMensMorrisLogic.py. 

### Training a model
To train a model, change the Game and NNetWrapper in the main.py file to the Ninemensmorris versions.
Or you can use the Notebook (NOTE: I used Google Drive to save the checkpoints, but had some issues in the process with saving checkpoints)

 
