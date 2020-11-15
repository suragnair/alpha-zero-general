# Dots And Boxes implementation for Alpha Zero General

[Alpha Zero General](https://github.com/suragnair/alpha-zero-general/) implementation of [Dots and Boxes](https://en.wikipedia.org/wiki/Dots_and_Boxes).

Read more details about this code in this [Medium Blog post](https://towardsdatascience.com/alphazero-a-novel-reinforcement-learning-algorithm-deployed-in-javascript-56018503ad18).

There are three Colab Notebooks <img height="20" style="vertical-align: middle; display:inline-block;" src="https://colab.research.google.com/img/colab_favicon.ico"> that will help you reproduce and follow the code along.

The first notebook lets you play against AlphaZero: <a href="https://colab.research.google.com/github/carlos-aguayo/alpha-zero-general/blob/dotsandboxes/dotsandboxes/Play%20Dots%20and%20Boxes%20using%20AlphaZero%20General.ipynb"><strong>Play Dots and Boxes using AlphaZero General.ipynb</strong></a>

You can also play against the JavaScript version here:
<a href="https://carlos-aguayo.github.io/alphazero"><strong>https://carlos-aguayo.github.io/alphazero</strong></a>

The second notebook show you the training process: <a href="https://colab.research.google.com/github/carlos-aguayo/alpha-zero-general/blob/dotsandboxes/dotsandboxes/Train%20Dots%20and%20Boxes%20using%20AlphaZero%20General.ipynb"><strong>Train Dots and Boxes using AlphaZero General.ipynb</strong></a>

The third notebook converts the trained model to be used with TensorFlow.js
<a href="https://colab.research.google.com/github/carlos-aguayo/alpha-zero-general/blob/dotsandboxes/dotsandboxes/Convert%20Keras%20Model%20for%20use%20with%20Tensorflow.js.ipynb"><strong>Convert Keras Model for use with Tensorflow.js.ipynb (View in Colab)</strong></a>

This code has also been ported to Javascript and available here:
<a href="https://github.com/carlos-aguayo/carlos-aguayo.github.io/tree/master/alphazero"><strong>https://github.com/carlos-aguayo/carlos-aguayo.github.io/tree/master/alphazero</strong></a>

You can also run things locally. To start training a model for Dots and Boxes:
```bash
python main-dotsandboxes.py
```
To play against AlphaZero general:
```bash
python pit-dotsandboxes.py
```

### Contributors and Credits
* [Carlos Aguayo](https://github.com/carlos-aguayo)

The implementation is based on the game of Othello ([https://github.com/suragnair/alpha-zero-general/tree/master/othello](https://towardsdatascience.com/alphazero-a-novel-reinforcement-learning-algorithm-deployed-in-javascript-56018503ad18)) courtesy of [Surag Nair](https://github.com/suragnair).