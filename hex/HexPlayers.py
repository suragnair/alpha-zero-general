import numpy as np

class RandomPlayer():
    def __init__(self, game) -> None:
        self.game = game

    def play(self, positions):
        action = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(positions, 1)
        while valids[action] != 1:
            action = np.random.randint(self.game.getActionSize())
        return action