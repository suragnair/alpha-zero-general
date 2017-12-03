from Coach import Coach
from OthelloGame import OthelloGame
from NNet import NNetWrapper as nn

if __name__=="__main__":
    g = OthelloGame(6)
    nnet = nn(g)
    c = Coach(g, nn)
    c.learn()
