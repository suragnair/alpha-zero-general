import Arena
from MCTS import MCTS
from ultimatetictactoe.UltimateTicTacToeGame import UltimateTicTacToeGame
from ultimatetictactoe.UltimateTicTacToePlayers import *
from ultimatetictactoe.keras.NNet import NNetWrapper as NNet


import numpy as np
from utils import *
import os
from subprocess import Popen, PIPE
"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

minimax_vs_alphazero = True

g = UltimateTicTacToeGame()

# all players
rp = RandomPlayer(g).play
hp = HumanUltimateTicTacToePlayer(g).play

# p = Popen(['ultimatettt.exe', 'X', '1'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
# while True:
#     out = p.stdout.readline()
#     print(out)
#     if not p.poll:
#         break
#     p.stdin.write(b"4 4\n")
#     p.stdin.flush()
# exit()

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp/','best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
def n1p(x, proc, verbose=False):
    result = mcts1.getActionProb(x, temp=0)
    if verbose:
        s = mcts1.game.stringRepresentation(x)
        Ps = np.reshape(mcts1.Ps[s][:81], (9, 9))
        print('\n'.join(['\t'.join(['{:4}'.format(item) for item in row])
                         for row in Ps]))
    pos = np.argmax(result)
    if minimax_vs_alphazero:
        str = "{} {}".format(int(pos/9), pos%9).encode() + b"\n"
        proc.stdin.write(str)
        proc.stdin.flush()
    return pos

def n2p(x, proc, verbose=False):
    result = mcts2.getActionProb(x, temp=0)
    if verbose:
        s = mcts2.game.stringRepresentation(x)
        Ps = np.reshape(mcts2.Ps[s][:81], (9, 9))
        print('\n'.join(['\t'.join(['{:4}'.format(item) for item in row])
                         for row in Ps]))
    pos = np.argmax(result)
    return pos

if minimax_vs_alphazero:
    player2 = hp
else:
    n2 = NNet(g)
    n2.load_checkpoint('./temp/','best.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(n1p, player2, g, display=UltimateTicTacToeGame.display)

max_depth = 1
results = [[[0, 0], [0, 0], [0, 0]] for _ in range(max_depth)]
while True:
    for depth in range(max_depth):
        print()
        print("Depth", depth)
        result = arena.playGamesVsMinimax(20, str(depth), verbose=False)
        for a in range(len(result)):
            for b in range(len(result[a])):
                results[depth][a][b] += result[a][b]
    print()
    for res in results:
        for a in res:
            print(*a, sep='\t', end='\t')
        print()

