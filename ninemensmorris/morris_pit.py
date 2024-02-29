"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
human_vs_cpu = False

g = NineMensMorrisGame()

# all players
rp = RandomPlayer(g).play

# nnet players
n1 = NNetWrapper(g)
n1.load_checkpoint('/content/drive/My Drive/training/20it/','best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

player2 = rp  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena(n1p, player2, g, display=NineMensMorrisGame.display)

print(arena.playGames(20, verbose=True))
