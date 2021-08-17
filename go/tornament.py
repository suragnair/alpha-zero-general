import Arena
from MCTS import MCTS
from go.Game import Game
from go.GoPlayers import *
from go.pytorch.NNet import NNetWrapper as NNet
import numpy as np
import random
import operator
from utils import *

"""
use thisss script to play any two agents against each other, or play manually with
any agent.
"""


args = dotdict({
    'size': 5,                  #board size
    'numMCTSSims': 2,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 2,         # Number of games to play during arena play to determine if new net will be accepted.
                                # Total matches: arenaCompare*len(playerList)*(len(playerList) - 1)
    'cpuct': 1.1,
    'arenaNumMCTSSims': 2,      # simulations for arena
    'instinctArena': True,      # if set true reset Arena's MTCL tree each time
    'balancedGame': False,      # if balanced, black should win over 6 scores
    'resignThreshold': -0.999999, # No Use. Resign Only in self-play Training


})


g = Game(args)

rp = RandomPlayer(g).play
hp = HumanGoPlayer(g).play


def createNetPlayer(dirr, tarName, sim=args.numMCTSSims, cpuct=args.cpuct):
    n = NNet(g)
    n.load_checkpoint(dirr, tarName)
    mcts = MCTS(g, n, args)
    player = lambda x: np.argmax(mcts.getActionProb(x, temp=0, instinctPlay=args.instinctArena)[0])
    return player

def playGame(player1, str1, player2, str2):
    arena = Arena.Arena(player1, player2, g)
    x, y, z, xb = arena.playGames(args.arenaCompare, verbose=False)
    print(str1, " win: ", x)
    print(str2, " win: ", y)
    print(str1, " win black: ", xb)
    return x, y

def tournament(playList):
    tournamentResult = dict.fromkeys(playList, 0)
    for a in playList:
        for b in playList:
            if a is not b:
                aWin, bWin = playGame(a, 'p1', b, 'p2')           
                tournamentResult[a] += (aWin - bWin + args.arenaCompare)/2
                tournamentResult[b] += (bWin - aWin + args.arenaCompare)/2
    print (
    '''
    く__,.ヘヽ.　　　　/　,ー､ 〉
    　　　＼ , !-─‐-i　/　/´
    　　　 　 ／｀ｰ　　　 L/／｀ヽ､
    　　 　 /　 ／,　 /|　 ,　 ,　　    ,
    　　　ｲ 　/ /-‐/　ｉ　L_ ﾊ ヽ!　 i
    　　　 ﾚ ﾍ 7ｲ｀ﾄ　 ﾚ-ﾄ､!ハ|　 |
    　　　　 !,/7 ✪　　 ´i✪ﾊiソ| 　 |　　　
    　　　　 |.从　　_　　 ,,,, / |./ 　 |
    　　　　 ﾚ| i＞.､,,__　_,.イ / 　.i 　|
    　　　 ﾚ| | / k_７_/ﾚヽ,　ﾊ.　|
    　　　 | |/i 〈|/　 i　,.ﾍ |　i　|
    　　　.|/ /　ｉ： 　 ﾍ!　　＼　|
    　　　 　 　 kヽ､ﾊ 　 _,.ﾍ､ 　 /､!
    　　　 !〈//｀Ｔ´, ＼ ｀7ｰr
    　　　 ﾚヽL__|___i,___,ンﾚ|ノ
    　　　 　　　ﾄ-,/　|___./
    　　　 　　　ｰ　　!_,.:
    '''
    )
    

    return list(tournamentResult.values())
best = createNetPlayer('./temp/',"best.pth.tar")
def_27 = createNetPlayer('./temp/',"def_27.pth.tar")
def_52 = createNetPlayer('./temp/',"def_52.pth.tar")
def_77 = createNetPlayer('./temp/',"def_77.pth.tar")
def_102 = createNetPlayer('./temp/',"def_102.pth.tar")
cha_27 = createNetPlayer('./temp/',"cha_27.pth.tar")
cha_52 = createNetPlayer('./temp/',"cha_52.pth.tar")
cha_77 = createNetPlayer('./temp/',"cha_77.pth.tar")
cha_102 = createNetPlayer('./temp/',"cha_102.pth.tar")
one = createNetPlayer('./temp/',"one.pth.tar")
five = createNetPlayer('./temp/',"five.pth.tar")
three = createNetPlayer('./temp/',"three.pth.tar")
no_noise = createNetPlayer('./temp/',"no_noise.pth.tar")
playerList = [def_27, def_52, def_77, def_102, cha_27, cha_52, cha_77, cha_102, best]
#playerList = [best, five, cha_27, cha_52]
result = tournament(playerList)
print(result)


            
               
                

