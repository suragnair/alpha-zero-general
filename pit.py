from Arena import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame, display
from othello.OthelloPlayers import *
from othello.tensorflow.NNet import NNetWrapper as NNet
import os
import numpy as np
import tensorflow as tf
import multiprocessing
from utils import *
from pytorch_classification.utils import Bar, AverageMeter

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

def Async_Play(game,args,iter_num,bar):
    bar.suffix = "iter:{i}/{x} | Total: {total:} | ETA: {eta:}".format(i=iter_num+1,x=args.numPlayGames,total=bar.elapsed_td, eta=bar.eta_td)
    bar.next()

    # set gpu
    if(args.multiGPU):
        if(iter_num%2==0):
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU

    # set gpu growth
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True  
    sess = tf.Session(config=config)

    # create NN
    model1 = NNet(game)
    model2 = NNet(game)

    # try load weight
    try:
        model1.load_checkpoint(folder=args.model1Folder, filename=args.model1FileName)
    except:
        print("load model1 fail")
        pass
    try:
        model2.load_checkpoint(folder=args.model2Folder, filename=args.model2FileName)
    except:
        print("load model2 fail")
        pass

    # create MCTS
    mcts1 = MCTS(game, model1, args)
    mcts2 = MCTS(game, model2, args)

    # each process play 2 games
    arena = Arena(lambda x: np.argmax(mcts1.getActionProb(x, temp=0)),lambda x: np.argmax(mcts2.getActionProb(x, temp=0)), game)
    arena.displayBar = False
    oneWon,twoWon, draws = arena.playGames(2)
    return oneWon,twoWon, draws

if __name__=="__main__":
    """
    Before using multiprocessing, please check 2 things before use this script.
    1. The number of PlayPool should not over your CPU's core number.
    2. Make sure all Neural Network which each process created can store in VRAM at same time. Check your NN size before use this.
    """
    args = dotdict({
    'numMCTSSims': 25,
    'cpuct': 1,

    'multiGPU': False,  # multiGPU only support 2 GPUs.
    'setGPU': '0',
    'numPlayGames': 4,  # total num should x2, because each process play 2 games.
    'numPlayPool': 4,   # num of processes pool.

    'model1Folder': './temp/',
    'model1FileName': 'best.pth.tar',
    'model2Folder': './temp/',
    'model2FileName': 'best.pth.tar',

    })

    def ParallelPlay(g):
        bar = Bar('Play', max=args.numPlayGames)
        pool = multiprocessing.Pool(processes=args.numPlayPool)
        res = []
        result = []
        for i in range(args.numPlayGames):
            res.append(pool.apply_async(Async_Play,args=(g,args,i,bar)))
        pool.close()
        pool.join()

        oneWon = 0
        twoWon = 0
        draws = 0
        for i in res:
            result.append(i.get())
        for i in result:
            oneWon += i[0]
            twoWon += i[1]
            draws += i[2]
        print("Model 1 Win:",oneWon," Model 2 Win:",twoWon," Draw:",draws)


    g = OthelloGame(6)

    # parallel version
    #ParallelPlay(g)

    # single process version
    # all players
    rp = RandomPlayer(g).play
    gp = GreedyOthelloPlayer(g).play
    hp = HumanOthelloPlayer(g).play

    # nnet players
    n1 = NNet(g)
    n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
    args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    #n2 = NNet(g)
    #n2.load_checkpoint('/dev/8x50x25/','best.pth.tar')
    #args2 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
    #mcts2 = MCTS(g, n2, args2)
    #n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    arena = Arena.Arena(n1p, hp, g, display=display)
    print(arena.playGames(2, verbose=True))