from tkinter import *
import numpy as np
from queue import *
import sys
import threading

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a

class HumanOthelloPlayer():
    def __init__(self, game):
        self.game = game
        self.game = game
        self.inputQueue = Queue(1)
        self.root = Tk()
        self.screen = Canvas(self.root, width=500, height=600, background="#444",highlightthickness=0)
        self.screen.pack()
        self.screen.create_rectangle(50,50,450,450,outline="#111")
        self.gap = 400/game.n
        for i in range(game.n):
            lineShift = 50+self.gap*(i+1)
            #Horizontal line
            self.screen.create_line(50,lineShift,450,lineShift,fill="#111")
            #Vertical line
            self.screen.create_line(lineShift,50,lineShift,450,fill="#111")
        #self.screen.bind("<Button-1>", self.clickHandle)
        self.screen.update()
        
    def clickHandle(self, event):
        y = int((event.x-50)/self.gap)
        x = int((event.y-50)/self.gap)
        #self.inputQueue.put((x.y), False)
        print((x,y))
    
    def playerName(self, player):
        if player == 1:
            return "Black"
        else:
             return "White"   
    
    def refresh(self, board, player=1, ended=False):
        if ended:
            p = player
            s = self.game.getScore(board, p)
            if s == 0:
                self.screen.create_text(250,550,anchor="c",font=("Consolas",20), text="Tie!")
            elif s < 0:
                p = -p
            self.screen.create_text(250,550,anchor="c",font=("Consolas",20), tgas="txt",text=self.playerName(p) + " won!")
            self.screen.delete("txt")
            input()
            return
                
        self.screen.delete("highlight")
        self.screen.delete("tile")
        for x in range(self.game.n):
            for y in range(self.game.n):
                if board[y][x] == -1:
                    # white
                    self.screen.create_oval(54+self.gap*x,54+self.gap*y,46+self.gap*(x+1),46+self.gap*(y+1),tags="tile {0}-{1}".format(x,y),fill="#aaa",outline="#aaa")
                    self.screen.create_oval(54+self.gap*x,54+self.gap*y,46+self.gap*(x+1),46+self.gap*(y+1),tags="tile {0}-{1}".format(x,y),fill="#fff",outline="#fff")
                elif board[y][x] == 1:
                    # black
                    self.screen.create_oval(54+self.gap*x,54+self.gap*y,46+self.gap*(x+1),46+self.gap*(y+1),tags="tile {0}-{1}".format(x,y),fill="#000",outline="#000")
                    self.screen.create_oval(54+self.gap*x,54+self.gap*y,46+self.gap*(x+1),46+self.gap*(y+1),tags="tile {0}-{1}".format(x,y),fill="#111",outline="#111")
        
        valid = self.game.getValidMoves(board, player)
        for i in range(len(valid)):
            if valid[i]:
                (y, x) = (int(i/self.game.n), int(i%self.game.n))
                self.screen.create_oval(68+self.gap*x,68+self.gap*y,32+self.gap*(x+1),32+self.gap*(y+1),tags="highlight",fill="#008000",outline="#008000")
        self.screen.update()

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i/self.game.n), int(i%self.game.n))
        while True:
            a = input()
            x,y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x!= -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a

class GreedyOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]
