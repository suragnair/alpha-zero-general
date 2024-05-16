#https://en.wikipedia.org/wiki/Tafl_games

class Tafl:
    size=0
    board=[]
    pieces=[]
    def expandeighth(self,size,eighth):
        hs=size//2
        aquarter=eighth.copy()
        for b in eighth:
            if b[0]!=b[1]: aquarter.extend([[b[1],b[0],b[2]]])
        whole=aquarter.copy()
        for b in aquarter:
            if (b[0]!=hs): whole.extend([[size-b[0]-1,b[1],b[2]]])
            if (b[1]!=hs): whole.extend([[b[0],size-b[1]-1,b[2]]])
            if (b[0]!=hs and b[1]!=hs): whole.extend([[size-b[0]-1,size-b[1]-1,b[2]]])
        return whole    


class Brandubh(Tafl):
   def __init__(self): 
     self.size=7   
     self.board=self.expandeighth(self.size,[[0,0,1],[3,3,2]])
     self.pieces=self.expandeighth(self.size,[[3,0,-1],[3,1,-1],[3,2,1],[3,3,2]])

class ArdRi(Tafl):
   def __init__(self): 
     self.size=7   
     self.board=self.expandeighth(self.size,[[0,0,1],[3,3,2]])
     self.pieces=self.expandeighth(self.size,[[2,0,-1],[3,0,-1],[3,1,-1],[3,2,1],[2,2,1],[3,3,2]])

class Tablut(Tafl):
   def __init__(self): 
     self.size=9
     self.board=self.expandeighth(self.size,[[0,0,1],[4,4,2]])
     self.pieces=self.expandeighth(self.size,[[3,0,-1],[4,0,-1],[4,1,-1],[4,2,1],[4,3,1],[4,4,2]])

class Tawlbwrdd(Tafl):
   def __init__(self): 
     self.size=11   
     self.board=self.expandeighth(self.size,[[0,0,1],[5,5,2]])
     self.pieces=self.expandeighth(self.size,[[4,0,-1],[5,0,-1],[4,1,-1],[5,2,-1],[5,3,1],[5,4,1],[4,4,1],[5,5,2]])

class Hnefatafl(Tafl):
   def __init__(self): 
     self.size=11  
     self.board=self.expandeighth(self.size,[[0,0,1],[5,5,2]])
     self.pieces=self.expandeighth(self.size,[[3,0,-1],[4,0,-1],[5,0,-1],[5,1,-1],[5,3,1],[5,4,1],[4,4,1],[5,5,2]])

class AleaEvangelii(Tafl):
   def __init__(self): 
     self.size=19  
     self.board=self.expandeighth(self.size,[[0,0,1],[9,9,2]])
     self.pieces=self.expandeighth(self.size,[[2,0,-1],[5,0,-1],[5,2,-1],[7,3,-1],[9,3,-1],[6,4,-1],[5,5,-1],[8,4,1],[9,6,1],[8,7,1],[9,8,1],[9,9,2]])



