import numpy as np
from .GameVariants import Tafl

class Board():


    def __init__(self, gv):
      self.size=gv.size  
      self.width=gv.size
      self.height=gv.size
      self.board=gv.board #[x,y,type]
      self.pieces=gv.pieces #[x,y,type]
      self.time=0
      self.done=0

    def __str__(self):
        return str(self.getPlayerToMove()) + ''.join(str(r) for v in self.getImage() for r in v) 

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return np.array(self.getImage())[index]

    def astype(self,t):
        return np.array(self.getImage()).astype(t)

    def getCopy(self):
      gv=Tafl()
      gv.size=self.size
      gv.board=np.copy(np.array(self.board)).tolist()
      gv.pieces=np.copy(np.array(self.pieces)).tolist()
      b = Board(gv)
      b.time=self.time
      b.done=self.done
      return b


    def countDiff(self, color):
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        count = 0
        for p in self.pieces:
            if p[0] >= 0:
               if p[2]*color > 0:
                   count += 1
               else:
                   count -= 1
        return count

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        return self._getValidMoves(color)
     
    def has_legal_moves(self, color):
        vm = self._getValidMoves(color)
        if len(vm)>0: return True
        return False


    def execute_move(self, move, color):
        """Perform the given move on the board.
        color gives the color pf the piece to play (1=white,-1=black)
        """
        x1,y1,x2,y2 = move
        pieceno = self._getPieceNo(x1,y1)
        legal = self._isLegalMove(pieceno,x2,y2)
        if legal>=0:
           #print("Accepted move: ",move) 
           self._moveByPieceNo(pieceno,x2,y2)
        #else:
           #print("Illegal move:",move,legal)
   
    def getImage(self):
        image = [[0 for col in range(self.width)] for row in range(self.height)]
        for item in self.board:
            image[item[1]][item[0]] = item[2]*10
        for piece in self.pieces:
            if piece[0] >= 0: image[piece[1]][piece[0]] = piece[2] + image[piece[1]][piece[0]]
        return image

    def getPlayerToMove(self):
        return -(self.time%2*2-1)


################## Internal methods ##################

    def _isLegalMove(self,pieceno,x2,y2):
      try:

         if x2 < 0 or y2 < 0 or x2 >= self.width or y2 > self.height: return -1
         
         piece = self.pieces[pieceno]
         x1=piece[0]
         y1=piece[1]
         if x1<0: return -2 #piece was captured
         if x1 != x2 and y1 != y2: return -3 #must move in straight line
         if x1 == x2 and y1 == y2: return -4 #no move

         piecetype = piece[2]
         if (piecetype == -1 and self.time%2 == 0) or (piecetype != -1 and self.time%2 == 1): return -5 #wrong player

         for item in self.board:
            if item[0] == x2 and item[1] == y2 and item[2] > 0:
                if piecetype != 2: return -10 #forbidden space
         for apiece in self.pieces:
            if y1==y2 and y1 == apiece[1] and ((x1 < apiece[0] and x2 >= apiece[0]) or (x1 > apiece[0] and x2 <= apiece[0])): return -20 #interposing piece
            if x1==x2 and x1 == apiece[0] and ((y1 < apiece[1] and y2 >= apiece[1]) or (y1 > apiece[1] and y2 <= apiece[1])): return -20 #interposing piece

         return 0 # legal move
      except Exception as ex:
         print("error in islegalmove ",ex,pieceno,x2,y2)
         raise

   
    def _getCaptures(self,pieceno,x2,y2):
       #Assumes was already checked for legal move
       captures=[]
       piece=self.pieces[pieceno]
       piecetype = piece[2]
       for apiece in self.pieces:
          if piecetype*apiece[2] < 0:
             d1 = apiece[0]-x2 
             d2 = apiece[1]-y2
             if (abs(d1)==1 and d2==0) or (abs(d2)==1 and d1==0): 
                 for bpiece in self.pieces:
                    if piecetype*bpiece[2] > 0 and not(piece[0]==bpiece[0] and piece[1]==bpiece[1]):
                       e1 = bpiece[0]-apiece[0]
                       e2 = bpiece[1]-apiece[1]
                       if d1==e1 and d2==e2:
                          captures.extend([apiece])
       return captures

    # returns code for invalid mode (<0) or number of pieces captured
    def _moveByPieceNo(self,pieceno,x2,y2):
      
      legal = self._isLegalMove(pieceno,x2,y2)
      if legal != 0: return legal

      self.time = self.time + 1

      piece=self.pieces[pieceno]
      piece[0]=x2
      piece[1]=y2
      caps = self._getCaptures(pieceno,x2,y2)
      #print("Captures = ",caps)
      for c in caps:
          c[0]=-99

      self.done = self._getWinLose()
      
      return len(caps)
        


    def _getWinLose(self):
       if self.time > 50: return -1
       for apiece in self.pieces:
           if apiece[2]==2 and apiece[0] > -1:
               for item in self.board:
                 if item[0]==apiece[0] and item[1]==apiece[1] and item[2]==1:
                     return 1 #white won
               return 0 # no winner
       return -1  #white lost
   
    def _getPieceNo(self,x,y):
       for pieceno in range(len(self.pieces)):
           piece=self.pieces[pieceno]
           if piece[0]==x and piece[1]==y: return pieceno
       return -1    
   
    def _getValidMoves(self,player):
       moves=[]
       for pieceno in range(len(self.pieces)):
           piece=self.pieces[pieceno]
           if piece[2]*player > 0:
              #print("checking pieceno ",pieceno,piece)
              for x in range(0,self.width):
                  if self._isLegalMove(pieceno,x,piece[1])>=0:moves.extend([[piece[0],piece[1],x,piece[1]]])
              for y in range(0,self.height):
                  if self._isLegalMove(pieceno,piece[0],y)>=0:moves.extend([[piece[0],piece[1],piece[0],y]])
       #print("moves ",moves)
       return moves


