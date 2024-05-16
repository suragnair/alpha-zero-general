# Santorini Rules and Structures

[Santorini](https://en.wikipedia.org/wiki/Santorini_(game)) is a two-player competative board game that takes place on a 5x5 board. Each player has two character pieces, (an infinite number of) cubic building pieces as well as dome pieces. 
Players begin the game by alternatingly placing their character pieces on an empty board space. At this point the board is empty except for the 4 player pieces (2 for each player). 

### Gameplay 
__The gameplay__ involves a player moving one of their pieces one space in any direction (including diagonal) and *then* building a block on a space adjacent to the location they end up at. 

When a block is placed on top of a space, the height of the space is increased by one. 

Multiple blocks can be stacked on top of one another, so if the board itself has a height of zero, a space with two blocks on it has a height of 2. 

A piece can move down any amount of height, but cannot move up by more than 1 in a turn. So a piece cannot move from the board onto the space with two blocks on it, without first moving onto a space next to that one with one block on it, and then waiting for the opponent to perform their turn.

If there are already three blocks on a space, instead of another block, a dome can be built on that space. Pieces cannot stand on, nor build on spaces which have domes. See [this photo](https://www.1843magazine.com/sites/default/files/styles/il_manual_crop_16_9/public/Santorini-header-V3.jpg) for an image of the domed roofs in the actual town of Santorini, Italy.  

### Winning
A player **wins** when they move a piece onto a space of height 3 (having three blocks on it, and no dome), or when their opponent is unable to make a legal move.

### Legal Move requirements
The following must be true about a player's move for it to be legal:
* The player cannot move onto, or build on, any space with another player's piece in it. 
* The player cannot move onto, or build on, any space with a dome on it, regardless of who placed the dome. 
* The block must be built in a space adjacent to where the piece that was moved ended up. The player cannot move one piece and build next to another piece. The player cannot move one piece and build on a space that was adjacent to where the piece started, but not adjacent to where the piece ended.
* The player must move and then must build. If the player either cannot move a piece, or cannot build after moving, that player loses. It is not possible for a player to move a piece onto a winner square, and then be unable to build, because the player will always have the option to build on space they moved from.
* The player cannot move a piece onto a square that is more than 1 block higher than the square the piece moved from.
* The player cannot pass their turn. 
* The player cannot place a build a block on a space that already has three blocks on it. The player can only build a dome on that space, or build elsewhere. 

## Implementation specifics:
Instead of treating the board as 5x5 grid, the board is treated as an array of shape (2,5,5). The first 5x5 subarray contains the locations of each player's characters. The second contains the heights of each space on the board. A player's characters are referred to internally as 1 and 2 (with -1 and -2 being the opponents), and visually represented as O and U for player 1, and X and Y for player 2 (player -1). The visualizations can be easily changed in SantorniGame.py.

Arbitrary nxn sized boards are supported. The default is 5x5. Calling SantoriniGame(n) (see Main.py) will create games with an nxn shaped board. Internally this will be (2,n,n).

By default, Each player's pieces are positioned around the center of the board at the start of the game, rather than having them place their own pieces. This is done to simplify the game for the neural net, since the game complexity is rather high, and the net struggles considerably to learn how to play (well). As an alternative, players can have their pieces placed randomly by setting true_random_placement=True when calling SantoriniGame: SantoriniGame(board_size, true_random_placement=True). At present, no way for players to place their pieces at the start of the game has been implemented.  


    """
    A Santorini Board of default shape: (2,5,5)
    
    Board logic:
        board shape: (2, self.n, self.n)
           [[[ 0,  0,  0,  0,  0],
             [ 0,  0,  1,  0,  0],
             [ 0, -1,  0, -2,  0],
             [ 0,  0,  2,  0,  0],
             [ 0,  0,  0,  0,  0]]
            
            [[ 0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0]]]
    
    BOARD[0]: character locations
    board[0] shape = (self.n,self.n) here this is (5,5)
    Cannonical version of this board shows 
    a player with their pieces as +1, +2 and opponents as -1, -2
        
    LOCATIONS: 
        Locations are given as (x,y) (ROW, COLUMN) coordinates,
        e.g. the 1 in board[0] is at location (1,2), and the 2 at (3,2), whereas
        the -1 is at location (2,1), and the -2 at (2,3)
    
    ACTIONS: 
        Actions are stored as list of tuples of the form:
            action = [piece_location, move_location, build_location]
                     [(x1,y1),        (x2, y2),      (x3, y3)]
    
    BOARD 1: Location heights
        board shape: (self.n,self.n)
        Cannonical board shows player height of each board space.
        The height of each space ranges from 0,...,4 (this is independent of self.n)
 
