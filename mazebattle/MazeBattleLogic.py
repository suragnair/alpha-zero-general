import random

"""
Board class for the game of MazeBattle.
Default board size is 10x10.
0 - empty
1 - player1
2 - player2
3 - player1 starting point
4 - player2 starting point
5 - wall


directions

812
703
654

Actions will be used with a direction. For example

(2, 3) - Will build or repair the wall on the right

valid moves vector will be booleans:

[stay, move1, .. , move8, build1, .., build8, break1, .., break8, shoot1, .., shoot8]

therefore vector size --> 33

"""


class Point:

    def __init__(self, x, y):
        '''Defines x and y variables'''
        self.x = x
        self.y = y

    def add_point(self, point: "Point"):
        return Point(self.x + point.x, self.y + point.y)

    @staticmethod
    def random(minValue, maxValue):
        x = random.randint(minValue, maxValue)
        y = random.randint(minValue, maxValue)
        return Point(x, y)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False


# from bkcharts.attributes import color
class Board:
    TAG_PLAYER1 = 1
    TAG_PLAYER2 = -1
    TAG_EMPTY = 2
    TAG_WALL_0_HIT = 3
    TAG_WALL_1_HIT = 4
    TAG_PLAYER1_STARTING_POINT = 5
    TAG_PLAYER2_STARTING_POINT = 6
    TAG_BULLET_1 = 71
    TAG_BULLET_2 = 72
    TAG_BULLET_3 = 73
    TAG_BULLET_4 = 74
    TAG_BULLET_5 = 75
    TAG_BULLET_6 = 76
    TAG_BULLET_7 = 77
    TAG_BULLET_8 = 78

    ACTION_STAY = 0
    ACTION_MOVE = 1
    ACTION_BUILD_WALL = 2
    ACTION_BREAK_WALL = 3
    ACTION_SHOOT = 4

    def __init__(self, n=20, wallPercent=.35):
        """Set up initial board configuration."""

        self.n = n
        # Create the empty board array.
        self.board = [None] * self.n
        for i in range(self.n):
            self.board[i] = [self.TAG_EMPTY] * self.n
        self.p1StartPoint = Point.random(0, n)
        self.p2StartPoint = Point.random(0, n)
        self.walls = round(wallPercent * (n ** 2))
        currentWalls = 0
        while currentWalls < self.walls:
            wallPoint = Point.random(0, n)
            while self.board[wallPoint.x][wallPoint.y] != self.TAG_EMPTY:
                wallPoint = Point.random(0, n)
            self.board[wallPoint.x][wallPoint.y] = self.TAG_WALL_1_HIT if bool(
                random.getrandbits(1)) else self.TAG_WALL_0_HIT
            currentWalls += 1
        while self.p1StartPoint == self.p2StartPoint:
            self.p2StartPoint = Point.random(0, n)
        self.board[self.p1StartPoint.x][self.p1StartPoint.y] = self.TAG_PLAYER1_STARTING_POINT
        self.board[self.p2StartPoint.x][self.p2StartPoint.y] = self.TAG_PLAYER2_STARTING_POINT

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.board[index]

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for player1, -1 for player2)
        @param color not used and came from previous version.        
        """
        moves = [1]  # stores the legal moves.
        # moves.add((self.ACTION_STAY, 0))  # Add no move action

        # Get surrounding coordinates
        playerPos = self.find_player(color)
        surroundings = self.get_surrounding_points(playerPos)
        toTags = self.positions_to_tags(surroundings)

        # MOVE
        movePositions = filter(lambda tag: self.can_move(color, tag), toTags)
        moveMoves = zip([self.ACTION_MOVE] * len(movePositions), movePositions)

        # BUILD
        buildPositions = filter(lambda tag: self.can_build(color, tag), toTags)
        buildMoves = zip([self.ACTION_BUILD_WALL] * len(movePositions), buildPositions)

        # BREAK
        breakPositions = filter(lambda tag: self.can_break(color, tag), toTags)
        breakMoves = zip([self.ACTION_BREAK_WALL] * len(movePositions), breakPositions)

        # SHOOT
        shootPositions = filter(lambda tag: self.can_shoot(color, tag), toTags)
        shootMoves = zip([self.ACTION_SHOOT] * len(movePositions), shootPositions)

        moves.update(moveMoves)
        moves.update(buildMoves)
        moves.update(breakMoves)
        moves.update(shootMoves)
        return list(moves)

    def get_surrounding_points(self, point: Point):
        '''
        directions

        812
        703
        654

        8 (-1, -1)
        1 (-1, 0)
        2 ( -1, 1)
        3 (0, 1)
        7 (0, -1)
        6 (1, -1)
        5 (1, 0)
        4 (1, 1)

        '''
        points = [point.add_point(Point(-1, -1)), point.add_point(Point(-1, 0)), point.add_point(Point(-1, 1)),
                  point.add_point(Point(0, 1)), point.add_point(Point(0, -1)), point.add_point(Point(1, -1)),
                  point.add_point(Point(1, 0)), point.add_point(Point(1, 1))]
        return filter(self.valid_position, points)

    def positions_to_tags(self, positions):
        toTags = []
        for position in positions:
            toTags.append(self[position.x][position.y])
        return toTags

    def find_player(self, color):
        playerPos = None
        for x in range(self.n):
            for y in range(self.n):
                if self[x][y] == color:
                    playerPos = Point(x, y)
                elif playerPos is None and ((self[x][y] == self.TAG_PLAYER1_STARTING_POINT and color == 1) or (
                        self[x][y] == self.TAG_PLAYER2_STARTING_POINT and color == -1)):
                    playerPos = Point(x, y)
        return playerPos

    def is_different_player_start(self, color, toTag):
        return (toTag == self.TAG_PLAYER2_STARTING_POINT and color == 1) or (
                toTag == self.TAG_PLAYER1_STARTING_POINT and color == -1)

    def is_different_player(self, color, toTag):
        return (toTag == self.TAG_PLAYER2 and color == 1) or (
                toTag == self.TAG_PLAYER1 and color == -1)

    def can_move(self, color, toTag):
        return toTag == self.TAG_EMPTY or self.is_different_player_start(color, toTag) or toTag >= self.TAG_BULLET_1

    def can_shoot(self, color, toTag):
        return toTag == self.TAG_EMPTY or self.is_different_player(color, toTag)

    def can_build(self, color, toTag):
        return toTag == self.TAG_EMPTY or toTag == self.TAG_WALL_1_HIT

    def can_break(self, color, toTag):
        return toTag == self.TAG_WALL_1_HIT or toTag == self.TAG_WALL_0_HIT

    def valid_position(self, point: Point):
        return (0 <= point.x <= self.n) and (0 <= point.y <= self.n)

    def is_win(self, color):
        """Check whether the given player has reached a point next to other player starting point
        @param color (1=white,-1=black)
        """
        for x in range(self.n):
            for y in range(self.n):
                if self[x][y] == color:
                    surroundingPoints = self.get_surrounding_points(Point(x, y))
                    toTags = self.positions_to_tags(surroundingPoints)
                    if (color == 1):
                        return len(filter(lambda toTag: toTag == self.TAG_PLAYER2_STARTING_POINT, toTags)) > 0
                    else:
                        return len(filter(lambda toTag: toTag == self.TAG_PLAYER1_STARTING_POINT, toTags)) > 0

    def updateBullets(self):
        '''
                directions

                812
                703
                654

                8 (-1, -1)
                1 (-1, 0)
                2 ( -1, 1)
                3 (0, 1)
                7 (0, -1)
                6 (1, -1)
                5 (1, 0)
                4 (1, 1)

        '''
        for x in range(self.n):
            for y in range(self.n):
                currentBulletPoint = None
                nextBulletPoint = None
                if self[x][y] == self.TAG_BULLET_1:
                    currentBulletPoint = Point(x, y)
                    nextBulletPoint = currentBulletPoint.add_point(Point(-1, 0))
                elif self[x][y] == self.TAG_BULLET_2:
                    currentBulletPoint = Point(x, y)
                    nextBulletPoint = currentBulletPoint.add_point(Point(-1, 1))
                elif self[x][y] == self.TAG_BULLET_3:
                    currentBulletPoint = Point(x, y)
                    nextBulletPoint = currentBulletPoint.add_point(Point(0, 1))
                elif self[x][y] == self.TAG_BULLET_4:
                    currentBulletPoint = Point(x, y)
                    nextBulletPoint = currentBulletPoint.add_point(Point(1, 1))
                elif self[x][y] == self.TAG_BULLET_5:
                    currentBulletPoint = Point(x, y)
                    nextBulletPoint = currentBulletPoint.add_point(Point(1, 0))
                elif self[x][y] == self.TAG_BULLET_6:
                    currentBulletPoint = Point(x, y)
                    nextBulletPoint = currentBulletPoint.add_point(Point(1, -1))
                elif self[x][y] == self.TAG_BULLET_7:
                    currentBulletPoint = Point(x, y)
                    nextBulletPoint = currentBulletPoint.add_point(Point(0, -1))
                elif self[x][y] == self.TAG_BULLET_8:
                    currentBulletPoint = Point(x, y)
                    nextBulletPoint = currentBulletPoint.add_point(Point(-1, -1))

                if (currentBulletPoint is not None) and self.valid_position(nextBulletPoint):
                    toTag = self[nextBulletPoint.x][nextBulletPoint.y]
                    if toTag == self.TAG_PLAYER2 or toTag == self.TAG_PLAYER1:
                        self[nextBulletPoint.x][nextBulletPoint.y] = self.TAG_EMPTY  # Player killed
                    else:
                        self[nextBulletPoint.x][nextBulletPoint.y] = self[currentBulletPoint.x][
                            currentBulletPoint.y]  # Continue the bullet
                self[currentBulletPoint.x][currentBulletPoint.y] = self.TAG_EMPTY  # Remove bullet from current square

    def exchange_board(self, color):
        copied = self[:][:]
        if color == 1:
            return copied
        for x in range(self.n):
            for y in range(self.n):
                if copied[x][y] == 1:
                    copied[x][y] = -1
                elif copied[x][y] == -1:
                    copied[x][y] = 1
                elif copied[x][y] == copied.TAG_PLAYER2_STARTING_POINT:
                    copied[x][y] = copied.TAG_PLAYER1_STARTING_POINT
                elif copied[x][y] == copied.TAG_PLAYER1_STARTING_POINT:
                    copied[x][y] = copied.TAG_PLAYER2_STARTING_POINT

    def execute_move(self, move, color):
        """Perform the given move on the board; 
        color gives the color pf the piece to play (1=white,-1=black)
        """
        '''
        directions

        812
        703
        654

        8 (-1, -1)
        1 (-1, 0)
        2 ( -1, 1)
        3 (0, 1)
        7 (0, -1)
        6 (1, -1)
        5 (1, 0)
        4 (1, 1)

        '''
        playerPos = self.find_player(color)
        (action, direction) = move
        vDif = None
        shootDir = None
        if direction == 1:
            vDif = Point(-1, 0)
            shootDir = self.TAG_BULLET_1
        elif direction == 2:
            vDif = Point(-1, 1)
            shootDir = self.TAG_BULLET_2
        elif direction == 3:
            vDif = Point(0, 1)
            shootDir = self.TAG_BULLET_3
        elif direction == 4:
            vDif = Point(1, 1)
            shootDir = self.TAG_BULLET_4
        elif direction == 5:
            vDif = Point(1, 0)
            shootDir = self.TAG_BULLET_5
        elif direction == 6:
            vDif = Point(1, -1)
            shootDir = self.TAG_BULLET_6
        elif direction == 7:
            vDif = Point(0, -1)
            shootDir = self.TAG_BULLET_7
        elif direction == 8:
            vDif = Point(-1, -1)
            shootDir = self.TAG_BULLET_8

        affectedPoint = playerPos.add_point(vDif)
        if action == self.ACTION_SHOOT:
            self[affectedPoint.x][affectedPoint.y] = shootDir
        elif action == self.ACTION_BREAK_WALL:
            self[affectedPoint.x][affectedPoint.y] -= 1
        elif action == self.ACTION_BUILD_WALL:
            self[affectedPoint.x][affectedPoint.y] = self.TAG_WALL_1_HIT
        elif action == self.ACTION_MOVE:
            if self[playerPos.x][playerPos.y] == color:
                self[playerPos.x][playerPos.y] = self.TAG_EMPTY
            self[affectedPoint.x][affectedPoint.y] = color
        self.updateBullets()
