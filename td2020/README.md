# TD2020 - RTS Game
*Jernej Habjan 2018*

This is implementation of RTS game in Alpha Zero General wrapper created by Surag Nair in [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general).
Game visualisation is also presented in PyGame and Unreal Engine 4.
## Quick info
| Name | Official name |
| --- | --- |
| Actor | Any board figure including neutral figures like Gold |
## Defining game rules:
- Grid of squared size (6x6, 8x8...),
- 1 unit per field
### Actors
#### Actor types
- **Gold** - Resource source unit
- **Worker** - Unit that can gather, return resources and build all types of buildings
- **Barracks** - Building that can build units of type Rifle Unit
- **Rifle Unit** - Units that can attack enemy units
- **Town Hall** - Building that produces Worker units
#### Actor costs
- **Gold** - 0
- **Worker** - 1
- **Barracks** - 4
- **Rifle Unit**  - 2
- **Town Hall** - 7
#### Actor health
- **Gold** - 1
- **Worker** - 1
- **Barracks** - 2
- **Rifle Unit** - 2
- **Town Hall** - 3
#### Actor actions
- **Gold** - [],
- **Worker** - [idle, up, down, left, right, barracks, npc],
- **Barracks** - [idle, rifle_infantry, npc],
- **Rifle Unit** - [idle, up, down, left, right, attack, npc],
- **Town Hall** -[idle, npc],

### Actions
- **idle**: Pass turn
- **up**: Move up 1 field if its empty
- **down**: Move down 1 field if its empty
- **right**: Move right 1 field if its empty
- **left**: Move left 1 field if its empty
- **mine_resources**: Mine gold resources if standing nearby
- **return_resources**: Return gold resources if carrying gold to Town Hall
- **attack**: Attack nearby unit
- **npc**: Build worker character
- **rifle_infantry**: Build attack unit
- **barracks**: Build building that produces attacking units
- **town_hall**: Build building that produces workers and is resource deposit
### Tile encoding
Each actor is encoded using following 6 properties:
- **Player Name**: [-1,0,1] (player -1, empty field, player 1)
- **Actor Type**: [1-5] Numerically encoded Actor Type of written above
- **Health**: [1-31] Current actor health - when actor has 0 health, it gets destroyed
- **Carry**: [0,1] If Worker unit is carrying resources or not - it gets set when worker uses mine_resources near resource source and gets removed when worker uses return_resources near resources drain actor.
- **Money**: [0-*] Current amount of money that this player has at current time (When money updates, it updates on every players actor)
- **Time**: [*-0 or 0-8191] Countdown time that gets updated on every board tile when move gets executed. Also timer that increases, and special milestones, health is decreased for all units by formula.
#### One Hot Tile Encoding:
Now each actor is encoded with binary vector of length 27.
- **Player Name**: [2] - 00(neutral), 01(1) or 10(-1),
- **Actor Type**: [4] - 4 bit,
- **Health**: [5] - 5 bit. This much because of life decrease every milestone in getNextState
- **Carry**: [1] - 1 bit,
- **Money**: [5] - 5 bits (32 aka 4 town halls or 32 workers) [every unit has the same for player]
- **Time**: [13] - 2^13 8192 (this should be enough for total game)

### Board presentation
Board is presented as 3D integer array of dimensions width, height, 6 (which represents number of properties in Tile encoding)
### Action checking sequence
Following actions are checked and executed in some order:
- mine_resources
- return_resources
- attack
- npc
- rifle_infantry
- barracks
- town_hall
```python
coords = [(x - 1, y + 1),
          (x, y + 1),
          (x + 1, y + 1),
          (x - 1, y),
          (x + 1, y),
          (x - 1, y - 1),
          (x, y - 1),
          (x + 1, y - 1)]
for n_x, n_y in coords:
    # check action condition or execute action
```
- When first tile is free in these coordinates, building or unit is spawned there.
- When first enemy in this tile sequence is chosen, it is attacked.

This results in building units and buildings towards lower-left corner, as units progress towards x-1, y+1 as this tile is free, expanding towards upper-right corner only when all other options are used.
- Fix for this would be to make actions for each of these coordinates for each of actions that use them, resulting in much higher action space.
### Game end
Instance of game is finished in following conditions:
- One of players does not have any available moves left (board is populated or one player is surrounded),
- One players' actors get destroyed,
- When remaining time reaches 0
### Initial board configuration
Board gets setup with a town hall for each player in the middle, with 2 patches of resource source actor - Minerals.
Each mineral patch is assigned its player (-1,1), because value 0 game recognises as unpopulated area.
Each player then starts with some amount of gold (1 or 20 or 100...).
## Learning
- Learning of this game is complicated, because of end game conditions.
Learning wrapper expects game to finish using MCTS simulations, but python might run into max recursion depth exceeded exception,
because player is repeating same move multiple times.
- This can be solved using timeouts, as where simulation gets stopped when we run out of remaining moves, but can lead to inaccurate MCTS tree,
because nodes do not get properly evaluated during backpropagation.

Proper end condition must be found or change of source is needed in order to exclude timeouts, because they are not returning best resuts.
### Model composition

### Possible learning idea
Idea is to incrementally learn model by changing end game condition.
First start learning model on simple end game condition like producing workers and when model is successfully creating workers, add another condition on top of that already learnt model.
- Possible problem might occur because of model size
## Players
### Greedy Player
Greedy player calculates score by summing health of all his actors.
### Human Player
#### PyGame

In PyGame, user can interact with his figures via keyboard or mouse.

User must first select figure with *left mouse click* and then choose one of actions written in canvas.
Other figure can be selected by clicking another figure or deselecting it with *right mouse button* on empty tile.
- Moving: User can move workers and infantry by 1 square in all 4 directions if they are empty by clicking on one of 4 corresponding tiles.
- Attacking: With selected infantry unit, user can attack enemy units that are in range.
- Gathering and returning resources: With worker selected, user can mine resources by clicking *right mouse button* on Gold actor if in range.
This goes the same when returning resources, but worker must be nearby Town Hall actor.
- Building: For building units and buildings, user must use one of keyboard shortcuts written on canvas.
- Idle: User can press space to idle with selected actor.
#### Console
Type one of listed commands seperated by space and press Enter.


## PyGame visual presentation
![Example of PyGame](https://i.imgur.com/b4olJTx.png)
## Unreal Engine visual presentation
- Used repository [JernejHabjan/TrumpDefense2020](https://github.com/JernejHabjan/TrumpDefense2020)
- Python communication plugin:  [getnamo/tensorflow-ue4](https://github.com/getnamo/tensorflow-ue4)

Unreal Engine handles visual presentation of playing custom RTS game agains computer opponent, which is querying actions via plugin to python prebuilt model.

### Usage of prebuilt model
Game sends game state info from engine to Python TensorFlow model, which then in return after number of MCTS simulations, preset by pre-learnt model returns best action for that state.
