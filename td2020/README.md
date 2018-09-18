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
- **Barracks** - 3
- **Rifle Unit** - 2
- **Town Hall** - 4
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
- **Health**: [1-4] Current actor health - when actor has 0 health, it gets destroyed
- **Carry**: [0,1] If Worker unit is carrying resources or not - it gets set when worker uses mine_resources near resource source and gets removed when worker uses return_resources near resources drain actor.
- **Money**: [0-*] Current amount of money that this player has at current time (When money updates, it updates on every players actor)
- **Remaining time**: [*-0] Countdown time that gets updated on every board tile when move gets executed
### Board presentation
Board is presented as 3D integer array of dimensions width, height, 6 (which represents number of properties in Tile encoding)
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

## PyGame visual presentation
Quick visual presentation of Python game in PyGame
## Unreal Engine visual presentation
- Used repository [JernejHabjan/TrumpDefense2020](https://github.com/JernejHabjan/TrumpDefense2020)
- Python communication plugin:  [getnamo/tensorflow-ue4](https://github.com/getnamo/tensorflow-ue4)

Unreal Engine handles visual presentation of playing custom RTS game agains computer opponent, which is querying actions via plugin to python prebuilt model.

### Usage of prebuilt model
Game sends game state info from engine to Python TensorFlow model, which then in return after number of MCTS simulations, preset by pre-learnt model returns best action for that state.
