# RTS Game
*Jernej Habjan 2019*

This is a [diploma thesis project](https://github.com/JernejHabjan/Diploma-Thesis), which is an implementation of RTS game in Alpha Zero General wrapper created by Surag Nair in [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general).
Game visualisation is also presented in PyGame and Unreal Engine 4.

## Game description
### Game state
- Pieces aren't overwriting other piece with attack move, as is done in chess, but are damaging other piece by some amount.
- 1 piece per tile (with grid of size 8x8 or 6x6)
- Player wins if he defeats all enemy players pieces
- Actions are 1 tick operations, unlike for example StarCraft, where training of units or constructing buildings takes multiple game states/ game ticks.

Lets define those pieces/ actors:
- Gold patch: Source of gold that workers can gather
- Worker: Unit that can construct buildings and gather minerals
- Barracks: Building that can train infantry
- Infantry: Unit that can attack enemy units
- Town Hall: Building that creates workers and is resource deposit for gold minerals.

And their actions:
- Move (4 directions): Workers and Infantry can move in one of 4 directions if that move is valid
- Gather gold (8 directions): Workers can gather gold if they are standing near that patch and aren't carrying gold
- Return gold (8 directions): Workers can return gold if they are standing near Town Hall and are carrying gold
- Attack (4 directions): Infantry can attack enemy piece in one of 4 directions. Health is then decreased for attack damage. If new health is below 0, actor is then removed. If that is last actor in game, Attacking player wins.
- Create Worker (4 directions): Town hall can create new Worker on one of 4 tiles if they're empty.
- Create Infantry (4 directions): Barracks can create new Infantry on one of 4 tiles if they're empty.
- Create Barracks (4 directions): Worker can create new Barracks on one of 4 tiles if they're empty.
- Create Town Hall (4 directions): Worker can create new Town Hall on one of 4 tiles if they're empty.
- Heal (4 directions): Actor can heal another friendly actor by some amount if they have that amount of gold and friendly actors' health isn't full.

Some moves like gather gold, return gold, attack, create worker, create barracks, create town hall, heal, are following simple for loop sequence:
```
coordinates = [(x - 1, y + 1),
    (x, y + 1),
    (x + 1, y + 1),
    (x - 1, y),
    (x + 1, y),
    (x - 1, y - 1),
    (x, y - 1),
    (x + 1, y - 1)]
for new_x, new_y in coordinates:
    # check if action is valid
    if (condition == ok):
        return new_x, new_y
```
This means that actions will first try to execute in (x-1, y+1) tile. If that action isn't valid, it will continue through for loop until the end - see photo below.
![Photo of action execution for loop](http://prntscr.com/mh96y0)

Pieces/actors have special properties:

|Actor type | Actions | Health points | Building cost|
|-----------|---------|---------------|--------------|
| Gold patch| /       | 10            | 0            |
| Worker    | move, build barracks, build town hall, gather and return gold minerals, heal       | 10            | 1            |
| Barracks  | create infantry, heal       | 10            | 4            |
| Infantry  | move, attack, heal       | 20            | 2            |
| Town Hall | create worker, heal       | 30            | 7            |

### Encoders
Each tile is encoded by N-dimensional vector, even if that tile is empty:

*Numeric Encoder*: - 6 encoding numbers (player number, actor type, health_points, is_carrying_gold, player_gold, time_playing)

*OneHot Encoder*: - expands those 6 numbers, so they're encoded binary (resulting in faster learning): player number 2 bits, actor type 3 bits, health_points 5 bits, is_carrying_gold 1 bit, player_gold 5 bits, time_playing 11 bits
### Game end
End game is determined if player does not have any actors left or cannot execute any action or time runs out.
Example with damage function was made, but it was not returning any good results (see code source for more info).
Now timeout is used, which ends game at N ticks (200 in our example) and evaluates winner with some elo function:
- sum of players money
- sum of players units' health
- sum of both

### Visualizations
Game can be played using CMD, PyGame or [Unreal Engine 4](https://github.com/JernejHabjan/TrumpDefense2020).
![Pygame](http://prntscr.com/mh9c5f)
![Unreal Engine 4](http://prntscr.com/mh9ci8)
### Results
For learning results, see releases [Here](https://github.com/JernejHabjan/alpha-zero-general/releases/tag/1.0.0) and [Here](https://github.com/JernejHabjan/alpha-zero-general/releases/tag/1.0.1).


## Requirements
- Recommended Python 3.6 (3.7 is not supported by TensorFlow by the time of writing(December 2018))
- Required TensorFlow (recommended 1.9)
- Optional Pygame (board outputs can be displayed also in console if Pygame is not used)
- Module can be connected via rts/visualization/rts_ue4.py to [UE4](https://github.com/JernejHabjan/TrumpDefense2020) using Tensorflow-ue4 v0.8.0 for UE4.19 https://github.com/getnamo/tensorflow-ue4/releases/tag/0.8.0
## Files
Main files to start learning and pitting:
- rts/learn.py
- rts/pit.py
- rts/src/config_class.py

# Install instructions
download git cmd
> https://git-scm.com/downloads
open git bash cmd
```
git clone https://github.com/JernejHabjan/alpha-zero-general.git
```
run install script in 
> alpha-zero-general/rts/install.sh

## Tensorflow-Gpu installation (Optional):
```pip install 'tensorflow-gpu==1.8'```
### TensorFlow and CUDA
Install cuda:
- Install cuda files:
- cuda_9.0.176_win10
- cuda_9.0.176.1_windows
- cuda_9.0.176.2_windows
- cuda_9.0.176.3_windows
- Extract this folder and add it to Cuda path to corresponding folders:
    - cudnn-9.0-windows10-x64-v7.1

vertify cuda installation:
- ```nvcc --version```

- make sure its added to env variables:
```
- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin
- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp
```

## Graphviz and pydot(Optional):
```
pip install graphviz
pip install pydot
```
Download Graphviz executable from [Here](https://graphviz.gitlab.io/_pages/Download/Download_windows.html)

Add path to environmental variables and !important! restart Pycharm
>C:\Program Files (x86)\Graphviz2.38\bin


# Running
## Setup pit and learn config:
- alpha_zero_general/rts/config.py -> CONFIG
## For pit:
- download release:
>https://github.com/JernejHabjan/alpha-zero-general/releases
- extract
- place extracted files in folder to
>alpha_zero_general/temp/
- and overwrite config file in
>alpha-zero-general/rts/src/config_class.py
- navigate to 
>C:\Users\USER\alpha-zero-general\rts
- run ```python pit.py```
## for learn:
- navigate to 
>C:\Users\USER\alpha-zero-general\rts
- run > ```python learn.py```
## Ue4:
    download latest release https://github.com/JernejHabjan/TrumpDefense2020/releases
    run
