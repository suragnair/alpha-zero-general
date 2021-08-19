# Alpha Zero General ---- GO Implementation

While it takes some modification of folders allocation to start the training process,
the pit.py should be ready to use.
Code need further cleaning up, and a lot of explaination of optimazation is omissed here. (coming soon)
### Contributions -- add-ons
* Game logic files for Go
* Trained models for 5*5 Go and 9*9 Go. (9*9 is in process of training)
* An asynchronous version of the code- parallel processes for self-play, neural net training and model comparison. 

### Modifications
1. Modified arena. Instead of pitting only with previous weight, it creates x new new weight and hold a tournament b/w x new and current best. It will accept the weight with highest wins.

2. Coach. I split it into three files for self-play, network training + pitting, and the last one that updating the best. The pickle library is used for self-play coach so that it supports multi-processing to generate examples a lot faster.


3. Monte-Carlo search. 

  3.1.
  For each moves, instead of limiting the max number of simulations to stop searching, we limit the max depth of the tree. 

  When the tree is increasing in depth, its impact on the Q value keep decreasing, so its passibility to change the selected move of first level will also decrease. This modifications aim to diminish useless searches and increase amounts of searches for those faltering between two or three moves. When generating game examples, I think it make more sense to stop the search when it hit certain depth than make rigid number of search regardless it will make any difference or not.

  3.2
  Tree pruning by limiting the number of children. 

  Original code assigned a nearly zero staring value for each uninitialized moves, and these nearly zero value will be compared with moves which already have Q values. As a result, when Q value approximate to 1, only one of two moves of highest policy value will be considered, as for 
  Q approximate to -1, all valid moves will be considered since 0 is larger than -1. That is, it has great chance to omit correct move when Q is large, it will waste computing power on useless searches when Q is small. 

  To prune the tree, we only use moves of top 5 policy, masking out all other moves. All of 5 moves will be initialized. In this way, we diminish Q value’s impact on decision making, provide a way to get more control in MTCL search.

4. Noise
In training, noise are generated using a way similar to the kata-go paper: fluctuating the limit maximum depth of each moves.


### Train a 9*9 Game model (in process)

To start training a model for Go, run three python script concurentlly:

To generate game examples. (can run multiple times of this script in a concurent manner)
```bash
python runSelfplay.py
```

To generate network weight using examples.
```bash
python runTraining4tars.py
```
For training 9*9 model, run this line 4 times and do not forget to modify the name of network's tar file each time in the Coach.

When 4 network weight is ready, update the best network weight with the best of these 4.
```bash
python runTrainingBest.py
```
### Play with model
```bash
python pit.py
```
All models are in temp folder

### The temp folder
We trained severfor 5x5 Go (parameters and iterations see report).  There is also one 9x9 Go model at the date of Aug 16. At this point, the 9x9 model cannot win over advanced admature Go player, but it is keep making progress with 4 1080Ti GPU. You can play a game against it using ```pit.py```. 



### Contributors and Credits
* [Jiageng Zheng](https://github.com/jiz322) Contribute Go implementation, Parallel optimazation, MCLT optimazation, Arena optimazation, various modifications(more details coming soon), and the training for 5*5 and 9*9 Go models.
* [Ted Huang](https://github.com/teddy57320) Credits for original Go [logic files](https://github.com/teddy57320/go).

