# Alpha Zero General ---- GO Implementation

While it takes some modification of folders allocation to start the training process,
the pit.py should be ready to use.
Code need further cleaning up, and a lot of explaination of optimazation is omissed here. (coming soon)
### Contributions -- add-ons
* Game logic files for Go
* Pre-trained models for 5*5 Go and 9*9 Go. (9*9 is in process of training)
* An asynchronous version of the code- parallel processes for self-play, neural net training and model comparison. 

### Modifications
* Arena logic
* MCTS logic

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

