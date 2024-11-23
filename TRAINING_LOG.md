### 6*6_numeps_100_num_mcts_sims_25_2_input_channel - A
- trained for 2 days. around 40 iterations
- self-competation, the latest 9 iterations, always draw. (40 games in total, 38+ draws)

However, it can stll not win the author:

Turn  26 Player  -1
   0 1 2 3 4 5 
-----------------------
0 |- O - - X - |
1 |- - O O X - |
2 |O X X X O O |
3 |O X O O O - |
4 |X O O - O X |
5 |- X X - X X |
-----------------------
[ 0 0] [ 0 2] [ 0 3] [ 0 5] [ 1 0] [ 1 1] [ 1 5] [ 3 5] [ 4 3] [ 5 0] [ 5 3]  


The AI (0) failed to detect they shall put the stone on [3 5].


Compete the 2 channel model with the 1 channel model, both uses 50 simulations for each step. The outcome is 

channel 1 model wins / loss /draw: 6 / 5 / 9 

performs nearly the same.


Analysis:

Loss_pi=8.34e-01, Loss_v=2.89e-02


### 6*6_numeps_100_num_mcts_sims_100_2_input_channel - B
- trained for one night. 13 iterations
- around 30 / 40 draw rate. Not saturated yet.

When comparing B with A in 10 games, A's win / loss / draw rate is 1 / 4 / 15

Analysis:
 - We did not see a significant improvement in B even after increasing the simulation depths in MCTS. 

Next steps
 - Check if there is any bugs in the code.
 - continue to train B until it saturates (40 draws in each iteration)



 -----------------------
[ 0 0] [ 0 1] [ 0 2] [ 0 3] [ 0 5] [ 1 0] [ 1 1] [ 1 2] [ 1 5] [ 2 0] [ 2 5] [ 3 0] [ 3 5] [ 4 0] [ 4 3] [ 4 4] [ 4 5] [ 5 0] [ 5 1] [ 5 2] [ 5 3] [ 5 4] [ 5 5] 4 5
Turn  15 Player  1
   0 1 2 3 4 5 
-----------------------
0 |- - - - X - |
1 |- - - O O - |
2 |- X X X O - |
3 |- X O O O - |
4 |- O X - - - |
5 |- - - - X - |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |0.00 0.00 0.00 0.00 0.00 0.00 |
1 |0.00 0.00 0.00 0.00 0.00 0.00 |
2 |0.00 0.00 0.00 0.00 0.00 0.00 |
3 |0.00 0.00 0.00 0.00 0.00 1.00 | # HERE IS A BUG!
4 |0.00 0.00 0.00 0.00 0.00 0.00 |
5 |0.00 0.00 0.00 0.00 0.00 0.00 |
-----------------------
Turn  16 Player  -1
   0 1 2 3 4 5 
-----------------------
0 |- - - - X - |
1 |- - - O O - |
2 |- X X X O - |
3 |- X O O O O |
4 |- O X - - - |
5 |- - - - X - |
-----------------------
[ 0 0] [ 0 1] [ 0 2] [ 0 3] [ 0 5] [ 1 0] [ 1 1] [ 1 2] [ 1 5] [ 2 0] [ 2 5] [ 3 0] [ 4 0] [ 4 3] [ 4 4] [ 4 5] [ 5 0] [ 5 1] [ 5 2] [ 5 3] [ 5 5] 

2 2
3 2
1 2
2 4
1 3
4 0
4 5


Turn  21 Player  1
   0 1 2 3 4 5 
-----------------------
0 |- - - - X - |
1 |- - O O O X |
2 |- X X X O O |
3 |- X O O O O |
4 |- O X - - - |
5 |- X X - X - |
-----------------------
AI must place stone on (5,5), (3, 5) or (0, 5)

-----------------------
0 |0.00 0.00 0.00 0.00 0.00 0.00 | # MCTS visit count
1 |0.00 0.00 0.00 0.00 0.00 0.00 |
2 |26.00 0.00 0.00 0.00 0.00 0.00 |
3 |0.00 0.00 0.00 0.00 0.00 0.00 |
4 |42.00 0.00 0.00 0.00 0.00 0.00 |
5 |0.00 0.00 0.00 31.00 0.00 0.00 |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |0.00 0.00 0.00 0.00 0.00 0.00 | # move prob reported by MCTS
1 |0.00 0.00 0.00 0.00 0.00 0.00 |
2 |0.26 0.00 0.00 0.00 0.00 0.00 |
3 |0.00 0.00 0.00 0.00 0.00 0.00 |
4 |0.42 0.00 0.00 0.00 0.00 0.00 |
5 |0.00 0.00 0.00 0.31 0.00 0.00 |
-----------------------
   0 1 2 3 4 5 
-----------------------
0 |0.00 0.00 0.00 0.00 0.00 0.00 | # move prob reported by policy network
1 |0.00 0.00 0.00 0.00 0.00 0.00 |
2 |0.27 0.00 0.00 0.00 0.00 0.00 |
3 |0.00 0.00 0.00 0.00 0.00 0.00 |
4 |0.41 0.00 0.00 0.00 0.00 0.00 |
5 |0.00 0.00 0.00 0.31 0.00 0.00 |
-----------------------
Turn  22 Player  -1

However, AI selects to play on (0, 4), which makes it losing the game.
   0 1 2 3 4 5 
-----------------------
0 |- - - - X - |
1 |- - O O O X |
2 |- X X X O O |
3 |- X O O O O |
4 |O O X - - - |
5 |- X X - X - |


## 6*6_numeps_100_num_mcts_sims_100_2_input_channel_128_channels
45 iterations
It beats the old model by 

player 1 wins / loss /draw: 0 / 8 / 12

Player 1 is the old model.
 7708/7708 [01:24<00:00, 91.66it/s, Loss_pi=1.94e+00, Loss_v=1.09e-01]

The previous model's issue is overfitting. After reducing the NN net filters from 512 to 128, 
the model performance dramatically improves.

## 15*15_numeps_100_num_mcts_sims_25_temp_15_input_channels_2_channels_128

Has over fitting issue. The test error is 100% higher than the training error.  

```EPOCH ::: 10
Training Net: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2490/2490 [02:18<00:00, 17.95it/s, Loss_pi=2.69e+00, Loss_v=1.00e-01]
2024-11-22 23:30:28 Bos-Mac-mini.local gomoku.pytorch.NNet[78352] INFO Test Losses - Policy Loss: 5.0656, Value Loss: 0.1131
```

Implemented the early stopping logic. 
