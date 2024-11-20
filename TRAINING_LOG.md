6*6_numeps_100_num_mcts_sims_25_2_input_channel 
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

