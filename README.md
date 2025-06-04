Deep-learning-aided dismantling of interdependent networks
==============================================================
This code is aimed to provide the implementation of **Deep-learning-aided dismantling of interdependent networks** for network dismantling for both synthetic and real-world interconenected multilayer networks.

![im1](a.png)


# DEMO
Please run the ``run.sh`` file as a demo to use our algorithm and model.
Run the following command to start traning the model with unit removal cost. The data used for training are synthetic graphs. With default parameters, training on a V100 gpu takes almost 24 hours.
~~~~ {.sourceCode .shell}
./run.sh MultiDismantler_unit_cost train
~~~~ 
Run the following command to test the pre-trained model with unit removal cost in real dataset, and the output of the code includes the sequence of deleted nodes, the removal cost during the dismantling process, the changes in the normalized LMCC, and the value of AUDC. The results are stored in the "results/unitcost/MultiDismantler_real" folder.
~~~~ {.sourceCode .shell}
./run.sh MultiDismantler_unit_cost testReal
~~~~ 
Run this following command to test the trained model with unit removal cost in the synthetic dataset generated with different network size of nodes and various synthetic network controling parameters $g$, $\gamma$, and $k$, and the output of the code includes the average value of AUDC.The results are stored in the "results/unitcost/MultiDismantler_syn/" folder.
~~~~ {.sourceCode .shell}
./run.sh MultiDismantler_unit_cost testSynthetic
~~~~ 
Run this command to output the Normalized LMCC decline curve of real network with degree node remmoval cost.The results are stored in the "results/unitcost/MultiDismantler_audc" folder.
~~~~ {.sourceCode .shell}
./run.sh MultiDismantler_unit_cost drawLmcc
~~~~ 
The only difference between the following commands and the above commands is that unit cost is replaced with degree cost.
~~~~ {.sourceCode .shell}
./run.sh MultiDismantler_degree_cost train
~~~~ 
~~~~ {.sourceCode .shell}
./run.sh MultiDismantler_degree_cost testReal
~~~~ 
~~~~ {.sourceCode .shell}
./run.sh MultiDismantler_degree_cost testSynthetic
~~~~ 
~~~~ {.sourceCode .shell}
./run.sh MultiDismantler_degree_cost drawLmcc
~~~~ 


