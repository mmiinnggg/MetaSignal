# 1. Introduction

This code corresponds to the article _'MetaSignal: Meta Reinforcement Learning for Traffic Signal Control via Fourier Basis Approximation'_, which has been submitted to _IEEE T-ITS_ and is currently under review.

# 2. Requirements

|Type|Name| Version |
|---|---|---------|
|language|python| 3.7     |
|simulation platform|CityFlow| 1.1.0   |
|other | itertools | 5.0.0  |
|other | pickle | 0.4.2 |
|other | collections | 3.0.0  |
|other | tqdm | 4.65.0 |
|other | numpy | 1.21.5  |
|other | math | --  |
|other | matplotlib |  3.5.3  |
|other | argparse | --  |
|other | datetime | --   |
|other | time | --  |
|other | csv | --  |
|other | json | --  |

- CityFlow Installation Guide: https://cityflow.readthedocs.io/en/latest/install.html

# 3. Code details

The files are functioning like this:

* ``runexp.py``

The main logic for completing the algo-training.
 > The **arg '--dataset'** requires special attention, and it should be consistent with the dataset being used. For example, the datasets corresponding to road networks such as Jinan, Hangzhou, Syn3x3, and Syn4x4 should have names that respectively match '--dataset==jinan', '--dataset==hangzhou', '--dataset==3x3', and '--dataset==4x4'.
 
* ``agent.py``

Simulate the reinforcement learning agent, get current traffic state and call the agent to make decision by interact with cityflow_env.

* ``cityflow_env.py``

Define a simulator environment to interact with the simulator and obtain needed data like features.

* ``utility.py``

Collect other used functions.

* ``metric\travel_time.py``

Numerical metrics for different agents to compared under the same standard.

* ``data.zip``

Containing all the used traffic file and road networks datasets. 

> The **storage path -- "dir"** to each dataset, as written in its corresponding JSON file, should be accurately specified based on your local machine's configuration.
