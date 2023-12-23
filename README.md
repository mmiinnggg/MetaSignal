# 1. Introduction

This code corresponds to the article _'MetaSignal: Meta Reinforcement Learning for Traffic Signal Control via Fourier Basis Approximation'_, which has been submitted to _IEEE T-ITS_ and is currently under review.

# 2. Requirements

- This code has been tested on Python 3.7, and compatibility with other versions is not guaranteed. It is recommended to use Python versions 3.5 and above.
- For installing CityFlow, it is recommended to follow the instructions provided at https://cityflow.readthedocs.io/en/latest/install.html.
  
|Name| Version |
|---|---------|
|Keras| v2.3.1   |
|tensorflow-gpu| 1.14.0  |
|CityFlow| 1.1.0   |
| tqdm | 4.65.0 |
| numpy | 1.21.5  |
| matplotlib |  3.5.3  |



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

Containing all the used traffic file and road networks datasets. When extracting the 'data.zip' file, the resulting files will be stored in the 'project dir/data' directory.

> The **storage path -- "dir"** to each dataset, as written in its corresponding JSON file, should be accurately specified based on your local machine's configuration.
