# 1. Introduction

This code corresponds to the article _'MetaSignal: Meta Reinforcement Learning for Traffic Signal Control via Fourier Basis Approximation'_, which has been submitted to _IEEE T-ITS_ and is currently under review..

# 2. Requirements

`python3.7`, `cityflow`, `pandas`, `numpy`

[`cityflow`](https://github.com/cityflow-project/CityFlow.git) needs a linux environment, and we run the code on WSL2.

# 3. Code details

The files are functioning like this:

* ``runexp.py``

The main logic for completing the algo-training.

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


