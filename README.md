# Cumulative Prospect Theoretic Reinforcement Learning

This repository gathers the code used to demonstrate the CPT policy gradient algorithm. 

The main training function for our algorithm is the `train` function in the `training.py` file. In the same file is the `train_spsa` function implementing the CPT-SPSA-G algorithm with tabular policies. The `envs.py` file contains the various environnements used for the experiments, while `policies.py` consists of various shapes of neural networks.

The figures that appear in the paper are reproducible by running the various notebooks in this repository - in some cases, to reduce runtime, we have changed the number of runs and/or the number of steps but it suffices to change them back to get the exact same results.
