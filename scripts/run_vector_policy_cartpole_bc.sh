#!/bin/bash
python models/vector_policy_bc.py --mode test --ngf 128 --n_actions 2 --batch_size 10 --checkpoint cartpole_bc --env CartPole-v1 --n_dims 4


