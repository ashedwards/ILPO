#!/bin/bash
python models/vector_policy_bco.py --mode test --n_actions 2 --real_actions 2 --batch_size 32 --env CartPole-v1 --exp_dir results/cartpole_bco --n_dims 4 --input_dir final_models/cartpole/cartpole.txt

