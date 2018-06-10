#!/bin/bash
python models/vector_policy.py --mode test --n_actions 2 --real_actions 2 --batch_size 10  --checkpoint cartpole_ilpo --env CartPole-v1 --exp_dir results/cartpole --n_dims 4

