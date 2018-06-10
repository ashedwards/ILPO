#!/bin/bash
python models/vector_policy.py --mode test --n_actions 3 --real_actions 3 --batch_size 10  --checkpoint acrobot_ilpo --env Acrobot-v1 --exp_dir results/acrobot --n_dims 6

