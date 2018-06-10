#!/bin/bash
python models/vector_policy_bc.py --mode test --ngf 128 --n_actions 3 --batch_size 10 --checkpoint acrobot_bc --env Acrobot-v1 --n_dims 6


