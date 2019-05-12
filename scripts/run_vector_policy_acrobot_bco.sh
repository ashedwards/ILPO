#!/bin/bash
python models/vector_policy_bco.py --mode test --n_actions 3 --real_actions 3 --batch_size 32 --env Acrobot-v1 --exp_dir results/acrobot_bco --n_dims 6 --input_dir final_models/acrobot/acrobot.txt

