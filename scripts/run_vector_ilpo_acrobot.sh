#!/bin/bash
python models/vector_ilpo.py --mode train --input_dir final_models/acrobot  --n_actions 3 --batch_size 32 --output_dir acrobot_ilpo --max_epochs 100 --exp_dir results/acrobot

