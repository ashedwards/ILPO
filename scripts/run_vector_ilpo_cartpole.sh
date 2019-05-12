#!/bin/bash
python models/vector_ilpo.py --mode train --input_dir final_models/cartpole --n_actions 2 --batch_size 32 --output_dir cartpole_ilpo --max_epochs 100

