#!/bin/bash
python models/vector_bc.py --mode train --input_dir final_models/cartpole_bc --n_actions 2 --batch_size 32 --output_dir cartpole_bc --max_epochs 1000 
