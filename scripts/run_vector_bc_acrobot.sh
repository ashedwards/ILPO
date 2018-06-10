#!/bin/bash
python models/vector_bc.py --mode train --input_dir final_models/acrobot_bc  --n_actions 3 --batch_size 32 --output_dir acrobot_bc --max_epochs 1000 
