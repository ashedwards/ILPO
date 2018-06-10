#!/bin/bash
python models/image_bc.py --mode train --ngf 15 --ndf 15 --input_dir final_models/thor --n_actions 3 --batch_size 100 --output_dir bc --max_epochs 5

