#!/bin/bash
python models/image_ilpo.py --mode train --ngf 15 --ndf 15 --input_dir final_models/thor/AB --n_actions 3 --batch_size 100 --output_dir ilpo --max_epochs 5

