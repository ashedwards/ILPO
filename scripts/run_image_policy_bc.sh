#!/bin/bash
python models/image_policy_bc.py --mode test --ngf 15 --n_actions 3 --batch_size 32 --checkpoint bc --env ThorFridge-v0  --policy_lr .0003

