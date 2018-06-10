#!/bin/bash
python models/run_image_policy.py --mode test --ngf 15 --n_actions 3 --batch_size 32 --checkpoint ilpo --env ThorFridge-v0  --policy_lr .0009

