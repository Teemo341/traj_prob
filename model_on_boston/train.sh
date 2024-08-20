#!/bin/bash

python train.py --block_size 52 --n_layer 8 --batch_size 512 --test_size 500 --max_iters 200000  --learning_rate 1e-1 --lr_drop_rate 0.5 --load_dir_id 0 --save_freq 5 --eval_freq 100 2>&1 | tee train.out