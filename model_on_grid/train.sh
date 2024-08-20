#!/bin/bash

python train.py --n_layer 8 --batch_size 1024 --test_size 1000 --max_iters 50000 --save_freq 5 2>&1 | tee train.out