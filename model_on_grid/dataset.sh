#!/bin/bash

python dataset.py --data_dir ./data_temp --simulation_num 1000 2>&1 | tee dataset.out