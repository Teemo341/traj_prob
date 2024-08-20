#!/bin/bash

python dataset.py --data_dir ./data_temp --simulation_num 5000000 2>&1 | tee dataset.out