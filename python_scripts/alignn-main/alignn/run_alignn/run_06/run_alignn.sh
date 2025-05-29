#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python3.8 ../generate_data_reg.py ./binary_metals_oh.pkl ./ init
python3.8 ../../train_folder.py --root_dir "./" --config "../config.json" --output_dir=temp > log.out
