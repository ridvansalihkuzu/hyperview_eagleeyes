#!/bin/bash -x


docker run  --init --rm --shm-size=16gb --gpus device=0  -v "${PWD}:/app" ridvansalih/hyperview  python main_hyper_view_training.py -m 0 -c 5 -l 0.1000 -b 16 -w 224  --num-epochs 99 -p --in-data 'data/'


