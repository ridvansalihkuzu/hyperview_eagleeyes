#!/bin/bash -x


docker run  --init --rm --shm-size=16gb --gpus device=0  -v "${PWD}:/app" ridvansalih/hyperview  python main_hyper_view_training.py -m 0 -c 5 -l 0.1000 -b 16 -w 224  --num-epochs 99 -p --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
docker run  --init --rm --shm-size=16gb --gpus device=0  -v "${PWD}:/app" ridvansalih/hyperview  python main_hyper_view_training.py -m 0 -c 5 -l 0.0100 -b 16 -w 224  --num-epochs 99 -p --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
docker run  --init --rm --shm-size=16gb --gpus device=0  -v "${PWD}:/app" ridvansalih/hyperview  python main_hyper_view_training.py -m 0 -c 5 -l 0.0010 -b 16 -w 224  --num-epochs 99 -p --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
docker run  --init --rm --shm-size=16gb --gpus device=0  -v "${PWD}:/app" ridvansalih/hyperview  python main_hyper_view_training.py -m 0 -c 5 -l 0.0001 -b 16 -w 224  --num-epochs 99 -p --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &

