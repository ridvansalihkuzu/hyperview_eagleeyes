#!/bin/bash -x


docker run  --init --rm --shm-size=16gb --gpus device=0  -v "${PWD}:/app" ridvansalih/clip  python main_hyper_view_training.py --target-index 0 1 2 3 &
docker run  --init --rm --shm-size=16gb --gpus device=0  -v "${PWD}:/app" ridvansalih/clip  python main_hyper_view_training.py --target-index 0 &
docker run  --init --rm --shm-size=16gb --gpus device=0  -v "${PWD}:/app" ridvansalih/clip  python main_hyper_view_training.py --target-index 1 &
docker run  --init --rm --shm-size=16gb --gpus device=0  -v "${PWD}:/app" ridvansalih/clip  python main_hyper_view_training.py --target-index 2 &
docker run  --init --rm --shm-size=16gb --gpus device=0  -v "${PWD}:/app" ridvansalih/clip  python main_hyper_view_training.py --target-index 3 &

