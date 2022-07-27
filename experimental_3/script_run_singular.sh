#!/bin/bash -x

module load Stages/2022

srun -N1 -p booster --account=hai_cons_ee --gres gpu:4 --time=01:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_hyper_view_training.py -m 0 -c 5 -l 0.1000 -b 16 -w 224  --num-epochs 99 -p --in-data 'data/'