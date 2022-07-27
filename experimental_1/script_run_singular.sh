#!/bin/bash -x

module load Stages/2022

srun --exclusive --account=hai_eagles --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=16 --time=23:50:00  singularity exec --bind "${PWD}:/mnt" --nv ../clip_latest.sif python main_hyper_view_training.py --target-index 0 1 2 3 &
srun --exclusive --account=hai_eagles --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=16 --time=23:50:00  singularity exec --bind "${PWD}:/mnt" --nv ../clip_latest.sif python main_hyper_view_training.py --target-index 0 &
srun --exclusive --account=hai_eagles --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=16 --time=23:50:00  singularity exec --bind "${PWD}:/mnt" --nv ../clip_latest.sif python main_hyper_view_training.py --target-index 1 &
srun --exclusive --account=hai_eagles --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=16 --time=23:50:00  singularity exec --bind "${PWD}:/mnt" --nv ../clip_latest.sif python main_hyper_view_training.py --target-index 2 &
srun --exclusive --account=hai_eagles --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=16 --time=23:50:00  singularity exec --bind "${PWD}:/mnt" --nv ../clip_latest.sif python main_hyper_view_training.py --target-index 3 &

