#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:quadro_rtx_6000:3

module purge
module restore PT
chmod +x /home/physics/phubdf/Classifier_pytorch_num.py

srun /home/physics/phubdf/Classifier_pytorch_num.py
