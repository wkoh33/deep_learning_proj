#!/bin/bash
#SBATCH -JSlurmTrainModel                    # Job name

#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

#SBATCH  --time=08:00:00                                    # Duration of the job (Ex: 15 mins)

#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=END,FAIL              # Mail preferences
#SBATCH --mail-user=wkoh33@gatech.edu        # E-mail address for notifications

module load anaconda3/2022.05.0.1                           # Load module dependencies
conda activate proj

GATECH_EMAIL=wkoh33
export ML_DATA=/home/hice1/$GATECH_EMAIL/scratch/data
export PYTHONPATH=/home/hice1/$GATECH_EMAIL/deep_learning_proj

# srun python fixmatch.py \
#     --filters=32 \
#     --batch=512 \
#     --dataset=cifar10.3@10-1 \
#     --arch=resnet \
#     --train_dir ./experiments/fixmatch

srun python fixmatch.py \
    --batch=512 \
    --dataset=cifar10.3@10-1 \
    --arch=squeezenet \
    --train_dir ./experiments/fixmatch