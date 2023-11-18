#!/bin/bash
#SBATCH -JSlurmPythonExample                    # Job name

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

#SBATCH -t5                                    # Duration of the job (Ex: 15 mins)

#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=FAIL              # Mail preferences
#SBATCH --mail-user=wkoh33@gatech.edu        # E-mail address for notifications

module load anaconda3/2022.05.0.1                           # Load module dependencies
conda activate proj
conda env list

GATECH_EMAIL=wkoh33
export ML_DATA=/home/hice1/$GATECH_EMAIL/scratch/data
export PYTHONPATH=/home/hice1/$GATECH_EMAIL/deep_learning_proj
export CUDA_VISIBLE_DEVICES=0
echo $ML_DATA
echo $PYTHONPATH

srun python fixmatch.py --filters=3 --dataset=cifar10.3@10-1 --train_dir ./experiments/fixmatch
# srun python test.py                             # Example Process