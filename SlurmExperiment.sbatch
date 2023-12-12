#!/bin/bash
#SBATCH -JSlurmTrainModel                    # Job name

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=24

#SBATCH  --time=16:00:00                                    # Duration of the job (Ex: 15 mins)

#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=END,FAIL              # Mail preferences
#SBATCH --mail-user=wkoh33@gatech.edu        # E-mail address for notifications

module load anaconda3/2022.05.0.1                           # Load module dependencies
conda activate proj

GATECH_EMAIL=wkoh33
export ML_DATA=/home/hice1/$GATECH_EMAIL/scratch/data
export PYTHONPATH=/home/hice1/$GATECH_EMAIL/deep_learning_proj

# TEST
srun python fixmatch.py \
    --lr=0.02 \
    --confidence=0.90 \
    --dataset=cifar10.3@250-1 \
    --arch=squeezenetmini1 \
    --train_dir ./experiments/fixmatch