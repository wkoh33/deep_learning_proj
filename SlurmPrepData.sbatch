#!/bin/bash
#SBATCH -JSlurmPrepData                    # Job name

#SBATCH -N1 --ntasks-per-node=8                 # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=20G                        # Memory per core
#SBATCH  --time=03:00:00                        # Duration of the job (Ex: 15 mins)

#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=FAIL              # Mail preferences
#SBATCH --mail-user=fho30@gatech.edu        # E-mail address for notifications

module load anaconda3/2022.05.0.1                           # Load module dependencies
conda activate proj

GATECH_EMAIL=fho30
export ML_DATA=/home/hice1/$GATECH_EMAIL/scratch/data
export PYTHONPATH=/home/hice1/$GATECH_EMAIL/deep_learning_proj

# Run bash script
./prepare_data.sh
