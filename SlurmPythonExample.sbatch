#!/bin/bash
#SBATCH -JSlurmPythonExample                    # Job name
#SBATCH -N1 --ntasks-per-node=4                 # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=1G                        # Memory per core
#SBATCH -t15                                    # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=FAIL              # Mail preferences
#SBATCH --mail-user=wkoh33@gatech.edu        # E-mail address for notifications

module load anaconda3/2022.05.0.1                           # Load module dependencies
conda activate proj
conda env list
srun python test.py                             # Example Process