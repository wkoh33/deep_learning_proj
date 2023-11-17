#!/bin/bash

# Check env variables are ok
if [ -z "$ML_DATA" ]
then
      echo "\$ML_DATA is empty. Run set_env_variables.sh with your right path"
      exit 1
fi

if [ -z "$PYTHONPATH" ]
then
      echo "\$PYTHONPATH is empty. Run set_env_variables.sh with your right path"
      exit 1
fi

# Get dataset
python3 ./scripts/create_datasets.py cifar10