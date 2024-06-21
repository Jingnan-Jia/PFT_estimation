# PFT regression

Folder `lung_functiion` # the main source code. 
- `modules` # provides the necessary modules (networks, losses, hyper-parameter setting, dataloaders, data augmentation, tools, etc.)
- `scripts` # provides the main scripts for network training and validation.
  - run.py  # main script to run the code

------


## How to run the code?
`cd lung_functiion/scripts` at first, then there are 2 ways to train the models:
1. `sbatch script.sh` to submit job to slurm in your linux server.
2. `run.py --epochs=300 --mode='train' ... ` more arguments can be found in `set_args.py`.

## How to download the dataset?
The datasets used in this project can be viewed and downloaded by applying requests at Zenodo (https://zenodo.org/records/12120376). In accordance with local and institutional guidance and legal requirements, some terms of use are required to check during the application. After we carefully approve your application, we will send you an email with a valid downloading link with the password.”
