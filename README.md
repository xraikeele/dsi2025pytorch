# Keele DSI Week 2025
## Using PyTorch on the GreenHPC cluster

This repository contains some sample code to try out PyTorch on a SLURM-managed HPC cluster

The main files are:

- `dsi_cifar.py`: a simple use case of training a ResNet on the CIFAR-10 dataset on a single node
- `dsi_cifar_torchrun.py`: a distributed version of the same code, to be launched with `torchrun`
- `dsi_cifar.slurm`: a SLURM script to launch `dsi_cifar_torchrun.py` on a few nodes

### Creating your python environment
The files above are meant to be run within a conda envronment. 

My suggestion is to initialise your environment with [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html), then install most packages with `pip`.

The `envs` folder contains the requirements for both `conda` and `pip` for the code in this repo.

You can initialise your own environment as follows:

1. Create a new conda environment with a name of your choice

    `conda env create --file envs/dsi_environment.yml --name <your_env_name>`

    (this will install Python 11 and a few basic packages, including `pip`)

2. Activate the environment

    `conda activate <your_env_name>`

3. Using the local `pip`, install all other required packages

    `pip install -r envs/dsi_requirements.txt`