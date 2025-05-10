#!/bin/bash
#SBATCH --job-name=dsi_cifar
#SBATCH --gres=gpu:4                # Request 4 GPUs on each node
#SBATCH --partition=gpu             # Use the GPU partition
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1         # This would be the torchrun process (keep it to 1)
#SBATCH --gpus-per-node=4           # Request 4 GPUs per node
#SBATCH --cpus-per-task=4           # Request 4 CPUs per task
#SBATCH --output=dsi_cifar-%j.out
#SBATCH --error=dsi_cifar-%j.err
#SBATCH --time=00:10:00             # Set a time limit (10 minutes)
#SBATCH --mem=16G                    # Request 16GB of memory per node

module use /opt/nvidia/hpc_sdk/modulefiles
module load nvhpc-hpcx-cuda12/24.5

# You can use the following command to save the current module environment to a file
# module save dsi2025       # This will save the current module environment to $HOME/.lmod.d/dsi2025
# Then you can simply restore your module environment using:
# module restore dsi2025

source /etc/profile.d/conda.sh

conda activate /home/dsi2025/envs/dsi2025env

# export OMP_NUM_THREADS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Setting WORLDSIZE to $SLURM_NTASKS tasks"
echo "Running on $(hostname)"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)

srun bash -c '
torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1) \
    --master_port=29500 \
    /home/dsi2025/dsi_cifar_torchrun.py
'

echo All done
conda deactivate
