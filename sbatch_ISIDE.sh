#!/bin/bash
# Sample GPU batch script
# Job parameters
# ----------------------------
#SBATCH --job-name=DF_tf_test
#SBATCH --output=tf_output.txt
#SBATCH --error=tf_output.txt

# Job resources
# ----------------------------
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1000
#SBATCH --time=0:01:00
#SBATCH --gres=gpu:2

# Cluster partition / queue
# ----------------------------
# Check partitions: sinfo -a
#SBATCH --partition=debug

# Operations
echo "Job started at $(date)"

# Job steps

# Check Nvidia
#nvidia-smi

# Test singularity container with NVIDIA GPU enviroment
singularity run --nv tensorflow_test.sif

echo "Job completed at $(date)"
