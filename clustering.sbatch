#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=medium

#SBATCH --mail-type=END

#SBATCH --time=32:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=32G

#SBATCH --gres=gpu

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate alex_env

echo "Starting clustering"
python3 clustering.py
echo "---DONE"