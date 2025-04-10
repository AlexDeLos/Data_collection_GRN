#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=medium

#SBATCH --mail-type=END

#SBATCH --time=42:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=2

#SBATCH --mem=24G

#SBATCH --gres=gpu

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate alex_env

echo "Starting clustering"
python3 clustering.py
echo "---DONE"