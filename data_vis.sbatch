#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=medium

#SBATCH --mail-type=END

#SBATCH --time=1:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=30G

#SBATCH --gres=gpu

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate alex_env

echo "Starting data vis"
python3 data_analisys.py
echo "---DONE"