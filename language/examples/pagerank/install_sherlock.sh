#!/bin/bash
#
#SBATCH --job-name=pagerank
#SBATCH --output=res_%j.txt
#SBATCH --time=10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6000MB
#SBATCH --gres gpu:4
#SBATCH -p aaiken

./install.py --cuda --gasnet
