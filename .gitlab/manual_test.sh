#!/bin/bash
#SBATCH -A UMS036
#SBATCH -t 2:00:00
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -c 56
#SBATCH -p batch

set -e
set -x

source .gitlab/manual_env_frontier.sh

.gitlab/test_script.sh
