#!/bin/bash
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -n | --nodes)
    NUM_NODES="$2"
    shift # past argument
    shift # past value
    ;;
    -g | --gpus)
    NUM_GPUS="$2"
    shift # pass argument
    shift # pass value
    ;;
    -c | --cpus)
    NUM_CPUS="$2"
    shift # pass argument
    shift # pass value
    ;;
    -p | --program)
    EXTENSION="$2"
    shift # pass argument
    shift # pass value
    ;;
    *)
    shift # pass argument
    ;;
esac
done

if [ -z "$NUM_NODES" ]; then echo "Warning: --nodes is missing"; exit; fi
if [ -z "$NUM_GPUS" ]; then echo "Warning: --gpus is missing"; exit; fi
if [ -z "$NUM_CPUS" ]; then echo "Warning: --cpus is missing"; exit; fi
if [ -z "$EXTENSION" ]; then echo "Warning: --file is missing"; exit; fi

if [ "$EXTENSION" == "mapping" ]; then
  echo "Running PageRank demo with mapping optimization..."
elif [ "$EXTENSION" == "baseline" ]; then
  echo "Running PageRank baseline demo..."
elif [ "$EXTENSION" == "optimized" ]; then
  echo "Running PageRank demo with optimized CUDA kernel..."
else
  echo "Warning: --program must be one of baseline, mapping, optimized"
  exit
fi

if [ $NUM_GPUS -eq 0 ]; then
    NUM_PARTS=$((NUM_CPUS * NUM_NODES))
else
    NUM_PARTS=$((NUM_GPUS * NUM_NODES))
fi

if [ $NUM_GPUS -lt 8 ]; then
    NUM_CPUS_TO_ALLOC=8
else
    NUM_CPUS_TO_ALLOC=$((NUM_CPUS))
fi

echo "Total partitions = ${NUM_PARTS}"

cat > submit_pr.sh <<EOL
#!/bin/bash
#
#SBATCH --job-name=pagerank
#SBATCH --export=TERRA_PATH,INCLUDE_PATH,PATH,LD_LIBRARY_PATH,CUDA_HOME,LG_RT_DIR,C_PATH
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=6000MB
#SBATCH -p aaiken
EOL

echo "#SBATCH --output=res_%j_${EXTENSION}_nodes${NUM_NODES}_gpus${NUM_GPUS}_cpus${NUM_CPUS}.txt" >> submit_pr.sh
echo "#SBATCH --nodes=${NUM_NODES}" >> submit_pr.sh
echo "#SBATCH --gres gpu:4" >> submit_pr.sh
echo "#SBATCH --cpus-per-task=16" >> submit_pr.sh
echo "" >> submit_pr.sh

if [ $NUM_GPUS -gt 0 ]; then
    echo "LAUNCHER='srun -n ${NUM_NODES}' ~/legion/language/regent.py ~/legion/language/examples/pagerank/pagerank_${EXTENSION}.rg -graph ~/legion/language/examples/pagerank/twitter.gr -ni 10 -nw ${NUM_PARTS} -ll:gpu ${NUM_GPUS} -ll:fsize 15000 -ll:zsize 20000 -ll:cpu ${NUM_CPUS} -lg:prof ${NUM_NODES} -lg:prof_logfile prof%.gz" >> submit_pr.sh
else
    echo "LAUNCHER='srun -n ${NUM_NODES}' ~/legion/language/regent.py ~/legion/language/examples/pagerank/pagerank_${EXTENSION}.rg -graph ~/legion/language/examples/pagerank/twitter.gr -ni 10 -nw ${NUM_PARTS} -ll:csize 20000 -ll:cpu ${NUM_CPUS} -lg:prof ${NUM_NODES} -lg:prof_logfile prof%.gz" >> submit_pr.sh
fi

sbatch submit_pr.sh
squeue | grep pagerank
