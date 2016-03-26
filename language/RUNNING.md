Sample Circuit command:
Without NUMA:
LD_LIBRARY_PATH="." aprun -n2 -N1 -cc none -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./circuit.spmd20 -npp 2500 -wpp 10000 -l 5 -p 40 -ll:gsize 0 -ll:csize 8192 -ll:cpu 21 -ll:dma 2 -hl:prof 2 -level legion_prof=2 -logfile prof_2x1x20_%.log

LD_LIBRARY_PATH="." aprun -n4 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./circuit.spmd10 -npp 2500 -wpp 10000 -l 5 -p 40 -ll:gsize 0 -ll:csize 8192 -ll:cpu 11 -ll:dma 2 -hl:prof 4 -level legion_prof=2 -logfile prof_2x2x10_%.log

MiniAero:
Without NUMA:
LD_LIBRARY_PATH="." aprun -n2 -N1 -cc none -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./miniaero.spmd16 -blocks 32 -mesh 256x1024x4 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 30 -output_frequency 31 -hl:sched 256 -ll:cpu 16 -ll:dma 2 -ll:util 2 -ll:csize 8192 -ll:rsize 0 -ll:gsize 0 -hl:prof 2 -level legion_prof=2 -logfile prof_2x16_%.log

With NUMA:
LD_LIBRARY_PATH="." aprun -n4 -S1 -cc numa_node -e GASNET_NETWORKDEPTH=64 -e GASNET_NETWORKDEPTH_TOTAL=384 ./miniaero.spmd8 -blocks 32 -mesh 256x1024x4 -x_length 2 -y_length 0.2 -z_length 1 -ramp 0 -dt 1e-8 -viscous -second_order -time_steps 30 -output_frequency 31 -hl:sched 256 -ll:cpu 8 -ll:dma 2 -ll:csize 8192 -ll:rsize 0 -ll:gsize 0 -hl:prof 4 -level legion_prof=2 -logfile prof_2x2x8_%.log
