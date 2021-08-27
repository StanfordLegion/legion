These are instructions for reproducing results from the paper *Index
Launches: Scalable, Flexible Representation of Parallel Task Groups*,
published in SC'21. The experiments were performed on the [Piz
Daint](https://www.cscs.ch/computers/piz-daint/) supercomputer at
[CSCS](https://www.cscs.ch/). For reproduction, accounts can be
requested via their various [allocation
schemes](https://www.cscs.ch/user-lab/allocation-schemes/). The steps
below assume an account has been obtained.

These files are permanently archived at
[10.5281/zenodo.5164404](https://doi.org/10.5281/zenodo.5164404), and
are also archived under the tag
[papers/index-launch-sc21](https://github.com/StanfordLegion/legion/tree/papers/index-launch-sc21)
in the main Legion repository. The latter is assumed in the
instructions below, but either method may be used to obtain the files.

# Setup Instructions for Piz Daint

Put the following in `~/.bashrc`:

```bash
if [[ $(hostname) == daint* ]]; then
    module load daint-gpu

    module unload PrgEnv-cray
    module load PrgEnv-gnu/6.0.9
    module swap gcc/10.1.0 gcc/8.3.0 # CUDA 11 doesn't support GCC 10

    module load cudatoolkit

    export CC=cc
    export CXX=CC
fi
```

Then run the following:

```bash
git clone -b papers/index-launch-sc21 https://github.com/StanfordLegion/legion.git
cd legion/language
./sc21_scripts/setup.sh
./sc21_scripts/build_stencil.sh stencil.run1
./sc21_scripts/build_circuit.sh circuit.run1
cp -r stencil.run1 stencil.run1_strong
cp -r circuit.run1 circuit.run1_strong
(cd stencil.run1 && for n in 1 2 4 8 16; do sbatch --nodes $n sbatch_stencil.sh; done)
(cd circuit.run1 && for n in 1 2 4 8 16; do sbatch --nodes $n sbatch_circuit.sh; done)
(cd stencil.run1_strong && for n in 1 2 4 8 16; do sbatch --nodes $n sbatch_stencil_strong.sh; done)
(cd circuit.run1_strong && for n in 1 2 4 8 16; do sbatch --nodes $n sbatch_circuit_strong.sh; done)
```

Feel free to run larger node counts after the small node count results
are verified.

Run directories are designed to be self-contained, so you can rebuild
Legion and prepare new runs while the old ones are in the queue. Note
that if you do rebuild Legion (and want those changes to propagate to
the runs), you will need to rebuild the run directories as well.

## Analyzing Results

The runs above will produce output files in directories for each
app/configuration/node count. These files can be processed
automatically to assist in data collection. For example:

```bash
cd circuit.run1
../sc21_scripts/parse_results.py
```

This command will print (to stdout) a tsv-formatted file, with columns
for system (really, configuration), nodes, procs_per_node, rep (i.e.,
repetition number), and elapsed time (in seconds). In most cases this
file can be directly copied-and-pasted from a terminal emulator into a
spreadsheet program. For our analysis, we used Google Docs.

The paper reports numbers in units of throughput (cells per second or
similar). This is straightforward to compute in a spreedsheet based on
the elapsed time (reported by the script) and problem sizes, numbers
of timesteps (determined by the run scripts in this directory). The
relevant spreadsheet formulas are included below for
convenience. (They assume a spreadsheet program like Google Docs where
columns are lettered A, B, ... and rows are numbered 1, 2, ....)

Finally a pivot table can be created to average the repetitions with a
given configuration. In Google Docs this can be done by selecting the
entire sheet, then going to Data > Pivot Table, and hitting
Create. Set rows to nodes (disable totals), columns to system (again
disable totals), and values to throughput (summarize by AVERAGE). Also
add a filter on throughput with "Cell is not empty" to hide empty rows
and columns.

### Circuit Weak Scaling

| system | nodes | procs_per_node | rep | elapsed_time | wires  | time_steps | throughput               |
|--------|-------|----------------|-----|--------------|--------|------------|--------------------------|
|    ... |   ... |            ... | ... |          ... | 200000 |         50 | = F2 * G2 / E2 / 1000000 |

(This same problem size, and therefore the same formulas, were used
for the overdecomposed experiment.)

### Circuit Strong Scaling

| system | nodes | procs_per_node | rep | elapsed_time | wires   | time_steps | throughput               |
|--------|-------|----------------|-----|--------------|---------|------------|--------------------------|
|    ... |   ... |            ... | ... |          ... | 5120000 |         50 | = F2 * G2 / E2 / 1000000 |


### Stencil Weak Scaling

| system | nodes | procs_per_node | rep | elapsed_time | points    | time_steps | throughput                  |
|--------|-------|----------------|-----|--------------|-----------|------------|-----------------------------|
|    ... |   ... |            ... | ... |          ... | 900000000 |         50 | = F2 * G2 / E2 / 1000000000 |

### Stencil Strong Scaling

| system | nodes | procs_per_node | rep | elapsed_time | points    | time_steps | throughput                  |
|--------|-------|----------------|-----|--------------|-----------|------------|-----------------------------|
|    ... |   ... |            ... | ... |          ... | 900000000 |         50 | = F2 * G2 / E2 / 1000000000 |

## Soleil-X Full Instructions

Building:

```bash
source soleil-x/env.sh
cd soleil-x/src
rm -f soleil.o && REGENT_FLAGS="-findex-launch 1 -findex-launch-dynamic 0 -foverride-demand-index-launch 1" make soleil.exec && mv soleil.exec soleil.exec.idx-no-check
rm -f soleil.o && REGENT_FLAGS="-findex-launch 1 -findex-launch-dynamic 1" make soleil.exec && mv soleil.exec soleil.exec.idx-dyn-check
rm -f soleil.o && REGENT_FLAGS="-findex-launch 0" make soleil.exec && mv soleil.exec soleil.exec.noidx
```

Setting up strong/weak scaling problem sizes:

```bash
# cd soleil-x/testcases/ws-pizdaint/hit-flow # flow only
# cd soleil-x/testcases/ws-pizdaint/hit # flow and particles
cd soleil-x/testcases/ws-pizdaint/hit_dom # flow, particles and dom
mkdir strong
cd strong
../../../../scripts/scale_up.py --strong-scale -n 10 ../stag-10.json
cd ..
mkdir weak
cd weak
../../../../scripts/scale_up.py -n 10 ../stag-10.json
cd ..
```

Running:

```bash
source soleil-x/env.sh
cd soleil-x/src

mkdir -p runs

# strong scaling
for n in 1 2; do for r in 0 1 2 3 4; do REP=$r EXECUTABLE=soleil.exec.idx-no-check SCRATCH=$PWD/runs ./run.sh -i ../testcases/ws-pizdaint/hit_dom/strong/$n.json; done; done
for n in 1 2; do for r in 0 1 2 3 4; do REP=$r EXECUTABLE=soleil.exec.idx-dyn-check SCRATCH=$PWD/runs ./run.sh -i ../testcases/ws-pizdaint/hit_dom/strong/$n.json; done; done
for n in 1 2; do for r in 0 1 2 3 4; do REP=$r EXECUTABLE=soleil.exec.noidx SCRATCH=$PWD/runs ./run.sh -i ../testcases/ws-pizdaint/hit_dom/strong/$n.json; done; done

# weak scaling
for n in 1 2; do for r in 0 1 2 3 4; do REP=$r EXECUTABLE=soleil.exec.idx-no-check SCRATCH=$PWD/runs ./run.sh -i ../testcases/ws-pizdaint/hit_dom/weak/$n.json; done; done
for n in 1 2; do for r in 0 1 2 3 4; do REP=$r EXECUTABLE=soleil.exec.idx-dyn-check SCRATCH=$PWD/runs ./run.sh -i ../testcases/ws-pizdaint/hit_dom/weak/$n.json; done; done
for n in 1 2; do for r in 0 1 2 3 4; do REP=$r EXECUTABLE=soleil.exec.noidx SCRATCH=$PWD/runs ./run.sh -i ../testcases/ws-pizdaint/hit_dom/weak/$n.json; done; done
```

## Soleil-X DOM Instructions

Building:

```bash
source soleil-x/env.sh
cd soleil-x/src
rm -f dom_host.o && REGENT_FLAGS="-findex-launch 1 -findex-launch-dynamic 0 -foverride-demand-index-launch 1" make dom_host.exec && mv dom_host.exec dom_host.exec.idx-no-check
rm -f dom_host.o && REGENT_FLAGS="-findex-launch 1 -findex-launch-dynamic 1" make dom_host.exec && mv dom_host.exec dom_host.exec.idx-dyn-check
rm -f dom_host.o && REGENT_FLAGS="-findex-launch 0" make dom_host.exec && mv dom_host.exec dom_host.exec.noidx
```

Setting up weak scaling problem sizes:

```bash
cd soleil-x/testcases/ws-pizdaint/dom
../../../scripts/scale_up.py -n 10 base.json
```

Running:

```bash
source soleil-x/env.sh
cd soleil-x/src
for n in 1 2; do EXECUTABLE=dom_host.exec.idx-no-check ./run.sh -i ../testcases/ws-pizdaint/dom/$n.json; done
for n in 1 2; do EXECUTABLE=dom_host.exec.idx-dyn-check ./run.sh -i ../testcases/ws-pizdaint/dom/$n.json; done
for n in 1 2; do EXECUTABLE=dom_host.exec.noidx ./run.sh -i ../testcases/ws-pizdaint/dom/$n.json; done
```

## Dynamic Projection Functor Checks

All files for these measurements are in `bitmask-tests/`.

`self.sh` will benchmark the dynamic self-check. Results for different projections functors and launch domain sizes can be obtained by changing the two marked lines in `self.rg`. Examples of all four projection functors that appear in the paper are also provided.

Similarly, `cross.sh` will benchmark the dynamic cross-check. The test is setup to simulate `N` arguments on the same partition, with the first argument being read-write and all others being read-only. Results for different number of arguments and launch domain sizes can be obtained by changing the two marked lines in `cross.rg`.

Note that the paper reports the average of the five executions done by each script.

# Miscellaneous Notes

## Pennant

Pennant was not used in the final paper due to triggered bugs in the
Legion runtime. However, if desired, you can follow the instructions
below to build/run.

Note: Pennant gets miscompiled if you don't build it on a compute
node. Use the following commands to get an interactive shell on a
compute node.

```bash
salloc --nodes 1 --constraint=gpu --time=01:00:00 --mail-type=ALL -A d108
srun --cpu-bind none --pty bash
```

Then build:

```bash
./sc21_scripts/build_pennant.sh pennant.run1
cp -r pennant.run1 pennant.run1_strong
```

Exit the job. Then from the head node, launch the runs as usual:

```bash
(cd pennant.run1 && for n in 1 2 4 8 16; do sbatch --nodes $n sbatch_pennant.sh; done)
(cd pennant.run1_strong && for n in 1 2 4 8 16; do sbatch --nodes $n sbatch_pennant_strong.sh; done)
```
