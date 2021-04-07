# Setup instructions for Piz Daint

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
git clone -b regent-index-launch-sc21 https://gitlab.com/StanfordLegion/legion.git
cd legion/language
./sc21_scripts/setup.sh
./sc21_scripts/build_stencil.sh stencil.run1
./sc21_scripts/build_circuit.sh circuit.run1
./sc21_scripts/build_pennant.sh pennant.run1
cp -r stencil.run1 stencil.run1_strong
cp -r circuit.run1 circuit.run1_strong
cp -r pennant.run1 pennant.run1_strong
(cd stencil.run1 && for n in 1 2 4 8 16; do sbatch --nodes $n sbatch_stencil.sh; done)
(cd circuit.run1 && for n in 1 2 4 8 16; do sbatch --nodes $n sbatch_circuit.sh; done)
(cd pennant.run1 && for n in 1 2 4 8 16; do sbatch --nodes $n sbatch_pennant.sh; done)
(cd stencil.run1_strong && for n in 1 2 4 8 16; do sbatch --nodes $n sbatch_stencil_strong.sh; done)
(cd circuit.run1_strong && for n in 1 2 4 8 16; do sbatch --nodes $n sbatch_circuit_strong.sh; done)
(cd pennant.run1_strong && for n in 1 2 4 8 16; do sbatch --nodes $n sbatch_pennant_strong.sh; done)
```

Feel free to run larger node counts after the small node count results
are verified.

Run directories are designed to be self-contained, so you can rebuild
Legion and prepare new runs while the old ones are in the queue. Note
that if you do rebuild Legion (and want those changes to propagate to
the runs), you will need to rebuild the run directories as well.

### Pennant SIGILL Workaround

Pennant gets miscompiled if you don't build it on a compute node:

```bash
salloc --nodes 1 --constraint=gpu --time=01:00:00 --mail-type=ALL -A d108
srun --cpu-bind none --pty bash
```

And then build as normal.

### Soleil-X DOM Instructions

Building:

```bash
source soleil-x/env.sh
cd soleil-x/src
rm -f dom_host.o && REGENT_FLAGS="-findex-launch 1 -findex-launch-dynamic 0 -foverride-demand-index-launch 1" make dom_host.exec && mv dom_host.exec dom_host.exec.idx-no-check
rm -f dom_host.o && REGENT_FLAGS="-findex-launch 1 -findex-launch-dynamic 1" make dom_host.exec && mv dom_host.exec dom_host.exec.idx-dyn-check
rm -f dom_host.o && REGENT_FLAGS="-findex-launch 0" make dom_host.exec && mv dom_host.exec dom_host.exec.noidx
```

Running:

```bash
source soleil-x/env.sh
cd soleil-x/src
EXECUTABLE=dom_host.exec.idx-no-check SCRATCH=$SCRATCH/dom.idx-no-check ./run.sh -i ../testcases/ws-pizdaint/dom/1.json
EXECUTABLE=dom_host.exec.idx-dyn-check SCRATCH=$SCRATCH/dom.idx-dyn-check ./run.sh -i ../testcases/ws-pizdaint/dom/1.json
EXECUTABLE=dom_host.exec.noidx SCRATCH=$SCRATCH/dom.noidx ./run.sh -i ../testcases/ws-pizdaint/dom/1.json
```
