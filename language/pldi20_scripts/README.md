# Setup instructions for Piz Daint

Put the following in `~/.bashrc`:

```bash
if [[ $(hostname) == daint* ]]; then
    module load daint-gpu

    module unload PrgEnv-cray
    module load PrgEnv-gnu/6.0.4

    module load cudatoolkit

    export CC=cc
    export CXX=CC
fi
```

Then run the following:

```bash
git clone -b regent-dcr-pldi2020 https://gitlab.com/StanfordLegion/legion.git
cd legion/language
./pldi20_scripts/setup.sh
./pldi20_scripts/build_stencil.sh stencil.run1
./pldi20_scripts/build_circuit.sh circuit.run1
./pldi20_scripts/build_pennant.sh pennant.run1
(cd stencil.run1 && for n in 1 2 4 16; do sbatch --nodes $n sbatch_stencil.sh; done)
(cd circuit.run1 && for n in 1 2 4 16; do sbatch --nodes $n sbatch_circuit.sh; done)
(cd pennant.run1 && for n in 1 2 4 16; do sbatch --nodes $n sbatch_pennant.sh; done)
```

Feel free to run larger node counts after the small node count results
are verified.

In order to reproduce non-CTL Python results, set the value of
`_constant_time_launches` to `False`. (Note: Best to do this in a
fresh build directory to avoid clobbering other results.)
