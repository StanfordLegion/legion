# Setup

 0. Add to `.bashrc`:
    ```
    module load system git
    module load llvm/3.8.1
    ```
 1. Setup SSH key
 2. ```
    git clone -b master https://github.com/StanfordLegion/legion.git
    cd legion/tools/nightly_perf/sherlock
    ./setup.sh
    ```
 3. Create file `legion/tools/nightly_perf/sherlock/env.sh` (defines `PERF_ACCESS_TOKEN`)
 4. Run `sbatch sbatch_once.sh` and make sure it works
 5. Run `sbatch sbatch_recurring.sh`
