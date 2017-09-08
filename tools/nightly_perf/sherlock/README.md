# Setup

 1. Setup SSH key
 2. ```
    git clone -b master https://github.com/StanfordLegion/legion.git
    ```
 3. Create file `legion/tools/nightly_perf/sherlock/env.sh` (defines `PERF_ACCESS_TOKEN`)
 4. Run `sbatch sbatch_once.sh` and make sure it works
 5. Run `sbatch sbatch_recurring.sh`
