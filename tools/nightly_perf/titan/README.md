# Setup

 1. Setup SSH key
 2. ```
    git clone -b master https://github.com/StanfordLegion/legion.git
    cd legion/tools/nightly_perf/titan
    ./setup.sh
    ```
 3. Create file `legion/tools/nightly_perf/titan/env.sh` (defines `PERF_ACCESS_TOKEN`)
 4. Run `qsub qsub_once.sh` and make sure it works
