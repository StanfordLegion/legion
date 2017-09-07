# Setup

 1. Create a user `github-perf-nightly`
 2. Copy SSH key
 3. ```
    git config --global user.name "Legion Testing Automation"
    git config --global user.email "legion-testing@googlegroups.com"
    ```
 4. ```
    git clone -b master https://github.com/StanfordLegion/legion.git
    ```
 5. FIXME: Run `legion/language/install.py` once to work around Terra install on n0002
 6. Create file `legion/tools/nightly_perf/sapling/env.sh`
 7. Install crontab from `crontab.txt`
