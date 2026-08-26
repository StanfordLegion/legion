#!/bin/zsh
# Launch the matrix driver detached from the invoking session so long TLC
# runs (user-authorized, up to ~4h) are not tied to the caller's lifetime.
cd "$(dirname "$0")"
nohup ./run_matrix2.sh > run_matrix.log 2>&1 &
disown
sleep 2
echo "driver pid: $(cat driver.lock 2>/dev/null || echo none)"
