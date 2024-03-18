#!/bin/bash

# Copyright 2024 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Usage:
#
# Run ./my_app ... and get backtraces after 10 minutes:
#     ./run_with_backtraces.sh ./my_app ...
#
# Same, with a 20 minute wait:
#     WAIT_BEFORE_BACKTRACE=20m ./run_with_backtraces.sh ./my_app ...
#
# Note: if you do not want the script to kill your application, set:
#     KILL_PROCESS_AFTER_BACKTRACE=0

set -e

"$@" &
pid=$!

wait ${WAIT_BEFORE_BACKTRACE:-10m}

gdb -p $pid -ex 'set confirm off' -ex 'set height 0' -ex 'set width 0' -ex 'thread apply all backtrace' -ex 'quit'

# wait for backtraces to complete and then kill the process
if [[ ${KILL_PROCESS_AFTER_BACKTRACE:-1} = 1 ]]; then
    wait 2m
    kill $pid
    wait 1m
    kill -9 $pid
fi
