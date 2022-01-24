#!/usr/bin/env python3

# Copyright 2022 Stanford University, NVIDIA Corporation
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

import fileinput
import re

boring_strings = [
    'Realm::Runtime::wait_for_shutdown',
]

def is_interesting(backtrace):
    if all(['Legion::Internal' not in line for line in backtrace]):
        return False
    for line in backtrace:
        for s in boring_strings:
            if s in line:
                return False
    return True

backtrace = []
for line in fileinput.input():
    line = line[:-1].strip()
    if len(backtrace) == 0:
        if line.startswith('Thread '):
            # start of a new backtrace
            backtrace.append(line)
    elif re.search(r'^#[0-9]+ ', line) is not None:
        # next line of a backtrace
        backtrace.append(line)
    else:
        # end of a backtrace
        if is_interesting(backtrace):
            print('\n'.join(backtrace))
            print
        backtrace = []

