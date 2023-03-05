#!/usr/bin/env python3

# Copyright 2023 Stanford University
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

from __future__ import print_function

import pygion
from pygion import task, RW

@task(privileges=[RW])
def hello_subregion(R):
    print('Subtask ran successfully')

@task
def main():
    R = pygion.Region([4, 4], {'x': pygion.float64})
    pygion.fill(R, 'x', 0)
    P = pygion.Partition.equal(R, [2, 2])
    hello_subregion(P[0, 0]) # this should work
    try:
        hello_subregion(P) # this should fail
    except TypeError:
        print('Test passed')
    else:
        assert False, 'Test failed'

if __name__ == '__main__':
    main()
