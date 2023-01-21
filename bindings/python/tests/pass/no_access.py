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
from pygion import task, ID, R, N

@task(privileges=[N])
def hello(R, i):
    print("hello from point %s (region %s)" % (i, R.ispace.bounds))

@task(privileges=[N])
def hello2(R, i):
    print("hello2 from point %s (region %s)" % (i, R.ispace.bounds))
    hello(R, i)

@task
def main():
    R = pygion.Region([4], {'x': pygion.float64})
    P = pygion.Partition.equal(R, [4])
    pygion.fill(R, 'x', 0)

    hello2(P[0], 0)

    for i in pygion.IndexLaunch([4]):
        hello2(P[i], i)

    pygion.index_launch([4], hello2, P[ID], ID)

    # FIXME: This is needed in nopaint to avoid a race with region deletion
    pygion.execution_fence()

if __name__ == '__main__':
    main()
