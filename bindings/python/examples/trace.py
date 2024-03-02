#!/usr/bin/env python3

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

from __future__ import print_function

import numpy

import pygion
from pygion import index_launch, task, IndexLaunch, ID, Partition, R, Region, RW, Trace

@task(privileges=[R])
def look(R, i):
    print(R.x)

@task(privileges=[RW])
def incr(R, i):
    numpy.add(R.x, 1, out=R.x)

@task
def main():
    R = Region([4, 4], {'x': pygion.float64})
    P = Partition.equal(R, [2, 2])
    pygion.fill(R, 'x', 0)

    trace1 = Trace()
    for t in range(5):
        with trace1:
            for i in IndexLaunch([2, 2]):
                look(R, i)

            for i in IndexLaunch([2, 2]):
                incr(P[i], i)

    trace2 = Trace()
    for t in range(5):
        with trace2:
            index_launch([2, 2], look, R, ID)
            index_launch([2, 2], incr, P[ID], ID)

if __name__ == '__main__':
    main()
