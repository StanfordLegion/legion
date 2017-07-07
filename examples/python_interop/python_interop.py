#!/usr/bin/env python

# Copyright 2017 Stanford University
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

import legion
import numpy

# This task is defined in C++. See init_task in python_interop.cc.
init = legion.extern_task(task_id=3, privileges=[legion.RW])

@legion.task
def hello(i):
    print('hello %s' % i)

# Define a Python task. This task takes two arguments: a region and a
# number, and increments every element of the region by that number.
@legion.task(privileges=[legion.RW])
def inc(R, step):
    # The fields of regions are numpy arrays, so you can call normal
    # numpy methods on them. Be careful about where the output is
    # directed if you want to avoid making extra copies of the data.
    print(R.x)
    numpy.add(R.x, step, out=R.x)
    print(R.x)
    return 42

# Define the main Python task. This task is called from C++. See
# top_level_task in python_iterop.cc.
@legion.task
def main_task():
    # Create a 2D index space of size 4x4.
    I = legion.Ispace.create([4, 4])

    # Create a field space with a single field x of type float64. For
    # interop with C++, we have to choose an explicit field ID here
    # (in this case, 1). We could leave this out if the code were pure
    # Python.
    F = legion.Fspace.create({'x': (legion.float64, 1)})

    # Create a region from I and F and launch two tasks.
    R = legion.Region.create(I, F)
    init(R)
    child_result = inc(R, 1)
    print("child task returned", child_result.get())
    print("main_task done")

    for i in legion.IndexLaunch([3]):
        print('queue %s' % i)
        hello(i)
