#!/usr/bin/env python

# Copyright 2018 Stanford University
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
from legion import task, RW
import numpy

# Define a Python task. This task takes one argument: a region. The
# privileges on this task indicate that the task will read and write
# the region.
@task(privileges=[RW])
def init(R):
    # The fields of regions are numpy arrays, so you can call normal
    # numpy methods on them.
    R.x.fill(0)

# It's also possible to pass other arguments to a task, as long as
# those arguments are pickleable. The second argument here is just a
# number.
@task(privileges=[RW])
def inc(R, step):
    print(R.x)
    # When using regions, be careful about where the output is
    # directed if you want to avoid making extra copies of the data.
    numpy.add(R.x, step, out=R.x)
    print(R.x)
    return 42

# Define the main task. This task is called first.
@task(top_level=True)
def main():
    # Create a 2D index space of size 4x4.
    I = legion.Ispace.create([4, 4])

    # Create a field space with a single field x of type float64.
    F = legion.Fspace.create({'x': legion.float64})

    # Create a region from I and F.
    R = legion.Region.create(I, F)

    # This could have also been done with the following shortand, and
    # Legion will automatically create an index space and field space.
    R2 = legion.Region.create([4, 4], {'x': legion.float64})

    # Launch two tasks. The second task will depend on the first,
    # since they both write R.
    init(R)
    child_result = inc(R, 1)

    # Note: when a task runs, it returns a future. To get the value of
    # the future, you have to block. However, in idiomatic Legion code
    # it would be more common to pass the future to another task
    # (without blocking).
    print("child task returned", child_result)
    print("child task future contains", child_result.get())
    print("main_task done")
