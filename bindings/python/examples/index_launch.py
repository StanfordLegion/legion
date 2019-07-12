#!/usr/bin/env python

# Copyright 2019 Stanford University
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
from legion import task, ID, R

@task
def hi(i):
    print("hello %s" % i)
    return i

@task(privileges=[R])
def hello(R, i):
    print("hello from point %s (region %s)" % (i, R.ispace.bounds))

@task
def main():
    futures = []
    for i in legion.IndexLaunch(10):
        futures.append(hi(i))
    for i, future in enumerate(futures):
        print("got %s" % future.get())
        assert int(future.get()) == i

    # Same in 2 dimensions.
    futures = []
    for point in legion.IndexLaunch([3, 3]):
        futures.append(hi(point))
    for i, point in enumerate(legion.Domain.create([3, 3])):
        assert futures[i].get() == point

    R = legion.Region.create([4, 4], {'x': legion.float64})
    P = legion.Partition.create_equal(R, [2, 2])
    legion.fill(R, 'x', 0)

    for i in legion.IndexLaunch([2, 2]):
        hello(R, i)

    for i in legion.IndexLaunch([2, 2]):
        hello(P[i], i)

    # Again, with a more explicit syntax.
    # ID is the name of the (implicit) loop variable.
    futures = legion.index_launch([3, 3], hi, ID)
    for point in legion.Domain.create([3, 3]):
        assert futures[point].get() == point

    legion.index_launch([2, 2], hello, R, ID)
    legion.index_launch([2, 2], hello, P[ID], ID)

if __name__ == '__legion_main__':
    main()
