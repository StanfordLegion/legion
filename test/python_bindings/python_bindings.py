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

init = legion.extern_task(task_id=3, privileges=[legion.RW])

@legion.task
def f(x, y, z):
    print("inside task f%s" % ((x, y, z),))
    return x+1

@legion.task(privileges=[legion.RW], leaf=True)
def inc(R, step):
    print("inside task inc%s" % ((R, step),))

    # Sanity check that the values written by init are here, and that
    # they follow the same array ordering.
    print(R.x)
    for x in xrange(0, 4):
        for y in xrange(0, 4):
            assert int(R.x[x][y]) == x*4 + y

    numpy.add(R.x, step, out=R.x)
    print(R.x)

@legion.task(privileges=[legion.RW], leaf=True)
def fill(S, value):
    print("inside task fill%s" % ((S, value),))

    S.x.fill(value)
    S.y.fill(value)
    print(S.x[0:10])
    print(S.y[0:10])

@legion.task(privileges=[legion.RW], leaf=True)
def saxpy(S, a):
    print("inside task saxpy%s" % ((S, a),))

    print(S.x[0:10])
    print(S.y[0:10])
    numpy.add(S.y, a*S.x, out=S.y)
    print(S.y[0:10])

@legion.task(inner=True)
def main_task():
    print("inside main_task()")

    x = f(1, "asdf", True)
    print("result of f is %s" % x.get())

    R = legion.Region.create([4, 4], {'x': (legion.float64, 1)})
    init(R)
    inc(R, 1)

    S = legion.Region.create([1000], {'x': legion.float64, 'y': legion.float64})
    fill(S, 10)
    saxpy(S, 2)
