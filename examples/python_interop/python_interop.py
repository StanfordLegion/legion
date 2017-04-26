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

@legion.task
def f(ctx, x, y, z):
    print("inside task f%s" % ((x, y, z),))
    return x+1

@legion.task(privileges = [legion.RW])
def g(ctx, R):
    print("inside task g%s" % ((R,),))

    # Use generic accessors to fill the array with some pattern. This
    # is to sanity check that NumPy is using the same indexing as
    # Legion.

    point = legion.ffi.new("legion_point_2d_t *")
    value = legion.ffi.new("double *")
    for x in xrange(0, 4):
        for y in xrange(0, 4):
            point.x[0] = x
            point.x[1] = y
            value[0] = float(x*4 + y)
            legion.c.legion_accessor_generic_write_domain_point(
                R.x.accessor,
                legion.c.legion_domain_point_from_point_2d(point[0]),
                value, legion.ffi.sizeof("double"))

    # If we access elements through the NumPy array, we should get the
    # same values back.
    print(R.x)
    for x in xrange(0, 4):
        for y in xrange(0, 4):
            assert int(R.x[x][y]) == x*4 + y

    ones = numpy.ones([4, 4])
    numpy.add(R.x, ones, out=R.x)
    print(R.x)

    R.x.fill(1)
    print(R.x)

@legion.task
def main_task(ctx):
    print("inside main_task()")

    x = f(ctx, 1, "asdf", True)
    print("result of f is %s" % x)

    R = legion.Region.create(ctx, [4, 4], {'x': legion.double})
    print("region %s" % R)
    # print("field %s" % R.x)

    g(ctx, R)
