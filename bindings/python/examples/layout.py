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
from pygion import task, AOS_C, AOS_F, Fspace, Ispace, LayoutConstraint, Region, RW, SOA_C, SOA_F
import numpy

@task(privileges=[RW], layout=[LayoutConstraint(order=SOA_F, dim=2)])
def test_SOA_F(R):
    print('test SOA_F layout:')
    print(R.x.flags)
    assert R.x.flags['F_CONTIGUOUS']
    assert not R.x.flags['C_CONTIGUOUS']
    values = numpy.array(
        [[ 0, 10],
         [20, 30]],
        dtype=R.x.dtype)
    numpy.add(R.x,   values, out=R.x)
    numpy.add(R.y, 2*values, out=R.y)

@task(privileges=[RW], layout=[LayoutConstraint(order=AOS_F, dim=2)])
def test_AOS_F(R):
    print('test AOS_F layout:')
    print(R.x.flags)
    assert not R.x.flags['C_CONTIGUOUS']
    values = numpy.array(
        [[  0, 100],
         [200, 300]],
        dtype=R.x.dtype)
    numpy.add(R.x, values, out=R.x)
    numpy.add(R.y, 2*values, out=R.y)

@task(privileges=[RW], layout=[LayoutConstraint(order=SOA_C, dim=2)])
def test_SOA_C(R):
    print('test SOA_C layout:')
    print(R.x.flags)
    assert R.x.flags['C_CONTIGUOUS']
    assert not R.x.flags['F_CONTIGUOUS']
    values = numpy.array(
        [[   0, 1000],
         [2000, 3000]],
        dtype=R.x.dtype)
    numpy.add(R.x, values, out=R.x)
    numpy.add(R.y, 2*values, out=R.y)

@task(privileges=[RW], layout=[LayoutConstraint(order=AOS_C, dim=2)])
def test_AOS_C(R):
    print('test AOS_C layout:')
    print(R.x.flags)
    assert not R.x.flags['F_CONTIGUOUS']
    values = numpy.array(
        [[    0, 10000],
         [20000, 30000]],
        dtype=R.x.dtype)
    numpy.add(R.x, values, out=R.x)
    numpy.add(R.y, 2*values, out=R.y)

@task
def main():
    I = Ispace([2, 2])
    F = Fspace({'x': pygion.int64, 'y': pygion.int64})
    R = Region(I, F)
    pygion.fill(R, 'x', 0)
    pygion.fill(R, 'y', 1)

    test_SOA_F(R)
    test_AOS_F(R)
    test_SOA_C(R)
    test_AOS_C(R)

    print(R.x)
    print(R.y)

    assert R.x[0,0] ==     0
    assert R.x[0,1] == 11110
    assert R.x[1,0] == 22220
    assert R.x[1,1] == 33330

    assert R.y[0,0] ==     1
    assert R.y[0,1] == 22221
    assert R.y[1,0] == 44441
    assert R.y[1,1] == 66661

if __name__ == '__main__':
    main()
