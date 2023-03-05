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
from pygion import task, R, Region, Reduce
import numpy

fspace = {
    'i16': pygion.int16,
    'i32': pygion.int32,
    'i64': pygion.int64,
    'u16': pygion.uint16,
    'u32': pygion.uint32,
    'u64': pygion.uint64,
    'f32': pygion.float32,
    'f64': pygion.float64,
    'c64': pygion.complex64,
    'c128': pygion.complex128,
}

fields_except_c128 = tuple(set(fspace.keys()) - set(['c128']))

@task(privileges=[Reduce('+')])
def inc(R, step):
    for field_name, field in R.items():
        numpy.add(field, step, out=field)

@task(privileges=[Reduce('*', *fields_except_c128)])
def mul(R, fact):
    for field_name, field in R.items():
        numpy.multiply(field, fact, out=field)

@task(privileges=[R])
def check(R, expected):
    for field_name, field in R.items():
        print(field_name, field)
        assert field[0] == expected

@task(privileges=[R(*fields_except_c128)])
def check_except_c128(R, expected):
    for field_name, field in R.items():
        print(field_name, field)
        assert field[0] == expected

@task
def main():
    R = Region([4], fspace)
    for field_name in R.keys():
        pygion.fill(R, field_name, 0)
    inc(R, 20)
    check(R, 20)
    mul(R, 3)
    check_except_c128(R, 60)

if __name__ == '__main__':
    main()
