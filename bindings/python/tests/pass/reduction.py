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
from legion import task, R, Region, Reduce
import numpy

@task(privileges=[Reduce('+')])
def inc(R, step):
    for field_name, field in R.items():
        numpy.add(field, step, out=field)

@task(privileges=[Reduce('*')])
def mul(R, fact):
    for field_name, field in R.items():
        numpy.multiply(field, fact, out=field)

@task(privileges=[R])
def check(R, expected):
    for field_name, field in R.items():
        print(field_name, field)
        assert field[0] == expected

@task
def main():
    R = Region([4], {
        'i16': legion.int16,
        'i32': legion.int32,
        'i64': legion.int64,
        'u16': legion.uint16,
        'u32': legion.uint32,
        'u64': legion.uint64,
        'f32': legion.float32,
        'f64': legion.float64,
        'c64': legion.complex64,
        # 'c128': legion.complex128,
    })
    for field_name in R.keys():
        legion.fill(R, field_name, 0)
    inc(R, 20)
    check(R, 20)
    mul(R, 3)
    check(R, 60)

if __name__ == '__main__':
    main()
