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
from legion import task

@task
def main():
    R = legion.Region.create(
        [10],
        {
            'b': legion.bool_,
            'c64': legion.complex64,
            'f16': legion.float16,
            'f32': legion.float32,
            'f64': legion.float64,
            'i8': legion.int8,
            'i16': legion.int16,
            'i32': legion.int32,
            'i64': legion.int64,
            'u8': legion.uint8,
            'u16': legion.uint16,
            'u32': legion.uint32,
            'u64': legion.uint64,
        })
    R.b.fill(False)
    R.c64.fill(1+2j)
    R.f16.fill(1.23)
    R.f32.fill(3.45)
    R.f64.fill(6.78)
    R.i8.fill(-1)
    R.i16.fill(-123)
    R.i32.fill(-123456)
    R.i64.fill(-123456789)
    R.u8.fill(1)
    R.u16.fill(123)
    R.u32.fill(123456)
    R.u64.fill(123456789)
