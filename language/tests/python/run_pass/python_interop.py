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
from pygion import Region, RW, int32, void

@pygion.task(task_id=2,
    argument_types=[],
    return_type=int32,
    calling_convention='regent')
def hello():
    print('hello from Python')
    return 123

@pygion.task(
    task_id=3,
    argument_types=[Region, int32],
    privileges=[RW],
    return_type=void,
    calling_convention='regent')
def inc(R, x):
    print(R.x)
    R.x[:] += x
    print(R.x)
