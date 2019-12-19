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
from legion import task, Fspace, Ispace, Region, RW
import numpy

@task
def main():
    R = Region([4, 4], {'x': legion.int32, 'y': legion.int32, 'z': legion.int32, 'w': legion.int32})
    legion.fill(R, 'x', 1)
    legion.fill(R, 'y', 20)
    legion.fill(R, ['z', 'w'], 100)

    legion.copy(R, ['x', 'y'], R, ['z', 'w'], redop='+')
    legion.copy(R, 'x', R, 'y', redop='+')
    legion.copy(R, 'y', R, 'x')

    assert R.x[0, 0] == 21
    assert R.y[0, 0] == 21
    assert R.z[0, 0] == 101
    assert R.w[0, 0] == 120

if __name__ == '__main__':
    main()
