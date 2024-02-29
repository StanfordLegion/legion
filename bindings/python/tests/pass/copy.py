#!/usr/bin/env python3

# Copyright 2024 Stanford University
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
from pygion import task, Fspace, Ispace, Region, RW
import numpy

@task
def main():
    R = Region([4, 4], {'x': pygion.int32, 'y': pygion.int32, 'z': pygion.int32, 'w': pygion.int32})
    pygion.fill(R, 'x', 1)
    pygion.fill(R, 'y', 20)
    pygion.fill(R, ['z', 'w'], 100)

    pygion.copy(R, ['x', 'y'], R, ['z', 'w'], redop='+')
    pygion.copy(R, 'x', R, 'y', redop='+')
    pygion.copy(R, 'y', R, 'x')

    assert R.x[0, 0] == 21
    assert R.y[0, 0] == 21
    assert R.z[0, 0] == 101
    assert R.w[0, 0] == 120

if __name__ == '__main__':
    main()
