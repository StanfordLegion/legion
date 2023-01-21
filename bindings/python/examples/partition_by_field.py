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
from pygion import task, Partition, Region, RW, WD
import numpy as np

@task(privileges=[WD])
def init_field(R):
    coloring = np.array(
        [[([0, 1],), ([1, 0],), ([0, 1],), ([1, 0],)],
         [([1, 1],), ([1, 0],), ([0, 1],), ([1, 1],)],
         [([0, 0],), ([1, 1],), ([1, 1],), ([0, 0],)],
         [([0, 0],), ([1, 1],), ([1, 1],), ([0, 0],)]],
        dtype=R.color.dtype)
    np.copyto(R.color, coloring, casting='no')

@task
def main():
    R = Region([4, 4], {'color': pygion.int2d})
    init_field(R)

    P = Partition.by_field(R, 'color', [2, 2])

    assert P.color_space.volume == 4

    print('Parent region has volume %s' % R.ispace.volume)
    assert R.ispace.volume == 16
    assert P[0, 0].ispace.volume == 4
    assert P[0, 1].ispace.volume == 3
    assert P[1, 0].ispace.volume == 3
    assert P[1, 1].ispace.volume == 6

if __name__ == '__main__':
    main()
