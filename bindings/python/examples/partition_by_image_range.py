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
    empty_rect = ([0, 0], [-1, -1])
    rects = np.array(
        [[([0, 0], [-1, -1]), ([0, 0], [ 3,  3]), ([0, 0], [-1, -1]), ([0, 0], [-1, -1])],
         [([0, 0], [-1, -1]), ([0, 0], [-1, -1]), ([0, 0], [-1, -1]), ([0, 0], [-1, -1])],
         [([1, 0], [ 2,  0]), ([0, 0], [-1, -1]), ([0, 0], [-1, -1]), ([0, 0], [-1, -1])],
         [([0, 0], [-1, -1]), ([0, 0], [-1, -1]), ([0, 0], [-1, -1]), ([1, 2], [3, 3])]],
        dtype=R.rect.dtype)
    np.copyto(R.rect, rects, casting='no')

@task
def main():
    R = Region([4, 4], {'rect': pygion.rect2d})
    init_field(R)

    P = Partition.restrict(R, [2, 2], np.eye(2)*2, [2, 2])
    Q = Partition.image(R, P, 'rect', [2, 2])

    assert P.color_space.volume == 4
    assert P[0, 0].ispace.volume == 4
    assert P[0, 1].ispace.volume == 4
    assert P[1, 0].ispace.volume == 4
    assert P[1, 1].ispace.volume == 4

    assert Q[0, 0].ispace.volume == 16
    assert Q[0, 1].ispace.volume == 0
    assert Q[1, 0].ispace.volume == 2
    assert Q[1, 1].ispace.volume == 6

if __name__ == '__main__':
    main()
