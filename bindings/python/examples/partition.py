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
from pygion import task, Ipartition, Partition, Region, RW
import numpy as np

@task(privileges=[RW])
def check_subregion(R):
    print('Subregion has volume %s extent %s bounds %s' % (
        R.ispace.volume, R.ispace.domain.extent, R.ispace.bounds))
    assert np.array_equal(R.x.shape, R.ispace.domain.extent)
    return R.ispace.volume

@task
def main():
    R = Region([4, 4], {'x': pygion.float64})
    pygion.fill(R, 'x', 0)

    # Create a partition of R.
    P = Partition.equal(R, [2, 2])

    # Same as above, broken explicitly into two steps.
    IP2 = Ipartition.equal(R.ispace, [2, 2])
    P2 = Partition(R, IP2)

    assert P.color_space.volume == 4

    # Grab a subregion of P.
    R00 = P[0, 0]

    print('Parent region has volume %s' % R.ispace.volume)
    assert R.ispace.volume == 16
    assert check_subregion(R00).get() == 4

    # Partition the subregion again.
    P00 = Partition.equal(R00, [2, 2])
    total_volume = 0
    for x in range(2):
        for y in range(2):
            R00xy = P00[x, y]
            total_volume += check_subregion(R00xy).get()
    assert total_volume == 4

    # An easy way to iterate subregions:
    for Rij in P:
        assert Rij.ispace.volume == 4

if __name__ == '__main__':
    main()
