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
from pygion import task, Domain, Partition, Region, RW
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
    colors = [2, 2]
    transform = [[2, 0], [0, 2]]
    extent = [2, 2]
    P = Partition.restrict(R, colors, transform, extent)

    # Again, with different parameters.
    colors2 = [3]
    transform2 = [[1], [2]]
    extent2 = Domain([2, 2], [-1, -1])
    P2 = Partition.restrict(R, colors2, transform2, extent2)

    assert P.color_space.volume == 4
    assert P2.color_space.volume == 3

    # Grab a subregion of P.
    R00 = P[0, 0]

    print('Parent region has volume %s' % R.ispace.volume)
    assert R.ispace.volume == 16
    assert check_subregion(R00).get() == 4
    for Rij in P:
        assert check_subregion(Rij).get() == 4
    assert check_subregion(P2[0]).get() == 1
    assert check_subregion(P2[1]).get() == 4
    assert check_subregion(P2[2]).get() == 2

if __name__ == '__main__':
    main()
