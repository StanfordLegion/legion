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
from legion import task, RW

@task(privileges=[RW])
def hello_subregion(R):
    print('Subregion has volume %s' % R.ispace.volume)
    return R.ispace.volume

@task
def main():
    R = legion.Region.create([4, 4], {'x': legion.float64})

    # Create a partition of R.
    colors = [2, 2]
    transform = [[2, 0], [0, 2]]
    extent = [2, 2]
    P = legion.Partition.create_by_restriction(R, colors, transform, extent)

    # Again, with different parameters.
    colors2 = [3]
    transform2 = [[1], [2]]
    extent2 = legion.Domain.create([2, 2], [-1, -1])
    P2 = legion.Partition.create_by_restriction(R, colors2, transform2, extent2)

    assert P.color_space.volume == 4
    assert P2.color_space.volume == 3

    # Grab a subregion of P.
    R00 = P[0, 0]

    print('Parent region has volume %s' % R.ispace.volume)
    assert R.ispace.volume == 16
    assert hello_subregion(R00).get() == 4
    for Rij in P:
        assert Rij.ispace.volume == 4
    assert P2[0].ispace.volume == 1
    assert P2[1].ispace.volume == 4
    assert P2[2].ispace.volume == 2

if __name__ == '__legion_main__':
    main()
