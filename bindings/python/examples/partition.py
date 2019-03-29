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
    P = legion.Partition.create_equal(R, [2, 2])

    # Same as above, broken explicitly into two steps.
    IP2 = legion.Ipartition.create_equal(R.ispace, [2, 2])
    P2 = legion.Partition.create(R, IP2)

    assert P.color_space.volume == 4

    # Grab a subregion of P.
    R00 = P[0, 0]

    print('Parent region has volume %s' % R.ispace.volume)
    assert R.ispace.volume == 16
    assert hello_subregion(R00).get() == 4

    # Partition the subregion again.
    P00 = legion.Partition.create_equal(R00, [2, 2])
    total_volume = 0
    for x in range(2):
        for y in range(2):
            R00xy = P00[x, y]
            total_volume += hello_subregion(R00xy).get()
    assert total_volume == 4

if __name__ == '__legion_main__':
    main()
