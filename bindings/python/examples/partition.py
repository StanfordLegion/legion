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

@task
def main():
    R = legion.Region.create([4, 4], {'x': legion.float64})

    # Create a partition of R.
    P = legion.Partition.create_equal(R, [2, 2])

    # Same as above, broken explicitly into two steps.
    IP2 = legion.Ipartition.create_equal(R.ispace, [2, 2])
    P2 = legion.Partition.create(R, IP2)

    assert P.color_space.volume == 4

    P2.destroy()
    IP2.destroy()
    P.destroy()
    R.destroy()

if __name__ == '__legion_main__':
    main()
