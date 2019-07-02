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
import numpy as np

@task
def main():
    d1 = legion.DomainPoint.create([1])
    d2 = legion.DomainPoint.create([1, 2])
    d3 = legion.DomainPoint.create([1, 2, 3])

    print(d1, repr(d1), d1.point)
    print(d2, repr(d2), d2.point)
    print(d3, repr(d3), d3.point)

    assert np.array_equal(d1.point, [1])
    assert np.array_equal(d2.point, [1, 2])
    assert np.array_equal(d3.point, [1, 2, 3])

    assert d1 == legion.DomainPoint.create([1])
    assert d1 != legion.DomainPoint.create([0])
    assert d1 != legion.DomainPoint.create([1, 2])

if __name__ == '__legion_main__':
    main()
