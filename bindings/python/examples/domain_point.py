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
from pygion import task, DomainPoint, RW
import numpy as np

@task
def main():
    d1 = DomainPoint([1])
    d2 = DomainPoint([1, 2])
    d3 = DomainPoint([1, 2, 3])

    print(d1, repr(d1), d1.point)
    print(d2, repr(d2), d2.point)
    print(d3, repr(d3), d3.point)

    assert np.array_equal(d1.point, [1])
    assert np.array_equal(d2.point, [1, 2])
    assert np.array_equal(d3.point, [1, 2, 3])

    assert d1 == DomainPoint([1])
    assert d1 != DomainPoint([0])
    assert d1 != DomainPoint([1, 2])

if __name__ == '__main__':
    main()
