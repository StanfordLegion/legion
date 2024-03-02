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
from pygion import task, Domain, RW

@task
def main():
    d = Domain(10)

    print(d, d.bounds)

    t = 0
    for x in d:
        print(x)
        t += int(x)
    assert t == 45

    d2 = Domain([3, 3], [1, 1])

    print(d2, d2.bounds)

    t2 = 0
    for x in d2:
        print(x)
        t2 += x[0] * x[1]
    assert t2 == 36


    d3 = Domain([4, 5, 6], [-1, 0, 1])

    print(d3, d3.bounds)

    t3 = 0
    for x in d3:
        t3 += x[0] * x[1] * x[2]
    assert t3 == 420

if __name__ == '__main__':
    main()
