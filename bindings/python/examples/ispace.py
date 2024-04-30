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
from pygion import task, Ispace, RW

@task
def main():
    d = Ispace(10)
    t = 0
    for x in d:
        print(x)
        t += int(x)
    assert t == 45

    d2 = Ispace([3, 3], [1, 1])
    t2 = 0
    for x in d2:
        print(x)
        t2 += x[0] * x[1]
    assert t2 == 36

if __name__ == '__main__':
    main()
