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
from pygion import task, R, RW, WD, N, Reduce
import numpy

@task
def main():
    assert R == R
    assert RW == RW
    assert WD == WD
    assert N == N
    assert Reduce('+') == Reduce('+')

    assert not (R == RW)
    assert not (R == WD)
    assert not (R == N)
    assert not (R == Reduce('+'))

    assert R != RW
    assert R != WD
    assert R != N
    assert Reduce('+') != Reduce('*')

    assert not (R != R)
    assert not (RW != RW)
    assert not (WD != WD)
    assert not (N != N)

    assert R('x') == R('x')
    assert R('x') != R('y')
    assert R('x') != R

    assert R + R == R
    assert R + RW == RW
    assert R + WD == WD

    assert not (R + RW == R)

    assert R('x') + R('y') == R('x', 'y')
    assert R('x') + R('y') != R('y', 'x')

    assert R('x', 'y') + RW('y') == R('x') + RW('y')
    assert R('x') + RW('x', 'y') == RW('x', 'y')

    assert R + Reduce('+') == RW
    assert Reduce('+') + Reduce('*') == RW
    assert Reduce('+') + WD == WD

    assert R('x', 'y') + Reduce('+', 'y', 'z') == R('x') + RW('y') + Reduce('+', 'z')

    print(R('x'))
    print(R('x') + RW('y'))
    print(RW('x') + RW('y'))

if __name__ == '__main__':
    main()
