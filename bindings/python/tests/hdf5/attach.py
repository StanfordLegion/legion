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

import h5py
import pygion
from pygion import acquire, attach_hdf5, task, Fspace, Ispace, Region, R
import numpy

def generate_hdf5_file(filename, dims):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('x', dims, dtype='i4')
        f.create_dataset('uu', dims, dtype='i4')
        f.create_dataset('z', dims, dtype='i4')
        f.create_dataset('w', dims, dtype='i4')

@task(privileges=[R])
def print_region(R):
    print(R.x)

@task
def main():
    R = Region([4, 4], {'x': pygion.int32, 'uu': pygion.int32, 'z': pygion.int32, 'w': pygion.int32})

    generate_hdf5_file('test.h5', [4, 4])

    with attach_hdf5(R, 'test.h5', {'x': 'x', 'uu': 'uu', 'z': 'z', 'w': 'w'}, pygion.file_read_only):
        with acquire(R, ['x', 'uu', 'z']):
            print_region(R)

if __name__ == '__main__':
    main()
