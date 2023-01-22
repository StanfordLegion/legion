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
from pygion import task, Future, Region, WD
import numpy

# Define a custom struct type.
pygion.ffi.cdef(r'''
typedef struct mystruct {
  int x;
  double y;
  int8_t z;
} mystruct;
''')
mystruct_np = numpy.dtype([('x', numpy.intc), ('y', numpy.double), ('z', numpy.byte)], align=True)
mystruct = pygion.Type(mystruct_np, 'mystruct')

@task(privileges=[WD])
def region_with_struct(R):
    R.myvalue[0] = (123, 3.14, 65)
    print(R.myvalue[0])

@task
def main():
    myvalue_root = pygion.ffi.new('mystruct *')
    myvalue = myvalue_root[0]
    myvalue.x = 123
    myvalue.y = 3.14
    myvalue.z = 65

    # Make a future with the custom struct type.
    g = Future(myvalue, mystruct)
    print("value of g.get() is %s" % g.get())
    assert g.get().x == 123

    # Make a region with the custom struct type.
    R = Region([4], {'myvalue': mystruct})
    region_with_struct(R)

if __name__ == '__main__':
    main()
