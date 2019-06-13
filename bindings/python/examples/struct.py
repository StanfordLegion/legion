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
import numpy

# Define a custom struct type.
legion.ffi.cdef(r'''
typedef struct mystruct {
  int x;
  double y;
  char z;
} mystruct;
''')
mystruct_np = numpy.dtype([('x', numpy.intc), ('y', numpy.double), ('z', numpy.byte)])
mystruct = legion.Type(mystruct_np, 'mystruct')

print(mystruct.size)

@task
def main():
    myvalue_root = legion.ffi.new('mystruct *')
    myvalue = myvalue_root[0]
    myvalue.x = 123
    myvalue.y = 3.14
    myvalue.z = str('A')

    # Make a future with the custom struct type.
    g = legion.Future(myvalue, mystruct)
    print("value of g.get() is %s" % g.get())
    assert g.get().x == 123

    # Make a region with the custom struct type.
    R = legion.Region.create([4], {'myvalue': mystruct})

    # Hack: This is what we need to do to coerce a CFFI value into a NumPy value.
    R.myvalue[0] = legion.ffi.buffer(myvalue_root)

    print(R.myvalue[0])

if __name__ == '__legion_main__':
    main()
