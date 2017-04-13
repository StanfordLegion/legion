#!/usr/bin/env python

# Copyright 2017 Stanford University
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

import cffi
import os
import subprocess
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
runtime_dir = os.path.join(root_dir, "runtime")
legion_dir = os.path.join(runtime_dir, "legion")

header = subprocess.check_output(["gcc", "-I", runtime_dir, "-I", legion_dir, "-E", "-P", os.path.join(legion_dir, "legion_c.h")])

ffi = cffi.FFI()
ffi.cdef(header)
c = ffi.dlopen(None)

def legion_task(body):
    def wrapper(args, user_data, proc):
        arg_ptr = ffi.new("char[]", bytes(args))
        arg_size = len(args)

        task = ffi.new("legion_task_t *")
        raw_regions = ffi.new("legion_physical_region_t **")
        num_regions = ffi.new("unsigned *")
        context = ffi.new("legion_context_t *")
        runtime = ffi.new("legion_runtime_t *")
        c.legion_task_preamble(arg_ptr, arg_size, proc,
                               task, raw_regions, num_regions, context, runtime)

        regions = []
        for i in xrange(num_regions[0]):
            regions.append(raw_regions[0][i])

        value = body(task[0], regions, context[0], runtime[0])
        assert(value is None) # FIXME: Support return values

        c.legion_task_postamble(runtime[0], context[0], ffi.NULL, 0)
    return wrapper

@legion_task
def main_task(task, regions, context, runtime):
    print("%x" % c.legion_runtime_get_executing_processor(runtime, context).id)
