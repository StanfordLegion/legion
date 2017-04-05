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
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
legion_dir = os.path.join(root_dir, "runtime", "legion")

ffi = cffi.FFI()
ffi.cdef(r"""
  typedef unsigned long long legion_lowlevel_id_t;

  typedef struct legion_runtime_t { void *impl; } legion_runtime_t;
  typedef struct legion_context_t { void *impl; } legion_context_t;
  typedef struct legion_task_t { void *impl; } legion_task_t;
  typedef struct legion_physical_region_t { void *impl; } legion_physical_region_t;

  typedef struct legion_processor_t {
    legion_lowlevel_id_t id;
  } legion_processor_t;

  legion_processor_t
  legion_runtime_get_executing_processor(legion_runtime_t runtime,
                                         legion_context_t ctx);

  void
  legion_task_preamble(
    const void *data,
    size_t datalen,
    legion_lowlevel_id_t proc_id,
    legion_task_t *taskptr,
    const legion_physical_region_t **regionptr,
    unsigned * num_regions_ptr,
    legion_context_t * ctxptr,
    legion_runtime_t * runtimeptr);

  void
  legion_task_postamble(
    legion_runtime_t runtime,
    legion_context_t ctx,
    const void *retval,
    size_t retsize);
""")
c = ffi.dlopen(None)

def legion_task(body):
    def wrapper(args, user_data, proc):
        arg_ptr = ffi.new("char[]", bytes(args))
        arg_size = len(args)

        task = ffi.new("legion_task_t *")
        regions = ffi.new("legion_physical_region_t **")
        num_regions = ffi.new("unsigned *")
        context = ffi.new("legion_context_t *")
        runtime = ffi.new("legion_runtime_t *")
        c.legion_task_preamble(arg_ptr, arg_size, proc,
                               task, regions, num_regions, context, runtime)

        value = body(task[0], regions[0], num_regions[0],
                     context[0], runtime[0])
        assert(value is None) # FIXME: Support return values

        c.legion_task_postamble(runtime[0], context[0], ffi.NULL, 0)
    return wrapper

@legion_task
def main_task(task, regions, num_regions, context, runtime):
    print("%x" % c.legion_runtime_get_executing_processor(runtime, context).id)
