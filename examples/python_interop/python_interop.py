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

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
legion_dir = os.path.join(root_dir, "runtime", "legion")

ffi = cffi.FFI()
ffi.cdef(r"""
void *dlopen(const char *filename, int flags);
char *dlerror(void);
void *dlsym(void *handle, const char *symbol);
int dlclose(void *handle);
""")
c = ffi.dlopen(None)

c.dlerror() # Clear any previous error
handle = c.dlopen(ffi.NULL, ffi.RTLD_NOW)
print("handle", handle)
if not handle:
    print("error", ffi.string(c.dlerror()))

ffi.cdef(r"""
  typedef unsigned long long legion_lowlevel_id_t;

  typedef struct legion_runtime_t { void *impl; } legion_runtime_t;
  typedef struct legion_context_t { void *impl; } legion_context_t;

  typedef struct legion_processor_t {
    legion_lowlevel_id_t id;
  } legion_processor_t;

  legion_processor_t
  legion_runtime_get_executing_processor(legion_runtime_t runtime,
                                         legion_context_t ctx);
""")

c.dlerror()
legion_runtime_get_executing_processor = c.dlsym(handle, "legion_runtime_get_executing_processor")
print("dlsym returned", legion_runtime_get_executing_processor)
if not legion_runtime_get_executing_processor:
    print("error", ffi.string(c.dlerror()))

legion_runtime_get_executing_processor = ffi.cast(
    "legion_processor_t (*)(legion_runtime_t, legion_context_t)",
    legion_runtime_get_executing_processor)

def main_task(args, user_data, proc):
    print("hello from task1 args=({!r}) userdata=({!r}) proc=({:x})".format(
        args, user_data, proc))
    legion_runtime_get_executing_processor()
