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

class Context(object):
    __slots__ = ['context', 'runtime', 'task', 'regions']
    def __init__(self, context, runtime, task, regions):
        self.context = context
        self.runtime = runtime
        self.task = task
        self.regions = regions

class Task (object):
    __slots__ = ['body', 'task_id']

    def __init__(self, body, register=True):
        self.body = body
        self.task_id = None
        if register:
            self.register()

    def __call__(self, *args):
        # Hack: This entrypoint needs to be able to handle both being
        # called in user code (to launch a task) and as the task
        # wrapper when the task itself executes. Unfortunately isn't a
        # good way to disentangle these. Detect if we're in the task
        # wrapper case by checking the number and types of arguments.
        if len(args) == 3 and \
           isinstance(args[0], bytearray) and \
           isinstance(args[1], bytearray) and \
           isinstance(args[2], long):
            self.execute_task(*args)
        else:
            self.spawn_task(*args)

    def spawn_task(self, ctx, *args):
        assert(isinstance(ctx, Context))
        assert(len(args) == 0)
        print("spawn_task")

        task_args = ffi.new("legion_task_argument_t *")
        task_args[0].args = ffi.NULL
        task_args[0].arglen = 0
        launcher = c.legion_task_launcher_create(
            self.task_id, task_args[0], c.legion_predicate_true(), 0, 0)
        c.legion_task_launcher_execute(ctx.runtime, ctx.context, launcher)
        c.legion_task_launcher_destroy(launcher)

    def execute_task(self, args, user_data, proc):
        arg_ptr = ffi.new("char[]", bytes(args))
        arg_size = len(args)

        task = ffi.new("legion_task_t *")
        raw_regions = ffi.new("legion_physical_region_t **")
        num_regions = ffi.new("unsigned *")
        context = ffi.new("legion_context_t *")
        runtime = ffi.new("legion_runtime_t *")
        c.legion_task_preamble(
            arg_ptr, arg_size, proc,
            task, raw_regions, num_regions, context, runtime)

        regions = []
        for i in xrange(num_regions[0]):
            regions.append(raw_regions[0][i])

        ctx = Context(context[0], runtime[0], task[0], regions)

        value = self.body(ctx)
        assert(value is None) # FIXME: Support return values

        c.legion_task_postamble(runtime[0], context[0], ffi.NULL, 0)

    def register(self):
        assert(self.task_id is None)

        execution_constraints = c.legion_execution_constraint_set_create()
        c.legion_execution_constraint_set_add_processor_constraint(
            execution_constraints, c.PY_PROC)

        layout_constraints = c.legion_task_layout_constraint_set_create()
        # FIXME: Add layout constraints

        options = ffi.new("legion_task_config_options_t *")
        options[0].leaf = False
        options[0].inner = False
        options[0].idempotent = False

        task_id = c.legion_runtime_preregister_task_variant_python_source(
            ffi.cast("legion_task_id_t", -1), # AUTO_GENERATE_ID
            '%s.%s' % (self.body.__module__, self.body.__name__),
            execution_constraints,
            layout_constraints,
            options[0],
            self.body.__module__,
            self.body.__name__,
            ffi.NULL,
            0)

        c.legion_execution_constraint_set_destroy(execution_constraints)
        c.legion_task_layout_constraint_set_destroy(layout_constraints)

        self.task_id = task_id
        return self

def task(body=None, **kwargs):
    if body is None:
        return lambda body: task(body, **kwargs)
    return Task(body, **kwargs)
