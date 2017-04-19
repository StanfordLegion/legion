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
import cPickle
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

class Future(object):
    __slots__ = ['handle']
    def __init__(self, handle):
        self.handle = c.legion_future_copy(handle)

    def __del__(self):
        c.legion_future_destroy(self.handle)

    def get(self):
        value_ptr = c.legion_future_get_untyped_pointer(self.handle)
        value_size = c.legion_future_get_untyped_size(self.handle)
        assert value_size > 0
        value_str = ffi.unpack(value_ptr, value_size)
        value = cPickle.loads(value_str)
        return value

class Type(object):
    __slots__ = ['size']
    def __init__(self, size):
        self.size = size
double = Type(8)

class Ispace(object):
    __slots__ = ['ctx', 'handle']

    def __init__(self, ctx, handle):
        self.ctx = ctx
        self.handle = handle

    @staticmethod
    def create(ctx, extent, start=None):
        if start is not None:
            assert len(start) == len(extent)
        else:
            start = [0 for _ in extent]
        assert 1 <= len(extent) <= 3
        rect = ffi.new("legion_rect_{}d_t *".format(len(extent)))
        for i in xrange(len(extent)):
            rect[0].lo.x[i] = start[i]
            rect[0].hi.x[i] = start[i] + extent[i] - 1
        domain = getattr(c, "legion_domain_from_rect_{}d".format(len(extent)))(rect[0])
        handle = c.legion_index_space_create_domain(ctx.runtime, ctx.context, domain)
        return Ispace(ctx, handle)

class Fspace(object):
    __slots__ = ['ctx', 'handle', 'field_ids']

    def __init__(self, ctx, handle, field_ids):
        self.ctx = ctx
        self.handle = handle
        self.field_ids = field_ids

    @staticmethod
    def create(ctx, fields):
        handle = c.legion_field_space_create(ctx.runtime, ctx.context)
        alloc = c.legion_field_allocator_create(
            ctx.runtime, ctx.context, handle)
        field_ids = {}
        for field_name, field_type in fields.items():
            field_id = c.legion_field_allocator_allocate_field(
                alloc, field_type.size,
                ffi.cast("legion_field_id_t", -1)) # AUTO_GENERATE_ID
            c.legion_field_id_attach_name(
                ctx.runtime, handle, field_id, field_name, False)
            field_ids[field_name] = field_id
        c.legion_field_allocator_destroy(alloc)
        return Fspace(ctx, handle, field_ids)

class Region(object):
    __slots__ = ['ctx', 'handle', 'ispace', 'fspace']

    def __init__(self, ctx, handle, ispace, fspace):
        self.ctx = ctx
        self.handle = handle
        self.ispace = ispace
        self.fspace = fspace

    @staticmethod
    def create(ctx, ispace, fspace):
        if not isinstance(ispace, Ispace):
            ispace = Ispace.create(ctx, ispace)
        if not isinstance(fspace, Fspace):
            fspace = Fspace.create(ctx, fspace)
        handle = c.legion_logical_region_create(
            ctx.runtime, ctx.context, ispace.handle, fspace.handle)
        return Region(ctx, handle, ispace, fspace)

    def __getattr__(self, field_name):
        if field_name in self.fspace.field_ids:
            return RegionField(self.ctx, self, field_name)
        else:
            raise AttributeError()

class RegionField(object):
    def __init__(self, ctx, region, field_name):
        pass # FIXME

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
            return self.execute_task(*args)
        else:
            return self.spawn_task(*args)

    def spawn_task(self, ctx, *args):
        assert(isinstance(ctx, Context))

        # Encode arguments in Pickle format.
        arg_str = bytes(cPickle.dumps(args))
        task_args = ffi.new("legion_task_argument_t *")
        task_args[0].args = ffi.new("char[]", arg_str)
        task_args[0].arglen = len(arg_str)

        # Launch the task.
        launcher = c.legion_task_launcher_create(
            self.task_id, task_args[0], c.legion_predicate_true(), 0, 0)
        result = c.legion_task_launcher_execute(
            ctx.runtime, ctx.context, launcher)
        c.legion_task_launcher_destroy(launcher)

        # Build future of result.
        future = Future(result)
        c.legion_future_destroy(result)
        return future

    def execute_task(self, raw_args, user_data, proc):
        raw_arg_ptr = ffi.new("char[]", bytes(raw_args))
        raw_arg_size = len(raw_args)

        # Execute preamble to obtain Legion API context.
        task = ffi.new("legion_task_t *")
        raw_regions = ffi.new("legion_physical_region_t **")
        num_regions = ffi.new("unsigned *")
        context = ffi.new("legion_context_t *")
        runtime = ffi.new("legion_runtime_t *")
        c.legion_task_preamble(
            raw_arg_ptr, raw_arg_size, proc,
            task, raw_regions, num_regions, context, runtime)

        # Decode arguments from Pickle format.
        arg_ptr = ffi.cast("char *", c.legion_task_get_args(task[0]))
        arg_size = c.legion_task_get_arglen(task[0])
        if arg_size > 0:
            arg_str = ffi.unpack(arg_ptr, arg_size)
            args = cPickle.loads(arg_str)
        else:
            args = ()

        # Unpack regions.
        regions = []
        for i in xrange(num_regions[0]):
            regions.append(raw_regions[0][i])

        # Build context.
        ctx = Context(context[0], runtime[0], task[0], regions)

        # Execute task body.
        result = self.body(ctx, *args)

        # Encode result in Pickle format.
        if result is not None:
            result_str = bytes(cPickle.dumps(result))
            result_size = len(result_str)
            result_ptr = ffi.new("char[]", result_size)
        else:
            result_size = 0
            result_ptr = ffi.NULL

        # Execute postamble.
        c.legion_task_postamble(runtime[0], context[0], result_ptr, result_size)

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
