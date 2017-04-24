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

_pickle_version = 0 # Use latest Pickle protocol

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
runtime_dir = os.path.join(root_dir, 'runtime')
legion_dir = os.path.join(runtime_dir, 'legion')

header = subprocess.check_output(['gcc', '-I', runtime_dir, '-I', legion_dir, '-E', '-P', os.path.join(legion_dir, 'legion_c.h')])

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

    def __reduce__(self):
        return (Type, (self.size,))

# Pre-defined Types
double = Type(8)

class Privilege(object):
    __slots__ = ['read', 'write', 'discard']

    def __init__(self, read=False, write=False, discard=False):
        self.read = read
        self.write = write
        self.discard = discard

    def __call__(self, fields):
        return PrivilegeFields(self, fields)

    def _legion_privilege(self):
        bits = 0
        if self.discard:
            assert self.write
            bits |= 2 # WRITE_DISCARD
        else:
            if self.write: bits = 7 # READ_WRITE
            elif self.read: bits = 1 # READ_ONLY
        return bits

class PrivilegeFields(Privilege):
    __slots__ = ['read', 'write', 'discard', 'fields']

    def __init__(self, privilege, fields):
        Privilege.__init__(self, privilege.read, privilege.write, privilege.discard)
        self.fields = fields

# Pre-defined Privileges
N = Privilege()
R = Privilege(read=True)
RO = Privilege(read=True)
RW = Privilege(read=True, write=True)
WD = Privilege(write=True, discard=True)

# Hack: Can't pickle static methods.
def _Ispace_unpickle(ispace_tid, ispace_id):
    handle = ffi.new('legion_index_space_t *')
    handle[0].tid = ispace_tid
    handle[0].id = ispace_id
    return Ispace(None, handle[0])

class Ispace(object):
    __slots__ = ['ctx', 'handle']

    def __init__(self, ctx, handle):
        self.ctx = ctx
        # Important: Copy handle. Do NOT assume ownership.
        self.handle = ffi.new('legion_index_space_t *', handle)

    def __reduce__(self):
        return (_Ispace_unpickle,
                (self.handle[0].tid,
                 self.handle[0].id))

    def _legion_set_context(self, ctx):
        self.ctx = ctx

    @staticmethod
    def create(ctx, extent, start=None):
        if start is not None:
            assert len(start) == len(extent)
        else:
            start = [0 for _ in extent]
        assert 1 <= len(extent) <= 3
        rect = ffi.new('legion_rect_{}d_t *'.format(len(extent)))
        for i in xrange(len(extent)):
            rect[0].lo.x[i] = start[i]
            rect[0].hi.x[i] = start[i] + extent[i] - 1
        domain = getattr(c, 'legion_domain_from_rect_{}d'.format(len(extent)))(rect[0])
        handle = c.legion_index_space_create_domain(ctx.runtime, ctx.context, domain)
        return Ispace(ctx, handle)

# Hack: Can't pickle static methods.
def _Fspace_unpickle(fspace_id, field_ids, field_types):
    handle = ffi.new('legion_field_space_t *')
    handle[0].id = fspace_id
    return Fspace(None, handle[0], field_ids, field_types)

class Fspace(object):
    __slots__ = ['ctx', 'handle', 'field_ids', 'field_types']

    def __init__(self, ctx, handle, field_ids, field_types):
        self.ctx = ctx
        # Important: Copy handle. Do NOT assume ownership.
        self.handle = ffi.new('legion_field_space_t *', handle)
        self.field_ids = field_ids
        self.field_types = field_types

    def __reduce__(self):
        return (_Fspace_unpickle,
                (self.handle[0].id,
                 self.field_ids,
                 self.field_types))

    def _legion_set_context(self, ctx):
        self.ctx = ctx

    @staticmethod
    def create(ctx, fields):
        handle = c.legion_field_space_create(ctx.runtime, ctx.context)
        alloc = c.legion_field_allocator_create(
            ctx.runtime, ctx.context, handle)
        field_ids = {}
        field_types = {}
        for field_name, field_type in fields.items():
            field_id = c.legion_field_allocator_allocate_field(
                alloc, field_type.size,
                ffi.cast('legion_field_id_t', -1)) # AUTO_GENERATE_ID
            c.legion_field_id_attach_name(
                ctx.runtime, handle, field_id, field_name, False)
            field_ids[field_name] = field_id
            field_types[field_name] = field_type
        c.legion_field_allocator_destroy(alloc)
        return Fspace(ctx, handle, field_ids, field_types)

# Hack: Can't pickle static methods.
def _Region_unpickle(tree_id, ispace, fspace):
    handle = ffi.new('legion_logical_region_t *')
    handle[0].tree_id = tree_id
    handle[0].index_space.tid = ispace.handle[0].tid
    handle[0].index_space.id = ispace.handle[0].id
    handle[0].field_space.id = fspace.handle[0].id

    return Region(None, handle[0], ispace, fspace)

class Region(object):
    __slots__ = ['ctx', 'handle', 'ispace', 'fspace',
                 'instances', 'privileges', 'instance_wrappers']

    def __init__(self, ctx, handle, ispace, fspace):
        self.ctx = ctx
        # Important: Copy handle. Do NOT assume ownership.
        self.handle = ffi.new('legion_logical_region_t *', handle)
        self.ispace = ispace
        self.fspace = fspace
        self.instances = {}
        self.privileges = {}
        self.instance_wrappers = {}

    def __reduce__(self):
        return (_Region_unpickle,
                (self.handle[0].tree_id,
                 self.ispace,
                 self.fspace))

    def _legion_set_context(self, ctx):
        self.ctx = ctx

    @staticmethod
    def create(ctx, ispace, fspace):
        if not isinstance(ispace, Ispace):
            ispace = Ispace.create(ctx, ispace)
        if not isinstance(fspace, Fspace):
            fspace = Fspace.create(ctx, fspace)
        handle = c.legion_logical_region_create(
            ctx.runtime, ctx.context, ispace.handle[0], fspace.handle[0])
        return Region(ctx, handle, ispace, fspace)

    def set_instance(self, field_name, instance, privilege):
        assert field_name not in self.instances
        self.instances[field_name] = instance
        self.privileges[field_name] = privilege

    def __getattr__(self, field_name):
        print(self.fspace.field_ids)
        print(self.instances)
        if field_name in self.fspace.field_ids and \
           field_name in self.instances:
            if field_name not in self.instance_wrappers:
                self.instance_wrappers[field_name] = RegionField(
                    self.ctx, self, field_name)
            return self.instance_wrappers[field_name]
        else:
            raise AttributeError()

class RegionField(object):
    __slots__ = ['ctx', 'region', 'field_name', 'accessor']

    def __init__(self, ctx, region, field_name):
        self.ctx = ctx
        self.region = region
        self.field_name = field_name
        self.accessor = None

    def _as_ndarray(self):
        assert self.accessor is None

        domain = c.legion_index_space_get_domain(
            self.ctx.runtime, self.region.ispace.handle[0])
        dim = domain.dim
        rect = getattr(c, 'legion_domain_get_rect_{}d'.format(dim))(domain)

        subrect = ffi.new('legion_rect_{}d_t *'.format(dim))
        offsets = ffi.new('legion_byte_offset_t[]', dim)

        instance = self.region.instances[self.field_name]
        self.accessor = c.legion_physical_region_get_field_accessor_generic(
            instance, self.region.fspace.field_ids[self.field_name])
        base_ptr = getattr(c, 'legion_accessor_generic_raw_rect_ptr_{}d'.format(dim))(
            self.accessor, rect, subrect, offsets)
        assert base_ptr
        for i in xrange(dim):
            assert subrect[0].lo.x[i] == rect.lo.x[i]
            assert subrect[0].hi.x[i] == rect.hi.x[i]
        assert offsets[0].offset == self.region.fspace.field_types[self.field_name].size

class Task (object):
    __slots__ = ['body', 'privileges', 'task_id']

    def __init__(self, body, privileges=None, register=True):
        self.body = body
        if privileges is not None:
            privileges = [(x if x is not None else N) for x in privileges]
        self.privileges = privileges
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
        arg_str = cPickle.dumps(args, protocol=_pickle_version)
        task_args = ffi.new('legion_task_argument_t *')
        task_args_buffer = ffi.new('char[]', arg_str)
        task_args[0].args = task_args_buffer
        task_args[0].arglen = len(arg_str)

        # Construct the task launcher.
        launcher = c.legion_task_launcher_create(
            self.task_id, task_args[0], c.legion_predicate_true(), 0, 0)
        if self.privileges is not None:
            assert(len(self.privileges) == len(args))
        for i, arg in zip(range(len(args)), args):
            if isinstance(arg, Region):
                priv = self.privileges[i]
                req = c.legion_task_launcher_add_region_requirement_logical_region(
                    launcher, arg.handle[0],
                    priv._legion_privilege(),
                    0, # EXCLUSIVE
                    arg.handle[0], 0, False)
                if hasattr(priv, 'fields'):
                    assert set(priv.fields) <= set(arg.fspace.field_ids.keys())
                for name, fid in arg.fspace.field_ids.items():
                    if not hasattr(priv, 'fields') or name in priv.fields:
                        c.legion_task_launcher_add_field(
                            launcher, req, fid, True)

        # Launch the task.
        result = c.legion_task_launcher_execute(
            ctx.runtime, ctx.context, launcher)
        c.legion_task_launcher_destroy(launcher)

        # Build future of result.
        future = Future(result)
        c.legion_future_destroy(result)
        return future

    def execute_task(self, raw_args, user_data, proc):
        raw_arg_ptr = ffi.new('char[]', bytes(raw_args))
        raw_arg_size = len(raw_args)

        # Execute preamble to obtain Legion API context.
        task = ffi.new('legion_task_t *')
        raw_regions = ffi.new('legion_physical_region_t **')
        num_regions = ffi.new('unsigned *')
        context = ffi.new('legion_context_t *')
        runtime = ffi.new('legion_runtime_t *')
        c.legion_task_preamble(
            raw_arg_ptr, raw_arg_size, proc,
            task, raw_regions, num_regions, context, runtime)

        # Decode arguments from Pickle format.
        arg_ptr = ffi.cast('char *', c.legion_task_get_args(task[0]))
        arg_size = c.legion_task_get_arglen(task[0])
        if arg_size > 0:
            args = cPickle.loads(ffi.unpack(arg_ptr, arg_size))
        else:
            args = ()

        # Unpack regions.
        regions = []
        for i in xrange(num_regions[0]):
            regions.append(raw_regions[0][i])

        # Unpack physical regions.
        if self.privileges is not None:
            assert num_regions[0] == len(self.privileges)
            req = 0
            for i, arg in zip(range(len(args)), args):
                if isinstance(arg, Region):
                    instance = raw_regions[0][req]
                    req += 1

                    priv = self.privileges[i]
                    if hasattr(priv, 'fields'):
                        assert set(priv.fields) <= set(arg.fspace.field_ids.keys())
                    for name, fid in arg.fspace.field_ids.items():
                        if not hasattr(priv, 'fields') or name in priv.fields:
                            arg.set_instance(name, instance, priv)

        # Build context.
        ctx = Context(context[0], runtime[0], task[0], regions)

        # Update context in any arguments that require it.
        for arg in args:
            if hasattr(arg, '_legion_set_context'):
                arg._legion_set_context(ctx)

        # Execute task body.
        result = self.body(ctx, *args)

        # Encode result in Pickle format.
        if result is not None:
            result_str = cPickle.dumps(result, protocol=_pickle_version)
            result_size = len(result_str)
            result_ptr = ffi.new('char[]', result_size)
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

        options = ffi.new('legion_task_config_options_t *')
        options[0].leaf = False
        options[0].inner = False
        options[0].idempotent = False

        task_id = c.legion_runtime_preregister_task_variant_python_source(
            ffi.cast('legion_task_id_t', -1), # AUTO_GENERATE_ID
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
