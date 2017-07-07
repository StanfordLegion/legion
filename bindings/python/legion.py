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
import collections
import numpy
import os
import re
import subprocess
import sys
import threading

_pickle_version = 0 # Use latest Pickle protocol

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
runtime_dir = os.path.join(root_dir, 'runtime')
legion_dir = os.path.join(runtime_dir, 'legion')

header = subprocess.check_output(['gcc', '-I', runtime_dir, '-I', legion_dir, '-E', '-P', os.path.join(legion_dir, 'legion_c.h')])

# Hack: Fix for Ubuntu 16.04 versions of standard library headers:
header = re.sub(r'typedef struct {.+?} max_align_t;', '', header, flags=re.DOTALL)

ffi = cffi.FFI()
ffi.cdef(header)
c = ffi.dlopen(None)

# The Legion context is stored in thread-local storage. This assumes
# that the Python processor maintains the invariant that every task
# corresponds to one and only one thread.
_my = threading.local()

class Context(object):
    __slots__ = ['context', 'runtime', 'task', 'regions', 'current_launch']
    def __init__(self, context, runtime, task, regions):
        self.context = context
        self.runtime = runtime
        self.task = task
        self.regions = regions
        self.current_launch = None
    def begin_launch(self, launch):
        assert self.current_launch == None
        self.current_launch = launch
    def end_launch(self, launch):
        assert self.current_launch is not None
        self.current_launch = None

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
        value_str = ffi.unpack(ffi.cast('char *',value_ptr), value_size)
        value = cPickle.loads(value_str)
        return value

class Type(object):
    __slots__ = ['numpy_type', 'size']

    def __init__(self, numpy_type):
        self.numpy_type = numpy_type
        self.size = numpy.dtype(numpy_type).itemsize

    def __reduce__(self):
        return (Type, (self.numpy_type,))

# Pre-defined Types
float16 = Type(numpy.float16)
float32 = Type(numpy.float32)
float64 = Type(numpy.float64)
int16 = Type(numpy.int16)
int32 = Type(numpy.int32)
int64 = Type(numpy.int64)
uint16 = Type(numpy.uint16)
uint32 = Type(numpy.uint32)
uint64 = Type(numpy.uint64)

class Privilege(object):
    __slots__ = ['read', 'write', 'discard']

    def __init__(self, read=False, write=False, discard=False):
        self.read = read
        self.write = write
        self.discard = discard

    def _fields(self):
        return (self.read, self.write, self.discard)

    def __eq__(self, other):
        return isinstance(other, Privilege) and self._fields() == other._fields()

    def __cmp__(self, other):
        assert isinstance(other, Privilege)
        return self._fields().__cmp__(other._fields())

    def __hash__(self):
        return hash(self._fields())

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
    return Ispace(handle[0])

class Ispace(object):
    __slots__ = ['handle']

    def __init__(self, handle):
        # Important: Copy handle. Do NOT assume ownership.
        self.handle = ffi.new('legion_index_space_t *', handle)

    def __reduce__(self):
        return (_Ispace_unpickle,
                (self.handle[0].tid,
                 self.handle[0].id))

    @staticmethod
    def create(extent, start=None):
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
        handle = c.legion_index_space_create_domain(_my.ctx.runtime, _my.ctx.context, domain)
        return Ispace(handle)

# Hack: Can't pickle static methods.
def _Fspace_unpickle(fspace_id, field_ids, field_types):
    handle = ffi.new('legion_field_space_t *')
    handle[0].id = fspace_id
    return Fspace(handle[0], field_ids, field_types)

class Fspace(object):
    __slots__ = ['handle', 'field_ids', 'field_types']

    def __init__(self, handle, field_ids, field_types):
        # Important: Copy handle. Do NOT assume ownership.
        self.handle = ffi.new('legion_field_space_t *', handle)
        self.field_ids = field_ids
        self.field_types = field_types

    def __reduce__(self):
        return (_Fspace_unpickle,
                (self.handle[0].id,
                 self.field_ids,
                 self.field_types))

    @staticmethod
    def create(fields):
        handle = c.legion_field_space_create(_my.ctx.runtime, _my.ctx.context)
        alloc = c.legion_field_allocator_create(
            _my.ctx.runtime, _my.ctx.context, handle)
        field_ids = {}
        field_types = {}
        for field_name, field_entry in fields.items():
            try:
                field_type, field_id = field_entry
            except TypeError:
                field_type = field_entry
                field_id = ffi.cast('legion_field_id_t', -1) # AUTO_GENERATE_ID
            field_id = c.legion_field_allocator_allocate_field(
                alloc, field_type.size, field_id)
            c.legion_field_id_attach_name(
                _my.ctx.runtime, handle, field_id, field_name, False)
            field_ids[field_name] = field_id
            field_types[field_name] = field_type
        c.legion_field_allocator_destroy(alloc)
        return Fspace(handle, field_ids, field_types)

# Hack: Can't pickle static methods.
def _Region_unpickle(tree_id, ispace, fspace):
    handle = ffi.new('legion_logical_region_t *')
    handle[0].tree_id = tree_id
    handle[0].index_space.tid = ispace.handle[0].tid
    handle[0].index_space.id = ispace.handle[0].id
    handle[0].field_space.id = fspace.handle[0].id

    return Region(handle[0], ispace, fspace)

class Region(object):
    __slots__ = ['handle', 'ispace', 'fspace',
                 'instances', 'privileges', 'instance_wrappers']

    def __init__(self, handle, ispace, fspace):
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

    @staticmethod
    def create(ispace, fspace):
        if not isinstance(ispace, Ispace):
            ispace = Ispace.create(ispace)
        if not isinstance(fspace, Fspace):
            fspace = Fspace.create(fspace)
        handle = c.legion_logical_region_create(
            _my.ctx.runtime, _my.ctx.context, ispace.handle[0], fspace.handle[0])
        result = Region(handle, ispace, fspace)
        for field_name in fspace.field_ids.keys():
            result.set_privilege(field_name, RW)
        return result

    def destroy(self):
        # This is not something you want to have happen in a
        # destructor, since regions may outlive the lifetime of the handle.
        c.legion_logical_region_destroy(
            _my.ctx.runtime, _my.ctx.context, self.handle[0])
        # Clear out references. Technically unnecessary but avoids abuse.
        del self.instance_wrappers
        del self.instances
        del self.handle
        del self.ispace
        del self.fspace

    def set_privilege(self, field_name, privilege):
        assert field_name not in self.privileges
        self.privileges[field_name] = privilege

    def set_instance(self, field_name, instance, privilege=None):
        assert field_name not in self.instances
        self.instances[field_name] = instance
        if privilege is not None:
            assert field_name not in self.privileges
            self.privileges[field_name] = privilege

    def map_inline(self):
        fields_by_privilege = collections.defaultdict(set)
        for field_name, privilege in self.privileges.iteritems():
            fields_by_privilege[privilege].add(field_name)
        for privilege, field_names  in fields_by_privilege.iteritems():
            launcher = c.legion_inline_launcher_create_logical_region(
                self.handle[0],
                privilege._legion_privilege(), 0, # EXCLUSIVE
                self.handle[0],
                0, False, 0, 0)
            for field_name in field_names:
                c.legion_inline_launcher_add_field(
                    launcher, self.fspace.field_ids[field_name], True)
            instance = c.legion_inline_launcher_execute(
                _my.ctx.runtime, _my.ctx.context, launcher)
            for field_name in field_names:
                self.set_instance(field_name, instance)

    def __getattr__(self, field_name):
        if field_name in self.fspace.field_ids:
            if field_name not in self.instances:
                if self.privileges[field_name] is None:
                    raise Exception('Invalid attempt to access field "%s" without privileges' % field_name)
                self.map_inline()
            if field_name not in self.instance_wrappers:
                self.instance_wrappers[field_name] = RegionField(
                    self, field_name)
            return self.instance_wrappers[field_name]
        else:
            raise AttributeError()

class RegionField(numpy.ndarray):
    # NumPy requires us to implement __new__ for subclasses of ndarray:
    # https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    def __new__(cls, region, field_name):
        accessor = RegionField._get_accessor(region, field_name)
        initializer = RegionField._get_array_initializer(region, field_name, accessor)
        obj = numpy.asarray(initializer).view(cls)

        obj.accessor = accessor
        return obj

    @staticmethod
    def _get_accessor(region, field_name):
        # Note: the accessor needs to be kept alive, to make sure to
        # save the result of this function in an instance variable.
        instance = region.instances[field_name]
        return c.legion_physical_region_get_field_accessor_generic(
            instance, region.fspace.field_ids[field_name])

    @staticmethod
    def _get_base_and_stride(region, field_name, accessor):
        domain = c.legion_index_space_get_domain(
            _my.ctx.runtime, region.ispace.handle[0])
        dim = domain.dim
        rect = getattr(c, 'legion_domain_get_rect_{}d'.format(dim))(domain)
        subrect = ffi.new('legion_rect_{}d_t *'.format(dim))
        offsets = ffi.new('legion_byte_offset_t[]', dim)

        base_ptr = getattr(c, 'legion_accessor_generic_raw_rect_ptr_{}d'.format(dim))(
            accessor, rect, subrect, offsets)
        assert base_ptr
        for i in xrange(dim):
            assert subrect[0].lo.x[i] == rect.lo.x[i]
            assert subrect[0].hi.x[i] == rect.hi.x[i]
        assert offsets[0].offset == region.fspace.field_types[field_name].size

        shape = tuple(rect.hi.x[i] - rect.lo.x[i] + 1 for i in xrange(dim))
        strides = tuple(offsets[i].offset for i in xrange(dim))

        return base_ptr, shape, strides

    @staticmethod
    def _get_array_initializer(region, field_name, accessor):
        base_ptr, shape, strides = RegionField._get_base_and_stride(
            region, field_name, accessor)
        field_type = region.fspace.field_types[field_name]

        # Numpy doesn't know about CFFI pointers, so we have to cast
        # this to a Python long before we can hand it off to Numpy.
        base_ptr = long(ffi.cast("size_t", base_ptr))

        return _RegionNdarray(shape, field_type, base_ptr, strides, False)

# This is a dummy object that is only used as an initializer for the
# RegionField object above. It is thrown away as soon as the
# RegionField is constructed.
class _RegionNdarray(object):
    __slots__ = ['__array_interface__']
    def __init__(self, shape, field_type, base_ptr, strides, read_only):
        # See: https://docs.scipy.org/doc/numpy/reference/arrays.interface.html
        self.__array_interface__ = {
            'version': 3,
            'shape': shape,
            'typestr': numpy.dtype(field_type.numpy_type).str,
            'data': (base_ptr, read_only),
            'strides': strides,
        }

class ExternTask(object):
    __slots__ = ['privileges', 'task_id']

    def __init__(self, task_id, privileges=None):
        if privileges is not None:
            privileges = [(x if x is not None else N) for x in privileges]
        self.privileges = privileges

        assert isinstance(task_id, int)
        self.task_id = task_id

    def __call__(self, *args):
        return self.spawn_task(*args)

    def spawn_task(self, *args):
        launcher = _TaskLauncher(
            task_id=self.task_id,
            privileges=self.privileges,
            calling_convention=None)
        return launcher.spawn_task(*args)

def extern_task(**kwargs):
    return ExternTask(**kwargs)

class Task (object):
    __slots__ = ['body', 'privileges', 'leaf', 'inner', 'idempotent', 'task_id']

    def __init__(self, body, privileges=None,
                 leaf=False, inner=False, idempotent=False,
                 register=True):
        self.body = body
        if privileges is not None:
            privileges = [(x if x is not None else N) for x in privileges]
        self.privileges = privileges
        self.leaf = bool(leaf)
        self.inner = bool(inner)
        self.idempotent = bool(idempotent)
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

    def spawn_task(self, *args):
        launcher = _TaskLauncher(
            task_id=self.task_id,
            privileges=self.privileges,
            calling_convention='python')
        return launcher.spawn_task(*args)

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
            req = 0
            for i, arg in zip(range(len(args)), args):
                if isinstance(arg, Region):
                    assert req < num_regions[0] and req < len(self.privileges)
                    instance = raw_regions[0][req]
                    req += 1

                    priv = self.privileges[i]
                    if hasattr(priv, 'fields'):
                        assert set(priv.fields) <= set(arg.fspace.field_ids.keys())
                    for name, fid in arg.fspace.field_ids.items():
                        if not hasattr(priv, 'fields') or name in priv.fields:
                            arg.set_instance(name, instance, priv)
            assert req == num_regions[0]

        # Build context.
        ctx = Context(context[0], runtime[0], task[0], regions)

        # Ensure that we're not getting tangled up in another
        # thread. There should be exactly one thread per task.
        try:
            _my.ctx
        except AttributeError:
            pass
        else:
            raise Exception('thread-local context already set')

        # Store context in thread-local storage.
        _my.ctx = ctx

        # Execute task body.
        result = self.body(*args)

        # Encode result in Pickle format.
        if result is not None:
            result_str = cPickle.dumps(result, protocol=_pickle_version)
            result_size = len(result_str)
            result_ptr = ffi.new('char[]', result_size)
            ffi.buffer(result_ptr, result_size)[:] = result_str
        else:
            result_size = 0
            result_ptr = ffi.NULL

        # Execute postamble.
        c.legion_task_postamble(runtime[0], context[0], result_ptr, result_size)

        # Clear thread-local storage.
        del _my.ctx

    def register(self):
        assert(self.task_id is None)

        execution_constraints = c.legion_execution_constraint_set_create()
        c.legion_execution_constraint_set_add_processor_constraint(
            execution_constraints, c.PY_PROC)

        layout_constraints = c.legion_task_layout_constraint_set_create()
        # FIXME: Add layout constraints

        options = ffi.new('legion_task_config_options_t *')
        options[0].leaf = self.leaf
        options[0].inner = self.inner
        options[0].idempotent = self.idempotent

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

class _TaskLauncher(object):
    __slots__ = ['task_id', 'privileges', 'calling_convention']

    def __init__(self, task_id, privileges, calling_convention):
        self.task_id = task_id
        self.privileges = privileges
        self.calling_convention = calling_convention

    def spawn_task(self, *args):
        assert(isinstance(_my.ctx, Context))

        # Do any required preprocessing on arguments.
        args = [
            arg._legion_preprocess_task_argument()
            if hasattr(arg, '_legion_preprocess_task_argument') else arg
            for arg in args]

        # Encode task arguments.
        task_args = ffi.new('legion_task_argument_t *')
        if self.calling_convention == 'python':
            arg_str = cPickle.dumps(args, protocol=_pickle_version)
            task_args_buffer = ffi.new('char[]', arg_str)
            task_args[0].args = task_args_buffer
            task_args[0].arglen = len(arg_str)
        else:
            # FIXME: External tasks need a dedicated calling
            # convention to permit the passing of task arguments.
            task_args[0].args = ffi.NULL
            task_args[0].arglen = 0

        # Construct the task launcher.
        launcher = c.legion_task_launcher_create(
            self.task_id, task_args[0], c.legion_predicate_true(), 0, 0)
        for i, arg in zip(range(len(args)), args):
            if isinstance(arg, Region):
                assert i < len(self.privileges)
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
            elif self.calling_convention is None:
                # FIXME: Task arguments aren't being encoded AT ALL;
                # at least throw an exception so that the user knows
                raise Exception('External tasks do not support non-region arguments')

        # Launch the task.
        result = c.legion_task_launcher_execute(
            _my.ctx.runtime, _my.ctx.context, launcher)
        c.legion_task_launcher_destroy(launcher)

        # Build future of result.
        future = Future(result)
        c.legion_future_destroy(result)
        return future

class _IndexValue(object):
    __slots__ = ['value']
    def __init__(self, value):
        self.value = value
    def __int__(self):
        return self.value
    def __index__(self):
        return self.value
    def __str__(self):
        return str(self.value)
    def __repr__(self):
        return repr(self.value)
    def _legion_preprocess_task_argument(self):
        return self.value

class IndexLaunch(object):
    __slots__ = ['extent']
    def __init__(self, extent):
        assert len(extent) == 1
        self.extent = extent
    def __iter__(self):
        _my.ctx.begin_launch(self)
        point = _IndexValue(None)
        for i in xrange(self.extent[0]):
            point.value = i
            yield point
        _my.ctx.end_launch(self)
        self.launch()
    def launch(self):
        print('TODO: launch')
