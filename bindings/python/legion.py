#!/usr/bin/env python

# Copyright 2018 Stanford University
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

from __future__ import absolute_import, division, print_function, unicode_literals

import cffi
try:
    import cPickle as pickle
except ImportError:
    import pickle
import collections
import itertools
import numpy
import os
import re
import subprocess
import sys
import threading

# Python 3.x compatibility:
try:
    long # Python 2
except NameError:
    long = int  # Python 3

try:
    xrange # Python 2
except NameError:
    xrange = range # Python 3

try:
    zip_longest = itertools.izip_longest # Python 2
except:
    zip_longest = itertools.zip_longest # Python 3

_pickle_version = pickle.HIGHEST_PROTOCOL # Use latest Pickle protocol

def find_legion_header():
    def try_prefix(prefix_dir):
        legion_h_path = os.path.join(prefix_dir, 'legion.h')
        if os.path.exists(legion_h_path):
            return prefix_dir, legion_h_path

    # For in-source builds, find the header relative to the bindings
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    runtime_dir = os.path.join(root_dir, 'runtime')
    result = try_prefix(runtime_dir)
    if result:
        return result

    # If this was installed to a non-standard prefix, we might be able
    # to guess from the directory structures
    if os.path.basename(root_dir) == 'lib':
        include_dir = os.path.join(os.path.dirname(root_dir), 'include')
        result = try_prefix(include_dir)
        if result:
            return result

    # Otherwise we have to hope that Legion is installed in a standard location
    result = try_prefix('/usr/include')
    if result:
        return result

    result = try_prefix('/usr/local/include')
    if result:
        return result

    raise Exception('Unable to locate legion.h header file')

prefix_dir, legion_h_path = find_legion_header()
header = subprocess.check_output(['gcc', '-I', prefix_dir, '-E', '-P', legion_h_path]).decode('utf-8')

# Hack: Fix for Ubuntu 16.04 versions of standard library headers:
header = re.sub(r'typedef struct {.+?} max_align_t;', '', header, flags=re.DOTALL)

ffi = cffi.FFI()
ffi.cdef(header)
c = ffi.dlopen(None)
max_legion_python_tasks = 1000000
next_legion_task_id = c.legion_runtime_generate_library_task_ids(
                        c.legion_runtime_get_runtime(),
                        os.path.basename(__file__).encode('utf-8'),
                        max_legion_python_tasks)
max_legion_task_id = next_legion_task_id + max_legion_python_tasks

# Returns true if this module is running inside of a Legion
# executable. If false, then other Legion functionality should not be
# expected to work.
def inside_legion_executable():
    try:
        c.legion_get_current_time_in_micros()
    except AttributeError:
        return False
    else:
        return True

def input_args(filter_runtime_options=False):
    raw_args = c.legion_runtime_get_input_args()

    args = []
    for i in range(raw_args.argc):
        args.append(ffi.string(raw_args.argv[i]))

    if filter_runtime_options:
        i = 1 # Skip program name

        prefixes = ['-lg:', '-hl:', '-realm:', '-ll:', '-cuda:', '-numa:',
                    '-dm:', '-bishop:']
        while i < len(args):
            match = False
            for prefix in prefixes:
                if args[i].startswith(prefix):
                    match = True
                    break
            if args[i] == '-level':
                match = True
            if args[i] == '-logfile':
                match = True
            if match:
                args.pop(i)
                args.pop(i) # Assume that every option has an argument
                continue
            i += 1
    return args

# The Legion context is stored in thread-local storage. This assumes
# that the Python processor maintains the invariant that every task
# corresponds to one and only one thread.
_my = threading.local()

class Context(object):
    __slots__ = ['context_root', 'context', 'runtime_root', 'runtime',
                 'task_root', 'task', 'regions', 'current_launch']
    def __init__(self, context_root, runtime_root, task_root, regions):
        self.context_root = context_root
        self.context = self.context_root[0]
        self.runtime_root = runtime_root
        self.runtime = self.runtime_root[0]
        self.task_root = task_root
        self.task = self.task_root[0]
        self.regions = regions
        self.current_launch = None
    def begin_launch(self, launch):
        assert self.current_launch == None
        self.current_launch = launch
    def end_launch(self, launch):
        assert self.current_launch == launch
        self.current_launch = None

class DomainPoint(object):
    __slots__ = ['impl']
    def __init__(self, value):
        assert(isinstance(value, _IndexValue))
        self.impl = ffi.new('legion_domain_point_t *')
        self.impl[0].dim = 1
        self.impl[0].point_data[0] = int(value)
    def raw_value(self):
        return self.impl[0]

class Domain(object):
    __slots__ = ['impl']
    def __init__(self, extent, start=None):
        if start is not None:
            assert len(start) == len(extent)
        else:
            start = [0 for _ in extent]
        assert 1 <= len(extent) <= 3
        rect = ffi.new('legion_rect_{}d_t *'.format(len(extent)))
        for i in xrange(len(extent)):
            rect[0].lo.x[i] = start[i]
            rect[0].hi.x[i] = start[i] + extent[i] - 1
        self.impl = getattr(c, 'legion_domain_from_rect_{}d'.format(len(extent)))(rect[0])
    def raw_value(self):
        return self.impl

class Future(object):
    __slots__ = ['handle', 'value_type', 'argument_number']
    def __init__(self, value, value_type=None, argument_number=None):
        if value is None:
            self.handle = None
        elif isinstance(value, Future):
            self.handle = c.legion_future_copy(value.handle)
            if value_type is None:
                value_type = value.value_type
        elif value_type is not None:
            value_ptr = ffi.new(ffi.getctype(value_type.cffi_type, '*'), value)
            value_size = ffi.sizeof(value_type.cffi_type)
            self.handle = c.legion_future_from_untyped_pointer(_my.ctx.runtime, value_ptr, value_size)
        else:
            value_str = pickle.dumps(value, protocol=_pickle_version)
            value_size = len(value_str)
            value_ptr = ffi.new('char[]', value_size)
            ffi.buffer(value_ptr, value_size)[:] = value_str
            self.handle = c.legion_future_from_untyped_pointer(_my.ctx.runtime, value_ptr, value_size)

        self.value_type = value_type
        self.argument_number = argument_number

    @staticmethod
    def from_cdata(value, *args, **kwargs):
        result = Future(None, *args, **kwargs)
        result.handle = c.legion_future_copy(value)
        return result

    @staticmethod
    def from_buffer(value, *args, **kwargs):
        result = Future(None, *args, **kwargs)
        result.handle = c.legion_future_from_untyped_pointer(_my.ctx.runtime, ffi.from_buffer(value), len(value))
        return result

    def __del__(self):
        if self.handle is not None:
            c.legion_future_destroy(self.handle)

    def __reduce__(self):
        if self.argument_number is None:
            raise Exception('Cannot pickle a Future except when used as a task argument')
        return (Future, (None, self.value_type, self.argument_number))

    def resolve_handle(self):
        if self.handle is None and self.argument_number is not None:
            self.handle = c.legion_future_copy(
                c.legion_task_get_future(_my.ctx.task, self.argument_number))

    def get(self):
        self.resolve_handle()

        if self.handle is None:
            return
        if self.value_type is None:
            value_ptr = c.legion_future_get_untyped_pointer(self.handle)
            value_size = c.legion_future_get_untyped_size(self.handle)
            assert value_size > 0
            value_str = ffi.unpack(ffi.cast('char *', value_ptr), value_size)
            value = pickle.loads(value_str)
            return value
        else:
            expected_size = ffi.sizeof(self.value_type.cffi_type)

            value_ptr = c.legion_future_get_untyped_pointer(self.handle)
            value_size = c.legion_future_get_untyped_size(self.handle)
            assert value_size == expected_size
            value = ffi.cast(ffi.getctype(self.value_type.cffi_type, '*'), value_ptr)[0]
            return value

    def get_buffer(self):
        self.resolve_handle()

        if self.handle is None:
            return
        value_ptr = c.legion_future_get_untyped_pointer(self.handle)
        value_size = c.legion_future_get_untyped_size(self.handle)
        return ffi.buffer(value_ptr, value_size)

class FutureMap(object):
    __slots__ = ['handle']
    def __init__(self, handle):
        self.handle = c.legion_future_map_copy(handle)

    def __del__(self):
        c.legion_future_map_destroy(self.handle)

    def __getitem__(self, point):
        domain_point = DomainPoint(_IndexValue(point))
        return Future.from_cdata(c.legion_future_map_get_future(self.handle, domain_point.raw_value()))

class Type(object):
    __slots__ = ['numpy_type', 'cffi_type', 'size']

    def __init__(self, numpy_type, cffi_type):
        self.numpy_type = numpy_type
        self.cffi_type = cffi_type
        self.size = numpy.dtype(numpy_type).itemsize

    def __reduce__(self):
        return (Type, (self.numpy_type, self.cffi_type))

# Pre-defined Types
float16 = Type(numpy.float16, 'short float')
float32 = Type(numpy.float32, 'float')
float64 = Type(numpy.float64, 'double')
int16 = Type(numpy.int16, 'int16_t')
int32 = Type(numpy.int32, 'int32_t')
int64 = Type(numpy.int64, 'int64_t')
uint16 = Type(numpy.uint16, 'uint16_t')
uint32 = Type(numpy.uint32, 'uint32_t')
uint64 = Type(numpy.uint64, 'uint64_t')

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
def _Ispace_unpickle(ispace_tid, ispace_id, ispace_type_tag):
    handle = ffi.new('legion_index_space_t *')
    handle[0].tid = ispace_tid
    handle[0].id = ispace_id
    handle[0].type_tag = ispace_type_tag
    return Ispace(handle[0])

class Ispace(object):
    __slots__ = ['handle']

    def __init__(self, handle):
        # Important: Copy handle. Do NOT assume ownership.
        self.handle = ffi.new('legion_index_space_t *', handle)

    def __reduce__(self):
        return (_Ispace_unpickle,
                (self.handle[0].tid,
                 self.handle[0].id,
                 self.handle[0].type_tag))

    @staticmethod
    def create(extent, start=None):
        domain = Domain(extent, start=start).raw_value()
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
                _my.ctx.runtime, handle, field_id, field_name.encode('utf-8'), False)
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
            _my.ctx.runtime, _my.ctx.context, ispace.handle[0], fspace.handle[0], False)
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
        domain = c.legion_index_space_get_domain(
            _my.ctx.runtime, region.ispace.handle[0])
        dim = domain.dim
        get_accessor = getattr(c, 'legion_physical_region_get_field_accessor_array_{}d'.format(dim))
        return get_accessor(instance, region.fspace.field_ids[field_name])

    @staticmethod
    def _get_base_and_stride(region, field_name, accessor):
        domain = c.legion_index_space_get_domain(
            _my.ctx.runtime, region.ispace.handle[0])
        dim = domain.dim
        rect = getattr(c, 'legion_domain_get_rect_{}d'.format(dim))(domain)
        subrect = ffi.new('legion_rect_{}d_t *'.format(dim))
        offsets = ffi.new('legion_byte_offset_t[]', dim)

        base_ptr = getattr(c, 'legion_accessor_array_{}d_raw_rect_ptr'.format(dim))(
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
    __slots__ = ['privileges', 'calling_convention', 'task_id']

    def __init__(self, task_id, privileges=None):
        if privileges is not None:
            privileges = [(x if x is not None else N) for x in privileges]
        self.privileges = privileges
        self.calling_convention = None
        assert isinstance(task_id, int)
        self.task_id = task_id

    def __call__(self, *args):
        return self.spawn_task(*args)

    def spawn_task(self, *args):
        if _my.ctx.current_launch:
            return _my.ctx.current_launch.spawn_task(self, *args)
        return TaskLaunch().spawn_task(self, *args)

def extern_task(**kwargs):
    return ExternTask(**kwargs)

def get_qualname(fn):
    # Python >= 3.3 only
    try:
        return fn.__qualname__.split('.')
    except AttributeError:
        pass

    # Python < 3.3
    try:
        import qualname
        return qualname.qualname(fn).split('.')
    except ImportError:
        pass

    # Hack: Issue error if we're wrapping a class method and failed to
    # get the qualname
    import inspect
    context = [x[0].f_code.co_name for x in inspect.stack()
               if '__module__' in x[0].f_code.co_names and
               inspect.getmodule(x[0].f_code).__name__ != __name__]
    if len(context) > 0:
        raise Exception('To use a task defined in a class, please upgrade to Python >= 3.3 or install qualname (e.g. pip install qualname)')

    return [fn.__name__]

class Task (object):
    __slots__ = ['body', 'privileges', 'leaf', 'inner', 'idempotent', 'calling_convention', 'task_id', 'registered']

    def __init__(self, body, privileges=None,
                 leaf=False, inner=False, idempotent=False,
                 register=True, top_level=False):
        self.body = body
        if privileges is not None:
            privileges = [(x if x is not None else N) for x in privileges]
        self.privileges = privileges
        self.leaf = bool(leaf)
        self.inner = bool(inner)
        self.idempotent = bool(idempotent)
        self.calling_convention = 'python'
        self.task_id = None
        if register:
            self.register(top_level)

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
        if _my.ctx.current_launch:
            return _my.ctx.current_launch.spawn_task(self, *args)
        return TaskLaunch().spawn_task(self, *args)

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
        if c.legion_task_get_is_index_space(task[0]):
            arg_ptr = ffi.cast('char *', c.legion_task_get_local_args(task[0]))
            arg_size = c.legion_task_get_local_arglen(task[0])
        else:
            arg_ptr = ffi.cast('char *', c.legion_task_get_args(task[0]))
            arg_size = c.legion_task_get_arglen(task[0])

        if arg_size > 0 and c.legion_task_get_depth(task[0]) > 0:
            args = pickle.loads(ffi.unpack(arg_ptr, arg_size))
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
        ctx = Context(context, runtime, task, regions)

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
            result_str = pickle.dumps(result, protocol=_pickle_version)
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

    def register(self, top_level_task):
        assert(self.task_id is None)
        if not top_level_task:
            global next_legion_task_id
            task_id = next_legion_task_id 
            next_legion_task_id += 1
            # If we ever hit this then we need to allocate more task IDs
            assert task_id < max_legion_task_id 
        else:
            task_id = 1 # Predefined value for the top-level task

        execution_constraints = c.legion_execution_constraint_set_create()
        c.legion_execution_constraint_set_add_processor_constraint(
            execution_constraints, c.PY_PROC)

        layout_constraints = c.legion_task_layout_constraint_set_create()
        # FIXME: Add layout constraints

        options = ffi.new('legion_task_config_options_t *')
        options[0].leaf = self.leaf
        options[0].inner = self.inner
        options[0].idempotent = self.idempotent

        qualname = get_qualname(self.body)
        task_name = ('%s.%s' % (self.body.__module__, '.'.join(qualname)))

        c_qualname_comps = [ffi.new('char []', comp.encode('utf-8')) for comp in qualname]
        c_qualname = ffi.new('char *[]', c_qualname_comps)

        c.legion_runtime_register_task_variant_python_source_qualname(
            c.legion_runtime_get_runtime(),
            task_id,
            task_name.encode('utf-8'),
            False, # Global
            execution_constraints,
            layout_constraints,
            options[0],
            self.body.__module__.encode('utf-8'),
            c_qualname,
            len(qualname),
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

    def preprocess_args(self, args):
        return [
            arg._legion_preprocess_task_argument()
            if hasattr(arg, '_legion_preprocess_task_argument') else arg
            for arg in args]

    def gather_futures(self, args):
        normal = []
        futures = []
        for arg in args:
            if isinstance(arg, Future):
                arg = Future(arg, argument_number=len(futures))
                futures.append(arg)
            normal.append(arg)
        return normal, futures

    def encode_args(self, args):
        task_args = ffi.new('legion_task_argument_t *')
        task_args_buffer = None
        if self.calling_convention == 'python':
            arg_str = pickle.dumps(args, protocol=_pickle_version)
            task_args_buffer = ffi.new('char[]', arg_str)
            task_args[0].args = task_args_buffer
            task_args[0].arglen = len(arg_str)
        else:
            # FIXME: External tasks need a dedicated calling
            # convention to permit the passing of task arguments.
            task_args[0].args = ffi.NULL
            task_args[0].arglen = 0
        # WARNING: Need to return the interior buffer or else it will be GC'd
        return task_args, task_args_buffer

    def spawn_task(self, *args):
        assert(isinstance(_my.ctx, Context))

        args = self.preprocess_args(args)
        args, futures = self.gather_futures(args)
        task_args, _ = self.encode_args(args)

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
            elif isinstance(arg, Future):
                c.legion_task_launcher_add_future(launcher, arg.handle)
            elif self.calling_convention is None:
                # FIXME: Task arguments aren't being encoded AT ALL;
                # at least throw an exception so that the user knows
                raise Exception('External tasks do not support non-region arguments')

        # Launch the task.
        result = c.legion_task_launcher_execute(
            _my.ctx.runtime, _my.ctx.context, launcher)
        c.legion_task_launcher_destroy(launcher)

        # Build future of result.
        future = Future.from_cdata(result)
        c.legion_future_destroy(result)
        return future

class _IndexLauncher(_TaskLauncher):
    __slots__ = ['task_id', 'privileges', 'calling_convention',
                 'domain', 'local_args', 'future_map']

    def __init__(self, task_id, privileges, calling_convention, domain):
        super(_IndexLauncher, self).__init__(
            task_id, privileges, calling_convention)
        self.domain = domain
        self.local_args = c.legion_argument_map_create()
        self.future_map = None

    def __del__(self):
        c.legion_argument_map_destroy(self.local_args)

    def spawn_task(self, *args):
        raise Exception('IndexLaunch does not support spawn_task')

    def attach_local_args(self, index, *args):
        point = DomainPoint(index)
        args = self.preprocess_args(args)
        task_args, _ = self.encode_args(args)
        c.legion_argument_map_set_point(
            self.local_args, point.raw_value(), task_args[0], False)

    def launch(self):
        # All arguments are passed as local, so global is NULL.
        global_args = ffi.new('legion_task_argument_t *')
        global_args[0].args = ffi.NULL
        global_args[0].arglen = 0

        # Construct the task launcher.
        launcher = c.legion_index_launcher_create(
            self.task_id, self.domain.raw_value(),
            global_args[0], self.local_args,
            c.legion_predicate_true(), False, 0, 0)

        # Launch the task.
        result = c.legion_index_launcher_execute(
            _my.ctx.runtime, _my.ctx.context, launcher)
        c.legion_index_launcher_destroy(launcher)

        # Build future (map) of result.
        self.future_map = FutureMap(result)
        c.legion_future_map_destroy(result)

class TaskLaunch(object):
    __slots__ = []
    def spawn_task(self, task, *args):
        launcher = _TaskLauncher(
            task_id=task.task_id,
            privileges=task.privileges,
            calling_convention=task.calling_convention)
        return launcher.spawn_task(*args)

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

class _FuturePoint(object):
    __slots__ = ['launcher', 'point', 'future']
    def __init__(self, launcher, point):
        self.launcher = launcher
        self.point = point
        self.future = None
    def get(self):
        if self.launcher.future_map is None:
            raise Exception('Cannot retrieve a future from an index launch until the launch is complete')

        self.future = self.launcher.future_map[self.point]

        # Clear launcher and point
        del self.launcher
        del self.point

        return self.future.get()

class IndexLaunch(object):
    __slots__ = ['extent', 'domain', 'launcher', 'point',
                 'saved_task', 'saved_args']

    def __init__(self, extent):
        assert len(extent) == 1
        self.extent = extent
        self.domain = Domain(extent)
        self.launcher = None
        self.point = None
        self.saved_task = None
        self.saved_args = None

    def __iter__(self):
        _my.ctx.begin_launch(self)
        self.point = _IndexValue(None)
        for i in xrange(self.extent[0]):
            self.point.value = i
            yield self.point
        _my.ctx.end_launch(self)
        self.launch()

    def ensure_launcher(self, task):
        if self.launcher is None:
            self.launcher = _IndexLauncher(
                task_id=task.task_id,
                privileges=task.privileges,
                calling_convention=task.calling_convention,
                domain=self.domain)

    def check_compatibility(self, task, *args):
        # The tasks in a launch must conform to the following constraints:
        #   * Only one task can be launched.
        #   * The arguments must be compatible:
        #       * At a given argument position, the value must always
        #         be a region, or always not.
        #       * If a region, the value must be symbolic (i.e. able
        #         to be analyzed as a function of the index expression).

        if self.saved_task is None:
            self.saved_task = task
        if task != self.saved_task:
            raise Exception('An IndexLaunch may contain only one task launch')

        if self.saved_args is None:
            self.saved_args = args
        for arg, saved_arg in zip_longest(args, self.saved_args):
            # TODO: Add support for region arguments
            if isinstance(arg, Region) or isinstance(arg, RegionField):
                raise Exception('TODO: Support region arguments to an IndexLaunch')

    def spawn_task(self, task, *args):
        self.ensure_launcher(task)
        self.check_compatibility(task, *args)
        self.launcher.attach_local_args(self.point, *args)
        # TODO: attach region args
        # TODO: attach future args
        return _FuturePoint(self.launcher, int(self.point))

    def launch(self):
        self.launcher.launch()

@task(leaf=True)
def _dummy_task():
    return 1

def execution_fence(block=False):
    c.legion_runtime_issue_execution_fence(_my.ctx.runtime, _my.ctx.context)
    if block:
        _dummy_task().get()

class Tunable(object):
    # FIXME: Deduplicate this with DefaultMapper::DefaultTunables
    NODE_COUNT = 0
    LOCAL_CPUS = 1
    LOCAL_GPUS = 2
    LOCAL_IOS = 3
    LOCAL_OMPS = 4
    LOCAL_PYS = 5
    GLOBAL_CPUS = 6
    GLOBAL_GPUS = 7
    GLOBAL_IOS = 8
    GLOBAL_OMPS = 9
    GLOBAL_PYS = 10

    @staticmethod
    def select(tunable_id):
        result = c.legion_runtime_select_tunable_value(
            _my.ctx.runtime, _my.ctx.context, tunable_id, 0, 0)
        future = Future.from_cdata(result, value_type=uint64)
        c.legion_future_destroy(result)
        return future

