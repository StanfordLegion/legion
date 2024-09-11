#!/usr/bin/env python3

# Copyright 2024 Stanford University
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

try:
    import cPickle as pickle
except ImportError:
    import pickle
import collections
import contextlib
from io import StringIO
import itertools
import math
import numpy
import os
import re
import subprocess
import sys
import threading
import weakref

# Python 3.x compatibility:
try:
    long # Python 2
except NameError:
    long = int  # Python 3

try:
    basestring # Python 2
except NameError:
    basestring = str # Python 3

try:
    xrange # Python 2
except NameError:
    xrange = range # Python 3

try:
    imap = itertools.imap # Python 2
except:
    imap = map # Python 3

try:
    zip_longest = itertools.izip_longest # Python 2
except:
    zip_longest = itertools.zip_longest # Python 3

_pickle_version = pickle.HIGHEST_PROTOCOL # Use latest Pickle protocol

import legion_top
from legion_cffi import ffi, lib as c

_max_dim = None
for dim in range(1, 9):
    try:
        getattr(c, 'legion_domain_get_rect_{}d'.format(dim))
    except AttributeError:
        break
    _max_dim = dim
assert _max_dim is not None, 'Unable to detect LEGION_MAX_DIM'

AUTO_GENERATE_ID = c.legion_auto_generate_id()

# Duplicate enum values from legion_config.h since CFFI isn't smart
# enough to parse them directly.

EXTERNAL_HDF5_FILE = 1

NO_ACCESS       = 0x00000000
READ_PRIV       = 0x00000001
READ_ONLY       = 0x00000001 # READ_PRIV
WRITE_PRIV      = 0x00000002
REDUCE_PRIV     = 0x00000004
REDUCE          = 0x00000004 # REDUCE_PRIV
READ_WRITE      = 0x00000007 # READ_PRIV | WRITE_PRIV | REDUCE_PRIV
DISCARD_MASK    = 0x10000000 # For marking we don't need inputs
WRITE_ONLY      = 0x10000002 # WRITE_PRIV | DISCARD_MASK
WRITE_DISCARD   = 0x10000007 # READ_WRITE | DISCARD_MASK

# Note: don't use __file__ here, it may return either .py or .pyc and cause
# non-deterministic failures.
library_name = "pygion.py"
max_legion_python_tasks = 1000000
next_legion_task_id = c.legion_runtime_generate_library_task_ids(
                        c.legion_runtime_get_runtime(),
                        library_name.encode('utf-8'),
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

input_args = legion_top.input_args

is_script = c.legion_runtime_has_context()

# The Legion context is stored in thread-local storage. This assumes
# that the Python processor maintains the invariant that every task
# corresponds to one and only one thread.
_my = threading.local()

global_task_registration_barrier = None

class Context(object):
    __slots__ = ['context_root', 'context', 'runtime_root', 'runtime',
                 'task_root', 'task', 'regions',
                 'owned_objects', 'current_launch', 'next_trace_id']
    def __init__(self, context_root, runtime_root, task_root, regions):
        self.context_root = context_root
        self.context = self.context_root[0]
        self.runtime_root = runtime_root
        self.runtime = self.runtime_root[0]
        self.task_root = task_root
        self.task = self.task_root[0]
        self.regions = regions
        self.owned_objects = []
        self.current_launch = None
        self.next_trace_id = 0
    def track_object(self, obj):
        self.owned_objects.append(weakref.ref(obj))
    def begin_launch(self, launch):
        assert self.current_launch == None
        self.current_launch = launch
    def end_launch(self, launch):
        assert self.current_launch == launch
        self.current_launch = None

# Hack: Can't pickle static methods.
def _DomainPoint_unpickle(values):
    return DomainPoint(values)

class DomainPoint(object):
    __slots__ = [
        'handle',
        '_point', # cached properties
    ]
    def __init__(self, values, **kwargs):
        def parse_kwargs(_handle=None):
            return _handle
        handle = parse_kwargs(**kwargs)

        if values is not None:
            assert handle is None
            try:
                len(values)
            except TypeError:
                values = [values]
            assert 1 <= len(values) <= _max_dim
            self.handle = ffi.new('legion_domain_point_t *')
            self.handle[0].dim = len(values)
            for i, value in enumerate(values):
                self.handle[0].point_data[i] = value
        else:
            # Important: Copy handle. Do NOT assume ownership.
            assert handle is not None
            self.handle = ffi.new('legion_domain_point_t *', handle)

        self._point = None

    def __reduce__(self):
        return (_DomainPoint_unpickle,
                ([self.handle[0].point_data[i] for i in xrange(self.dim)],))

    def __int__(self):
        assert self.dim == 1
        return self.handle[0].point_data[0]

    def __index__(self):
        return self.__int__()

    def __getitem__(self, i):
        assert 0 <= i < self.dim
        return self.handle[0].point_data[i]

    def __eq__(self, other):
        if not isinstance(other, DomainPoint):
            return NotImplemented
        return numpy.array_equal(self.point, other.point)

    def __str__(self):
        if self.dim == 1:
            return str(int(self))
        return str(self.point)

    def __repr__(self):
        return 'DomainPoint({})'.format(self.point)

    @property
    def dim(self):
        return self.handle[0].dim

    @property
    def point(self):
        if self._point is None:
            self._point = self.asarray()
        return self._point

    @staticmethod
    def coerce(value):
        if not isinstance(value, DomainPoint):
            return DomainPoint(value)
        return value

    def raw_value(self):
        return self.handle[0]

    def asarray(self):
        return numpy.frombuffer(
            ffi.buffer(self.handle[0].point_data),
            count=self.dim,
            dtype=numpy.int64)

class Domain(object):
    __slots__ = [
        'handle',
        '_bounds', # cached properties
    ]
    def __init__(self, extent, start=None, **kwargs):
        def parse_kwargs(_handle=None):
            return _handle
        handle = parse_kwargs(**kwargs)

        if extent is not None:
            assert handle is None
            try:
                len(extent)
            except TypeError:
                extent = [extent]
            if start is not None:
                try:
                    len(start)
                except TypeError:
                    start = [start]
                assert len(start) == len(extent)
            else:
                start = [0 for _ in extent]
            assert 1 <= len(extent) <= _max_dim
            rect = ffi.new('legion_rect_{}d_t *'.format(len(extent)))
            for i in xrange(len(extent)):
                rect[0].lo.x[i] = start[i]
                rect[0].hi.x[i] = start[i] + extent[i] - 1
            handle = getattr(c, 'legion_domain_from_rect_{}d'.format(len(extent)))(rect[0])

        # Important: Copy handle. Do NOT assume ownership.
        assert handle is not None
        self.handle = ffi.new('legion_domain_t *', handle)
        self._bounds = None

    @property
    def dim(self):
        return self.handle[0].dim

    @property
    def volume(self):
        return c.legion_domain_get_volume(self.handle[0])

    @property
    def bounds(self):
        if self._bounds is None:
            self._bounds = self.asarray()
        return self._bounds

    @property
    def extent(self):
        bounds = self.bounds
        return bounds[1] - bounds[0] + 1

    @property
    def start(self):
        return self.bounds[0]

    @staticmethod
    def coerce(value):
        if not isinstance(value, Domain):
            return Domain(value)
        return value

    def __iter__(self):
        return imap(
            DomainPoint,
            itertools.product(
                *[xrange(
                    self.handle[0].rect_data[i],
                    self.handle[0].rect_data[i+self.dim] + 1)
                  for i in xrange(self.dim)]))

    def raw_value(self):
        return self.handle[0]

    def asarray(self):
        return numpy.frombuffer(
            ffi.buffer(self.handle[0].rect_data),
            count=self.dim * 2,
            dtype=numpy.int64
        ).reshape((2, self.dim))

class DomainTransform(object):
    __slots__ = ['handle']
    def __init__(self, matrix, **kwargs):
        def parse_kwargs(_handle=None):
            return _handle
        handle = parse_kwargs(**kwargs)

        if matrix is not None:
            assert handle is None
            matrix = numpy.asarray(matrix, dtype=numpy.int64)
            transform = ffi.new('legion_transform_{}x{}_t *'.format(*matrix.shape))
            ffi.buffer(transform[0].trans)[:] = matrix
            handle = getattr(c, 'legion_domain_transform_from_{}x{}'.format(*matrix.shape))(transform[0])

        # Important: Copy handle. Do NOT assume ownership.
        assert handle is not None
        self.handle = ffi.new('legion_domain_transform_t *', handle)

    @staticmethod
    def coerce(value):
        if not isinstance(value, DomainTransform):
            return DomainTransform(value)
        return value

    def raw_value(self):
        return self.handle[0]

class Future(object):
    __slots__ = ['handle', 'value_type', 'argument_number']
    def __init__(self, value, value_type=None, argument_number=None):
        if value is None:
            self.handle = None
        elif isinstance(value, Future):
            value.resolve_handle()
            self.handle = c.legion_future_copy(value.handle)
            if value_type is None:
                value_type = value.value_type
        elif value_type is not None:
            if value_type.size > 0:
                value_ptr = ffi.new(ffi.getctype(value_type.cffi_type, '*'), value)
            else:
                assert value is None
                value_ptr = ffi.NULL
            value_size = value_type.size
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
        elif self.value_type.size == 0:
            c.legion_future_get_void_result(self.handle)
        else:
            expected_size = ffi.sizeof(self.value_type.cffi_type)

            value_ptr = c.legion_future_get_untyped_pointer(self.handle)
            value_size = c.legion_future_get_untyped_size(self.handle)
            assert value_size == expected_size
            value = ffi.cast(ffi.getctype(self.value_type.cffi_type, '*'), value_ptr)[0]
            # Hack: Use closure to keep self alive as long as the value is live.
            if isinstance(value, ffi.CData):
                return ffi.gc(value, lambda x: self)
            return value

    def get_buffer(self):
        self.resolve_handle()

        if self.handle is None:
            return
        value_ptr = c.legion_future_get_untyped_pointer(self.handle)
        value_size = c.legion_future_get_untyped_size(self.handle)
        return ffi.buffer(value_ptr, value_size)

    def is_ready(self):
        self.resolve_handle()

        if self.handle is None:
            return True
        return c.legion_future_is_ready(self.handle)

class FutureMap(object):
    __slots__ = ['handle', 'value_type']
    def __init__(self, handle, value_type=None):
        self.handle = c.legion_future_map_copy(handle)
        self.value_type = value_type

    def __del__(self):
        c.legion_future_map_destroy(self.handle)

    def __getitem__(self, point):
        point = DomainPoint.coerce(point)
        return Future.from_cdata(
            c.legion_future_map_get_future(self.handle, point.raw_value()),
            value_type=self.value_type)

    def wait_all_results(self):
        c.legion_future_map_wait_all_results(self.handle)

_type_cache = {}

class Type(object):
    __slots__ = ['numpy_type', 'cffi_type', 'size']

    def __new__(cls, numpy_type, cffi_type):
        if cffi_type in _type_cache:
            return _type_cache[cffi_type]
        obj = super(Type, cls).__new__(cls)
        _type_cache[cffi_type] = obj
        return obj

    def __init__(self, numpy_type, cffi_type):
        assert (numpy_type is None) == (cffi_type is None)
        self.numpy_type = numpy_type
        self.cffi_type = cffi_type
        self.size = ffi.sizeof(cffi_type) if cffi_type is not None else 0

    def __reduce__(self):
        return (Type, (self.numpy_type, self.cffi_type))

    def __repr__(self):
        return 'Type(%s,%s)' % (repr(self.numpy_type), repr(self.cffi_type))

# Pre-defined Types
void = Type(None, None)
bool_ = Type(numpy.bool_, 'bool')
complex64 = Type(numpy.complex64, 'float _Complex')
complex128 = Type(numpy.complex128, 'double _Complex')
float32 = Type(numpy.float32, 'float')
float64 = Type(numpy.float64, 'double')
int8 = Type(numpy.int8, 'int8_t')
int16 = Type(numpy.int16, 'int16_t')
int32 = Type(numpy.int32, 'int32_t')
int64 = Type(numpy.int64, 'int64_t')
uint8 = Type(numpy.uint8, 'uint8_t')
uint16 = Type(numpy.uint16, 'uint16_t')
uint32 = Type(numpy.uint32, 'uint32_t')
uint64 = Type(numpy.uint64, 'uint64_t')

_type_semantic_tag = 54321 # keep in sync with Regent std_base.t
_type_ids = {
    101: int8,
    102: int16,
    103: int32,
    104: int64,
    105: uint8,
    106: uint16,
    107: uint32,
    108: uint64,
    109: float32,
    110: float64,
    111: bool_,
}

_rect_types = []
_base_id = max(_type_ids)
for dim in xrange(1, _max_dim + 1):
    itype = Type(
        numpy.dtype([('x', numpy.int64, (dim,))], align=True),
        'legion_point_{}d_t'.format(dim))
    globals()["int{}d".format(dim)] = itype
    _type_ids[_base_id + dim] = itype
    rtype = Type(
        numpy.dtype([('lo', numpy.int64, (dim,)), ('hi', numpy.int64, (dim,))], align=True),
        'legion_rect_{}d_t'.format(dim))
    globals()["rect{}d".format(dim)] = rtype
    _rect_types.append(rtype)
_rect_types = frozenset(_rect_types)

def is_rect_type(t):
    return t in _rect_types

_redop_ids = {}
def _fill_redop_ids():
    operators = ['+', '-', '*', '/', 'max', 'min']
    types = [bool_, int8, int16, int32, int64, uint8, uint16, uint32, uint64, None, float32, float64, None, complex64, complex128]
    # LEGION_MAX_APPLICATION_REDOP_ID + 1 is now a new base
    next_id = 1048577
    for operator in operators:
        _redop_ids[operator] = {}
        for type in types:
            if type is not None:
                _redop_ids[operator][type] = next_id
            next_id += 1
_fill_redop_ids()

class Privilege(object):
    __slots__ = ['read', 'write', 'discard', 'reduce', 'fields']

    def __init__(self, read=False, write=False, discard=False, reduce=False, fields=None):
        self.read = read
        self.write = write
        self.discard = discard
        self.reduce = reduce
        self.fields = fields

        if self.fields is not None:
            assert len(self.fields) > 0

        if self.discard:
            assert self.write

    def _fields(self):
        return (self.read, self.write, self.discard, self.reduce, self.fields)

    def __eq__(self, other):
        if not isinstance(other, Privilege):
            return NotImplemented
        return self._fields() == other._fields()

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self._fields())

    def __call__(self, *fields):
        assert self.fields is None
        return Privilege(self.read, self.write, self.discard, self.reduce, fields)

    def __add__(self, other):
        return PrivilegeComposite([self, other])

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.discard:
            priv = 'WD'
        elif self.write:
            priv = 'RW'
        elif self.read:
            priv = 'R'
        elif self.reduce:
            priv = 'Reduce(%s)' % self.reduce
        else:
            priv = 'N'
        if self.fields is not None:
            return '%s(%s)' % (priv, ', '.join(self.fields))
        return priv

    def _legion_privilege(self):
        bits = NO_ACCESS
        if self.reduce:
            assert False
        else:
            if self.write: bits = READ_WRITE
            elif self.read: bits = READ_ONLY
        if self.discard:
            bits |= DISCARD_MASK
        return bits

    def _legion_grouped_privileges(self, fspace):
        if self.fields:
            if not set(self.fields) <= set(fspace.keys()):
                raise Exception(
                    'Privilege fields ({}) are not a subset of fspace fields ({})'.format(
                        ' '.join(self.fields), ' '.join(fspace.keys())))
        fields = fspace.keys() if self.fields is None else self.fields
        if self.reduce:
            redop_ids = collections.OrderedDict()
            for i, field_name in enumerate(fields):
                redop_id = self._legion_redop_id(fspace.field_types[field_name])
                if redop_id not in redop_ids:
                    redop_ids[redop_id] = []
                redop_ids[redop_id].append(field_name)
            return [(self, None, redop_id, redop_fields) for redop_id, redop_fields in redop_ids.items()]
        else:
            return [(self, self._legion_privilege(), None, fields if self.read or self.write or self.reduce else [])]

    def _legion_redop_id(self, field_type):
        return _redop_ids[self.reduce][field_type]

class PrivilegeComposite(object):
    __slots__ = ['privileges']

    def __init__(self, privileges):
        self.privileges = self.normalize(privileges)

    @staticmethod
    def normalize(privileges):
        fields = collections.OrderedDict()
        read_set = set()
        write_set = set()
        discard_set = set()
        reduce_sets = collections.OrderedDict()

        for privilege in privileges:
            privilege_fields = privilege.fields if privilege.fields is not None else [None]
            fields.update([(x, True) for x in privilege_fields])
            if privilege.read:
                read_set.update(privilege_fields)
            if privilege.write:
                write_set.update(privilege_fields)
            if privilege.discard:
                discard_set.update(privilege_fields)
            if privilege.reduce:
                if privilege.reduce not in reduce_sets:
                    reduce_sets[privilege.reduce] = set()
                reduce_sets[privilege.reduce].update(privilege_fields)

        # Reductions combine with read/reduce privileges to upgrade to read-write.
        for op, reduce_set in reduce_sets.items():
            write_set.update(reduce_set & read_set)
            if None in read_set:
                write_set.update(reduce_set)
            for op2, reduce_set2 in reduce_sets.items():
                if op != op2:
                    write_set.update(reduce_set & reduce_set2)
                    if None in reduce_set2:
                        write_set.update(reduce_set)

        # Read/write/discard shadow reduction privileges.
        if None in read_set or None in write_set or None in discard_set:
            reduce_sets = collections.OrderedDict()
        else:
            for reduce_set in reduce_sets.values():
                reduce_set.difference_update(read_set, write_set, discard_set)

        # Discard shadows read/write.
        if None in discard_set:
            read_set = set()
            write_set = set()
            discard_set = set([None])
        else:
            read_set -= discard_set
            write_set -= discard_set

        # Write shadows read.
        if None in write_set:
            read_set = set()
            write_set = set([None])
        else:
            read_set -= write_set

        def filter_set(ctor, field_set):
            if None in field_set:
                return ctor
            return ctor(*filter(lambda x: x in field_set, fields.keys()))

        return tuple(
            ([filter_set(R, read_set)] if len(read_set) > 0 else []) +
            ([filter_set(RW, write_set)] if len(write_set) > 0 else []) +
            ([filter_set(WD, discard_set)] if len(discard_set) > 0 else []) +
            [filter_set(Reduce(op), reduce_set) for op, reduce_set in reduce_sets.items()])

    def __eq__(self, other):
        if len(self.privileges) == 1:
            return other == self.privileges[0]

        if not isinstance(other, PrivilegeComposite):
            return NotImplemented
        return self.privileges == other.privileges

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.privileges)

    def __add__(self, other):
        return PrivilegeComposite(self.privileges + (other,))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return ' + '.join(map(str, self.privileges))

    def _legion_grouped_privileges(self, fspace):
        return [x for privilege in self.privileges for x in privilege._legion_grouped_privileges(fspace)]

# Pre-defined Privileges
N = Privilege()
R = Privilege(read=True)
RO = Privilege(read=True)
RW = Privilege(read=True, write=True)
WD = Privilege(write=True, discard=True)

def Reduce(operator, *fields):
    return Privilege(reduce=operator, fields=fields if len(fields) > 0 else None)

class Disjointness(object):
    __slots__ = ['kind', 'value']

    def __init__(self, kind, value):
        self.kind = kind
        self.value = value

    def __eq__(self, other):
        return isinstance(other, Disjointness) and self.value == other.value

    def __cmp__(self, other):
        assert isinstance(other, Disjointness)
        return self.value.__cmp__(other.value)

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return self.kind

disjoint = Disjointness('disjoint', 0)
aliased = Disjointness('aliased', 1)
compute = Disjointness('compute', 2)
disjoint_complete = Disjointness('disjoint_complete', 3)
aliased_complete = Disjointness('aliased_complete', 4)
compute_complete = Disjointness('compute_complete', 5)
disjoint_incomplete = Disjointness('disjoint_incomplete', 6)
aliased_incomplete = Disjointness('aliased_incomplete', 7)
compute_incomplete = Disjointness('compute_incomplete', 8)

class FileMode(object):
    __slots__ = ['kind', 'value']

    def __init__(self, kind, value):
        self.kind = kind
        self.value = value

    def __eq__(self, other):
        return isinstance(other, FileMode) and self.value == other.value

    def __cmp__(self, other):
        assert isinstance(other, FileMode)
        return self.value.__cmp__(other.value)

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return self.kind

file_read_only = FileMode('read_only', 0)
file_read_write = FileMode('read_write', 1)
file_create = FileMode('create', 2)

class DimOrder(object):
    __slots__ = ['kind', 'order']

    def __init__(self, kind, order):
        self.kind = kind
        self.order = order

    def __str__(self):
        return self.kind

SOA_F = DimOrder('SOA_F', lambda dim: list(range(dim)) + ['F'])
SOA_C = DimOrder('SOA_C', lambda dim: list(reversed(range(dim))) + ['F'])
AOS_F = DimOrder('AOS_F', lambda dim: ['F'] + list(range(dim)))
AOS_C = DimOrder('AOS_C', lambda dim: ['F'] + list(reversed(range(dim))))

class LayoutConstraint(object):
    __slots__ = ['dim', 'order']

    def __init__(self, dim, order):
        self.dim = dim
        self.order = order

    def __str__(self):
        return '%s(dim=%s)' % (self.order, self.dim)

# Hack: Can't pickle static methods.
def _Ispace_unpickle(ispace_tid, ispace_id, ispace_type_tag, owned):
    handle = ffi.new('legion_index_space_t *')
    handle[0].tid = ispace_tid
    handle[0].id = ispace_id
    handle[0].type_tag = ispace_type_tag
    return Ispace(None, _handle=handle[0], _owned=owned)

class Ispace(object):
    __slots__ = [
        'handle', 'owned', 'escaped',
        '_domain', # cached properties
        '__weakref__', # allow weak references
    ]

    def __init__(self, extent, start=None, name=None, **kwargs):
        def parse_kwargs(_handle=None, _owned=False):
            return _handle, _owned
        handle, owned = parse_kwargs(**kwargs)

        if extent is not None:
            assert handle is None
            domain = Domain(extent, start=start).raw_value()
            handle = c.legion_index_space_create_domain(_my.ctx.runtime, _my.ctx.context, domain)
            if name is not None:
                c.legion_index_space_attach_name(_my.ctx.runtime, handle, name.encode('utf-8'), False)
            owned = True

        # Important: Copy handle. Do NOT assume ownership.
        assert handle is not None
        self.handle = ffi.new('legion_index_space_t *', handle)
        self.owned = owned
        self.escaped = False
        self._domain = None

        if self.owned:
            _my.ctx.track_object(self)

    def __del__(self):
        if self.owned and not self.escaped:
            self.destroy()

    def __reduce__(self):
        return (_Ispace_unpickle,
                (self.handle[0].tid,
                 self.handle[0].id,
                 self.handle[0].type_tag,
                 self.owned and self.escaped))

    def __iter__(self):
        return self.domain.__iter__()

    @property
    def domain(self):
        if self._domain is None:
            self._domain = Domain(None, _handle=c.legion_index_space_get_domain(_my.ctx.runtime, self.handle[0]))
        return self._domain

    @property
    def dim(self):
        return self.domain.dim

    @property
    def volume(self):
        return self.domain.volume

    @property
    def bounds(self):
        return self.domain.bounds

    @staticmethod
    def coerce(value):
        if not isinstance(value, Ispace):
            return Ispace(value)
        return value

    def destroy(self):
        assert self.owned and not self.escaped

        # This is not something you want to have happen in a
        # destructor, since fspaces may outlive the lifetime of the handle.
        c.legion_index_space_destroy(
            _my.ctx.runtime, _my.ctx.context, self.handle[0])
        # Clear out references. Technically unnecessary but avoids abuse.
        del self.handle

    def raw_value(self):
        return self.handle[0]

    @staticmethod
    def from_raw(handle):
        return _Ispace_unpickle(handle.tid, handle.id, handle.type_tag, False)

# Hack: Can't pickle static methods.
def _Fspace_unpickle(fspace_id, field_ids, field_types, owned):
    handle = ffi.new('legion_field_space_t *')
    handle[0].id = fspace_id
    return Fspace(None, _handle=handle[0], _field_ids=field_ids, _field_types=field_types, _owned=owned)

class Fspace(object):
    __slots__ = [
        'handle', 'field_ids', 'field_types',
        'owned', 'escaped',
        '__weakref__', # allow weak references
    ]

    def __init__(self, fields, name=None, **kwargs):
        def parse_kwargs(_handle=None, _field_ids=None, _field_types=None, _owned=False):
            return _handle, _field_ids, _field_types, _owned
        handle, field_ids, field_types, owned = parse_kwargs(**kwargs)

        if fields is not None:
            assert handle is None and field_ids is None and field_types is None
            handle = c.legion_field_space_create(_my.ctx.runtime, _my.ctx.context)
            if name is not None:
                c.legion_field_space_attach_name(_my.ctx.runtime, handle, name.encode('utf-8'), False)
            alloc = c.legion_field_allocator_create(
                _my.ctx.runtime, _my.ctx.context, handle)
            field_ids = collections.OrderedDict()
            field_types = collections.OrderedDict()
            for field_name, field_entry in fields.items():
                try:
                    field_type, field_id = field_entry
                except TypeError:
                    field_type = field_entry
                    field_id = ffi.cast('legion_field_id_t', AUTO_GENERATE_ID)
                field_id = c.legion_field_allocator_allocate_field(
                    alloc, field_type.size, field_id)
                c.legion_field_id_attach_name(
                    _my.ctx.runtime, handle, field_id, field_name.encode('utf-8'), False)
                field_ids[field_name] = field_id
                field_types[field_name] = field_type
            c.legion_field_allocator_destroy(alloc)
            owned = True

        # Important: Copy handle. Do NOT assume ownership.
        assert handle is not None and field_ids is not None and field_types is not None
        self.handle = ffi.new('legion_field_space_t *', handle)
        self.field_ids = field_ids
        self.field_types = field_types
        self.owned = owned
        self.escaped = False

        if owned:
            _my.ctx.track_object(self)

    def __del__(self):
        if self.owned and not self.escaped:
            self.destroy()

    def __reduce__(self):
        return (_Fspace_unpickle,
                (self.handle[0].id,
                 self.field_ids,
                 self.field_types,
                 self.owned and self.escaped))

    @staticmethod
    def coerce(value):
        if not isinstance(value, Fspace):
            return Fspace(value)
        return value

    def destroy(self):
        assert self.owned and not self.escaped

        # This is not something you want to have happen in a
        # destructor, since fspaces may outlive the lifetime of the handle.
        c.legion_field_space_destroy(
            _my.ctx.runtime, _my.ctx.context, self.handle[0])
        # Clear out references. Technically unnecessary but avoids abuse.
        del self.handle
        del self.field_ids
        del self.field_types

    def raw_value(self):
        return self.handle[0]

    @staticmethod
    def from_raw(handle):
        size = ffi.new('size_t *')
        raw_field_ids = c.legion_field_space_get_fields(
            _my.ctx.runtime, _my.ctx.context, handle, size)

        names = []
        raw_name = ffi.new('const char **')
        for i in range(size[0]):
            field_id = raw_field_ids[i]
            c.legion_field_id_retrieve_name(
                _my.ctx.runtime, handle, field_id, raw_name)
            names.append(ffi.string(raw_name[0]).decode('utf-8'))

        field_ids = collections.OrderedDict()
        for i, name in enumerate(names):
            field_ids[name] = raw_field_ids[i]

        field_types = collections.OrderedDict()
        raw_tag = ffi.new('const void **')
        tag_size = ffi.new('size_t *')
        for i, name in enumerate(names):
            field_id = field_ids[name]
            ok = c.legion_field_id_retrieve_semantic_information(
                _my.ctx.runtime, handle, field_id, _type_semantic_tag, raw_tag, tag_size, True, True)
            if ok:
                assert tag_size[0] == ffi.sizeof('uint32_t')
                tag = ffi.cast('uint32_t *', raw_tag[0])[0]
                field_types[name] = _type_ids[tag]

        return _Fspace_unpickle(handle.id, field_ids, field_types, False)

    def keys(self):
        return self.field_ids.keys()

# Hack: Can't pickle static methods.
def _Region_unpickle(ispace, fspace, tree_id, owned):
    handle = ffi.new('legion_logical_region_t *')
    handle[0].tree_id = tree_id
    handle[0].index_space = ispace.handle[0]
    handle[0].field_space = fspace.handle[0]

    return Region(ispace, fspace, _handle=handle[0], _owned=owned)

class Region(object):
    __slots__ = [
        'handle', 'ispace', 'fspace', 'parent',
        'instances', 'privileges', 'instance_wrappers',
        'owned', 'escaped',
        '__weakref__', # allow weak references
    ]

    # Make this speak the Type interface
    numpy_type = None
    cffi_type = 'legion_logical_region_t'
    size = ffi.sizeof(cffi_type)

    def __init__(self, ispace, fspace, name=None, **kwargs):
        def parse_kwargs(_handle=None, _parent=None, _owned=False):
            return _handle, _parent, _owned
        handle, parent, owned = parse_kwargs(**kwargs)

        if handle is None:
            assert parent is None
            ispace = Ispace.coerce(ispace)
            fspace = Fspace.coerce(fspace)
            handle = c.legion_logical_region_create(
                _my.ctx.runtime, _my.ctx.context, ispace.raw_value(), fspace.raw_value(), False)
            if name is not None:
                c.legion_logical_region_attach_name(_my.ctx.runtime, handle, name.encode('utf-8'), False)
            owned = True

        # Important: Copy handle. Do NOT assume ownership.
        assert handle is not None
        self.handle = ffi.new('legion_logical_region_t *', handle)
        self.ispace = ispace
        self.fspace = fspace
        self.parent = parent
        self.owned = owned
        self.escaped = False
        self.instances = {}
        self.privileges = {}
        self.instance_wrappers = {}

        if owned:
            _my.ctx.track_object(self)
            for field_name in fspace.field_ids.keys():
                self._set_privilege(field_name, RW)

    def __del__(self):
        if self.owned and not self.escaped:
            self.destroy()

    def __reduce__(self):
        return (_Region_unpickle,
                (self.ispace,
                 self.fspace,
                 self.handle[0].tree_id,
                 self.owned and self.escaped))

    def destroy(self):
        assert self.owned and not self.escaped

        # This is not something you want to have happen in a
        # destructor, since regions may outlive the lifetime of the handle.
        c.legion_logical_region_destroy(
            _my.ctx.runtime, _my.ctx.context, self.handle[0])
        # Clear out references. Technically unnecessary but avoids abuse.
        del self.parent
        del self.instance_wrappers
        del self.instances
        del self.handle
        del self.ispace
        del self.fspace

    @staticmethod
    def from_raw(handle):
        ispace = Ispace.from_raw(handle.index_space)
        fspace = Fspace.from_raw(handle.field_space)
        return _Region_unpickle(ispace, fspace, handle.tree_id, False)

    def raw_value(self):
        return self.handle[0]

    def keys(self):
        return self.fspace.keys()

    def values(self):
        for key in self.keys():
            if key in self.privileges and self.privileges[key] is not None:
                yield getattr(self, key)

    def items(self):
        for key in self.keys():
            if key in self.privileges and self.privileges[key] is not None:
                yield key, getattr(self, key)

    def _set_privilege(self, field_name, privilege):
        assert self.parent is None # not supported on subregions
        assert field_name not in self.privileges
        self.privileges[field_name] = privilege

    def _set_instance(self, field_name, instance, privilege=None):
        assert self.parent is None # not supported on subregions
        assert field_name not in self.instances
        self.instances[field_name] = instance
        if privilege is not None:
            self._set_privilege(field_name, privilege)

    def _clear_instance(self, field_name):
        assert self.parent is None # not supported on subregions
        if field_name in self.instances:
            # FIXME: need to determine when it is safe to destroy the
            # associated instance (may or may not be inline mapped)
            del self.instances[field_name]

    def _map_inline(self):
        assert self.parent is None # FIXME: support inline mapping subregions

        fields_by_privilege = collections.defaultdict(set)
        for field_name, privilege in self.privileges.items():
            fields_by_privilege[privilege].add(field_name)
        for privilege, field_names  in fields_by_privilege.items():
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
                self._set_instance(field_name, instance)

    def __getattr__(self, field_name):
        if field_name in self.fspace.field_ids:
            if field_name not in self.instances:
                if self.privileges[field_name] is None:
                    raise Exception('Invalid attempt to access field "%s" without privileges' % field_name)
                self._map_inline()
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
        if initializer is None:
            obj = numpy.empty(tuple(0 for i in xrange(region.ispace.dim))).view(
                dtype=region.fspace.field_types[field_name].numpy_type,
                type=cls)
        else:
            obj = numpy.asarray(initializer).view(
                dtype=region.fspace.field_types[field_name].numpy_type,
                type=cls)

        obj.accessor = accessor
        return obj

    @staticmethod
    def _get_accessor(region, field_name):
        # Note: the accessor needs to be kept alive, to make sure to
        # save the result of this function in an instance variable.
        instance = region.instances[field_name]
        dim = region.ispace.dim
        get_accessor = getattr(c, 'legion_physical_region_get_field_accessor_array_{}d'.format(dim))
        return get_accessor(instance, region.fspace.field_ids[field_name])

    @staticmethod
    def _get_base_and_stride(region, field_name, accessor):
        domain = region.ispace.domain
        dim = domain.dim
        if domain.volume < 1:
            return None, None, None

        rect = getattr(c, 'legion_domain_get_rect_{}d'.format(dim))(domain.raw_value())
        subrect = ffi.new('legion_rect_{}d_t *'.format(dim))
        offsets = ffi.new('legion_byte_offset_t[]', dim)

        base_ptr = getattr(c, 'legion_accessor_array_{}d_raw_rect_ptr'.format(dim))(
            accessor, rect, subrect, offsets)
        assert base_ptr
        for i in xrange(dim):
            assert subrect[0].lo.x[i] == rect.lo.x[i]
            assert subrect[0].hi.x[i] == rect.hi.x[i]

        shape = tuple(rect.hi.x[i] - rect.lo.x[i] + 1 for i in xrange(dim))
        strides = tuple(offsets[i].offset for i in xrange(dim))

        return base_ptr, shape, strides

    @staticmethod
    def _get_array_initializer(region, field_name, accessor):
        base_ptr, shape, strides = RegionField._get_base_and_stride(
            region, field_name, accessor)
        if base_ptr is None:
            return None

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

def fill(region, field_names, value):
    assert(isinstance(region, Region))
    if isinstance(field_names, basestring):
        field_names = [field_names]

    for field_name in field_names:
        field_id = region.fspace.field_ids[field_name]
        field_type = region.fspace.field_types[field_name]
        raw_value = ffi.new('{} *'.format(field_type.cffi_type), value)
        c.legion_runtime_fill_field(
            _my.ctx.runtime, _my.ctx.context,
            region.raw_value(), region.parent.raw_value() if region.parent is not None else region.raw_value(),
            field_id, raw_value, field_type.size,
            c.legion_predicate_true())

def copy(src_region, src_field_names, dst_region, dst_field_names, redop=None):
    assert(isinstance(src_region, Region))
    assert(isinstance(dst_region, Region))

    if isinstance(src_field_names, basestring):
        src_field_names = [src_field_names]
    if isinstance(dst_field_names, basestring):
        dst_field_names = [dst_field_names]

    launcher = c.legion_copy_launcher_create(c.legion_predicate_true(), 0, 0)

    if redop is None:
        src_groups = [src_field_names]
        dst_groups = [dst_field_names]
        add_dst_requirement = c.legion_copy_launcher_add_dst_region_requirement_logical_region
    else:
        src_groups = zip(src_field_names)
        dst_groups = zip(dst_field_names)
        add_dst_requirement = c.legion_copy_launcher_add_dst_region_requirement_logical_region_reduction

    for idx, group in enumerate(src_groups):
        c.legion_copy_launcher_add_src_region_requirement_logical_region(
            launcher,
            src_region.raw_value(),
            R._legion_privilege(), 0, # EXCLUSIVE
            src_region.parent.raw_value() if src_region.parent is not None else src_region.raw_value(),
            0, False)
        for src_field_name in group:
            src_field_id = src_region.fspace.field_ids[src_field_name]
            c.legion_copy_launcher_add_src_field(launcher, idx, src_field_id, True)

    for idx, group in enumerate(dst_groups):
        if redop is None:
            dst_privilege = RW._legion_privilege()
        else:
            dst_field_type = dst_region.fspace.field_types[group[0]]
            dst_privilege = Reduce(redop, [group[0]])._legion_redop_id(dst_field_type)
        add_dst_requirement(
            launcher,
            dst_region.raw_value(),
            dst_privilege, 0, # EXCLUSIVE
            dst_region.parent.raw_value() if dst_region.parent is not None else dst_region.raw_value(),
            0, False)
        for dst_field_name in group:
            dst_field_id = dst_region.fspace.field_ids[dst_field_name]
            c.legion_copy_launcher_add_dst_field(launcher, idx, dst_field_id, True)

    c.legion_copy_launcher_execute(_my.ctx.runtime, _my.ctx.context, launcher)

    c.legion_copy_launcher_destroy(launcher)

@contextlib.contextmanager
def attach_hdf5(region, filename, field_map, mode, restricted=True, mapped=False):
    assert(isinstance(region, Region))

    assert(isinstance(filename, basestring))
    filename = filename.encode('utf-8')

    raw_field_map = c.legion_field_map_create()
    encoded_values = [] # make sure these don't get deleted before the launcher
    for field_name, value in field_map.items():
        encoded_value = value.encode('utf-8')
        encoded_values.append(encoded_value)
        c.legion_field_map_insert(raw_field_map, region.fspace.field_ids[field_name], encoded_value)
        region._clear_instance(field_name)

    assert(isinstance(mode, FileMode))

    launcher = c.legion_attach_launcher_create(
        region.raw_value(),
        region.parent.raw_value() if region.parent is not None else region.raw_value(),
        EXTERNAL_HDF5_FILE)

    c.legion_attach_launcher_attach_hdf5(launcher, filename, raw_field_map, mode.value)
    c.legion_attach_launcher_set_restricted(launcher, restricted)
    c.legion_attach_launcher_set_mapped(launcher, mapped)

    instance = c.legion_attach_launcher_execute(
        _my.ctx.runtime, _my.ctx.context, launcher)

    c.legion_attach_launcher_destroy(launcher)
    c.legion_field_map_destroy(raw_field_map)

    yield

    c.legion_detach_external_resource(
        _my.ctx.runtime, _my.ctx.context, instance)

@contextlib.contextmanager
def acquire(region, field_names):
    assert(isinstance(region, Region))

    launcher = c.legion_acquire_launcher_create(
        region.raw_value(),
        region.parent.raw_value() if region.parent is not None else region.raw_value(),
        c.legion_predicate_true(), 0, 0)

    for field_name in field_names:
        c.legion_acquire_launcher_add_field(launcher, region.fspace.field_ids[field_name])

    c.legion_acquire_launcher_execute(_my.ctx.runtime, _my.ctx.context, launcher)
    c.legion_acquire_launcher_destroy(launcher)

    yield

    launcher = c.legion_release_launcher_create(
        region.raw_value(),
        region.parent.raw_value() if region.parent is not None else region.raw_value(),
        c.legion_predicate_true(), 0, 0)

    for field_name in field_names:
        c.legion_release_launcher_add_field(launcher, region.fspace.field_ids[field_name])

    c.legion_release_launcher_execute(_my.ctx.runtime, _my.ctx.context, launcher)
    c.legion_release_launcher_destroy(launcher)

# Hack: Can't pickle static methods.
def _Ipartition_unpickle(tid, id, type_tag, parent, color_space):
    handle = ffi.new('legion_index_partition_t *')
    handle[0].tid = tid
    handle[0].id = id
    handle[0].type_tag = type_tag

    return Ipartition(handle[0], parent, color_space)

class Ipartition(object):
    __slots__ = ['handle', 'parent', 'color_space']

    # Make this speak the Type interface
    numpy_type = None
    cffi_type = 'legion_index_partition_t'
    size = ffi.sizeof(cffi_type)

    def __init__(self, handle, parent, color_space):
        # Important: Copy handle. Do NOT assume ownership.
        self.handle = ffi.new('legion_index_partition_t *', handle)
        self.parent = parent
        self.color_space = color_space

    def __reduce__(self):
        return (_Ipartition_unpickle,
                (self.handle[0].tid, self.handle[0].id, self.handle[0].type_tag, self.parent, self.color_space))

    def __getitem__(self, point):
        if isinstance(point, SymbolicExpr):
            return SymbolicIndexAccess(self, point)
        point = DomainPoint.coerce(point)
        subspace = c.legion_index_partition_get_index_subspace_domain_point(
            _my.ctx.runtime, self.handle[0], point.raw_value())
        return Ispace(None, _handle=subspace)

    def __iter__(self):
        for point in self.color_space:
            yield self[point]

    @staticmethod
    def equal(ispace, color_space, granularity=1, color=AUTO_GENERATE_ID):
        assert isinstance(ispace, Ispace)
        color_space = Ispace.coerce(color_space)
        handle = c.legion_index_partition_create_equal(
            _my.ctx.runtime, _my.ctx.context,
            ispace.raw_value(), color_space.raw_value(), granularity, color)
        return Ipartition(handle, ispace, color_space)

    @staticmethod
    def by_field(region, field, color_space, color=AUTO_GENERATE_ID):
        assert isinstance(region, Region)
        color_space = Ispace.coerce(color_space)
        handle = c.legion_index_partition_create_by_field(
            _my.ctx.runtime, _my.ctx.context,
            region.raw_value(),
            region.parent.raw_value() if region.parent is not None else region.raw_value(),
            region.fspace.field_ids[field],
            color_space.raw_value(), color, 0, 0, disjoint.value, [ffi.cast("void*", 0), 0])
        return Ipartition(handle, region.ispace, color_space)

    @staticmethod
    def image(ispace, projection, field, color_space,
                        part_kind=compute, color=AUTO_GENERATE_ID):
        assert isinstance(ispace, Ispace)
        assert isinstance(projection, Partition)
        assert isinstance(part_kind, Disjointness)
        color_space = Ispace.coerce(color_space)
        parent = projection.parent
        if is_rect_type(parent.fspace.field_types[field]):
            create_by_image = c.legion_index_partition_create_by_image_range
        else:
            create_by_image = c.legion_index_partition_create_by_image
        handle = create_by_image(
            _my.ctx.runtime, _my.ctx.context,
            ispace.raw_value(), projection.raw_value(),
            parent.parent.raw_value() if parent.parent is not None else parent.raw_value(),
            parent.fspace.field_ids[field],
            color_space.raw_value(), part_kind.value, color, 0, 0, [ffi.cast("void*", 0), 0])
        return Ipartition(handle, parent.ispace, color_space)

    @staticmethod
    def preimage(projection, region, field, color_space,
                           part_kind=compute, color=AUTO_GENERATE_ID):
        assert isinstance(projection, Ipartition)
        assert isinstance(region, Region)
        assert isinstance(part_kind, Disjointness)
        color_space = Ispace.coerce(color_space)
        if is_rect_type(region.fspace.field_types[field]):
            create_by_preimage = c.legion_index_partition_create_by_preimage_range
        else:
            create_by_preimage = c.legion_index_partition_create_by_preimage
        handle = create_by_preimage(
            _my.ctx.runtime, _my.ctx.context,
            projection.raw_value(), region.raw_value(),
            region.parent.raw_value() if region.parent is not None else region.raw_value(),
            region.fspace.field_ids[field],
            color_space.raw_value(), part_kind.value, color, 0, 0, [ffi.cast("void*", 0), 0])
        return Ipartition(handle, region.ispace, color_space)

    @staticmethod
    def restrict(ispace, color_space, transform, extent,
                              part_kind=compute, color=AUTO_GENERATE_ID):
        assert isinstance(ispace, Ispace)
        assert isinstance(part_kind, Disjointness)
        color_space = Ispace.coerce(color_space)
        transform = DomainTransform.coerce(transform)
        extent = Domain.coerce(extent)
        handle = c.legion_index_partition_create_by_restriction(
            _my.ctx.runtime, _my.ctx.context,
            ispace.raw_value(), color_space.raw_value(), transform.raw_value(), extent.raw_value(), part_kind.value, color)
        return Ipartition(handle, ispace, color_space)

    @staticmethod
    def pending(ispace, color_space,
                       part_kind=compute, color=AUTO_GENERATE_ID):
        assert isinstance(ispace, Ispace)
        assert isinstance(part_kind, Disjointness)
        color_space = Ispace.coerce(color_space)
        handle = c.legion_index_partition_create_pending_partition(
            _my.ctx.runtime, _my.ctx.context,
            ispace.raw_value(), color_space.raw_value(), part_kind.value, color)
        return Ipartition(handle, ispace, color_space)

    # The following methods are for pending partitions only:
    def union(self, color, ispaces):
        color = DomainPoint.coerce(color)

        handles = ffi.new('legion_index_space_t[]', [ispace.raw_value() for ispace in ispaces])
        c.legion_index_partition_create_index_space_union_spaces(
            _my.ctx.runtime, _my.ctx.context,
            self.handle[0], color.raw_value(), handles, len(ispaces))

    def destroy(self):
        # This is not something you want to have happen in a
        # destructor, since partitions may outlive the lifetime of the handle.
        c.legion_index_partition_destroy(
            _my.ctx.runtime, _my.ctx.context, self.handle[0])
        # Clear out references. Technically unnecessary but avoids abuse.
        del self.handle
        del self.parent
        del self.color_space

    def raw_value(self):
        return self.handle[0]

# Hack: Can't pickle static methods.
def _Partition_unpickle(parent, ipartition):
    handle = ffi.new('legion_logical_partition_t *')
    handle[0].tree_id = parent.raw_value().tree_id
    handle[0].index_partition = ipartition.raw_value()
    handle[0].field_space = parent.fspace.raw_value()

    return Partition(parent, ipartition, _handle=handle[0])

class Partition(object):
    __slots__ = ['handle', 'parent', 'ipartition']

    # Make this speak the Type interface
    numpy_type = None
    cffi_type = 'legion_logical_partition_t'
    size = ffi.sizeof(cffi_type)

    def __init__(self, parent, ipartition, **kwargs):
        def parse_kwargs(_handle=None):
            return _handle
        handle = parse_kwargs(**kwargs)

        if handle is None:
            assert isinstance(parent, Region)
            assert isinstance(ipartition, Ipartition)
            handle = c.legion_logical_partition_create(
                _my.ctx.runtime, parent.raw_value(), ipartition.raw_value())

        # Important: Copy handle. Do NOT assume ownership.
        assert handle is not None
        self.handle = ffi.new('legion_logical_partition_t *', handle)
        self.parent = parent
        self.ipartition = ipartition

    def __reduce__(self):
        return (_Partition_unpickle,
                (self.parent,
                 self.ipartition))

    def __getitem__(self, point):
        if isinstance(point, SymbolicExpr):
            return SymbolicIndexAccess(self, point)
        point = DomainPoint.coerce(point)
        subspace = self.ipartition[point]
        subregion = c.legion_logical_partition_get_logical_subregion_by_color_domain_point(
            _my.ctx.runtime, self.handle[0], point.raw_value())
        return Region(subspace, self.parent.fspace, _handle=subregion,
                      _parent=self.parent.parent if self.parent.parent is not None else self.parent)

    def __iter__(self):
        for point in self.color_space:
            yield self[point]

    @property
    def color_space(self):
        return self.ipartition.color_space

    @staticmethod
    def equal(region, color_space, granularity=1, color=AUTO_GENERATE_ID):
        assert isinstance(region, Region)
        ipartition = Ipartition.equal(region.ispace, color_space, granularity, color)
        return Partition(region, ipartition)

    @staticmethod
    def by_field(region, field, color_space, color=AUTO_GENERATE_ID):
        assert isinstance(region, Region)
        ipartition = Ipartition.by_field(
            region, field, color_space, color)
        return Partition(region, ipartition)

    @staticmethod
    def image(region, projection, field, color_space,
                        part_kind=compute, color=AUTO_GENERATE_ID):
        assert isinstance(region, Region)
        ipartition = Ipartition.image(
            region.ispace, projection, field, color_space, part_kind, color)
        return Partition(region, ipartition)

    @staticmethod
    def preimage(projection, region, field, color_space,
                           part_kind=compute, color=AUTO_GENERATE_ID):
        assert isinstance(projection, Partition)
        ipartition = Ipartition.preimage(
            projection.ipartition, region, field, color_space, part_kind, color)
        return Partition(region, ipartition)

    @staticmethod
    def restrict(region, color_space, transform, extent,
                              part_kind=compute, color=AUTO_GENERATE_ID):
        assert isinstance(region, Region)
        ipartition = Ipartition.restrict(
            region.ispace, color_space, transform, extent, part_kind, color)
        return Partition(region, ipartition)

    @staticmethod
    def pending(region, color_space, part_kind=compute, color=AUTO_GENERATE_ID):
        assert isinstance(region, Region)
        ipartition = Ipartition.pending(
            region.ispace, color_space, part_kind, color)
        return Partition(region, ipartition)

    def union(self, color, regions):
        ispaces = [region.ispace for region in regions]
        self.ipartition.union(color, ispaces)

    def destroy(self):
        # This is not something you want to have happen in a
        # destructor, since partitions may outlive the lifetime of the handle.
        c.legion_logical_partition_destroy(
            _my.ctx.runtime, _my.ctx.context, self.handle[0])
        # Clear out references. Technically unnecessary but avoids abuse.
        del self.handle
        del self.parent
        del self.ipartition

    def raw_value(self):
        return self.handle[0]

def define_regent_argument_struct(task_id, argument_types, privileges, return_type, arguments=None):
    if argument_types is None:
        raise Exception('Arguments must be typed in extern Regent tasks')

    struct_name = 'task_args_%s' % task_id

    n_fields = int(math.ceil(len(argument_types)/64.))

    fields = ['uint64_t %s[%s];' % ('__map', n_fields)]
    for i, arg_type in enumerate(argument_types):
        arg_name = '__arg_%s' % i
        fields.append('%s %s;' % (arg_type.cffi_type, arg_name))
    if arguments is not None:
        for i, arg in enumerate(arguments):
            if isinstance(arg, Region) or (isinstance(arg, SymbolicExpr) and arg.is_region()):
                fields.append('legion_field_id_t __arg_%s_fields[%s];' % (i, len(arg.fspace.field_types)))

    struct = 'typedef struct %s { %s } %s;' % (struct_name, ' '.join(fields), struct_name)
    ffi.cdef(struct)

    return struct_name

class ExternTask(object):
    __slots__ = ['argument_types', 'privileges', 'return_type',
                 'calling_convention', 'task_id', '_argument_struct']

    def __init__(self, task_id, argument_types=None, privileges=None,
                 return_type=void, calling_convention=None):
        self.argument_types = argument_types
        self.privileges = privileges
        self.return_type = return_type
        self.calling_convention = calling_convention
        assert isinstance(task_id, int)
        self.task_id = task_id
        self._argument_struct = None

    def argument_struct(self, args=None):
        if self.calling_convention == 'regent' and self._argument_struct is None:
            self._argument_struct = define_regent_argument_struct(
                self.task_id, self.argument_types, self.privileges, self.return_type, args)
        return self._argument_struct

    def __call__(self, *args):
        return self.spawn_task(*args)

    def spawn_task(self, *args, **kwargs):
        if _my.ctx.current_launch:
            return _my.ctx.current_launch.spawn_task(self, *args, **kwargs)
        return TaskLaunch().spawn_task(self, *args, **kwargs)

def extern_task(**kwargs):
    return ExternTask(**kwargs)

class ExternTaskWrapper(object):
    # Note: Can't use __slots__ for this class because __qualname__
    # conflicts with the class variable.
    def __init__(self, thunk, name):
        self.thunk = thunk
        self.__name__ = name
        self.__qualname__ = name
    def __call__(self, *args, **kwargs):
        f = self.thunk(*args, **kwargs)
        if f.value_type != void:
            return f.get()

_next_wrapper_id = 1000
def extern_task_wrapper(privileges=None, return_type=void, **kwargs):
    global _next_wrapper_id
    extern = extern_task(privileges=privileges, return_type=return_type, **kwargs)
    wrapper_name = str('wrapper_task_%s' % _next_wrapper_id)
    _next_wrapper_id += 1
    wrapper = ExternTaskWrapper(extern, wrapper_name)
    task_wrapper = task(wrapper, privileges=privileges, return_type=return_type, inner=True)
    setattr(sys.modules[__name__], wrapper_name, task_wrapper)
    return task_wrapper

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

def _postprocess(arg, point):
    if hasattr(arg, '_legion_postprocess_task_argument'):
        return arg._legion_postprocess_task_argument(point)
    return arg

class Task (object):
    __slots__ = ['body', 'privileges', 'layout', 'argument_types',
                 'return_type', 'leaf', 'inner', 'idempotent',
                 'replicable', 'calling_convention', 'task_id',
                 'registered', '_argument_struct']

    def __init__(self, body, privileges=None, layout=None,
                 argument_types=None, return_type=None,
                 leaf=False, inner=False, idempotent=False, replicable=False,
                 calling_convention='python',
                 register=True, task_id=None, top_level=False):
        if calling_convention == 'regent':
            if argument_types is None:
                raise Exception('when calling_convention is "regent", argument_types must be defined')
            if return_type is None:
                raise Exception('when calling_convention is "regent", return_type must be defined')
        self.body = body
        self.privileges = privileges
        self.layout = layout
        self.argument_types = argument_types
        self.return_type = return_type
        self.leaf = bool(leaf)
        self.inner = bool(inner)
        self.idempotent = bool(idempotent)
        self.replicable = bool(replicable)
        self.calling_convention = calling_convention
        self.task_id = None
        self._argument_struct = None
        if register:
            self.register(task_id, top_level)
        self.argument_struct()

    def __call__(self, *args, **kwargs):
        # Hack: This entrypoint needs to be able to handle both being
        # called in user code (to launch a task) and as the task
        # wrapper when the task itself executes. Unfortunately isn't a
        # good way to disentangle these. Detect if we're in the task
        # wrapper case by checking the number and types of arguments.
        if len(args) == 3 and \
           isinstance(args[0], bytearray) and \
           isinstance(args[1], bytearray) and \
           isinstance(args[2], long):
            return self.execute_task(*args, **kwargs)
        else:
            return self.spawn_task(*args, **kwargs)

    def argument_struct(self, args=None):
        if self.calling_convention == 'regent' and self._argument_struct is None:
            self._argument_struct = define_regent_argument_struct(
                self.task_id, self.argument_types, self.privileges, self.return_type, args)
        return self._argument_struct

    def spawn_task(self, *args, **kwargs):
        if _my.ctx.current_launch:
            return _my.ctx.current_launch.spawn_task(self, *args, **kwargs)
        return TaskLaunch().spawn_task(self, *args, **kwargs)

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

        # Unpack regions.
        regions = []
        for i in xrange(num_regions[0]):
            regions.append(raw_regions[0][i])

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

        # Decode arguments from Pickle format.
        arg_ptr = ffi.cast('char *', c.legion_task_get_args(task[0]))
        arg_size = c.legion_task_get_arglen(task[0])
        if c.legion_task_get_is_index_space(task[0]) and arg_size == 0:
            arg_ptr = ffi.cast('char *', c.legion_task_get_local_args(task[0]))
            arg_size = c.legion_task_get_local_arglen(task[0])

        if self.calling_convention == 'python':
            if arg_size > 0 and c.legion_task_get_depth(task[0]) > 0:
                args = pickle.loads(ffi.unpack(arg_ptr, arg_size))
            else:
                args = ()
        elif self.calling_convention == 'regent':
            if len(self.argument_types) == 0:
                args = ()
            else:
                arg_struct = self.argument_struct()
                # We're not going to be able to unpack the field IDs
                # because we don't have the type info. So we'll just
                # unpack the initial struct and try to work from there.
                assert arg_size >= ffi.sizeof(arg_struct)
                arg_data = ffi.cast('%s *' % arg_struct, arg_ptr)
                future_map = getattr(arg_data, '__map')
                args = []
                for i, arg_type in enumerate(self.argument_types):
                    future = future_map[i // 64] & (1 << (i % 64)) != 0
                    if future:
                        assert False
                    else:
                        arg_name = '__arg_%s' % i
                        arg_value = getattr(arg_data, arg_name)
                        if hasattr(arg_type, 'from_raw'):
                            arg_value = arg_type.from_raw(arg_value)
                        args.append(arg_value)
        else:
            assert False

        # Postprocess arguments.
        point = DomainPoint(None, _handle=c.legion_task_get_index_point(task[0]))
        args = tuple(_postprocess(arg, point) for arg in args)

        # Unpack physical regions.
        if self.privileges is not None:
            req = 0
            for i, arg in zip(range(len(args)), args):
                if isinstance(arg, Region):
                    assert i < len(self.privileges)
                    groups = self.privileges[i]._legion_grouped_privileges(arg.fspace)
                    for priv, _, _, fields in groups:
                        assert req < num_regions[0]
                        instance = raw_regions[0][req]
                        req += 1

                        for field in fields:
                            arg._set_instance(field, instance, priv)
            assert req == num_regions[0]

        # Execute task body.
        result = self.body(*args)

        # Mark any remaining objects as escaped.
        for ref in ctx.owned_objects:
            obj = ref()
            if obj is not None:
                obj.escaped = True

        # Encode result.
        if not self.return_type:
            result_str = pickle.dumps(result, protocol=_pickle_version)
            result_size = len(result_str)
            result_ptr = ffi.new('char[]', result_size)
            ffi.buffer(result_ptr, result_size)[:] = result_str
        else:
            if self.return_type.size > 0:
                result_ptr = ffi.new(ffi.getctype(self.return_type.cffi_type, '*'), result)
            else:
                result_ptr = ffi.NULL
            result_size = self.return_type.size

        # Execute postamble.
        c.legion_task_postamble(runtime[0], context[0], result_ptr, result_size)

        # Clear thread-local storage.
        del _my.ctx

    def register(self, task_id, top_level_task):
        assert(self.task_id is None)

        if not task_id:
            global next_legion_task_id
            task_id = next_legion_task_id
            next_legion_task_id += 1
            # If we ever hit this then we need to allocate more task IDs
            assert task_id < max_legion_task_id

        execution_constraints = c.legion_execution_constraint_set_create()
        c.legion_execution_constraint_set_add_processor_constraint(
            execution_constraints, c.PY_PROC)

        layout_constraints = c.legion_task_layout_constraint_set_create()
        if self.layout is not None:
            for i, constraint in enumerate(self.layout):
                layout = c.legion_layout_constraint_set_create()
                dim = constraint.dim
                dims = ffi.new('legion_dimension_kind_t [%s]' % (dim + 1))
                for d_idx, d in enumerate(constraint.order.order(dim)):
                    if d == 'F':
                        d = 9 # DIM_F
                    dims[d_idx] = d
                c.legion_layout_constraint_set_add_ordering_constraint(
                    layout,
                    dims,
                    dim + 1,
                    True)
                layout_id = c.legion_layout_constraint_set_register(
                    c.legion_runtime_get_runtime(),
                    c.legion_field_space_no_space(),
                    layout,
                    str(constraint).encode('utf-8'))
                c.legion_layout_constraint_set_destroy(layout)
                c.legion_task_layout_constraint_set_add_layout_constraint(
                    layout_constraints,
                    i,
                    layout_id)

        options = ffi.new('legion_task_config_options_t *')
        options[0].leaf = self.leaf
        options[0].inner = self.inner
        options[0].idempotent = self.idempotent
        options[0].replicable = self.replicable

        qualname = get_qualname(self.body)
        task_name = ('%s.%s' % (self.body.__module__, '.'.join(qualname)))

        c_qualname_comps = [ffi.new('char []', comp.encode('utf-8')) for comp in qualname]
        c_qualname = ffi.new('char *[]', c_qualname_comps)

        global global_task_registration_barrier
        if global_task_registration_barrier is not None:
            c.legion_runtime_enable_scheduler_lock()
            c.legion_phase_barrier_arrive(_my.ctx.runtime, _my.ctx.context, global_task_registration_barrier, 1)
            global_task_registration_barrier = c.legion_phase_barrier_advance(_my.ctx.runtime, _my.ctx.context, global_task_registration_barrier)
            c.legion_phase_barrier_wait(_my.ctx.runtime, _my.ctx.context, global_task_registration_barrier)
            # Need to hold this through the end of registration.
            # c.legion_runtime_disable_scheduler_lock()

        c.legion_runtime_register_task_variant_python_source_qualname(
            c.legion_runtime_get_runtime(),
            task_id,
            task_name.encode('utf-8'),
            True, # self.replicable or not is_script, # Global
            execution_constraints,
            layout_constraints,
            options[0],
            self.body.__module__.encode('utf-8'),
            c_qualname,
            len(qualname),
            ffi.NULL,
            0)
        # If we're the top-level task then tell the runtime about our ID
        if top_level_task:
            c.legion_runtime_set_top_level_task_id(task_id)
        if global_task_registration_barrier is not None:
            c.legion_phase_barrier_arrive(_my.ctx.runtime, _my.ctx.context, global_task_registration_barrier, 1)
            global_task_registration_barrier = c.legion_phase_barrier_advance(_my.ctx.runtime, _my.ctx.context, global_task_registration_barrier)
            # c.legion_runtime_enable_scheduler_lock()
            c.legion_phase_barrier_wait(_my.ctx.runtime, _my.ctx.context, global_task_registration_barrier)
            c.legion_runtime_disable_scheduler_lock()

        c.legion_execution_constraint_set_destroy(execution_constraints)
        c.legion_task_layout_constraint_set_destroy(layout_constraints)

        self.task_id = task_id
        return self

def task(body=None, **kwargs):
    if body is None:
        return lambda body: task(body, **kwargs)
    return Task(body, **kwargs)

class _TaskLauncher(object):
    __slots__ = ['task']

    def __init__(self, task):
        self.task = task

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
        if self.task.calling_convention == 'python':
            arg_str = pickle.dumps(args, protocol=_pickle_version)
            task_args_buffer = ffi.new('char[]', arg_str)
            task_args[0].args = task_args_buffer
            task_args[0].arglen = len(arg_str)
        elif self.task.calling_convention == 'regent':
            arg_struct = self.task.argument_struct(args)
            task_args_buffer = ffi.new('%s*' % arg_struct)
            # Note: ffi.new returns zeroed memory
            for i, arg in enumerate(args):
                if isinstance(arg, Future):
                    getattr(task_args_buffer, '__map')[i // 64] |= 1 << (i % 64)
            for i, arg in enumerate(args):
                if not isinstance(arg, Future) and not (isinstance(arg, SymbolicExpr) and arg.is_region()):
                    arg_name = '__arg_%s' % i
                    arg_value = arg
                    if hasattr(arg, 'handle') and not isinstance(arg, DomainPoint):
                        arg_value = arg.handle[0]
                    setattr(task_args_buffer, arg_name, arg_value)
            for i, arg in enumerate(args):
                if isinstance(arg, Region) or (isinstance(arg, SymbolicExpr) and arg.is_region()):
                    arg_name = '__arg_%s_fields' % i
                    arg_slot = getattr(task_args_buffer, arg_name)
                    for j, field_id in enumerate(arg.fspace.field_ids.values()):
                        arg_slot[j] = field_id
            task_args[0].args = task_args_buffer
            task_args[0].arglen = ffi.sizeof(arg_struct)
        else:
            # FIXME: External tasks need a dedicated calling
            # convention to permit the passing of task arguments.
            task_args[0].args = ffi.NULL
            task_args[0].arglen = 0
        # WARNING: Need to return the interior buffer or else it will be GC'd
        return task_args, task_args_buffer

    def attach_region_requirements(self, launcher, args, is_index_launch):
        if is_index_launch:
            def add_region_normal(launcher, handle, *args):
                return c.legion_index_launcher_add_region_requirement_logical_region(
                    launcher, handle, 0, # projection
                    *args)
            def add_region_reduction(launcher, handle, *args):
                return c.legion_index_launcher_add_region_requirement_logical_region_reduction(
                    launcher, handle, 0, # projection
                    *args)
            add_partition_normal = c.legion_index_launcher_add_region_requirement_logical_partition
            add_partition_reduction = c.legion_index_launcher_add_region_requirement_logical_partition_reduction
            add_field = c.legion_index_launcher_add_field
        else:
            add_region_normal = c.legion_task_launcher_add_region_requirement_logical_region
            add_region_reduction = c.legion_task_launcher_add_region_requirement_logical_region_reduction
            add_field = c.legion_task_launcher_add_field

        for i, arg in zip(range(len(args)), args):
            if isinstance(arg, Region) or (isinstance(arg, SymbolicExpr) and arg.is_region()):
                if self.task.privileges is None or i >= len(self.task.privileges):
                    raise Exception('Privileges are required on all Region arguments')
                groups = self.task.privileges[i]._legion_grouped_privileges(arg.fspace)
                if isinstance(arg, Region):
                    parent = arg.parent if arg.parent is not None else arg
                    for _, priv, redop, fields in groups:
                        if redop is None:
                            req = add_region_normal(
                                launcher, arg.raw_value(),
                                priv, 0, # EXCLUSIVE
                                parent.raw_value(), 0, False)
                        else:
                            req = add_region_reduction(
                                launcher, arg.raw_value(),
                                redop, 0, # EXCLUSIVE
                                parent.raw_value(), 0, False)
                        for field in fields:
                            add_field(
                                launcher, req, arg.fspace.field_ids[field], True)
                elif isinstance(arg, SymbolicExpr):
                    # FIXME: Support non-trivial projection functors
                    assert isinstance(arg, SymbolicIndexAccess) and (isinstance(arg.index, SymbolicLoopIndex) or isinstance(arg.index, ConcreteLoopIndex))
                    proj_id = 0

                    parent = arg.parent if arg.parent is not None else arg
                    parent = parent.parent if parent.parent is not None else parent
                    for _, priv, redop, fields in groups:
                        if redop is None:
                            req = add_partition_normal(
                                launcher, arg.raw_value(), proj_id,
                                priv, 0, # EXCLUSIVE
                                parent.raw_value(), 0, False)
                        else:
                            req = add_partition_reduction(
                                launcher, arg.raw_value(), proj_id,
                                redop, 0, # EXCLUSIVE
                                parent.raw_value(), 0, False)
                        for field in fields:
                            add_field(
                                launcher, req, arg.fspace.field_ids[field], True)
            elif self.task.privileges is not None and i < len(self.task.privileges) and self.task.privileges[i]:
                raise TypeError('Privileges can only be specified for Region arguments, got %s' % type(arg))

    def spawn_task(self, *args, **kwargs):
        # Hack: workaround for Python 2 not having keyword-only arguments
        def validate_spawn_task_args(point=None, mapper=0, tag=0):
            return point, mapper, tag
        point, mapper, tag = validate_spawn_task_args(**kwargs)

        assert(isinstance(_my.ctx, Context))

        args, futures = self.gather_futures(args)
        args = self.preprocess_args(args)
        task_args, task_args_root = self.encode_args(args)

        # Construct the task launcher.
        launcher = c.legion_task_launcher_create(
            self.task.task_id, task_args[0], c.legion_predicate_true(), mapper, tag)
        if point is not None:
            point = DomainPoint.coerce(point)
            c.legion_task_launcher_set_point(launcher, point.raw_value())
        self.attach_region_requirements(launcher, args, False)
        for i, arg in zip(range(len(args)), args):
            if self.task.privileges is not None and i < len(self.task.privileges) and self.task.privileges[i] and not isinstance(arg, Region):
                raise TypeError('Privileges can only be specified for Region arguments, got %s' % type(arg))
            if isinstance(arg, Region):
                pass # Already attached above
            elif isinstance(arg, Future):
                c.legion_task_launcher_add_future(launcher, arg.handle)
            elif self.task.calling_convention is None:
                # FIXME: Task arguments aren't being encoded AT ALL;
                # at least throw an exception so that the user knows
                raise Exception('External tasks do not support non-region arguments')

        # Launch the task.
        if _my.ctx.current_launch is not None:
            return _my.ctx.current_launch.attach_task_launcher(
                launcher, point, root=task_args_root)

        result = c.legion_task_launcher_execute(
            _my.ctx.runtime, _my.ctx.context, launcher)
        c.legion_task_launcher_destroy(launcher)

        # Build future of result.
        future = Future.from_cdata(result, value_type=self.task.return_type)
        c.legion_future_destroy(result)
        return future

class _IndexLauncher(_TaskLauncher):
    __slots__ = ['task', 'domain', 'mapper', 'tag',
                 'global_args', 'local_args', 'region_args', 'future_args',
                 'reduction_op', 'future_map']

    def __init__(self, task, domain, mapper, tag):
        super(_IndexLauncher, self).__init__(task)
        self.domain = domain
        self.mapper = mapper
        self.tag = tag
        self.global_args = None
        self.local_args = c.legion_argument_map_create()
        self.region_args = None
        self.future_args = []
        self.reduction_op = None
        self.future_map = None

    def __del__(self):
        c.legion_argument_map_destroy(self.local_args)

    def spawn_task(self, *args, **kwargs):
        raise Exception('IndexLaunch does not support spawn_task')

    def attach_local_args(self, index, *args):
        task_args, _ = self.encode_args(self.preprocess_args(args))
        c.legion_argument_map_set_point(
            self.local_args, index.value.raw_value(), task_args[0], False)

    def attach_global_args(self, *args):
        assert self.global_args is None
        self.global_args = args

    def attach_region_args(self, *args):
        self.region_args = args

    def attach_future_args(self, *args):
        self.future_args = args

    def set_reduction_op(self, op):
        self.reduction_op = op

    def launch(self):
        # Encode global args (if any).
        if self.global_args is not None:
            global_args, global_args_root = self.encode_args(self.preprocess_args(self.global_args))
        else:
            global_args = ffi.new('legion_task_argument_t *')
            global_args[0].args = ffi.NULL
            global_args[0].arglen = 0
            global_args_root = None

        # Construct the task launcher.
        launcher = c.legion_index_launcher_create(
            self.task.task_id, self.domain.raw_value(),
            global_args[0], self.local_args,
            c.legion_predicate_true(), False, self.mapper, self.tag)

        assert (self.global_args is not None) != (self.region_args is not None)
        if self.global_args is not None:
            self.attach_region_requirements(launcher, self.global_args, True)
        if self.region_args is not None:
            self.attach_region_requirements(launcher, self.region_args, True)

        for arg in self.future_args:
            c.legion_index_launcher_add_future(launcher, arg.handle)

        # Launch the task.
        if _my.ctx.current_launch is not None:
            return _my.ctx.current_launch.attach_index_launcher(
                launcher, root=global_args_root)

        launch = c.legion_index_launcher_execute
        redop = []
        if self.reduction_op is not None:
            assert self.task.return_type is not None
            launch = c.legion_index_launcher_execute_reduction
            redop = [_redop_ids[self.reduction_op][self.task.return_type]]

        result = launch(
            _my.ctx.runtime, _my.ctx.context, launcher, *redop)
        c.legion_index_launcher_destroy(launcher)

        # Build future (map) of result.
        if self.reduction_op is not None:
            self.future_map = Future.from_cdata(result, value_type=self.task.return_type)
            c.legion_future_destroy(result)
        else:
            self.future_map = FutureMap(result, value_type=self.task.return_type)
            c.legion_future_map_destroy(result)

class _MustEpochLauncher(object):
    __slots__ = ['domain', 'launcher', 'roots', 'has_sublaunchers']

    def __init__(self, domain=None):
        self.domain = domain
        self.launcher = c.legion_must_epoch_launcher_create(0, 0)
        if self.domain is not None:
            c.legion_must_epoch_launcher_set_launch_domain(
                self.launcher, self.domain.raw_value())
        self.roots = []
        self.has_sublaunchers = False

    def __del__(self):
        c.legion_must_epoch_launcher_destroy(self.launcher)

    def spawn_task(self, *args, **kwargs):
        raise Exception('MustEpochLaunch does not support spawn_task')

    def attach_task_launcher(self, task_launcher, point, root=None):
        if point is None:
            raise Exception('MustEpochLauncher requires a point for each task')
        if root is not None:
            self.roots.append(root)
        c.legion_must_epoch_launcher_add_single_task(
            self.launcher, point.raw_value(), task_launcher)
        self.has_sublaunchers = True

    def attach_index_launcher(self, index_launcher, root=None):
        if root is not None:
            self.roots.append(root)
        c.legion_must_epoch_launcher_add_index_task(
            self.launcher, index_launcher)
        self.has_sublaunchers = True

    def launch(self):
        if not self.has_sublaunchers:
            raise Exception('MustEpochLaunch requires at least one point task to be executed')
        result = c.legion_must_epoch_launcher_execute(
            _my.ctx.runtime, _my.ctx.context, self.launcher)
        c.legion_future_map_destroy(result)

class TaskLaunch(object):
    __slots__ = []
    def spawn_task(self, task, *args, **kwargs):
        launcher = _TaskLauncher(task=task)
        return launcher.spawn_task(*args, **kwargs)

class _FuturePoint(object):
    __slots__ = ['launcher', 'point', 'future']
    def __init__(self, launcher, point):
        self.launcher = launcher
        self.point = point
        self.future = None
    def get(self):
        if self.future is not None:
            return self.future.get()

        if self.launcher.future_map is None:
            raise Exception('Cannot retrieve a future from an index launch until the launch is complete')

        self.future = self.launcher.future_map[self.point]

        # Clear launcher and point
        del self.launcher
        del self.point

        return self.future.get()

class SymbolicExpr(object):
    def is_region(self):
        return False

class SymbolicIndexAccess(SymbolicExpr):
    __slots__ = ['value', 'index']
    def __init__(self, value, index):
        self.value = value
        self.index = index
    def __str__(self):
        return '%s[%s]' % (self.value, self.index)
    def __repr__(self):
        return '%s[%s]' % (self.value, self.index)
    def _legion_preprocess_task_argument(self):
        if isinstance(self.index, ConcreteLoopIndex):
            return self.value[self.index._legion_preprocess_task_argument()]
        return self
    def _legion_postprocess_task_argument(self, point):
        result = _postprocess(self.value, point)[_postprocess(self.index, point)]
        # FIXME: Clear parent field of regions being used as projection requirements
        if isinstance(result, Region):
            result.parent = None
        return result
    def is_region(self):
        return isinstance(self.value, Partition)
    @property
    def parent(self):
        if self.is_region():
            return self.value.parent
        assert False
    @property
    def fspace(self):
        if self.is_region():
            return self.value.parent.fspace
        assert False
    def raw_value(self):
        if self.is_region():
            return self.value.raw_value()
        assert False

class SymbolicLoopIndex(SymbolicExpr):
    __slots__ = ['name']
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name
    def _legion_postprocess_task_argument(self, point):
        return point

ID = SymbolicLoopIndex('ID')

class ConcreteLoopIndex(SymbolicExpr):
    __slots__ = ['value']
    def __init__(self, value):
        self.value = value
    def __int__(self):
        return self.value.__int__()
    def __index__(self):
        return self.value.__index__()
    def __str__(self):
        return str(self.value)
    def __repr__(self):
        return repr(self.value)
    def _legion_preprocess_task_argument(self):
        return self.value

def index_launch(domain, task, *args, **kwargs):
    def parse_kwargs(reduce=None, mapper=0, tag=0):
        return reduce, mapper, tag
    reduce, mapper, tag = parse_kwargs(**kwargs)

    if isinstance(domain, Domain):
        domain = domain
    elif isinstance(domain, Ispace):
        domain = domain.domain
    else:
        domain = Domain(domain)
    launcher = _IndexLauncher(task=task, domain=domain, mapper=mapper, tag=tag)
    args, futures = launcher.gather_futures(args)
    launcher.attach_global_args(*args)
    launcher.attach_future_args(*futures)
    launcher.set_reduction_op(reduce)
    launcher.launch()
    return launcher.future_map

class IndexLaunch(object):
    __slots__ = ['domain', 'mapper', 'tag', 'launcher', 'point',
                 'saved_task', 'saved_args']

    def __init__(self, domain, **kwargs):
        # Hack: workaround for Python 2 not having keyword-only arguments
        def validate_spawn_task_args(mapper=0, tag=0):
            return mapper, tag
        mapper, tag = validate_spawn_task_args(**kwargs)

        if isinstance(domain, Domain):
            self.domain = domain
        elif isinstance(domain, Ispace):
            self.domain = domain.domain
        else:
            self.domain = Domain(domain)
        self.mapper = mapper
        self.tag = tag
        self.launcher = None
        self.point = None
        self.saved_task = None
        self.saved_args = None

    def __iter__(self):
        _my.ctx.begin_launch(self)
        self.point = ConcreteLoopIndex(None)
        for i in self.domain:
            self.point.value = i
            yield self.point
        _my.ctx.end_launch(self)
        self.launch()

    def ensure_launcher(self, task):
        if self.launcher is None:
            self.launcher = _IndexLauncher(
                task=task, domain=self.domain, mapper=self.mapper, tag=self.tag)

    def check_compatibility(self, task, *args):
        # The tasks in a launch must conform to the following constraints:
        #   * Only one task can be launched.
        #   * The arguments must be compatible:
        #       * At a given argument position, the value must always
        #         be a special value, or always not.
        #       * Special values include: regions and futures.
        #       * If a region, the value must be symbolic (i.e. able
        #         to be analyzed as a function of the index expression).
        #       * If a future, the values must be literally identical
        #         (i.e. each argument slot in the launch can only
        #         accept a single future value.)

        if self.saved_task is None:
            self.saved_task = task
        if task != self.saved_task:
            raise Exception('An IndexLaunch may contain only one task launch')

        if self.saved_args is None:
            self.saved_args = args
        for arg, saved_arg in zip_longest(args, self.saved_args):
            # TODO: Add support for region arguments
            if isinstance(arg, Region) or isinstance(arg, RegionField):
                if arg != saved_arg:
                    raise Exception('Region argument to IndexLaunch does not match previous value at this position')
            elif isinstance(arg, Future):
                if arg != saved_arg:
                    raise Exception('Future argument to IndexLaunch does not match previous value at this position')

    def spawn_task(self, task, *args):
        self.ensure_launcher(task)
        self.check_compatibility(task, *args)
        args, futures = self.launcher.gather_futures(args)
        self.launcher.attach_local_args(self.point, *args)
        self.launcher.attach_region_args(*args)
        self.launcher.attach_future_args(*futures)
        return _FuturePoint(self.launcher, self.point.value)

    def launch(self):
        self.launcher.launch()

class MustEpochLaunch(object):
    __slots__ = ['domain', 'launcher']

    def __init__(self, domain=None):
        if isinstance(domain, Domain):
            self.domain = domain
        elif isinstance(domain, Ispace):
            self.domain = ispace.domain
        elif domain is not None:
            self.domain = Domain(domain)
        else:
            self.domain = None
        self.launcher = None

    def __enter__(self):
        self.launcher = _MustEpochLauncher(domain=self.domain)
        _my.ctx.begin_launch(self)

    def __exit__(self, exc_type, exc_value, tb):
        _my.ctx.end_launch(self)
        if exc_value is None:
            self.launch()
            del self.launcher

    def spawn_task(self, *args, **kwargs):
        # TODO: Support index launches
        TaskLaunch().spawn_task(*args, **kwargs)

        # TODO: Support return values

    def attach_task_launcher(self, *args, **kwargs):
        self.launcher.attach_task_launcher(*args, **kwargs)

    def attach_index_launcher(self, *args, **kwargs):
        self.launcher.attach_index_launcher(*args, **kwargs)

    def launch(self):
        self.launcher.launch()

def execution_fence(block=False, future=False):
    f = Future.from_cdata(
        c.legion_runtime_issue_execution_fence(_my.ctx.runtime, _my.ctx.context),
        value_type=void)
    if block or future:
        if block:
            f.get()
        if future:
            return f

def print_once(*args, **kwargs):
    fd = (kwargs['file'] if 'file' in kwargs else sys.stdout).fileno()
    message = StringIO()
    kwargs['file'] = message
    print(*args, **kwargs)
    c.legion_runtime_print_once_fd(_my.ctx.runtime, _my.ctx.context, fd, 'w'.encode('utf-8'), message.getvalue().encode('utf-8'))

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

class Trace(object):
    __slots__ = ['trace_id']

    def __init__(self):
        self.trace_id = _my.ctx.next_trace_id
        _my.ctx.next_trace_id += 1

    def __enter__(self):
        c.legion_runtime_begin_trace(_my.ctx.runtime, _my.ctx.context, self.trace_id, True)

    def __exit__(self, exc_type, exc_value, tb):
        c.legion_runtime_end_trace(_my.ctx.runtime, _my.ctx.context, self.trace_id)

if is_script:
    _my.ctx = Context(
        legion_top.top_level.context,
        legion_top.top_level.runtime,
        legion_top.top_level.task,
        [])

    def _cleanup():
        del _my.ctx

    legion_top.add_cleanup_item(_cleanup)

    # FIXME: Really this should be the number of control replicated shards at this level
    c.legion_runtime_enable_scheduler_lock()
    num_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    c.legion_runtime_disable_scheduler_lock()

    if num_procs > 1 and not legion_top.is_control_replicated():
        raise RuntimeError("Pygion must be executed with control replication for multi-process runs")

    global_task_registration_barrier = c.legion_phase_barrier_create(_my.ctx.runtime, _my.ctx.context, num_procs)
