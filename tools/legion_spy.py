#!/usr/bin/env python

# Copyright 2016 Stanford University, NVIDIA Corporation
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

import subprocess
import sys, os, re, shutil
import string
import tempfile
from getopt import getopt

# These are imported from legion_types.h
NO_DEPENDENCE = 0
TRUE_DEPENDENCE = 1
ANTI_DEPENDENCE = 2
ATOMIC_DEPENDENCE = 3
SIMULTANEOUS_DEPENDENCE = 4
DEPENDENCE_TYPES = [
"no",
"true",
"anti",
"atomic",
"simultaneous",
]

NO_ACCESS  = 0x00000000
READ_ONLY  = 0x00000001
READ_WRITE = 0x00000007
WRITE_ONLY = 0x00000002
REDUCE     = 0x00000004

EXCLUSIVE = 0
ATOMIC = 1
SIMULTANEOUS = 2
RELAXED = 3

NO_OP_KIND = 0
SINGLE_TASK_KIND = 1
INDEX_TASK_KIND = 2
MAP_OP_KIND = 3
CLOSE_OP_KIND = 4
FENCE_OP_KIND = 5
COPY_OP_KIND = 6
FILL_OP_KIND = 7
ACQUIRE_OP_KIND = 8
RELEASE_OP_KIND = 9
DELETION_OP_KIND = 10
DEP_PART_KIND = 11
PENDING_PART_KIND = 12

OpNames = [
"Unknown Kind",
"Single Task",
"Index Task",
"Map Op",
"Close Op",
"Fence Op",
"Copy Op",
"Fill Op",
"Acquire Op",
"Release Op",
"Deletion Op",
"Dependent Partition Op",
"Pending Partition Op",
]


class Point(object):
    __slots__ = ['dim', 'values']
    def __init__(self, dim):
        self.dim = dim
        self.values = list()

    def add_value(self, val):
        self.values.append(val)
        assert(len(self.values) <= self.dim)

    def mk_string(self, start, delim, end):
        result = start
        first = True
        for val in self.values:
            if not first:
                result = result + delim
            result = result + str(val)
            first = False
        result = result + end
        return result

    def to_string(self):
        return self.mk_string("(", ",", ")")

    def to_dim_string(self):
        return self.mk_string("", "][", "")

    def to_simple_string(self):
        return self.mk_string("", "_", "")

    def __hash__(self):
        return hash(self.to_simple_string())

class Processor(object):
    __slots__ = ['state', 'uid', 'kind', 'mem_latency', 'mem_bandwidth']
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.kind = "Unknown"
        self.mem_latency = dict()
        self.mem_bandwidth = dict()

    def set_kind(self, kind):
        self.kind = kind

    def add_memory(self, mem, bandwidth, latency):
        self.mem_latency[mem] = latency
        self.mem_bandwidth[mem] = bandwidth

class Memory(object):
    __slots__ = ['state', 'uid', 'kind', 'capacity', 'proc_latency',
                 'proc_bandwidth', 'mem_latency', 'mem_bandwidth']
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.kind = "Unknown"
        self.capacity = -1
        self.proc_latency = dict()
        self.proc_bandwidth = dict()
        self.mem_latency = dict()
        self.mem_bandwidth = dict()

    def set_kind(self, kind):
        self.kind = kind

    def set_capacity(self, capacity):
        self.capacity = capacity

    def add_processor(self, proc, bandwidth, latency):
        self.proc_latency[proc] = latency
        self.proc_bandwidth[proc] = bandwidth

    def add_memory(self, mem, bandwidth, latency):
        assert mem is not self
        self.mem_latency[mem] = latency
        self.mem_bandwidth[mem] = bandwidth

class IndexSpace(object):
    __slots__ = ['state', 'uid', 'parent', 'color', 'children', 'instances', 'name']
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.parent = None
        self.color = Point(0) 
        self.children = dict()
        self.instances = dict()
        self.name = None

    def set_name(self, name):
        self.name = name

    def set_parent(self, parent, color):
        self.parent = parent
        self.color = color
        self.parent.add_child(self)
        # Update any instances
        for tid,region in self.instances.iteritems():
            parent = self.state.get_partition(parent.uid, 
                region.field_space.uid, tid)
            region.set_parent(parent)

    def add_child(self, child):
        self.children[child.color] = child

    def add_instance(self, tid, region):
        self.instances[tid] = region

    def __str__(self):
        if self.name is None:
            return "Index Space "+str(self.uid)
        else:
          return self.name

    __repr__ = __str__

class IndexPartition(object):
    __slots__ = ['state', 'uid', 'parent', 'color', 'children', 'instances', 
                 'disjoint', 'complete', 'name']
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.parent = None
        self.color = Point(0)
        self.children = dict()
        self.instances = dict()
        self.disjoint = False
        self.complete = False
        self.name = None

    def set_parent(self, parent, color):
        self.parent = parent
        self.color = color
        self.parent.add_child(self)
        # Update any instances
        for tid,partition in self.instances.iteritems():
            parent = self.state.get_region(parent.uid,
                partition.field_space.uid, tid)
            region.set_parent(parent)

    def set_disjoint(self, disjoint):
        self.disjoint = disjoint

    def set_complete(self, complete):
        self.complete = complete

    def set_name(self, name):
        self.name = name

    def add_child(self, child):
        self.children[child.color] = child

    def add_instance(self, tid, partition):
        self.instances[tid] = partition

    def __str__(self):
        if self.name is None:
            return "Index Partition: "+str(self.uid)
        else:
            return self.name

    __repr__ = __str__

class Field(object):
    __slots__ = ['space', 'fid', 'name']
    def __init__(self, space, fid):
        self.space = space
        self.fid = fid
        self.name = None

    def set_name(self, name):
        self.name = name

    def __str__(self):
        if self.name is None:
            return "Field "+str(self.fid)
        else:
            return self.name

    __repr__ = __str__

class FieldSpace(object):
    __slots__ = ['state', 'uid', 'name', 'fields']
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.name = None
        self.fields = dict()

    def set_name(self, name):
        self.name = name

    def get_field(self, fid):
        if fid in self.fields:
            return self.fields[fid]
        field = Field(self, fid)
        self.fields[fid] = field
        return field

    def __str__(self):
        if self.name is None:
            return "Field Space "+str(self.uid)
        else:
            return self.name

    __repr__ = __str__

class LogicalRegion(object):
    __slots__ = ['state', 'index_space', 'field_space', 'tree_id', 'name', 'parent']
    def __init__(self, state, iid, fid, tid):
        self.state = state
        self.index_space = iid
        self.field_space = fid
        self.tree_id = tid
        self.name = None
        self.parent = None
        self.index_space.add_instance(self.tree_id, self)

    def set_name(self, name):
        self.name = name

    def set_parent(self, parent):
        self.parent = parent

    def __str__(self):
        if self.name is None:
            return "Region (%d,%d,%d)" % (self.index_space.uid,
                self.field_space.uid,self.tree_id)
        else:
            return self.name

    __repr__ = __str__

class LogicalPartition(object):
    __slots__ = ['state', 'index_partition', 'field_space', 'tree_id', 'name', 'parent']
    def __init__(self, state, iid, fid, tid):
        self.state = state
        self.index_partition = iid
        self.field_space = fid
        self.tree_id = tid
        self.name = None 
        self.parent = None
        self.index_partition.add_instance(self.tree_id, self)

    def set_name(self, name):
        self.name = name

    def set_parent(self, parent):
        self.parent = parent

    def __str__(self):
        if self.name is None:
            return "Partition (%d,%d,%d)" % (self.index_partition.uid,
                self.field_space.uid, self.tid)
        else:
            return self.name

    __repr__ = __str__

class Requirement(object):
    __slots__ = ['state', 'index', 'is_reg', 'index_node', 'field_space', 'tid',
                 'logical_node', 'priv', 'coher', 'redop', 'fields']
    def __init__(self, state, index, is_reg, index_node, field_space, 
                 tid, logical_node, priv, coher, redop):
        self.state = state
        self.index = index
        self.is_reg = is_reg
        self.index_node = index_node
        self.field_space = field_space
        self.tid = tid
        self.logical_node = logical_node 
        self.priv = priv
        self.coher = coher
        self.redop = redop
        self.fields = list()

    def print_requirement(self):
        if self.is_reg:
            print "        Logical Region Requirement ("+hex(self.ispace)+","+\
                  str(self.fspace)+","+str(self.tid)+")"
        else:
            print "        Logical Partition Requirement ("+str(self.ispace)+","+\
                  str(self.fspace)+","+str(self.tid)+")"
        field_str = "          Fields: "
        first = True
        for f in self.fields:
            if first:
                field_str = field_str+str(f)
                first = False
            else:
                field_str = field_str+", "+str(f)
        print field_str
        print "        Privilege: "+self.get_privilege()
        print "        Coherence: "+self.get_coherence()

    def add_field(self, fid):
        field = self.field_space.get_field(fid)
        self.fields.append(field)

    def is_no_access(self):
        return self.priv == NO_ACCESS

    def is_read_only(self):
        return self.priv == READ_ONLY

    def has_write(self):
        return (self.priv == READ_WRITE) or (self.priv == REDUCE) or \
                (self.priv == WRITE_ONLY)

    def is_write(self):
        return (self.priv == READ_WRITE) or (self.priv == WRITE_ONLY)

    def is_read_write(self):
        return self.priv == READ_WRITE

    def is_write_only(self):
        return self.priv == WRITE_ONLY

    def is_reduce(self):
        return self.priv == REDUCE

    def is_exclusive(self):
        return self.coher == EXCLUSIVE

    def is_atomic(self):
        return self.coher == ATOMIC

    def is_simult(self):
        return self.coher == SIMULTANEOUS

    def is_relaxed(self):
        return self.coher == RELAXED

    def to_string(self):
        if self.is_reg:
            print "Region Requirement for ("+hex(self.ispace)+","+\
                  str(self.fspace)+","+str(self.tid)+")"
        else:
            print "Partition Requirement for ("+str(self.ispace)+","+\
                  str(self.fspace)+","+str(self.tid)+")"
        print "    Privilege: "+self.get_privilege()
        print "    Coherence: "+self.get_coherence()
        print "    Fields: "+" ".join(self.get_field_names())

    def get_privilege(self):
        if self.priv == NO_ACCESS:
            return "NO ACCESS"
        elif self.priv == READ_ONLY:
            return "READ-ONLY"
        elif self.priv == READ_WRITE:
            return "READ-WRITE"
        elif self.priv == WRITE_ONLY:
            return "WRITE-ONLY"
        else:
            assert self.priv == REDUCE
            return "REDUCE with Reduction Op "+str(self.redop)

    def get_coherence(self):
        if self.coher == EXCLUSIVE:
            return "EXCLUSIVE"
        elif self.coher == ATOMIC:
            return "ATOMIC"
        elif self.coher == SIMULTANEOUS:
            return "SIMULTANEOUS"
        else:
            assert self.coher == RELAXED
            return "RELAXED"

    def get_privilege_and_coherence(self):
        result = ''
        if self.priv == NO_ACCESS:
            result = result + "NA"
        elif self.priv == READ_ONLY:
            result = result + "RO"
        elif self.priv == READ_WRITE:
            result = result + "RW"
        elif self.priv == WRITE_ONLY:
            result = result + "WO"
        else:
            assert self.priv == REDUCE
            result = result + "Red"+str(self.redop)+'-'
        if self.coher == EXCLUSIVE:
            result = result + "E"
        elif self.coher == ATOMIC:
            result = result + "A"
        elif self.coher == SIMULTANEOUS:
            result = result + "S"
        else:
            assert self.coher == RELAXED
            result = result + "R"
        return result

class MappingDependence(object):
    __slots__ = ['op1', 'op2', 'idx1', 'idx2', 'dtype']
    def __init__(self, op1, op2, idx1, idx2, dtype):
        self.op1 = op1
        self.op2 = op2
        self.idx1 = idx1
        self.idx2 = idx2

    def __eq__(self,other):
        return (self.op1 is other.op1) and \
               (self.op2 is other.op2) and \
               (self.idx1 == other.idx1) and \
               (self.idx2 == other.idx2) and \
               (self.dtype == other.dtype)

    def __str__(self):
        return "index %d of %s and index %d of %s (type: %s)" % \
                (self.idx1, self.op1.get_name(),
                 self.idx2, self.op2.get_name(),
                 DEPENDENCE_TYPES[self.dtype])

    __repr__ = __str__

        
class Operation(object):
    __slots__ = ['state', 'uid', 'kind', 'context', 'name', 'reqs', 
                 'logical_incoming', 'logical_outgoing', 'task_id', 
                 'points', 'creator']
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.kind = NO_OP_KIND 
        self.context = None
        self.name = None
        self.reqs = None
        self.logical_incoming = None
        self.logical_outgoing = None
        # Only valid for tasks
        self.task_id = -1
        # Only valid for index tasks
        self.points = None
        # Only valid for close operations
        self.creator = None

    def set_name(self, name):
        self.name = name

    def __str__(self):
        if self.name is None:
            return OpNames[self.kind] + " " + str(self.uid)
        else:
            return self.name

    __repr__ = __str__

    def set_context(self, context, add=True):
        self.context = context
        if add:
          self.context.add_operation(self)

    def set_op_kind(self, kind):
        if self.kind == NO_OP_KIND:
            self.kind = kind
        else:
            assert self.kind is kind

    def set_task_id(self, task_id):
        assert self.kind == SINGLE_TASK_KIND or self.kind == INDEX_TASK_KIND
        self.task_id = task_id

    def set_creator(self, creator):
        assert self.kind == CLOSE_OP_KIND
        self.creator = creator

    def add_point_task(self, point):
        assert self.kind == INDEX_TASK_KIND
        # Initialize if necessary
        if self.points is None:
            self.points = dict()
        index_point = point.point
        if index_point in self.points:
            self.points[index_point] = self.state.alias_points(point,
                                              self.points[index_point])
        else:
            self.points[index_point] = point

    def add_requirement(self, requirement):
        if self.reqs is None:
            self.reqs = dict()
        assert requirement.index not in self.reqs
        self.reqs[requirement.index] = requirement

    def add_requirement_field(self, index, fid):
        assert self.reqs is not None
        assert index in self.reqs
        self.reqs[index].add_field(fid)

    def add_logical_incoming(self, dep):
        assert dep.op2 == self
        if self.logical_incoming is None:
            self.logical_incoming = set()
        self.logical_incoming.add(dep)

    def add_logical_outgoing(self, dep):
        assert dep.op1 == self
        if self.logical_outgoing is None:
            self.logical_outgoing = set()
        self.logical_outgoing.add(dep)

    def merge(self, other):
        if self.kind == NO_OP_KIND:
            self.kind = other.kind
        elif other.kind <> NO_OP_KIND:
            assert self.kind == other.kind
        if self.context is None:
            self.context = other.context
        elif other.context is not None:
            assert self.context == other.context
        if self.name is None:
            self.name = other.name
        if len(self.reqs) == 0:
            self.reqs = other.reqs
        else:
            assert len(other.reqs) == 0
        if self.task_id == -1:
            self.task_id = other.task_id
        elif other.task_id <> -1:
            assert self.task_id == other.task_id
        # Should only be called on point tasks
        assert len(self.points) == 0
        assert len(other.points) == 0
        

class Task(object):
    __slots__ = ['op', 'task_id', 'point', 'operations']
    def __init__(self, state, op):
        self.op = op
        self.task_id = -1 
        self.point = Point(0)
        self.operations = list()

    def __str__(self):
        return "Task %d (UID %d)" % (self.task_id, self.op.uid)

    __repr__ = __str__

    def set_task_id(self, task_id):
        self.task_id = task_id
        self.op.set_task_id(task_id)

    def add_operation(self, operation):
        self.operations.append(operation)

    def set_point(self, point):
        self.point = point

    def merge(self, other):
        if self.task_id == -1:
            self.task_id = other.task_id
        elif other.task_id <> -1:
            assert self.task_id == other.task_id
        if self.point.dim == 0:
            self.point = other.point
        elif other.point.dim <> 0:
            assert self.point == other.point
        if len(self.operations) == 0:
            self.operations = other.operations
        else:
            assert len(other.operations) == 0

prefix    = "\[(?P<node>[0-9]+) - (?P<thread>[0-9a-f]+)\] \{\w+\}\{legion_spy\}: "
prefix_pat              = re.compile(prefix)
# Patterns for the shape of the machine
proc_kind_pat           = re.compile(prefix+"Processor Kind (?P<kind>[0-9]+) (?P<name>[-$()\w. ])")
mem_kind_pat            = re.compile(prefix+"Memory Kind (?P<kind>[0-9]+) (?P<name>[-$()\w. ]+)")
processor_pat           = re.compile(prefix+"Processor (?P<pid>[0-9a-f]+) (?P<kind>[0-9]+)")
memory_pat              = re.compile(prefix+"Memory (?P<mid>[0-9a-f]+) (?P<capacity>[0-9]+) (?P<kind>[0-9]+)")
proc_mem_pat            = re.compile(prefix+"Processor Memory (?P<pid>[0-9a-f]+) (?P<mid>[0-9a-f]+) (?P<band>[0-9]+) (?P<lat>[0-9]+)")
mem_mem_pat             = re.compile(prefix+"Memory Memory (?P<mone>[0-9a-f]+) (?P<mtwo>[0-9a-f]+) (?P<band>[0-9]+) (?P<lat>[0-9]+)")
# Patterns for the shape of region trees
top_index_pat           = re.compile(prefix+"Index Space (?P<uid>[0-9a-f]+)")
index_name_pat          = re.compile(prefix+"Index Space Name (?P<uid>[0-9a-f]+) (?P<name>[-$()\w. ]+)")
index_part_pat          = re.compile(prefix+"Index Partition (?P<pid>[0-9a-f]+) (?P<uid>[0-9a-f]+) (?P<disjoint>[0-1]) (?P<dim>[0-9]+) (?P<val1>[0-9]+) (?P<val2>[0-9]+) (?P<val3>[0-9]+)")
index_part_name_pat     = re.compile(prefix+"Index Partition Name (?P<uid>[0-9a-f]+) (?P<name>[-$()\w. ]+)")
index_subspace_pat      = re.compile(prefix+"Index Subspace (?P<pid>[0-9a-f]+) (?P<uid>[0-9a-f]+) (?P<dim>[0-9]+) (?P<val1>[0-9]+) (?P<val2>[0-9]+) (?P<val3>[0-9]+)")
field_space_pat         = re.compile(prefix+"Field Space (?P<uid>[0-9]+)")
field_space_name_pat    = re.compile(prefix+"Field Space Name (?P<uid>[0-9]+) (?P<name>[-$()\w. ]+)")
field_create_pat        = re.compile(prefix+"Field Creation (?P<uid>[0-9]+) (?P<fid>[0-9]+)")
field_name_pat          = re.compile(prefix+"Field Name (?P<uid>[0-9]+) (?P<fid>[0-9]+) (?P<name>[-$()\w. ]+)")
region_pat              = re.compile(prefix+"Region (?P<iid>[0-9a-f]+) (?P<fid>[0-9]+) (?P<tid>[0-9]+)")
region_name_pat         = re.compile(prefix+"Logical Region Name (?P<iid>[0-9a-f]+) (?P<fid>[0-9]+) (?P<tid>[0-9]+) (?P<name>[-$()\w. ]+)")
partition_name_pat      = re.compile(prefix+"Logical Partition Name (?P<iid>[0-9a-f]+) (?P<fid>[0-9]+) (?P<tid>[0-9]+) (?P<name>[-$()\w. ]+)")
# Patterns for operations
top_task_pat             = re.compile(prefix+"Top Task (?P<tid>[0-9]+) (?P<uid>[0-9]+) (?P<name>[-$()\w. ]+)")
single_task_pat          = re.compile(prefix+"Individual Task (?P<ctx>[0-9]+) (?P<tid>[0-9]+) (?P<uid>[0-9]+) (?P<name>[-$()\w. ]+)")
index_task_pat           = re.compile(prefix+"Index Task (?P<ctx>[0-9]+) (?P<tid>[0-9]+) (?P<uid>[0-9]+) (?P<name>[-$()\w. ]+)")
mapping_pat              = re.compile(prefix+"Mapping Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
close_pat                = re.compile(prefix+"Close Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<is_inter>[0-1])")
close_creator_pat        = re.compile(prefix+"Close Operation Creator (?P<uid>[0-9]+) (?P<cuid>[0-9]+) (?P<idx>[0-9]+)")
fence_pat                = re.compile(prefix+"Fence Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
copy_op_pat              = re.compile(prefix+"Copy Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
fill_op_pat              = re.compile(prefix+"Fill Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
acquire_op_pat           = re.compile(prefix+"Acquire Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
release_op_pat           = re.compile(prefix+"Release Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
deletion_pat             = re.compile(prefix+"Deletion Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
dep_partition_op_pat     = re.compile(prefix+"Dependent Partition Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<pid>[0-9a-f]+) (?P<kind>[0-9]+)")
pending_partition_op_pat = re.compile(prefix+"Pending Partition Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
target_partition_pat     = re.compile(prefix+"Pending Partition Target (?P<uid>[0-9]+) (?P<pid>[0-9a-f]+) (?P<kind>[0-9]+)")
index_slice_pat          = re.compile(prefix+"Index Slice (?P<index>[0-9]+) (?P<slice>[0-9]+)")
slice_slice_pat          = re.compile(prefix+"Slice Slice (?P<slice1>[0-9]+) (?P<slice2>[0-9]+)")
slice_point_pat          = re.compile(prefix+"Slice Point (?P<slice>[0-9]+) (?P<point>[0-9]+) (?P<dim>[0-9]+) (?P<val1>[0-9]+) (?P<val2>[0-9]+) (?P<val3>[0-9]+)")
point_point_pat          = re.compile(prefix+"Point Point (?P<point1>[0-9]+) (?P<point2>[0-9]+)")
# Patterns for logical analysis and region requirements
requirement_pat         = re.compile(prefix+"Logical Requirement (?P<uid>[0-9]+) (?P<index>[0-9]+) (?P<is_reg>[0-1]) (?P<ispace>[0-9a-f]+) (?P<fspace>[0-9]+) (?P<tid>[0-9]+) (?P<priv>[0-9]+) (?P<coher>[0-9]+) (?P<redop>[0-9]+)")
req_field_pat           = re.compile(prefix+"Logical Requirement Field (?P<uid>[0-9]+) (?P<index>[0-9]+) (?P<fid>[0-9]+)")
mapping_dep_pat         = re.compile(prefix+"Mapping Dependence (?P<ctx>[0-9]+) (?P<prev_id>[0-9]+) (?P<pidx>[0-9]+) (?P<next_id>[0-9]+) (?P<nidx>[0-9]+) (?P<dtype>[0-9]+)")

def parse_legion_spy_line(line, state):
    # Quick test to see if the line is even worth considering
    m = prefix_pat.match(line)
    if m is None:
        return False
    # We order these regex matches by the frequency in which they
    # are likely to happen in order to improve parsing time

    # Region requirements and mapping dependences happen often
    m = requirement_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        is_reg = True if int(m.group('is_reg')) == 1 else False
        field_space = state.get_field_space(int(m.group('fspace')))
        tid = int(m.group('tid'))
        priv = int(m.group('priv'))
        coher = int(m.group('coher'))
        redop = int(m.group('redop'))
        if is_reg:
            index_space = state.get_index_space(int(m.group('ispace'),16))
            region = state.get_region(index_space.uid, field_space.uid, tid) 
            requirement = Requirement(state, int(m.group('index')), True,
                index_space, field_space, tid, region, priv, coher, redop)
            op.add_requirement(requirement)
        else:
            index_partition = state.get_index_partition(int(m.group('ispace'),16))
            partition = state.get_partition(index_partition.uid, field_space.uid, tid)
            requirement = Requirement(state, int(m.group('index')), False,
                index_partition, field_space, tid, partition, priv, coher, redop)
            op.add_requirement(requirement)
        return True
    m = req_field_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        index = int(m.group('index'))
        fid = int(m.group('fid'))
        op.add_requirement_field(index, fid)
        return True
    m = mapping_dep_pat.match(line)
    if m is not None:
        op1 = state.get_operation(int(m.group('prev_id')))
        op2 = state.get_operation(int(m.group('next_id')))
        dep = MappingDependence(op1, op2, int(m.group('pidx')),
            int(m.group('nidx')), int(m.group('dtype')))
        op2.add_logical_incoming(dep)
        op1.add_logical_outgoing(dep)
        return True
    # Operations near the top since they happen frequently
    m = top_task_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(SINGLE_TASK_KIND)
        op.set_name(m.group('name'))
        task = state.get_task(int(m.group('uid')))
        task.set_task_id(int(m.group('tid')))
        return True
    m = single_task_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(SINGLE_TASK_KIND)
        op.set_name(m.group('name'))
        task = state.get_task(int(m.group('uid')))
        task.set_task_id(int(m.group('tid')))
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context)
        return True
    m = index_task_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(INDEX_TASK_KIND)
        op.set_task_id(int(m.group('tid')))
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context)
        return True
    m = mapping_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(MAP_OP_KIND)
        op.set_name("Mapping Op "+m.group('uid'))
        context = state.get_task(int(m.group('ctx')))
        # only add this if it is a post close op
        is_inter = int(m.group('is_inter'))
        op.set_context(context, False if is_inter == 1 else True)
        return True
    m = close_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(CLOSE_OP_KIND)
        op.set_name("Close Op "+m.group('uid'))
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context)
        return True
    m = close_creator_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(CLOSE_OP_KIND)
        creator = state.get_operation(int(m.group('cuid')))
        op.set_creator(creator)
        return True

    m = index_slice_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('index')))
        state.slice_index[int(m.group('slice'))] = op
        return True
    m = slice_slice_pat.match(line)
    if m is not None:
        state.slice_slice[int(m.group('slice2'))] = int(m.group('slice1'))
        return True
    m = slice_point_pat.match(line)
    if m is not None:
        point = state.get_task(int(m.group('point')))
        dim = int(m.group('dim'))
        index_point = Point(dim)
        index_point.add_value(int(m.group('val1')))
        if dim > 1:
            index_point.add_value(int(m.group('val2')))
            if dim > 2:
                index_point.add_value(int(m.group('val3')))
        point.set_point(index_point)
        state.point_slice[point] = int(m.group('slice'))
        return True
    m = point_point_pat.match(line)
    if m is not None:
        p1 = state.get_task(int(m.group('point1')))
        p2 = state.get_task(int(m.group('point2')))
        state.alias_points(p1, p2)
        return True
    # Region tree shape patterns (near the bottom since they are infrequent)
    m = top_index_pat.match(line)
    if m is not None:
        state.get_index_space(int(m.group('uid'),16))
        return True
    m = index_name_pat.match(line)
    if m is not None:
        ispace = state.get_index_space(int(m.group('uid'),16))
        ispace.set_name(m.group('name'))
        return True
    m = index_part_pat.match(line)
    if m is not None:
        parent = state.get_index_space(int(m.group('pid'),16))
        part = state.get_index_partition(int(m.group('uid'),16))
        dim = int(m.group('dim'))
        color= Point(dim)
        color.add_value(int(m.group('val1')))
        if dim > 1:
            color.add_value(int(m.group('val2')))
            if dim > 2:
                color.add_value(int(m.group('val3')))
        part.set_parent(parent, color)
        part.set_disjoint(True if int(m.group('disjoint')) == 1 else False)
        return True
    m = index_part_name_pat.match(line)
    if m is not None:
        part = state.get_index_partition(int(m.group('uid'),16))
        part.set_name(m.group('name'))
        return True
    m = index_subspace_pat.match(line)
    if m is not None:
        ispace = state.get_index_space(int(m.group('uid'),16))
        parent = state.get_index_partition(int(m.group('pid'),16))
        dim = int(m.group('dim'))
        color= Point(dim)
        color.add_value(int(m.group('val1')))
        if dim > 1:
            color.add_value(int(m.group('val2')))
            if dim > 2:
                color.add_value(int(m.group('val3')))
        ispace.set_parent(parent, color)
        return True
    m = field_space_pat.match(line)
    if m is not None:
        state.get_field_space(int(m.group('uid'),16))
        return True
    m = field_space_name_pat.match(line)
    if m is not None:
        space = state.get_field_space(int(m.group('uid'),16))
        space.set_name(m.group('name'))
        return True
    m = field_create_pat.match(line)
    if m is not None:
        space = state.get_field_space(int(m.group('uid'),16))
        space.get_field(int(m.group('fid')))
        return True
    m = field_name_pat.match(line)
    if m is not None:
        space = state.get_field_space(int(m.group('uid'),16))
        field = space.get_field(int(m.group('fid')))
        field.set_name(m.group('name'))
        return True
    m = region_pat.match(line)
    if m is not None:
        state.get_region(int(m.group('iid'),16),
            int(m.group('fid')),int(m.group('tid')))
        return True
    m = region_name_pat.match(line)
    if m is not None:
        region = state.get_region(int(m.group('iid'),16),
            int(m.group('fid')),int(m.group('tid')))
        region.set_name(m.group('name'))
        return True
    m = partition_name_pat.match(line)
    if m is not None:
        partition = state.get_partition(int(m.group('iid'),16),
            int(m.group('fid')),int(m.group('tid')))
        partition.set_name(m.group('name'))
        return True
    # Machine kinds (at the bottom cause they are least likely)
    m = proc_kind_pat.match(line)
    if m is not None:
        state.record_processor_kind(int(m.group('kind')), m.group('name'))
        return True
    m = mem_kind_pat.match(line)
    if m is not None:
        state.record_memory_kind(int(m.group('kind')), m.group('name'))
        return True
    m = processor_pat.match(line)
    if m is not None:
        proc = state.get_processor(int(m.group('pid'),16))
        proc.set_kind(state.get_processor_kind(int(m.group('kind'))))
        return True
    m = memory_pat.match(line)
    if m is not None:
        mem = state.get_memory(int(m.group('mid'),16))
        mem.set_kind(state.get_memory_kind(int(m.group('kind'))))
        mem.set_capacity(int(m.group('capacity')))
        return True
    m = proc_mem_pat.match(line)
    if m is not None:
        proc = state.get_processor(int(m.group('pid'),16))
        mem = state.get_memory(int(m.group('mid'),16))
        bandwidth = int(m.group('band'))
        latency = int(m.group('lat'))
        proc.add_memory(mem, bandwidth, latency)
        mem.add_processor(proc, bandwidth, latency)
        return True
    m = mem_mem_pat.match(line)
    if m is not None:
        mem1 = state.get_memory(int(m.group('mone'),16))
        mem2 = state.get_memory(int(m.group('mtwo'),16))
        bandwidth = int(m.group('band'))
        latency = int(m.group('lat'))
        mem1.add_memory(mem2, bandwidth, latency)
        mem2.add_memory(mem1, bandwidth, latency)
        return True
      
    return False
        

class State(object):
    def __init__(self, verbose):
        self.verbose = verbose
        self.top_level_uid = None
        self.traverser_gen = 1
        # Machine things
        self.processors = dict()
        self.memories = dict()
        self.processor_kinds = dict()
        self.memory_kinds = dict()
        # Region tree things
        self.index_spaces = dict()
        self.index_partitions = dict()
        self.field_spaces = dict()
        self.regions = dict()
        self.partitions = dict()
        self.top_spaces = dict()
        self.trees = dict()
        # Logical things 
        self.ops = dict()
        self.tasks = dict()
        self.has_mapping_deps = False
        # Physical things 
        self.instances = dict()
        self.events = dict()
        # For parsing only
        self.slice_index = dict()
        self.slice_slice = dict()
        self.point_slice = dict()

    def parse_log_file(self, file_name):
        print 'Reading log file %s...' % file_name
        log = open(file_name, 'r')
        matches = 0
        for line in log:
            if parse_legion_spy_line(line, self):
                matches += 1
            elif self.verbose:
                print 'Skipping line: ' + line
        log.close()
        if matches == 0:
            print 'WARNING: file %s contained no valid lines!' % file_name
        if self.verbose:
            print 'Matched %d lines in %s' % (matches,file_name)
        return matches

    def post_parse(self):
        # Find the top-level index spaces
        num_index_trees = 0
        for space in self.index_spaces.itervalues():
            if space.parent is None:
                self.top_spaces[num_index_trees] = space
                num_index_trees += 1
        # Find the top-level regions
        for region in self.regions.itervalues():
            if region.parent is None:
                self.trees[region.tree_id] = region
        # Merge the points and slices
        for point,slice_ in self.point_slice.iteritems():
            while slice_ in self.slice_slice:
                slice_ = self.slice_slice[slice_]
            assert slice_ in self.slice_index
            self.slice_index[slice_].add_point_task(point)
        # We can delete some of these data structures now that we
        # no longer need them, go go garbage collection
        self.slice_index = None
        self.slice_slice = None
        self.point_slice = None
        if self.verbose:
            print "Found %d index space trees" % num_index_trees
            print "Found %d field spaces" % len(self.field_spaces)
            print "Found %d region trees" % len(self.trees)
            print ""
            print "Found %d tasks" % len(self.tasks)
            print "Found %d operations (including tasks)" % len(self.ops)
        logical_enabled = self.has_mapping_deps
        physical_enabled = len(self.events) > 0
        return logical_enabled,physical_enabled

    def alias_points(self, p1, p2):
        # These two tasks are aliased so merge them together 
        # Merge the operations first 
        p1.op.merge(p2.op)
        self.ops[p2.op.uid] = p1.op
        # Now merge the tasks and delete the other task
        p1.merge(p2)
        del self.tasks[p2.op]

    def perform_logical_checks(self):
        pass

    def perform_physical_checks(self):
        pass

    def make_region_tree_graphs(self):
        pass

    def make_machine_graphs(self):
        pass

    def make_dataflow_graphs(self, simplify_graphs):
        pass

    def make_event_graphs(self, simplify_graphs):
        pass

    def get_processor(self, proc_id):
        if proc_id in self.processors:
            return self.processors[proc_id]
        proc = Processor(self, proc_id)
        self.processors[proc_id] = proc
        return proc

    def get_memory(self, mem_id):
        if mem_id in self.memories:
            return self.memories[mem_id]
        mem = Memory(self, mem_id)
        self.memories[mem_id] = mem
        return mem

    def get_index_space(self, iid):
        if iid in self.index_spaces:
            return self.index_spaces[iid]
        result = IndexSpace(self, iid)
        self.index_spaces[iid] = result
        return result

    def get_index_partition(self, iid):
        if iid in self.index_partitions:
            return self.index_partitions[iid]
        result = IndexPartition(self, iid)
        self.index_partitions[iid] = result
        return result

    def get_field_space(self, fid):
        if fid in self.field_spaces:
            return self.field_spaces[fid]
        result = FieldSpace(self, fid)
        self.field_spaces[fid] = result
        return result

    def get_region(self, iid, fid, tid):
        key = (iid, fid, tid)
        if key in self.regions:
            return self.regions[key]
        ispace = self.get_index_space(iid)
        fspace = self.get_field_space(fid)
        result = LogicalRegion(self, ispace, fspace, tid)
        # See if we need to make the parent
        if ispace.parent is not None:
            parent = self.get_partition(ispace.parent.uid, fid, tid)
            result.set_parent(parent)
        self.regions[key] = result
        return result

    def get_partition(self, iid, fid, tid):
        key = (iid, fid, tid)
        if key in self.partitions:
            return self.partitions[key]
        ipart = self.get_index_partition(iid)
        fspace = self.get_field_space(fid)
        result = LogicalPartition(self, ipart, fspace, tid)
        parent = self.get_region(ipart.parent.uid, fid, tid)
        result.set_parent(parent)
        self.partitions[key] = result
        return result

    def get_operation(self, uid):
        if uid in self.ops:
            return self.ops[uid]
        result = Operation(self, uid)
        self.ops[uid] = result
        return result

    def get_task(self, uid):
        op = self.get_operation(uid)
        if op in self.tasks:
            return self.tasks[op]
        op.set_op_kind(SINGLE_TASK_KIND)
        result = Task(self, op)
        self.tasks[op] = result
        return result

    def record_processor_kind(self, kind, name):
        self.processor_kinds[kind] = name

    def record_memory_kind(self, kind, name):
        self.memory_kinds[kind] = name

    def get_processor_kind(self, kind):
        assert kind in self.processor_kinds
        return self.processor_kinds[kind]

    def get_memory_kind(self, kind):
        assert kind in self.memory_kinds
        return self.memory_kinds[kind]

def usage():
    print "Usage: "+sys.argv[0]+" [-l -p -r -m -d -e -s -k -v] <file_name>"
    print "  -l : perform logical checks"
    print "  -p : perform physical checks"
    print "  -r : make region tree graphs"
    print "  -m : make machine graphs"
    print "  -d : make dataflow graphs"
    print "  -e : make event graphs"
    print "  -s : generate simplified graphs"
    print "  -k : keep temporary files"
    print "  -v : verbose"
    sys.exit(1)

def main(temp_dir):
    if len(sys.argv) < 2:
        usage()

    opts, args = getopt(sys.argv[1:],'lprmdeskv')
    opts = dict(opts)
    if len(args) <> 1:
        usage()
    file_names = args

    logical_checks = False
    physical_checks = False
    region_tree_graphs = False
    machine_graphs = False
    dataflow_graphs = False
    event_graphs = False
    simplify_graphs = False
    keep_temp_files = False
    verbose = False
    for opt in opts:
        if opt == '-l':
            logical_checks = True
            continue
        if opt == '-p':
            physical_checks = True
            continue
        if opt == '-r':
            region_tree_graphs = True
            continue
        if opt == '-m':
            machine_graphs = True
            continue
        if opt == '-d':
            dataflow_graphs = True
            continue
        if opt == '-e':
            event_graphs = True
            continue;
        if opt == '-s':
            simplify_graphs = True
            continue
        if opt == '-k':
            keep_temp_files = True
            continue
        if opt == '-v':
            verbose = True
            continue

    state = State(verbose)
    total_matches = 0 
    for file_name in file_names:
        total_matches += state.parse_log_file(file_name)
    if verbose:
        print 'Matched %d lines across all files.' % total_matches
    if total_matches == 0:
        print 'No matches found! Exiting...'
        return
    logical_enabled,physical_enabled = state.post_parse()
    if logical_checks and not logical_enabled:
        print "WARNING: Requested logical analysis but logging information is "\
              "missing. Please compile the runtime with -DLEGION_SPY to enable "\
              "validation of the runtime." 
        logical_checks = False
    if physical_checks and not physical_enabled:
        print "WARNING: Requested physical analysis but logging information is "\
              "missing. Please compile the runtime with -DLEGION_SPY to enable "\
              "validation of the runtime."
        physical_checks = False
    if logical_checks:
        print "Performing logical checks..."
        state.perform_logical_checks()
    if physical_checks:
        print "Performing physical checks..."
        state.perform_physical_checks()
    if region_tree_graphs:
        print "Making region tree graphs..."
        state.make_region_tree_graphs()
    if machine_graphs:
        print "Making machine graphs..."
        state.make_machine_graphs()
    if dataflow_graphs:
        print "Making dataflow graphs..."
        state.make_dataflow_graphs(simplify_graphs)
    if event_graphs:
        print "Making event graphs..."
        state.make_event_graphs(simplify_graphs)

    print 'Legion Spy analysis complete.  Exiting...'
    if keep_temp_files:
        try:
            subprocess.check_call('cp', temp_dir+'.', '.')
        except:
            print 'WARNING: Unable to copy temporary files into current directory'

if __name__ == "__main__":
    temp_dir = tempfile.mkdtemp()+'/'
    try:
        sys.setrecursionlimit(10000)
        main(temp_dir)
    finally:
        shutil.rmtree(temp_dir)

# EOF

