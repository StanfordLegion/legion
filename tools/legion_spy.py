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
import sys, os, re, gc, shutil
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
DEP_PART_OP_KIND = 11
PENDING_PART_OP_KIND = 12

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
    __slots__ = ['state', 'uid', 'kind', 'mem_latency', 'mem_bandwidth', 'node_name']
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.kind = "Unknown"
        self.mem_latency = dict()
        self.mem_bandwidth = dict()
        self.node_name = 'proc_node_'+str(uid)

    def __str__(self):
        return self.kind + " Processor " + hex(self.uid)

    def set_kind(self, kind):
        self.kind = kind

    def add_memory(self, mem, bandwidth, latency):
        self.mem_latency[mem] = latency
        self.mem_bandwidth[mem] = bandwidth

    def print_node(self, printer):
        label = str(self)
        printer.println(self.node_name+' [label="'+label+
            '",shape=oval,fontsize=14,'+
            'fontcolor=black,fontname="Helvetica"];')

    def print_mem_edges(self, printer):
        for mem,band in self.mem_bandwidth.iteritems():
            label = 'bandwidth='+str(band)+',latency='+str(self.mem_latency[mem])
            printer.println(self.node_name+' -> ' + mem.node_name+ 
                ' [label="'+label+'",style=solid,color=black,penwidth=2];')
        

class Memory(object):
    __slots__ = ['state', 'uid', 'kind', 'capacity', 'proc_latency',
                 'proc_bandwidth', 'mem_latency', 'mem_bandwidth', 'node_name']
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.kind = "Unknown"
        self.capacity = -1
        self.proc_latency = dict()
        self.proc_bandwidth = dict()
        self.mem_latency = dict()
        self.mem_bandwidth = dict()
        self.node_name = 'mem_node_'+str(uid)

    def __str__(self):
        return self.kind + "Memory " + hex(self.uid)

    __repr__ = __str__

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

    def print_node(self, printer):
        label = str(self) 
        printer.println(self.node_name+' [label="'+label+
            '",shape=box,fontsize=14,'+
            'fontcolor=black,fontname="Helvetica"];')

    def print_mem_edges(self, printer):
        for mem,band in self.mem_bandwidth.iteritems():
            label = 'bandwidth='+str(band)+',latency='+str(self.mem_latency[mem])
            printer.println(self.node_name+' -> ' + mem.node_name+ 
                ' [label="'+label+'",style=solid,color=black,penwidth=2];')

class IndexSpace(object):
    __slots__ = ['state', 'uid', 'parent', 'color', 'children', 
                 'instances', 'name', 'independent_children',
                 'node_name']
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.parent = None
        self.color = Point(0) 
        self.children = dict()
        self.instances = dict()
        self.name = None
        self.independent_children = None
        self.node_name = 'index_space_node_'+str(uid)

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

    def add_independent_children(self, ip1, ip2):
        if self.independent_children is None:
            self.independent_chidlren = set()
        self.independent_children.add((ip1,ip2))
        self.independent_children.add((ip2,ip1))

    def __str__(self):
        if self.name is None:
            return "Index Space "+str(self.uid)
        else:
          return self.name

    __repr__ = __str__

    def print_link_to_parent(self, printer, parent):
        printer.println(parent+' -> '+ self.node_name+
                " [style=solid,color=black,penwidth=2];")

    def print_graph(self, printer):
        if self.name is not None:
            label = self.name + ' (ID: '+str(uid)
        else:
            if self.parent == None:
                label = 'index space '+hex(self.uid)
            else:
                color = None
                for c, child in self.parent.children.iteritems():
                    if child == self:
                        color = c
                        break
                assert color is not None
                label = 'subspace '+str(self.uid)+\
                        ' (color: ' + color.to_simple_string() +')'
        printer.println(self.node_name+' [label="'+label+
                '",shape=plaintext,fontsize=14,'+
                'fontcolor=black,fontname="Helvetica"];')
        # print links to children
        for child in self.children.itervalues():
            child.print_link_to_parent(printer, self.node_name)
        for child in self.children.itervalues():
            child.print_graph(printer)


class IndexPartition(object):
    __slots__ = ['state', 'uid', 'parent', 'color', 'children', 'instances', 
                 'disjoint', 'complete', 'name', 'independent_children',
                 'node_name']
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
        self.independent_children = None
        self.node_name = 'index_part_node_'+str(uid)

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

    def add_independent_children(self, is1, is2):
        if self.independent_children is None:
            self.independent_children = set()
        self.independent_children.add((is1,is2))
        self.independent_children.add((is2,is1))

    def __str__(self):
        if self.name is None:
            return "Index Partition: "+str(self.uid)
        else:
            return self.name

    __repr__ = __str__

    def print_link_to_parent(self, printer, parent):
        if self.disjoint:
            edge_label = '*'
        else:
            edge_label = ''
        printer.println(parent+' -> '+ self.node_name+
                ' [label="'+edge_label+'",style=dotted,color=black,penwidth=2];')

    def print_graph(self, printer):
        if self.name is not None:
            label = self.name + ' (ID: ' + str(self.uid) + ')'
        else:
            label = 'Index Partition '+str(self.uid)
        printer.println(self.node_name+' [label="'+label+
                '",shape=plaintext,fontsize=14,'+
                'fontcolor=black,fontname="times italic"];')
        # print links to children
        for child in self.children.itervalues():
            child.print_link_to_parent(printer, self.node_name)
        for child in self.children.itervalues():
            child.print_graph(printer)

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
            return self.name + " (FID=" + str(self.fid) + ")"

    __repr__ = __str__

class FieldSpace(object):
    __slots__ = ['state', 'uid', 'name', 'fields', 'node_name']
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.name = None
        self.fields = dict()
        self.node_name = 'field_space_node_'+str(uid)

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

    def print_graph(self, printer):
        if self.name is not None:
            label = self.name + ' (ID: '+str(self.uid) + ')'
        else:
            label = str(self)
        printer.println(self.node_name+' [label="'+label+
                '",shape=plaintext,fontsize=14,'+
                'fontcolor=black,fontname="Helvetica"];')

        for fid,field in self.fields.iteritems():
            field_id = "field_node_"+str(self.uid)+"_"+str(fid)
            if field.name is not None:
                field_name = field.name + '(FID: ' + str(fid) + ')'
            else:
                field_name = 'FID: ' + str(fid)
            printer.println(field_id+' [label="'+field_name+
                    '",shape=plaintext,fontsize=14,'+
                    'fontcolor=black,fontname="Helvetica"]')
            printer.println(self.node_name+' -> '+ field_id+
                    " [style=dotted,color=black,penwidth=2];")

class LogicalRegion(object):
    __slots__ = ['state', 'index_space', 'field_space', 'tree_id', 'children',
                 'name', 'parent', 'logical_state', 'physical_state', 'node_name',
                 'has_named_children']
    def __init__(self, state, iid, fid, tid):
        self.state = state
        self.index_space = iid
        self.field_space = fid
        self.tree_id = tid
        self.children = dict() 
        self.name = None
        self.parent = None
        self.logical_state = dict()
        self.physical_state = dict()
        self.index_space.add_instance(self.tree_id, self)
        self.node_name = 'region_node_'+str(self.index_space.uid)+\
            '_'+str(self.field_space.uid)+'_'+str(self.tree_id)
        self.has_named_children = False

    def set_name(self, name):
        self.name = name

    def set_parent(self, parent):
        self.parent = parent
        self.parent.add_child(self.index_space.color, self)

    def add_child(self, color, child):
        self.children[color] = child 

    def __str__(self):
        if self.name is None:
            return "Region (%d,%d,%d)" % (self.index_space.uid,
                self.field_space.uid,self.tree_id)
        else:
            return self.name

    __repr__ = __str__

    def reset_logical_state(self):
        if not not self.logical_state:
            self.logical_state = dict()

    def reset_physical_state(self):
        if not not self.logical_state:
            self.physical_state = dict()

    def compute_path(self, path, target):
        if self is not target:
            assert self.parent is not None
            self.parent.compute_path(path, target)
        path.append(self)

    def perform_logical_analysis(self, depth, path, op, req, field):
        assert self is path[depth]
        if field not in self.logical_state:
            self.logical_state[field] = LogicalState(self, field)
        if not self.logical_state[field].perform_logical_analysis(op, req):
            return False
        if (depth+1) < len(path):
            return path[depth+1].perform_logical_analysis(depth+1, path, op, req, field)
        return True

    def mark_named_children(self):
        if self.name is not None:
            self.has_named_children = True
        for child in self.children.itervalues():
            has_named_children = child.mark_named_children()
            self.has_named_children = self.has_named_children or has_named_children
        return self.has_named_children

    def print_link_to_parent(self, printer, parent):
        printer.println(parent+' -> '+ self.node_name+
                " [style=solid,color=black,penwidth=2];")

    def gen_id(self):
        return 'index: '+str(self.index_space.uid)+','+\
                'field: '+str(self.field_space.uid)+','+\
                'tree: '+str(self.tree_id)

    def print_node(self, printer):
        if self.name is not None:
            label = self.name+' ('+self.gen_id()+')'
        else:
            if self.parent == None:
                label = 'region ('+self.gen_id()+')'
            else:
                label = 'subregion ('+self.gen_id()+')'

        printer.println(self.node_name+' [label="'+label+
                '",shape=plaintext,fontsize=14,'+
                'fontcolor=black,fontname="Helvetica"];')

    def print_graph(self, printer, simplify_graph):
        if simplify_graph and not self.has_named_children:
            return
        # print node itself
        self.print_node(printer)
        # Instantiate the region tree if it isn't full
        if len(self.children) < len(self.index_space.children):
            for color,child in self.index_space.children.iteritems():
                if color not in self.children:
                    state.get_partition(child.uid, self.field_space.uid, self.tree_id)
        # print links to children
        for child in self.children.itervalues():
            if not simplify_graph or child.has_named_children:
                child.print_link_to_parent(printer, self.node_name)
        for child in self.children.itervalues():
            child.print_graph(printer, simplify_graph)

class LogicalPartition(object):
    __slots__ = ['state', 'index_partition', 'field_space', 'tree_id', 'children',
                 'name', 'parent', 'logical_state', 'physical_state', 'node_name',
                 'has_named_children']
    def __init__(self, state, iid, fid, tid):
        self.state = state
        self.index_partition = iid
        self.field_space = fid
        self.tree_id = tid
        self.children = dict()
        self.name = None 
        self.parent = None
        self.logical_state = dict()
        self.physical_state = dict()
        self.index_partition.add_instance(self.tree_id, self)
        self.node_name = 'part_node_'+str(self.index_partition.uid)+'_'+\
            str(self.field_space.uid)+'_'+str(self.tree_id)
        self.has_named_children = False

    def set_name(self, name):
        self.name = name

    def set_parent(self, parent):
        self.parent = parent
        self.parent.add_child(self.index_partition.color, self)

    def add_child(self, color, child):
        self.children[color] = child

    def __str__(self):
        if self.name is None:
            return "Partition (%d,%d,%d)" % (self.index_partition.uid,
                self.field_space.uid, self.tid)
        else:
            return self.name

    __repr__ = __str__

    def reset_logical_state(self):
        if not not self.logical_state:
            self.logical_state = dict()

    def reset_physical_state(self):
        if not not self.logical_state:
            self.physical_state = dict()

    def compute_path(self, path, target):
        if self is not target:
            assert self.parent is not None
            self.parent.compute_path(path, target)
        path.append(self)

    def perform_logical_analysis(self, depth, path, op, req, field):
        assert self is path[depth]
        if field not in self.logical_state:
            self.logical_state[field] = LogicalState(self, field)
        if not self.logical_state[field].perform_logical_analysis(op, req):
            return False
        if (depth+1) < len(path):
            return path[depth+1].perform_logical_analysis(depth+1, path, op, req, field)
        return True

    def mark_named_children(self):
        if self.name is not None:
            self.has_named_children = True
        for child in self.children.itervalues():
            has_named_children = child.mark_named_children()
            self.has_named_children = self.has_named_children or has_named_children
        return self.has_named_children

    def print_link_to_parent(self, printer, parent):
        if self.index_partition.disjoint:
            edge_label = '*'
        else:
            edge_label = ''
        printer.println(parent+' -> '+ self.node_name+
                ' [label="'+edge_label+'",style=dotted,color=black,penwidth=2];')

    def gen_id(self):
        return 'part: '+hex(self.index_partition.uid)+','+\
                'field: '+str(self.field_space.uid)+','+\
                'tree: '+str(self.tree_id)

    def print_node(self, printer):
        if self.name is not None:
            label = self.name+' ('+self.gen_id() +')'
        else:
            label = 'partition ('+self.gen_id() +')'
        printer.println(self.node_name+' [label="'+label+
                '",shape=plaintext,fontsize=14,'+
                'fontcolor=black,fontname="times italic"];')

    def print_graph(self, printer, simplify_graph):
        if simplify_graph and not self.has_named_children:
            return
        # print node itself
        self.print_node(printer)
        # instantiate the region tree if it isn't full
        if len(self.children) < len(self.index_partition.children):
            for color,child in self.index_partition.children.iteritems():
                if color not in self.children:
                    self.state.get_region(child.uid,
                        self.field_space.uid, self.tree_id)
        # print links to children
        for child in self.children.itervalues():
            child.print_link_to_parent(printer, self.node_name)
            if simplify_graph and not child.has_named_children:
                child.print_node(printer)
        for child in self.children.itervalues():
            child.print_graph(printer, simplify_graph)


class LogicalState(object):
    __slots__ = ['node', 'field']
    def __init__(self, node, field):
        self.node = node
        self.field = field

    def perform_logical_analysis(self, op, req):
        return True


class PhysicalState(object):
    __slots__ = ['node', 'field']
    def __init__(self, node, field):
        self.node = node 
        self.field = field


class Requirement(object):
    __slots__ = ['state', 'index', 'is_reg', 'index_node', 'field_space', 'tid',
                 'logical_node', 'priv', 'coher', 'redop', 'fields', 'parent']
    def __init__(self, state, index, is_reg, index_node, field_space, 
                 tid, logical_node, priv, coher, redop, parent):
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
        self.parent = parent
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
    __slots__ = ['state', 'uid', 'kind', 'context', 'name', 'reqs', 'mappings', 
                 'incoming', 'outgoing', 'start_event', 'finish_event', 'task_id', 
                 'points', 'creator', 'partition_kind', 'partition_node']
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.kind = NO_OP_KIND 
        self.context = None
        self.name = None
        self.reqs = None
        self.mappings = None
        self.incoming = None
        self.outgoing = None
        self.start_event = state.get_no_event() 
        self.finish_event = state.get_no_event()
        # Only valid for tasks
        self.task_id = -1
        # Only valid for index tasks
        self.points = None
        # Only valid for close operations
        self.creator = None
        # Only valid for pending partition operations
        self.partition_kind = None
        self.partition_node = None

    def set_name(self, name):
        self.name = name
        if self.points is not None:
            for point in self.points.itervalues():
                point.set_name(name)

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

    def set_events(self, start, finish):
        self.start_event = start
        self.finish_event = finish

    def set_task_id(self, task_id):
        assert self.kind == SINGLE_TASK_KIND or self.kind == INDEX_TASK_KIND
        self.task_id = task_id

    def set_creator(self, creator):
        assert self.kind == CLOSE_OP_KIND
        self.creator = creator

    def set_pending_partition_info(self, node, kind):
        assert self.kind == PENDING_PART_OP_KIND
        self.partition_node = node
        self.partition_kind = kind

    def add_point_task(self, point):
        assert self.kind == INDEX_TASK_KIND
        # Initialize if necessary
        if self.points is None:
            self.points = dict()
        point.op.set_name(self.name)
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

    def add_mapping_decision(self, index, fid, inst):
        if self.mappings is None:
            self.mappings = dict()
        if index not in self.mappings:
            self.mappings[index] = dict()
        self.mappings[index][fid] = inst

    def add_incoming(self, dep):
        assert dep.op2 == self
        if self.incoming is None:
            self.incoming = set()
        self.incoming.add(dep)

    def add_outgoing(self, dep):
        assert dep.op1 == self
        if self.outgoing is None:
            self.outgoing = set()
        self.outgoing.add(dep)

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
        if not self.reqs:
            self.reqs = other.reqs
        else:
            assert not other.reqs
        if self.task_id == -1:
            self.task_id = other.task_id
        elif other.task_id <> -1:
            assert self.task_id == other.task_id
        # Should only be called on point tasks
        assert not self.points
        assert not other.points

    def perform_logical_analysis(self):
        if self.reqs is None:
            return True
        # We need a context to do this
        assert self.context is not None
        success = True
        for idx,req in self.reqs.iteritems():
            # Special out for no access
            if req.priv is NO_ACCESS:
                continue
            # Compute the analysis path
            path = list()
            req.logical_node.compute_path(path, req.parent)
            assert not not path
            # Now do the traversal for each of the fields
            for field in req.fields:
                if not req.parent.perform_logical_analysis(0, path, self, req, field):
                    success = False
                    break
            # Early out
            if not success:
                break
        return success
        

class Task(object):
    __slots__ = ['op', 'point', 'operations', 'top']
    def __init__(self, state, op):
        self.op = op
        self.point = Point(0)
        self.operations = list()
        self.top = False

    def __str__(self):
        return str(self.op)

    __repr__ = __str__

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
        if not self.operations:
            self.operations = other.operations
        else:
            assert not other.operations

    def check_logical_dependence_analysis(self):
        # If we don't have any operations we are done
        if not self.operations or self.top:
            return True
        print 'Performing logical dependence analysis for %s...' % str(self)
        if self.op.state.verbose:
            print '  Analyzing %d operations...' % len(self.operations)
        # Iterate over all the operations in order and
        # have them perform their analysis
        success = True
        for op in self.operations:
            if not op.perform_logical_analysis():
                success = False
                break
        # Reset the logical state when we are done
        self.op.state.reset_logical_state()
        return success
    
    def check_logical_sanity(self):
        if not self.operations:
            return True
        success = True

        return success

class Instance(object):
    __slots__ = ['state', 'handle', 'memory', 'region', 'fields', 'redop']
    def __init__(self, state, handle):
        self.state = state
        self.handle = handle
        self.memory = None
        self.region = None
        self.fields = None
        self.redop = 0

    def __str__(self):
        return "Instance %s in %s" % (hex(self.handle), str(self.memory))

    def set_memory(self, memory):
        self.memory = memory

    def set_region(self, region):
        self.region = region

    def set_redop(self, redop):
        self.redop = redop

    def add_field(self, fid):
        # We better have a region at this point
        assert self.region is not None
        field = self.region.field_space.get_field(fid)
        if self.fields is None:
            self.fields = set()
        self.fields.add(field)

    # Only one virtual instance always with ID 0
    def is_virtual(self):
        return self.handle == 0

class EventHandle(object):
    __slots__ = ['uid', 'gen']
    def __init__(self, uid, gen):
        self.uid = uid
        self.gen = gen

    def __hash__(self):
        return hash((self.uid, self.gen))

    def __eq__(self, other):
        return (self.uid,self.gen) == (other.uid,other.gen)

    def __str__(self):
        return "ev(" + hex(self.uid) + "," + str(self.gen) + ")"

    __repr__ = __str__

    def exists(self):
        return (self.uid <> 0)

class Event(object):
    __slots__ = ['state', 'handle', 'phase_barrier', 'incoming', 'outgoing' ]
    def __init__(self, state, handle):
        self.state = state
        self.handle = handle
        self.phase_barrier = False
        self.incoming = None
        self.outgoing = None

    def __str__(self):
        return str(self.handle)

    __repr__ = __str__

    def add_incoming(self, prev):
        if self.incoming is None:
            self.incoming = set()
        self.incoming.add(prev)

    def add_outgoing(self, nex):
        if self.outgoing is None:
            self.outgoing = set()
        self.outgoing.add(nex)

class RealmCopy(object):
    __slots__ = ['state', 'creator', 'region', 'start_event', 'finish_event',
                 'src_fields', 'dst_fields', 'srcs', 'dsts', 'redops']
    def __init__(self, state, finish):
        self.state = state
        self.creator = None
        self.region = None
        self.start_event = state.get_no_event()
        self.finish_event = finish
        self.src_fields = list()
        self.dst_fields = list()
        self.srcs = list()
        self.dsts = list()
        self.redops = list()

    def set_creator(self, creator):
        self.creator = creator

    def set_start(self, start):
        self.start_event = start

    def set_region(self, region):
        self.region = region

    def add_field(self, src_fid, src, dst_fid, dst, redop):
        assert self.region is not None
        src_field = self.region.field_space.get_field(src_fid)
        dst_field = self.region.field_space.get_field(dst_fid)
        self.src_fields.append(src_field)
        self.dst_fields.append(dst_field)
        self.srcs.append(src)
        self.dsts.append(dst)
        self.redops.append(redop)

class RealmFill(object):
    __slots__ = ['state', 'creator', 'region', 'start_event', 'finish_event',
                 'fields', 'dsts', ]
    def __init__(self, state, finish):
        self.state = state
        self.creator = None
        self.region = None
        self.start_event = state.get_no_event()
        self.finish_event = finish
        self.fields = list()
        self.dsts = list()

    def set_creator(self, creator):
        self.creator = creator

    def set_start(self, start):
        self.start_event = start

    def set_region(self, region):
        self.region = region

    def add_field(self, fid, dst):
        assert self.region is not None
        field = self.region.field_space.get_field(fid)
        self.fields.append(field)
        self.dsts.append(dst)

class GraphPrinter(object):
    __slots__ = ['name', 'filename', 'out', 'depth', 'cluster_id']
    def __init__(self,path,name,direction='LR'):
        self.name = name
        self.filename = path+name+'.dot'
        self.out = open(self.filename,'w')
        self.depth = 0
        self.println('digraph "'+name+'"')
        self.println('{')
        self.down()
        #self.println('aspect = ".00001,100";')
        #self.println('ratio = 1;')
        #self.println('size = "10,10";')
        self.println('compound = true;')
        self.println('rankdir="'+direction+'";')
        self.println('size = "36,36";')
        self.cluster_id = 0

    def close(self):
        self.up()
        self.println('}')
        self.out.close()
        return self.filename

    def print_pdf_after_close(self, simplify):
        dot_file = self.close()
        pdf_file = self.name+".pdf"
        try:
            if simplify:
                tred = subprocess.Popen(['tred', dot_file], stdout=subprocess.PIPE)
                dot = subprocess.Popen(['dot', '-Tpdf', '-o', pdf_file], stdin=tred.stdout)
                if dot.wait() != 0:
                    raise Exception('DOT failed')
            else:
                subprocess.check_call(['dot', '-Tpdf', '-o', pdf_file, dot_file])
        except:
            print "WARNING: DOT failure, image for graph "+str(self.name)+" not generated"
            subprocess.call(['rm', '-f', 'core', pdf_file])

    def up(self):
        assert self.depth > 0
        self.depth = self.depth-1

    def down(self):
        self.depth = self.depth+1

    def start_new_cluster(self):
        self.println('subgraph cluster_' + str(self.cluster_id))
        self.cluster_id += 1
        self.println('{')
        self.down()

    def end_this_cluster(self):
        self.up()
        self.println('}')

    def println(self,string):
        for i in range(self.depth):
            self.out.write('  ')
        self.out.write(string)
        self.out.write('\n')

prefix    = "\[(?P<node>[0-9]+) - (?P<thread>[0-9a-f]+)\] \{\w+\}\{legion_spy\}: "
prefix_pat               = re.compile(prefix)
# Patterns for the shape of the machine
proc_kind_pat            = re.compile(
    prefix+"Processor Kind (?P<kind>[0-9]+) (?P<name>[-$()\w. ]+)")
mem_kind_pat             = re.compile(
    prefix+"Memory Kind (?P<kind>[0-9]+) (?P<name>[-$()\w. ]+)")
processor_pat            = re.compile(
    prefix+"Processor (?P<pid>[0-9a-f]+) (?P<kind>[0-9]+)")
memory_pat               = re.compile(
    prefix+"Memory (?P<mid>[0-9a-f]+) (?P<capacity>[0-9]+) (?P<kind>[0-9]+)")
proc_mem_pat             = re.compile(
    prefix+"Processor Memory (?P<pid>[0-9a-f]+) (?P<mid>[0-9a-f]+) (?P<band>[0-9]+) "+
           "(?P<lat>[0-9]+)")
mem_mem_pat              = re.compile(
    prefix+"Memory Memory (?P<mone>[0-9a-f]+) (?P<mtwo>[0-9a-f]+) (?P<band>[0-9]+) "+
           "(?P<lat>[0-9]+)")
# Patterns for the shape of region trees
top_index_pat            = re.compile(
    prefix+"Index Space (?P<uid>[0-9a-f]+)")
index_name_pat           = re.compile(
    prefix+"Index Space Name (?P<uid>[0-9a-f]+) (?P<name>[-$()\w. ]+)")
index_part_pat           = re.compile(
    prefix+"Index Partition (?P<pid>[0-9a-f]+) (?P<uid>[0-9a-f]+) (?P<disjoint>[0-1]) "+
           "(?P<dim>[0-9]+) (?P<val1>[0-9]+) (?P<val2>[0-9]+) (?P<val3>[0-9]+)")
index_part_name_pat      = re.compile(
    prefix+"Index Partition Name (?P<uid>[0-9a-f]+) (?P<name>[-$()\w. ]+)")
index_subspace_pat       = re.compile(
    prefix+"Index Subspace (?P<pid>[0-9a-f]+) (?P<uid>[0-9a-f]+) (?P<dim>[0-9]+) "+
           "(?P<val1>[0-9]+) (?P<val2>[0-9]+) (?P<val3>[0-9]+)")
field_space_pat          = re.compile(
    prefix+"Field Space (?P<uid>[0-9]+)")
field_space_name_pat     = re.compile(
    prefix+"Field Space Name (?P<uid>[0-9]+) (?P<name>[-$()\w. ]+)")
field_create_pat         = re.compile(
    prefix+"Field Creation (?P<uid>[0-9]+) (?P<fid>[0-9]+)")
field_name_pat           = re.compile(
    prefix+"Field Name (?P<uid>[0-9]+) (?P<fid>[0-9]+) (?P<name>[-$()\w. ]+)")
region_pat               = re.compile(
    prefix+"Region (?P<iid>[0-9a-f]+) (?P<fid>[0-9]+) (?P<tid>[0-9]+)")
region_name_pat          = re.compile(
    prefix+"Logical Region Name (?P<iid>[0-9a-f]+) (?P<fid>[0-9]+) (?P<tid>[0-9]+) "+
           "(?P<name>[-$()\w. ]+)")
partition_name_pat       = re.compile(
    prefix+"Logical Partition Name (?P<iid>[0-9a-f]+) (?P<fid>[0-9]+) (?P<tid>[0-9]+) "+
           "(?P<name>[-$()\w. ]+)")
partition_complete_pat   = re.compile(
    prefix+"Index Partition Complete (?P<iid>[0-9a-f]+)")
is_independence_pat      = re.compile(
    prefix+"Index Space Independence (?P<pid>[0-9a-f]+) (?P<is1>[0-9a-f]+) "+
           "(?P<is2>[0-9a-f]+)")
ip_independence_pat      = re.compile(
    prefix+"Index Partition Independence (?P<pid>[0-9a-f]+) (?P<ip1>[0-9a-f]+) "+
           "(?P<ip2>[0-9a-f]+)")
dominance_pat            = re.compile(
    prefix+"Index Dominance (?P<id1>[0-9a-f]+) (?P<id2>[0-9a-f]+) (?P<kind>[0-3])")
non_intersection_pat     = re.compile(
    prefix+"Index Non-Intersection (?P<id1>[0-9a-f]+) (?P<id2>[0-9a-f]+) (?P<kind>[0-3])")
# Patterns for operations
task_name_pat            = re.compile(
    prefix+"Task ID Name (?P<tid>[0-9]+) (?P<name>[-$()\w. ]+)")
top_task_pat             = re.compile(
    prefix+"Top Task (?P<tid>[0-9]+) (?P<uid>[0-9]+) (?P<name>[-$()\w. ]+)")
single_task_pat          = re.compile(
    prefix+"Individual Task (?P<ctx>[0-9]+) (?P<tid>[0-9]+) (?P<uid>[0-9]+) "+
           "(?P<name>[-$()\w. ]+)")
index_task_pat           = re.compile(
    prefix+"Index Task (?P<ctx>[0-9]+) (?P<tid>[0-9]+) (?P<uid>[0-9]+) "+
           "(?P<name>[-$()\w. ]+)")
mapping_pat              = re.compile(
    prefix+"Mapping Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
close_pat                = re.compile(
    prefix+"Close Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<is_inter>[0-1])")
close_creator_pat        = re.compile(
    prefix+"Close Operation Creator (?P<uid>[0-9]+) (?P<cuid>[0-9]+) (?P<idx>[0-9]+)")
fence_pat                = re.compile(
    prefix+"Fence Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
copy_op_pat              = re.compile(
    prefix+"Copy Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
fill_op_pat              = re.compile(
    prefix+"Fill Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
acquire_op_pat           = re.compile(
    prefix+"Acquire Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
release_op_pat           = re.compile(
    prefix+"Release Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
deletion_pat             = re.compile(
    prefix+"Deletion Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
dep_partition_op_pat     = re.compile(
    prefix+"Dependent Partition Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) "+
           "(?P<pid>[0-9a-f]+) (?P<kind>[0-9]+)")
pending_partition_op_pat = re.compile(
    prefix+"Pending Partition Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
target_partition_pat     = re.compile(
    prefix+"Pending Partition Target (?P<uid>[0-9]+) (?P<pid>[0-9a-f]+) (?P<kind>[0-9]+)")
index_slice_pat          = re.compile(
    prefix+"Index Slice (?P<index>[0-9]+) (?P<slice>[0-9]+)")
slice_slice_pat          = re.compile(
    prefix+"Slice Slice (?P<slice1>[0-9]+) (?P<slice2>[0-9]+)")
slice_point_pat          = re.compile(
    prefix+"Slice Point (?P<slice>[0-9]+) (?P<point>[0-9]+) (?P<dim>[0-9]+) "+
           "(?P<val1>[0-9]+) (?P<val2>[0-9]+) (?P<val3>[0-9]+)")
point_point_pat          = re.compile(
    prefix+"Point Point (?P<point1>[0-9]+) (?P<point2>[0-9]+)")
# Patterns for logical analysis and region requirements
requirement_pat         = re.compile(
    prefix+"Logical Requirement (?P<uid>[0-9]+) (?P<index>[0-9]+) (?P<is_reg>[0-1]) "+
           "(?P<ispace>[0-9a-f]+) (?P<fspace>[0-9]+) (?P<tid>[0-9]+) (?P<priv>[0-9]+) "+
           "(?P<coher>[0-9]+) (?P<redop>[0-9]+) (?P<pis>[0-9a-f]+)")
req_field_pat           = re.compile(
    prefix+"Logical Requirement Field (?P<uid>[0-9]+) (?P<index>[0-9]+) (?P<fid>[0-9]+)")
mapping_dep_pat         = re.compile(
    prefix+"Mapping Dependence (?P<ctx>[0-9]+) (?P<prev_id>[0-9]+) (?P<pidx>[0-9]+) "+
           "(?P<next_id>[0-9]+) (?P<nidx>[0-9]+) (?P<dtype>[0-9]+)")
# Physical instance and mapping decision patterns
instance_pat            = re.compile(
    prefix+"Physical Instance (?P<iid>[0-9a-f]+) (?P<mid>[0-9a-f]+) (?P<redop>[0-9]+)")
instance_region_pat     = re.compile(
    prefix+"Physical Instance Region (?P<iid>[0-9a-f]+) (?P<ispace>[0-9]+) "
           "(?P<fspace>[0-9]+) (?P<tid>[0-9]+)")
instance_field_pat      = re.compile(
    prefix+"Physical Instance Field (?P<iid>[0-9a-f]+) (?P<fid>[0-9]+)")
mapping_decision_pat    = re.compile(
    prefix+"Mapping Decision (?P<uid>[0-9]+) (?P<idx>[0-9]+) (?P<fid>[0-9]+) "
           "(?P<iid>[0-9a-f]+)")
# Physical event and operation patterns
event_dependence_pat     = re.compile(
    prefix+"Event Event (?P<id1>[0-9a-f]+) (?P<gen1>[0-9]+) (?P<id2>[0-9a-f]+) "+
           "(?P<gen2>[0-9]+)")
operation_event_pat     = re.compile(
    prefix+"Operation Events (?P<uid>[0-9]+) (?P<id1>[0-9a-f]+) (?P<gen1>[0-9]+) "+
           "(?P<id2>[0-9a-f]+) (?P<gen2>[0-9]+)")
realm_copy_pat          = re.compile(
    prefix+"Copy Events (?P<uid>[0-9]+) (?P<ispace>[0-9]+) (?P<fspace>[0-9]+) "+
           "(?P<tid>[0-9]+) (?P<preid>[0-9a-f]+) (?P<pregen>[0-9]+) "+
           "(?P<postid>[0-9a-f]+) (?P<postgen>[0-9]+)")
realm_copy_field_pat    = re.compile(
    prefix+"Copy Field (?P<id>[0-9a-f]+) (?P<gen>[0-9]+) (?P<srcfid>[0-9]+) "+
           "(?P<srcid>[0-9a-f]+) (?P<dstfid>[0-9]+) (?P<dstid>[0-9a-f]+) (?P<redop>[0-9]+)")
realm_fill_pat          = re.compile(
    prefix+"Fill Events (?P<uid>[0-9]+) (?P<ispace>[0-9]+) (?P<fspace>[0-9]+) "+
           "(?P<tid>[0-9]+) (?P<preid>[0-9a-f]+) (?P<pregen>[0-9]+) "+
           "(?P<postid>[0-9a-f]+) (?P<postgen>[0-9]+)")
realm_fill_field_pat    = re.compile(
    prefix+"Fill Field (?P<id>[0-9a-f]+) (?P<gen>[0-9]+) (?P<fid>[0-9]+) "+
           "(?P<dstid>[0-9a-f]+)")
phase_barrier_pat       = re.compile(
    prefix+"Phase Barrier (?P<iid>[0-9a-f]+)")

def parse_legion_spy_line(line, state):
    # Quick test to see if the line is even worth considering
    m = prefix_pat.match(line)
    if m is None:
        return False
    # We order these regex matches by the frequency in which they
    # are likely to happen in order to improve parsing time
    # Event stuff is very likely the most frequent stuff
    m = event_dependence_pat.match(line)
    if m is not None:
        e1 = state.get_event(int(m.group('id1'),16),int(m.group('gen1')))
        e2 = state.get_event(int(m.group('id2'),16),int(m.group('gen2')))
        e2.add_incoming(e1)
        e1.add_outgoing(e2)
        return True
    m = operation_event_pat.match(line)
    if m is not None:
        e1 = state.get_event(int(m.group('id1'),16),int(m.group('gen1')))
        e2 = state.get_event(int(m.group('id2'),16),int(m.group('gen2')))
        op = state.get_operation(int(m.group('uid')))
        op.set_events(e1, e2)
        return True
    m = realm_copy_pat.match(line)
    if m is not None:
        e1 = state.get_event(int(m.group('preid'),16),int(m.group('pregen')))
        e2 = state.get_event(int(m.group('postid'),16),int(m.group('postgen')))
        copy = state.get_realm_copy(e2)
        copy.set_start(e2)
        op = state.get_operation(int(m.group('uid')))
        copy.set_creator(op)
        region = state.get_region(int(m.group('ispace')), 
            int(m.group('fspace')), int(m.group('tid')))
        copy.set_region(region)
        return True
    m = realm_copy_field_pat.match(line)
    if m is not None:
        e = state.get_event(int(m.group('id'),16),int(m.group('gen')))
        copy = state.get_realm_copy(e)
        src = state.get_instance(int(m.group('srcid'),16))
        dst = state.get_instance(int(m.group('dstid'),16))
        copy.add_field(int(m.group('srcfid')), src, 
                       int(m.group('dstfid')), dst, int(m.group('redop')))
        return True
    m = realm_fill_pat.match(line)
    if m is not None:
        e1 = state.get_event(int(m.group('preid'),16),int(m.group('pregen')))
        e2 = state.get_event(int(m.group('postid'),16),int(m.group('postgen')))
        fill = state.get_realm_fill(e2)
        fill.set_start(e1)
        op = state.get_operation(int(m.group('uid')))
        fill.set_creator(op)
        region = state.get_region(int(m.group('ispace')), 
            int(m.group('fspace')), int(m.group('tid')))
        fill.set_region(region)
        return True
    m = realm_fill_field_pat.match(line)
    if m is not None:
        e = state.get_event(int(m.group('id'),16),int(m.group('gen')))
        fill = state.get_realm_fill(e)
        dst = state.get_instance(int(m.group('dstid'),16))
        fill.add_field(int(m.group('fid')), dst)
        return True
    m = phase_barrier_pat.match(line)
    if m is not None:
        if state.phase_barriers is None:
            state.phase_barriers = set()
        state.phase_barriers.add(int(m.group('iid'),16))
        return True
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
        parent_ispace = int(m.group('pis'),16)
        parent = state.get_region(int(m.group('pis'),16), field_space.uid, tid)
        if is_reg:
            index_space = state.get_index_space(int(m.group('ispace'),16))
            region = state.get_region(index_space.uid, field_space.uid, tid) 
            requirement = Requirement(state, int(m.group('index')), True,
                index_space, field_space, tid, region, 
                priv, coher, redop, parent)
            op.add_requirement(requirement)
        else:
            index_partition = state.get_index_partition(int(m.group('ispace'),16))
            partition = state.get_partition(index_partition.uid, field_space.uid, tid)
            requirement = Requirement(state, int(m.group('index')), False,
                index_partition, field_space, tid, partition, 
                priv, coher, redop, parent)
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
        op2.add_incoming(dep)
        op1.add_outgoing(dep)
        # Record that we found a mapping dependence
        state.has_mapping_deps = True
        return True
    # Physical Instances and Mapping decisions happen frequently too
    m = instance_pat.match(line)
    if m is not None:
        mem = state.get_memory(int(m.group('mid'),16))
        inst = state.get_instance(int(m.group('iid'),16))
        inst.set_memory(mem)
        inst.set_redop(int(m.group('redop')))
        return True
    m = instance_region_pat.match(line)
    if m is not None:
        inst = state.get_instance(int(m.group('iid'),16))
        region = state.get_region(int(m.group('ispace'),16), 
            int(m.group('fspace')), int(m.group('tid')))
        inst.set_region(region)
        return True
    m = instance_field_pat.match(line)
    if m is not None:
        inst = state.get_instance(int(m.group('iid'),16))
        inst.add_field(int(m.group('fid')))
        return True
    m = mapping_decision_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        inst = state.get_instance(int(m.group('iid'),16))
        op.add_mapping_decision(int(m.group('idx')),
            int(m.group('fid')), inst)
        return True
    # Operations near the top since they happen frequently
    m = task_name_pat.match(line)
    if m is not None:
        state.task_names[int(m.group('tid'))] = m.group('name')
        return True
    m = top_task_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(SINGLE_TASK_KIND)
        op.set_name(m.group('name'))
        op.set_task_id(int(m.group('tid')))
        if op.context is not None:
            op.context.top = True
        return True
    m = single_task_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(SINGLE_TASK_KIND)
        op.set_name(m.group('name'))
        op.set_task_id(int(m.group('uid')))
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context)
        return True
    m = index_task_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(INDEX_TASK_KIND)
        op.set_name(m.group('name'))
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
    m = fence_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(FENCE_OP_KIND)
        op.set_name("Fence Op "+m.group('uid'))
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context)
        return True
    m = copy_op_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(COPY_OP_KIND)
        op.set_name("Copy Op "+m.group('uid'))
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context)
        return True
    m = fill_op_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(FILL_OP_KIND)
        op.set_name("Fill Op "+m.group('uid'))
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context)
        return True
    m = acquire_op_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(ACQUIRE_OP_KIND)
        op.set_name("Acquire Op "+m.group('uid'))
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context)
        return True
    m = release_op_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(RELEASE_OP_KIND)
        op.set_name("Release Op "+m.group('uid'))
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context)
        return True
    m = deletion_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(DELETION_OP_KIND)
        op.set_name("Deletion Op "+m.group('uid'))
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context)
        return True
    m = dep_partition_op_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(DEP_PART_OP_KIND)
        op.set_name("Dependent Partition Op "+m.group('uid'))
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context)
        return True
    m = pending_partition_op_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(PENDING_PART_OP_KIND)
        op.set_name("Pending Partition Op "+m.group('uid'))
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context)
        return True
    m = target_partition_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid'))) 
        index_partition = state.get_index_partition(int(m.group('pid'),16))
        op.set_pending_partition_info(index_partition, int(m.group('kind')))
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
    m = partition_complete_pat.match(line)
    if m is not None:
        partition = state.get_index_partition(int(m.group('iid'),16))
        partition.set_complete(True)
        return True
    m = is_independence_pat.match(line)
    if m is not None:
        parent = state.get_index_partition(int(m.group('pid'),16))
        is1 = state.get_index_space(int(m.group('is1'),16))
        is2 = state.get_index_space(int(m.group('is2'),16))
        parent.add_independent_children(is1, is2)
        return True
    m = ip_independence_pat.match(line)
    if m is not None:
        parent = state.get_index_space(int(m.group('pid'),16))
        ip1 = state.get_index_partition(int(m.group('ip1'),16))
        ip2 = state.get_index_partition(int(m.group('ip2'),16))
        parent.add_independent_children(ip1, ip2)
        return True
    m = dominance_pat.match(line)
    if m is not None:
        kind = int(m.group('kind'))
        if kind is 0:
            is1 = state.get_index_space(int(m.group('id1'),16))
            is2 = state.get_index_space(int(m.group('id2'),16))
            state.dominance_pairs.add((is1,is2))
        elif kind is 1:
            is1 = state.get_index_space(int(m.group('id1'),16))
            ip2 = state.get_index_partition(int(m.group('id2'),16))
            state.dominance_pairs.add((is1,ip2))
        elif kind is 2:
            ip1 = state.get_index_partition(int(m.group('id1'),16))
            is2 = state.get_index_partition(int(m.group('id2'),16))
            state.dominance_pairs.add((ip1,is2))
        else:
            ip1 = state.get_index_partition(int(m.group('id1'),16))
            ip2 = state.get_index_partition(int(m.group('id2'),16))
            state.dominance_pairs.add((ip1,ip2))
        return True
    m = non_intersection_pat.match(line)
    if m is not None:
        kind = int(m.group('kind'))
        if kind is 0:
            is1 = state.get_index_space(int(m.group('id1'),16))
            is2 = state.get_index_space(int(m.group('id2'),16))
            state.non_intersection_pairs.add((is1,is2))
            state.non_intersection_pairs.add((is2,is1))
        elif kind is 1:
            is1 = state.get_index_space(int(m.group('id1'),16))
            ip2 = state.get_index_partition(int(m.group('id2'),16))
            state.non_intersection_pairs.add((is1,ip2))
            state.non_intersection_pairs.add((ip2,is1))
        elif kind is 2:
            ip1 = state.get_index_partition(int(m.group('id1'),16))
            is2 = state.get_index_partition(int(m.group('id2'),16))
            state.non_intersection_pairs.add((ip1,is2))
            state.non_intersection_pairs.add((is2,ip1))
        else:
            ip1 = state.get_index_partition(int(m.group('id1'),16))
            ip2 = state.get_index_partition(int(m.group('id2'),16))
            state.non_intersection_pairs.add((ip1,ip2))
            state.non_intersection_pairs.add((ip2,ip1))
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
    __slots__ = ['verbose', 'top_level_uid', 'traverser_gen', 'processors', 'memories',
                 'processor_kinds', 'memory_kinds', 'index_spaces', 'index_partitions',
                 'field_spaces', 'regions', 'partitions', 'top_spaces', 'trees',
                 'non_intersection_pairs', 'dominance_pairs', 'ops', 'tasks', 
                 'task_names', 'has_mapping_deps', 'instances', 'events', 'copies',
                 'fills', 'phase_barriers', 'no_event', 'slice_index', 'slice_slice', 
                 'point_slice']
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
        self.non_intersection_pairs = set()
        self.dominance_pairs = set()
        # Logical things 
        self.ops = dict()
        self.tasks = dict()
        self.task_names = dict()
        self.has_mapping_deps = False
        # Physical things 
        self.instances = dict()
        self.events = dict()
        self.copies = dict()
        self.fills = dict()
        self.phase_barriers = None
        self.no_event = Event(self, EventHandle(0,0))
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
        # Fill in any task names
        for task in self.tasks.itervalues():
            if task.op.task_id in self.task_names:
                task.op.set_name(self.task_names[task.task_id])
        # Check to see if we have any unknown operations
        unknown = None
        for op in self.ops.itervalues():
            if op.kind is NO_OP_KIND:
                unknown = op
                break
        if unknown is not None:
            print 'WARNING: operation %d has unknown operation kind!' % op.uid 
        # If we have any phase barriers, mark all the events of the phase barrier
        if self.phase_barriers is not None:
            for event in self.events:
                if event.handle.uid in self.phase_barriers:
                    event.phase_barrier = True
        # We can delete some of these data structures now that we
        # no longer need them, go go garbage collection
        self.slice_index = None
        self.slice_slice = None
        self.point_slice = None
        if self.verbose:
            print "Found %d processors" % len(self.processors)
            print "Found %d memories" % len(self.memories)
            print ""
            print "Found %d index space trees" % num_index_trees
            print "Found %d field spaces" % len(self.field_spaces)
            print "Found %d region trees" % len(self.trees)
            print ""
            print "Found %d tasks" % len(self.tasks)
            print "Found %d operations (including tasks)" % len(self.ops)
            print ""
            print "Found %d instances" % len(self.instances)
            print "Found %d events" % len(self.events)
            print "Found %d copies" % len(self.copies)
            print "Found %d fills" % len(self.fills)
        logical_enabled = self.has_mapping_deps
        physical_enabled = not not self.events
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
        # Run the full analysis first, this will confirm that
        # the runtime did what we thought it should do
        success = True
        for task in self.tasks.itervalues():
            if not task.check_logical_dependence_analysis():
                success = False
        # Skip the rest of the checks if something is wrong
        if not success:
            return
        # Run the old version of the checks that
        # is more of a sanity check on our algorithm that
        # doesn't depend on our implementation but doesn't
        # really tell us what it means if something goes wrong
        #for task in self.tasks:
        #    task.check_logical_sanity()

    def perform_physical_checks(self):
        pass

    def make_region_tree_graphs(self, path, simplify_graphs):
        index_space_printer = GraphPrinter(path, 'index_space_graph', 'TB')
        for node in self.index_spaces.itervalues():
            if node.parent is None:
                node.print_graph(index_space_printer)
        index_space_printer.print_pdf_after_close(simplify_graphs)

        field_space_printer = GraphPrinter(path, 'field_space_graph', 'TB')
        for node in self.field_spaces.itervalues():
            node.print_graph(field_space_printer)
        field_space_printer.print_pdf_after_close(simplify_graphs)

        region_graph_printer = GraphPrinter(path, 'region_graph', 'TB')
        for node in self.trees.itervalues():
            if simplify_graphs:
                node.mark_named_children()
            node.print_graph(region_graph_printer, simplify_graphs)
        region_graph_printer.print_pdf_after_close(simplify_graphs)

    def make_machine_graphs(self, path):
        machine_printer = GraphPrinter(path, 'machine_graph', 'TB')
        for proc in self.processors.itervalues():
            proc.print_node(machine_printer)
        for mem in self.memories.itervalues():
            mem.print_node(machine_printer)
        for proc in self.processors.itervalues():
            proc.print_mem_edges(machine_printer)
        for mem in self.memories.itervalues():
            mem.print_mem_edges(machine_printer)
        machine_printer.print_pdf_after_close(False)

    def make_dataflow_graphs(self, path, simplify_graphs):
        pass

    def make_event_graphs(self, path, simplify_graphs):
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

    def get_instance(self, iid):
        if iid in self.instances:
            return self.instances[iid]
        result = Instance(self, iid)
        self.instances[iid] = result
        return result

    def get_event(self, iid, gen):
        handle = EventHandle(iid, gen)
        if handle in self.events:
            return self.events[handle]
        result = Event(self, handle)
        self.events[handle] = result 
        return result 

    def get_no_event(self):
        return self.no_event

    def get_realm_copy(self, event):
        if event in self.copies:
            return self.copies[event]
        result = RealmCopy(self, event)
        self.copies[event] = result
        return result

    def get_realm_fill(self, event):
        if event in self.fills:
            return self.fills[event]
        result = RealmFill(self, event)
        self.fills[event] = result
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

    def reset_logical_state(self):
        for region in self.regions.itervalues():
            region.reset_logical_state()
        for partition in self.partitions.itervalues():
            partition.reset_logical_state()
        # Definitely run the garbage collector here
        gc.collect()

    def reset_physical_state(self):
        for region in self.regions.itervalues():
            region.reset_physical_state()
        for partition in self.partitions.itervalues():
            partition.reset_physical_state()
        # Definitely run the garbage collector here
        gc.collect()

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
        state.make_region_tree_graphs(temp_dir, simplify_graphs)
    if machine_graphs:
        print "Making machine graphs..."
        state.make_machine_graphs(temp_dir)
    if dataflow_graphs:
        print "Making dataflow graphs..."
        state.make_dataflow_graphs(temp_dir, simplify_graphs)
    if event_graphs:
        print "Making event graphs..."
        state.make_event_graphs(temp_dir, simplify_graphs)

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

