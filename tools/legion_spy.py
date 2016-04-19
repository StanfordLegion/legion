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
import sys, os, re, gc, shutil, copy
import string
import tempfile
from getopt import getopt,GetoptError
from array import *
from collections import deque

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
INTER_CLOSE_OP_KIND = 4
READ_ONLY_CLOSE_OP_KIND = 5
POST_CLOSE_OP_KIND = 6
FENCE_OP_KIND = 7
COPY_OP_KIND = 8
FILL_OP_KIND = 9
ACQUIRE_OP_KIND = 10 
RELEASE_OP_KIND = 11
DELETION_OP_KIND = 12
DEP_PART_OP_KIND = 13
PENDING_PART_OP_KIND = 14

OPEN_NONE = 0
OPEN_READ_ONLY = 1
OPEN_READ_WRITE = 2
OPEN_SINGLE_REDUCE = 3
OPEN_MULTI_REDUCE = 4

OpNames = [
"Unknown Kind",
"Single Task",
"Index Task",
"Map Op",
"Inter Close Op",
"Post Close Op",
"Fence Op",
"Copy Op",
"Fill Op",
"Acquire Op",
"Release Op",
"Deletion Op",
"Dependent Partition Op",
"Pending Partition Op",
]

def check_for_anti_dependence(req1, req2, actual):
    if req1.is_read_only():
        assert req2.has_write()
        return ANTI_DEPENDENCE
    else:
        if req2.is_write_only():
            return ANTI_DEPENDENCE
        else:
            return actual

def compute_dependence_type(req1, req2):
    if req1.is_no_access() or req2.is_no_access():
        return NO_DEPENDENCE
    elif req1.is_read_only() and req2.is_read_only():
        return NO_DEPENDENCE
    elif req1.is_reduce() and req2.is_reduce():
        if req1.redop == req2.redop:
            return NO_DEPENDENCE
        else:
            return TRUE_DEPENDENCE
    else:
        assert req1.has_write() or req2.has_write() 
        if req1.is_exclusive() or req2.is_exclusive():
            return check_for_anti_dependence(req1,req2,TRUE_DEPENDENCE)
        elif req1.is_atomic() or req2.is_atomic():
            if req1.is_atomic() and req2.is_atomic():
                return check_for_anti_dependence(req1,req1,ATOMIC_DEPENDENCE)
            elif (((not req1.is_atomic()) and req1.is_read_only()) or 
                  ((not req2.is_atomic()) and req2.is_read_only())):
                return NO_DEPENDENCE
            else:
                return check_for_anti_dependence(req1,req2,TRUE_DEPENDENCE)
        elif req1.is_simult() or req2.is_simult():
            return SIMULTANEOUS_DEPENDENCE
        elif req1.is_relaxed() and req2.is_relaxed():
            return req1,req2,SIMULTANEOUS_DEPENDENCE
        # Should never get here
        assert False
        return NO_DEPENDENCE

def check_preconditions(preconditions, op):
    assert op.reachable_cache is not None
    for pre in preconditions:
        # Make sure they are in the same context before testing them
        pre_ctx = pre.get_context()
        op_ctx = op.get_context()
        if pre_ctx is not op_ctx:
            # Need to make sure these are in the same context 
            pre_ancestor = pre
            op_ancestor = op
            # First get the contexts at the same depth 
            while pre_ctx.get_depth() < op_ctx.get_depth():
                op_ancestor = op_ancestor.get_context().op
                op_ctx = op_ancestor.get_context()
            while op_ctx.get_depth() < pre_ctx.get_depth():
                pre_ancestor =  pre_ancestor.get_context().op
                pre_ctx = pre_ancestor.get_context()
            # Contexts are the at the same depth, walk up until contexts are the same
            while pre_ctx is not op_ctx:
                op_ancestor = op_ancestor.get_context().op
                pre_ancestor = pre_ancestor.get_context().op
            # Special case, if they are the same op, then we are done :)
            if pre_ancestor is op_ancestor:
                continue
            # Now we can test for reachability
            assert op_ancestor.reachable_cache is not None
            if pre_ancestor not in op.reachable_cache:
                return pre
        else:
            # Easy, we can test these directly 
            if pre not in op.reachable_cache:
                return pre
    return None

class Point(object):
    __slots__ = ['dim', 'vals']
    def __init__(self, dim):
        self.dim = dim
        self.vals = array('i', (0,)*dim)

    def mk_string(self, start, delim, end):
        result = start
        first = True
        for i in range(self.dim):
            if not first:
                result = result + delim
            else:
                first = False
            result = result + str(self.vals[i])
        result = result + end
        return result

    def to_string(self):
        return self.mk_string("(", ",", ")")

    def to_dim_string(self):
        return self.mk_string("", "][", "")

    def __str__(self):
        return self.mk_string("", "_", "")

    __repr__ = __str__

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, point):
        assert self.dim == point.dim
        for i in range(self.dim):
            if self.vals[i] <> point.vals[i]:
                return False
        return True

    def copy(self):
        result = Point(self.dim)
        for i in range(self.dim):
            result.vals[i] = self.vals[i]
        return result;

class Rect(object):
    __slots__ = ['dim', 'lo', 'hi']
    def __init__(self, lo, hi):
        assert lo.dim == hi.dim
        self.dim = lo.dim
        self.lo = lo
        self.hi = hi

    def __str__(self):
        return str(self.lo) + "-" + str(self.hi)

    __repr__ = __str__

    def __hash__(self):
        return hash(str(self))

    def empty(self):
        for i in range(self.dim):
            if self.lo.vals[i] > self.hi.vals[i]:
                return True
        return False

    def dominates(self, rect):
        for i in range(self.dim):
            if rect.lo.vals[i] < self.lo.vals[i] or self.hi.vals[i] < rect.hi.vals[i]:
                return False
        return True

    def intersects(self, rect):
        # If they overlap in all dimensions then they intersect
        assert rect.dim == self.dim
        for i in range(self.dim):
            if rect.hi.vals[i] < self.lo.vals[i] or self.hi.vals[i] < rect.lo.vals[i]:
                return False
        return True

    def intersection(self, rect):
        lo = Point(self.dim)
        hi = Point(self.dim)
        for i in range(self.dim):
            lo.vals[i] = max(self.lo.vals[i], rect.lo.vals[i])
            hi.vals[i] = min(self.hi.vals[i], rect.hi.vals[i])
        return Rect(lo, hi)

    def contains_point(self, point):
        for i in range(point.dim):
            x = point.vals[i]
            if x < self.lo.vals[i] or self.hi.vals[i] < x:
                return False
        return True

    def iterator(self):
        if self.dim == 1:
            for x in range(self.lo.vals[0], self.hi.vals[0]+1):
                point = Point(1)
                point.vals[0] = x
                yield point
        elif self.dim == 2:
            for x in range(self.lo.vals[0], self.hi.vals[0]+1):
                for y in range(self.lo.vals[1], self.hi.vals[1]+1):
                    point = Point(2)
                    point.vals[0] = x
                    point.vals[1] = y
                    yield point
        elif self.dim == 3:
            for x in range(self.lo.vals[0], self.hi.vals[0]+1):
                for y in range(self.lo.vals[1], self.hi.vals[1]+1):
                    for z in range(self.lo.vals[2], self.hi.vals[2]+1):
                        point = Point(3)
                        point.vals[0] = x
                        point.vals[1] = y
                        point.vals[2] = z
                        yield point
        else:
            assert False

    def copy(self):
        return Rect(self.lo.copy(), self.hi.copy())

# A Collection of points and rectangles
class Shape(object):
    __slots__ = ['points', 'rects']
    def __init__(self):
        self.points = set()
        self.rects = set()

    def __str__(self):
        result = ''
        first = True
        for point in self.points:
            if first:
                first = False
            else:
                result += ' '
            result += str(point)
        for rect in self.rects:
            if first:
                first = False
            else:
                result += ' '
            result += str(rect)
        return result

    __repr__ = __str__

    def add_point(self, point):
        self.points.add(point)

    def add_rect(self, rect):
        self.rects.add(rect)

    def empty(self):
        return not self.points and not self.rects

    def copy(self):
        result = Shape()
        for point in self.points:
            result.points.add(point)
        for rect in self.rects:
            result.rects.add(rect)
        return result

    # Set intersection
    def __and__(self, other):
        result = self.copy()
        result &= other
        return result

    def __iand__(self, other):
        to_remove = list()
        for point in self.points:
            if point in other.points:
                continue
            has_point = False
            for rect in other.rects:
                if rect.contains_point(point):
                    has_point = True
                    break
            if not has_point:
                to_remove.append(point)
        for point in to_remove:
            self.points.remove(point)
        new_rects = set()
        for rect in self.rects:
            for orect in other.rects:
                new_rect = rect.intersection(orect)
                if not new_rect.empty():
                    new_rects.add(new_rect)
        self.rects = new_rects
        return self

    # Set union
    def __or__(self, other):
        result = self.copy()
        result |= other
        return result

    def __ior__(self, other):
        for point in other.points:
            self.points.add(point)
        for rect in other.rects:
            self.rects.add(rect)
        return self

    # Set substraction
    def __sub__(self, other):
        result = self.copy()
        result -= other
        return result

    def __isub__(self, other):
        to_remove = set()
        # Do rectangles first, then do points
        for rect in self.rects:
            for orect in other.rects:
                if rect.intersects(orect):
                    # See if it dominates
                    if orect.dominates(rect):
                        to_remove.add(rect)
                        break
                    else:
                        # break it into points
                        for point in rect.iterator():
                            if not orect.contains_point(point):
                                self.points.add(point)
                        to_remove.add(rect)
                        break
            if rect not in to_remove:
                # Check the rectangle agains the points
                for point in other.points:
                    if rect.contains_point(point):
                        # break it into points
                        for new_point in rect.iterator():
                            if new_point <> point:
                                self.points.add(new_point) 
                        to_remove.add(rect)
                        break
        if to_remove:
            for rect in to_remove:
                self.rects.remove(rect)
        to_remove = set()
        for point in self.points:
            if point in other.points:
                to_remove.add(point)
                continue
            for rect in other.rects:
                if rect.contains_point(point):
                    to_remove.add(point)
                    continue
        if to_remove:
            for point in to_remove:
                self.points.remove(point)
        return self

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
        return self.kind + " Memory " + hex(self.uid)

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
                 'depth', 'shape', 'node_name', 'intersections',
                 'dominated']
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.parent = None
        self.color = Point(0) 
        self.children = dict()
        self.instances = dict()
        self.name = None
        self.depth = 0
        self.shape = None
        self.node_name = 'index_space_node_'+str(uid)
        self.intersections = dict() 
        self.dominated = dict()

    def set_name(self, name):
        self.name = name

    def set_parent(self, parent, color):
        self.parent = parent
        self.depth = parent.depth+1
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

    def add_point(self, point):
        if self.shape is None:
            self.shape = Shape()
            self.shape.add_point(point.copy())
            # Help with dynamic allocation
            if self.parent is not None:
                self.parent.parent.add_point(point)
        else:
            update = Shape()
            update.add_point(point.copy())
            update -= self.shape
            if not update.empty():
                self.shape |= update
                # Help with dynamic allocation
                if self.parent is not None:
                    self.parent.parent.add_point(point)

    def add_rect(self, rect):
        if self.shape is None:
            self.shape = Shape()
            self.shape.add_rect(rect.copy())
            # Help with dynamic allocation
            if self.parent is not None:
                self.parent.parent.add_rect(rect)
        else:
            update = Shape()
            update.add_rect(rect.copy())
            update -= self.shape
            if not update.empty():
                self.shape |= update
                # Help with dynamic allocation
                if self.parent is not None:
                    self.parent.parent.add_rect(rec)

    def set_empty(self):
        assert self.shape is None
        self.shape = Shape()

    def __str__(self):
        if self.name is None:
            return "Index Space "+str(self.uid)
        else:
          return self.name + ' ('+str(self.uid)+')'

    __repr__ = __str__

    def are_all_children_disjoint(self):
        return False

    def are_children_disjoint(self, c1, c2):
        if c1.intersects(c2):
            return False
        return True

    def get_shape(self):
        if self.shape is None:
            print "No shape for "+str(self)
        assert self.shape is not None
        return self.shape

    def intersection(self, other):
        if other in self.intersections:
            if self.intersections is None:
                return Shape() # empty shape
            return self.intersections[other]
        intersection = self.get_shape() & other.get_shape()
        if intersection.empty():
            self.intersections[other] = None
            return intersection 
        self.intersections[other] = intersection
        return intersection

    def intersects(self, other):
        return self.intersection(other) is not None

    def dominates(self, other):
        if other in self.dominated:
            return self.dominated[other]
        non_dominated = other.get_shape() - self.get_shape()
        if non_dominated.empty():
            self.dominated[other] = True
            return True
        else:
            self.dominated[other] = False
            return False

    def is_complete(self):
        return False

    def get_num_children(self):
        return len(self.children)

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
                        ' (color: ' + str(color) +')'
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
                 'disjoint', 'complete', 'name', 'depth', 'shape', 'node_name',
                 'intersections', 'dominated']
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.parent = None
        self.color = Point(0)
        self.children = dict()
        self.instances = dict()
        self.disjoint = False
        self.complete = None 
        self.name = None
        self.depth = None
        self.shape = None
        self.node_name = 'index_part_node_'+str(uid)
        self.intersections = dict()
        self.dominated = dict()

    def set_parent(self, parent, color):
        self.parent = parent
        self.depth = parent.depth+1
        self.color = color
        self.parent.add_child(self)
        # Update any instances
        for tid,partition in self.instances.iteritems():
            parent = self.state.get_region(parent.uid,
                partition.field_space.uid, tid)
            region.set_parent(parent)

    def set_disjoint(self, disjoint):
        self.disjoint = disjoint

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
            return self.name + ' ('+str(self.uid)+')'

    __repr__ = __str__

    def are_all_children_disjoint(self):
        return self.disjoint

    def are_children_disjoint(self, c1, c2):
        if self.disjoint:
            return True
        if c1.intersects(c2):
            return False
        return True

    def get_shape(self):
        if self.shape is None:
            for child in self.children.itervalues():
                if self.shape is None:
                    self.shape = child.get_shape().copy()
                else:
                    self.shape |= child.get_shape()
        return self.shape

    def intersection(self, other):
        if other in self.intersections:
            if self.intersections[other] is None:
                return Shape() # empty shape
            return self.intersections[other]
        intersection = self.get_shape() & other.get_shape()
        if intersection.empty():
            self.intersections[other] = None
            return intersection 
        self.intersections[other] = intersection
        return intersection
        
    def intersects(self, other):
        return self.intersection(other) is not None

    def dominates(self, other):
        if other in self.dominated:
            return self.dominated[other]
        non_dominated = other.get_shape() - self.get_shape()
        if non_dominated.empty():
            self.dominated[other] = True
            return True
        else:
            self.dominated[other] = False
            return False

    def is_complete(self):
        if self.complete is None:
            # Figure out if this partition is complete or not 
            self.complete = (self.parent.get_shape() - self.get_shape()).empty()
        return self.complete

    def get_num_children(self):
        return len(self.children)

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
            return self.name + ' (' + str(self.fid) + ')'

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
            return self.name + ' ('+str(self.uid)+')'

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

    def has_all_children(self):
        return len(self.children) == len(self.index_space.children)

    def __str__(self):
        if self.name is None:
            return "Region (%d,%d,%d)" % (self.index_space.uid,
                self.field_space.uid,self.tree_id)
        else:
            return self.name + ' ('+str(self.index_space.uid)+','+\
                str(self.field_space.uid)+','+str(self.tree_id)+')'

    __repr__ = __str__

    def are_all_children_disjoint(self):
        return self.index_space.are_all_children_disjoint()

    def are_children_disjoint(self, c1, c2):
        return self.index_space.are_children_disjoint(c1.index_partition, 
                                                      c2.index_partition)

    def get_shape(self):
        return self.index_space.get_shape()

    def intersection(self, other):
        if isinstance(other, LogicalRegion):
            return self.index_space.intersection(other.index_space)
        else:
            return self.index_space.intersection(other.index_partition)

    def intersects(self, other):
        if isinstance(other, LogicalRegion):
            return self.index_space.intersects(other.index_space)
        else:
            return self.index_space.intersects(other.index_partition)

    def dominates(self, other):
        if isinstance(other, LogicalRegion):
            return self.index_space.dominates(other.index_space)
        else:
            return self.index_space.dominates(other.index_partition)

    def is_complete(self):
        return self.index_space.is_complete()

    def get_num_children(self):
        return self.index_space.get_num_children()

    def reset_logical_state(self):
        if self.logical_state:
            self.logical_state = dict()

    def reset_physical_state(self, depth):
        if self.physical_state and depth in self.physical_state:
            self.physical_state[depth] = dict()

    def compute_path(self, path, target):
        if self is not target:
            assert self.parent is not None
            self.parent.compute_path(path, target)
        path.append(self)

    def perform_logical_analysis(self, depth, path, op, req, field, 
                                 projecting, prev, checks):
        assert self is path[depth]
        if field not in self.logical_state:
            self.logical_state[field] = LogicalState(self, field)
        arrived = (depth+1) == len(path)
        next_child = path[depth+1] if not arrived else None
        if not self.logical_state[field].perform_logical_analysis(op, req, next_child, 
                                                                  projecting, prev, checks):
            return False
        if not arrived:
            return path[depth+1].perform_logical_analysis(depth+1, path, op, req, field, 
                                                          projecting, prev, checks)
        return True

    def perform_logical_fence(self, op, field, checks):
        if field not in self.logical_state:
            self.logical_state[field] = LogicalState(self, field)
        if not self.logical_state[field].perform_logical_fence(op, checks):
            return False
        for child in self.children.itervalues():
            if not child.perform_logical_fence(op, field, checks):
                return False
        return True

    def close_logical_tree(self, field, closed_users):
        if field not in self.logical_state:
            return
        self.logical_state[field].close_logical_tree(closed_users)

    def get_physical_state(self, depth, field):
        if depth not in self.physical_state:
            self.physical_state[depth] = dict()
        field_dict = self.physical_state[depth]
        if field not in field_dict:
            if self.parent is not None:
                parent_state = self.parent.get_physical_state(depth, field)
                field_dict[field] = PhysicalState(self, depth, field, parent_state)
            else:
                field_dict[field] = PhysicalState(self, depth, field, None)
        return field_dict[field]

    # Should only be called on regions and not on partitions
    def initialize_physical_state(self, depth, field, inst):
        physical_state = self.get_physical_state(depth, field)
        physical_state.initialize_physical_state(inst)

    def copy_physical_state(self, src_depth, dst_depth, field, clear_state, top=True):
        src_physical_state = self.get_physical_state(src_depth, field)
        dst_physical_state = self.get_physical_state(dst_depth, field)
        src_physical_state.move_to(dst_physical_state, clear_state, top)
        # Traverse down the tree
        for child in self.children.itervalues():
            child.copy_physical_state(src_depth, dst_depth, field, clear_state)

    # Should only be called on regions and not on partitions
    def perform_physical_analysis(self, depth, field, op, req, inst, 
                                  perform_checks, register = True):
        physical_state = self.get_physical_state(depth, field)
        return physical_state.perform_physical_analysis(op, req, inst,
                                                        perform_checks, register)

    # Should only be called on regions and not on partitions
    def perform_fill_analysis(self, depth, field, op, req, perform_checks):
        physical_state = self.get_physical_state(depth, field)
        return physical_state.perform_fill_analysis(op, req, perform_checks)

    def perform_physical_close(self, depth, field, op, req, inst, perform_checks):
        physical_state = self.get_physical_state(depth, field)
        return physical_state.perform_physical_close(op, req, inst,
                                                     perform_checks)

    def close_physical_tree(self, depth, field, target, op, perform_checks):
        for child in self.children.itervalues():
            if not child.perform_close_physical_tree(depth, field, target, 
                                                     op, perform_checks):
                return False
        return True

    def perform_close_physical_tree(self, depth, field, target, op, perform_checks):
        physical_state = self.get_physical_state(depth, field)
        return physical_state.close_physical_tree(target, op, perform_checks)

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
                    self.state.get_partition(child.uid, self.field_space.uid, self.tree_id)
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

    def has_all_children(self):
        return len(self.children) == len(self.index_partition.children)

    def __str__(self):
        if self.name is None:
            return "Partition (%d,%d,%d)" % (self.index_partition.uid,
                self.field_space.uid, self.tree_id)
        else:
            return self.name + ' ('+str(self.index_partition.uid)+','+\
                str(self.field_space.uid)+','+str(self.tree_id)+')'

    __repr__ = __str__

    def are_all_children_disjoint(self):
        return self.index_partition.are_all_children_disjoint()

    def are_children_disjoint(self, c1, c2):
        return self.index_partition.are_children_disjoint(c1.index_space, 
                                                          c2.index_space)

    def get_shape(self):
        return self.index_partition.get_shape()

    def intersection(self, other):
        if isinstance(other, LogicalRegion):
            return self.index_partition.intersection(other.index_space)
        else:
            return self.index_partition.intersection(other.index_partition)

    def intersects(self, other):
        if isinstance(other, LogicalRegion):
            return self.index_partition.intersects(other.index_space)
        else:
            return self.index_partition.intersects(other.index_partition)

    def dominates(self, other):
        if isinstance(other, LogicalRegion):
            return self.index_partition.dominates(other.index_space)
        else:
            return self.index_partition.dominates(other.index_partition)

    def is_complete(self):
        return self.index_partition.is_complete()

    def get_num_children(self):
        return self.index_partition.get_num_children()

    def reset_logical_state(self):
        if self.logical_state:
            self.logical_state = dict()

    def reset_physical_state(self, depth):
        if self.physical_state and depth in self.physical_state:
            self.physical_state[depth] = dict()

    def compute_path(self, path, target):
        if self is not target:
            assert self.parent is not None
            self.parent.compute_path(path, target)
        path.append(self)

    def perform_logical_analysis(self, depth, path, op, req, field, 
                                 projecting, prev, checks):
        assert self is path[depth]
        if field not in self.logical_state:
            self.logical_state[field] = LogicalState(self, field)
        arrived = (depth+1) == len(path)
        next_child = path[depth+1] if not arrived else None
        if not self.logical_state[field].perform_logical_analysis(op, req, next_child, 
                                                                  projecting, prev, checks):
            return False
        if not arrived:
            return path[depth+1].perform_logical_analysis(depth+1, path, op, req, field, 
                                                          projecting, prev, checks)
        return True

    def perform_logical_fence(self, op, field, checks):
        if field not in self.logical_state:
            self.logical_state[field] = LogicalState(self, field)
        if not self.logical_state[field].perform_logical_fence(op, checks):
            return False
        for child in self.children.itervalues():
            if not child.perform_logical_fence(op, field, checks):
                return False
        return True

    def close_logical_tree(self, field, closed_users):
        if field not in self.logical_state:
            return
        self.logical_state[field].close_logical_tree(closed_users)

    def get_physical_state(self, depth, field):
        if depth not in self.physical_state:
            self.physical_state[depth] = dict()
        field_dict = self.physical_state[depth]
        if field not in field_dict:
            assert self.parent is not None
            parent_state = self.parent.get_physical_state(depth, field)
            field_dict[field] = PhysicalState(self, depth, field, parent_state)
        return field_dict[field]

    def copy_physical_state(self, src_depth, dst_depth, field, clear_state):
        src_physical_state = self.get_physical_state(src_depth, field)
        dst_physical_state = self.get_physical_state(dst_depth, field)
        src_physical_state.move_to(dst_physical_state, clear_state, False)
        # Traverse down the tree
        for child in self.children.itervalues():
            child.copy_physical_state(src_depth, dst_depth, field, clear_state, False)

    def perform_physical_close(self, depth, field, op, req, inst, perform_checks):
        physical_state = self.get_physical_state(depth, field)
        return physical_state.perform_physical_close(op, req, inst,
                                                     perform_checks)

    def close_physical_tree(self, depth, field, target, op, perform_checks):
        for child in self.children.itervalues():
            if not child.perform_close_physical_tree(depth, field, target, 
                                                     op, perform_checks):
                return False
        return True

    def perform_close_physical_tree(self, depth, field, target, op, perform_checks):
        physical_state = self.get_physical_state(depth, field)
        return physical_state.close_physical_tree(target, op, perform_checks)

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
    __slots__ = ['node', 'field', 'open_mode', 'open_redop', 'open_children',
                 'current_epoch_users', 'previous_epoch_users', 'current_redop']
    def __init__(self, node, field):
        self.node = node
        self.field = field
        self.open_mode = OPEN_NONE
        self.open_redop = 0 # for open children reductions
        self.open_children = set()
        self.current_epoch_users = list()
        self.previous_epoch_users = list()
        self.current_redop = 0 # for reductions being done at this node

    def perform_logical_analysis(self, op, req, next_child, 
                                 projecting, previous_deps, perform_checks):
        arrived = next_child is None
        # Figure out if we need to check close operations or not
        if not arrived or not (projecting or req.is_write()):
            if not self.siphon_logical_children(op, req, next_child, 
                                                previous_deps, perform_checks):
                return False
        # Now do our analysis to figure out who we need to wait on locally
        if not arrived or not projecting:
            if not self.perform_epoch_analysis(op, req, perform_checks,
                                               arrived, previous_deps):
                return False
        if arrived and not projecting:
            # If we are doing a write, register dependences on all
            # open subtrees
            if req.is_write() and self.open_children:
                closed_users = list()
                for child in self.open_children:
                    child.close_logical_tree(self.field, closed_users)
                if perform_checks:
                    for prev_op,prev_req in closed_users:
                        if not op.has_mapping_dependence(req, prev_op, prev_req,
                                                        TRUE_DEPENDENCE, self.field):
                            return False
                else:
                    # Not performing checks so record the mapping dependences 
                    for prev_op,prev_req in closed_users:
                        dep = MappingDependence(prev_op, prev_req.index,
                                                op, req.index, TRUE_DEPENDENCE)
                        prev_op.add_outgoing(dep)
                        op.add_incoming(dep)
                # We closed all the children
                self.open_children = set()
            # Add ourselves as the current user
            self.current_epoch_users.append((op,req))
            # Record if we have outstanding reductions
            if req.redop <> 0:
                self.current_redop = req.redop                
        return True

    def perform_logical_fence(self, op, perform_checks):
        if perform_checks: 
            for prev_op,prev_req in self.current_epoch_users:
                if prev_op not in op.logical_incoming:
                    print "ERROR: missing logical fence dependence between "+\
                          str(prev_op)+" and "+str(op)
                    if self.node.state.assert_on_fail:
                        assert False
                    return False
        else:
            for prev_op,prev_req in self.current_epoch_users:
                dep = MappingDependence(prev_op, op, 0, 0, TRUE_DEPENDENCE)
                prev_op.add_outgoing(dep)
                op.add_incoming(dep)
        # Clear out the user lists
        self.current_epoch_users = list()
        self.previous_epoch_users = list()
        return True

    # Maybe not the most intuitive name for a method but it aligns with the runtime
    def siphon_logical_children(self, op, req, next_child, 
                                previous_deps, perform_checks):
        # First see if we have any reductions to flush 
        if self.current_redop <> 0 and self.current_redop <> req.redop:
            error_str = " in order to flush reductions with redop "+str(self.current_redop) 
            if not self.perform_close_operations(next_child, 
                                                 False, # allow next
                                                 False, # permit leave open
                                                 False, # read-only close
                                                 True, # flush reductions
                                                 op, req, previous_deps,
                                                 perform_checks, error_str):
                return False
        # Figure out what to do based on our open mode
        if self.open_mode == OPEN_NONE:
            # Only open if there is next child
            if next_child is not None:
                if req.is_read_only():
                    self.open_mode = OPEN_READ_ONLY
                elif req.is_write():
                    self.open_mode = OPEN_READ_WRITE
                else:
                    assert req.redop <> 0
                    self.open_mode = OPEN_SINGLE_REDUCE
                    self.open_redop = req.redop
        elif self.open_mode == OPEN_READ_ONLY:
            # If this is not also read-only, do the close operations
            if not req.is_read_only():
                error_str = " in order to transition from a READ_ONLY state to "+\
                    "READ_WRITE" if req.is_write() else "SINGLE_REDUCE"
                if not self.perform_close_operations(next_child,
                                                     True, # allow next
                                                     False, # permit leave open
                                                     True, # read only close
                                                     False, # flush reductions
                                                     op, req, previous_deps,
                                                     perform_checks, error_str):
                    return False
                # Upgrade the state
                if req.is_write():
                    self.open_mode = OPEN_READ_WRITE
                else:
                    self.open_mode = OPEN_SINGLE_REDUCE
                    self.open_redop = req.redop
        elif self.open_mode == OPEN_READ_WRITE:
            error_str = " in state READ_WRITE"
            if not self.perform_close_operations(next_child,
                                                 True, # allow next
                                                 False, # permit leave open
                                                 False, # read only close
                                                 False, # flush reductions
                                                 op, req, previous_deps,
                                                 perform_checks, error_str):
                return False
            # See if we closed everything
            if next_child is not None and self.open_mode == OPEN_NONE:
                if req.is_read_only():
                    self.open_mode = OPEN_READ_ONLY
                elif req.is_write():
                    self.open_mode = OPEN_READ_WRITE
                else:
                    assert req.redop <> 0
                    self.open_mode = OPEN_SINGLE_REDUCE
                    self.open_redop = req.redop
        elif self.open_mode == OPEN_SINGLE_REDUCE:
            if req.redop == self.open_redop:
                # Same reduction mode, see if we need to transition 
                if next_child is not None and next_child not in self.open_children:
                    # See if the new child overlaps with any children 
                    # if it does we have to go to multi reduce
                    for child in self.open_children:
                        if not self.node.are_children_disjoint(child, next_child):
                            self.open_mode = OPEN_MULTI_REDUCE
                            break
            else:
                error_str = " in state SINGLE_REDUCE"
                # Different reduction operations, do the closes
                if not self.perform_close_operations(next_child,
                                                     True, # allow next
                                                     False, # permit leave open
                                                     False, # read only close
                                                     False, # flush reductions
                                                     op, req, previous_deps,
                                                     perform_checks, error_str):
                    return False
                # go to read write
                if next_child is not None:
                    self.open_mode = OPEN_READ_WRITE
                    self.open_redop = 0
        elif self.open_mode == OPEN_MULTI_REDUCE:
            if req.redop <> self.open_redop:
                error_str = " in state MULTI_REDUCE"
                if not self.perform_close_operations(next_child,
                                                     False, # allow next
                                                     False, # permit leave open
                                                     False, # read only close
                                                     False, # flush reductions
                                                     op, req, previous_deps,
                                                     perform_checks, error_str):
                    return False
                # See if we closed everything
                if next_child is not None and self.open_mode == OPEN_NONE:
                    if req.is_read_only():
                        self.open_mode = OPEN_READ_ONLY
                    elif req.is_write():
                        self.open_mode = OPEN_READ_WRITE
                    else:
                        assert req.redop <> 0
                        self.open_mode = OPEN_SINGLE_REDUCE
                        self.open_redop = req.redop
        else:
            assert False # Very bad
        # if we made it this far and we have a next child, record it as open
        if next_child is not None:
            assert self.open_mode is not OPEN_NONE
            self.open_children.add(next_child)
        else:
            self.open_mode = OPEN_NONE
            self.open_redop = 0
        return True

    def find_close_operation(self, op, req, perform_checks, error_str):
        close = op.get_close_operation(req, self.node, self.field)
        if close is None:
            if perform_checks:
                print "ERROR: "+str(op)+" failed to generate a close "+\
                    "operation for field "+str(self.field)+" of region "+\
                    "requirement "+str(req.index)+" at "+str(self.node)+error_str
            else:
                print "ERROR: "+str(op)+" failed to generate a close "+\
                    "operation that we normally would have expected. This "+\
                    "is likely a runtime bug. Re-run with logical checks "+\
                    "to confirm."
            if self.node.state.assert_on_fail:
                assert False
        return close

    def perform_close_checks(self, close, closed_users, op, req, 
                             previous_deps, error_str):
        assert 0 in close.reqs
        close_req = close.reqs[0]
        # Check for dependences against all the closed users
        # as well as any users in the previous set
        for prev_op,prev_req in closed_users:
            # Check for replays
            if prev_op is op:
                op.need_logical_replay = True
                return False
            if not close.has_mapping_dependence(close_req, prev_op,
                            prev_req, TRUE_DEPENDENCE, self.field):
                print "ERROR: close operation "+str(close)+" generated by "+\
                      "field "+str(self.field)+" of region requirement "+\
                      str(req.index)+" of "+str(op)+" failed to find a "+\
                      "mapping dependence on previous operation "+\
                      str(prev_op)+"in sub-tree being closed"+error_str
                if self.node.state.assert_on_fail:
                    assert False
                return False
        for prev_op,prev_req in previous_deps:
            # Check for replays
            if prev_op is op:
                op.need_logical_replay = True
                return False
            if not close.has_mapping_dependence(close_req, prev_op,
                            prev_req, TRUE_DEPENDENCE, self.field):
                print "ERROR: close operation "+str(close)+" generated by "+\
                      "field "+str(self.field)+" of region requirement "+\
                      str(req.index)+" of "+str(op)+" failed to find a "+\
                      "mapping dependence on previous operation "+\
                      str(prev_op)+" from higher in the region tree"
                if self.node.state.assert_on_fail:
                    assert False
                return False
        return True

    def record_close_dependences(self, close, closed_users, previous_deps):
        assert 0 in close.reqs
        close_req = close.reqs[0]
        for prev_op,prev_req in closed_users:
            dep = MappingDependence(prev_op, close, prev_req.index, 
                                    close_req.index, TRUE_DEPENDENCE)
            prev_op.add_outgoing(dep)
            close.add_incoming(dep)
        for prev_op,prev_req in previous_deps:
            dep = MappingDependence(prev_op, close, prev_req.index,
                                    close_req.index, TRUE_DEPENDENCE)
            prev_op.add_outgoing(dep)
            close.add_incoming(dep)

    def perform_close_operations(self, next_child, allow_next, permit_leave_open, 
                                 read_only_close, flush_reductions, op, req, 
                                 previous_deps, perform_checks, error_str):
        # Handle special cases first, general case at the bottom
        if flush_reductions:
            assert not allow_next
            assert not permit_leave_open
            assert not read_only_close
            # If we are flushing reductions we have to close up all the open children
            close = self.find_close_operation(op, req, perform_checks, error_str)
            if close is None:
                return False
            assert close.is_close()
            # Make sure it is the right kind of close operation
            if read_only_close and close.kind <> READ_ONLY_CLOSE_OP_KIND:
                if self.node.state.assert_on_fail:
                    assert False
                return False
            if not read_only_close and close.kind <> INTER_CLOSE_OP_KIND:
                if self.node.state.assert_on_fail:
                    assert False
                return False
            closed_users = list()
            for child in self.open_children:
                child.close_logical_tree(self.field, closed_users)
            # clear the open children
            self.open_children = set() 
            if perform_checks:
                if not self.perform_close_checks(close, closed_users, op, req,
                                                 previous_deps, error_str):
                    return False
            else:
                self.record_close_dependences(close, closed_users, previous_deps)
            # Now do the dependence analysis to put the close op in
            # our set of current epoch users
            assert 0 in close.reqs
            if not self.perform_epoch_analysis(close, close.reqs[0],
                                               perform_checks, True, None, op):
                return False
            # Record the close operation in the current epoch
            self.current_epoch_users.append((close,close.reqs[0]))
        elif next_child is not None and self.node.are_all_children_disjoint():
            if not allow_next and next_child in self.open_children: 
                # We actually need to do the close, get the close operation
                close = self.find_close_operation(op, req, perform_checks, error_str)
                if close is None:
                    return False
                assert close.is_close()
                # Make sure it is the right kind of close operation
                if read_only_close and close.kind <> READ_ONLY_CLOSE_OP_KIND:
                    if self.node.state.assert_on_fail:
                        assert False
                    return False
                if not read_only_close and close.kind <> INTER_CLOSE_OP_KIND:
                    if self.node.state.assert_on_fail:
                        assert False
                    return False
                closed_users = list()
                next_child.close_logical_tree(self.field, closed_users)
                if perform_checks:
                    if not self.perform_close_checks(close, closed_users, op, req,
                                                     previous_deps, error_str):
                        return False
                else:
                    self.record_close_dependences(close, closed_users, previous_deps)
                assert 0 in close.reqs
                if not self.perform_epoch_analysis(close, close_reqs[0],
                                                   perform_checks, True, None, op):
                    return False
                # Record the close operation in the current epoch
                self.current_epoch_users.append((close,close_reqs[0]))
        else:
            close = None
            closed_users = None
            to_remove = list()
            for child in self.open_children:
                if allow_next and next_child is not None and child is next_child:
                    continue
                if next_child is not None and self.node.are_children_disjoint(
                                                              child, next_child):
                    continue
                if close is None:
                    close = self.find_close_operation(op, req, perform_checks, error_str)
                    if close is None:
                        return False
                    # Make sure it is the right kind of close operation
                    if read_only_close and close.kind <> READ_ONLY_CLOSE_OP_KIND:
                        if self.node.state.assert_on_fail:
                            assert False
                        return False
                    if not read_only_close and close.kind <> INTER_CLOSE_OP_KIND:
                        if self.node.state.assert_on_fail:
                            assert False
                        return False
                    closed_users = list()
                child.close_logical_tree(self.field, closed_users)
                if not permit_leave_open:
                    to_remove.append(child)
            if close is not None:
                if perform_checks:
                    if not self.perform_close_checks(close, closed_users, op, req,
                                                     previous_deps, error_str):
                        return False
                else:
                    self.record_close_dependences(close, closed_users, previous_deps)
                # Now do the dependence analysis to put the close op in
                # our set of current epoch users
                assert 0 in close.reqs
                if not self.perform_epoch_analysis(close, close.reqs[0],
                                                   perform_checks, True, None, op):
                    return False
                # Record the close operation in the current epoch
                self.current_epoch_users.append((close,close.reqs[0]))
            for child in to_remove:
                self.open_children.remove(child)
        if not self.open_children:
            self.open_mode = OPEN_NONE
            self.open_redop = 0
        return True

    def close_logical_tree(self, closed_users):
        # Save the closed users and then close the subtrees
        closed_users += self.current_epoch_users
        self.current_epoch_users = list()
        self.previous_epoch_users = list()
        for child in self.open_children:
            child.close_logical_tree(self.field, closed_users)
        self.open_children = set()
        self.open_mode = OPEN_NONE
        self.open_redop = 0
        self.current_redop = 0
        
    def perform_epoch_analysis(self, op, req, perform_checks, 
                               can_dominate, recording_set,
                               replay_op = None):
        dominates = True
        # Check the current users first
        for prev_op,prev_req in self.current_epoch_users:
            dep_type = compute_dependence_type(prev_req, req) 
            if dep_type is NO_DEPENDENCE:
                dominates = False
                continue
            # If we have a replay op, see if we've hit it
            if replay_op is not None and prev_op is replay_op:
                replay_op.need_logical_replay = True
                return False
            # Check to see if it has the mapping dependence
            if perform_checks:
                if not op.has_mapping_dependence(req, prev_op, prev_req, 
                                                 dep_type, self.field):
                    return False
            else:
                # Not performing checks so record the mapping dependence
                dep = MappingDependence(prev_op, op, prev_req.index,
                                        req.index, dep_type)
                prev_op.add_outgoing(dep)
                op.add_incoming(dep)
            if recording_set is not None:
                recording_set.append((prev_op,prev_req))
        if not dominates:
            for prev_op,prev_req in self.previous_epoch_users:
                dep_type = compute_dependence_type(prev_req, req)
                if dep_type is NO_DEPENDENCE:
                    continue
                # If we have a replay op, see if we've hit it
                if replay_op is not None and prev_op is replay_op:
                    replay_op.need_logical_replay = True
                    return False
                if perform_checks:
                    if not op.has_mapping_dependence(req, prev_op, prev_req, 
                                                     dep_type, self.field):
                        return False
                else:
                    # Not performing checks so record the mapping dependence
                    dep = MappingDependence(prev_op, op, prev_req.index,
                                            req.index, dep_type)
                    prev_op.add_outgoing(dep)
                    op.add_incoming(dep)   
                if recording_set is not None:
                    recording_set.append((prev_op,prev_req))
        if can_dominate and dominates:
            # Filter back the users
            self.previous_epoch_users = self.current_epoch_users
            self.current_epoch_users = list()
            self.current_redop = 0 # no more reductions in the current list
        return True


class PhysicalState(object):
    __slots__ = ['node', 'depth', 'field', 'parent', 'dirty', 'redop', 
                 'valid_instances', 'reduction_instances']
    def __init__(self, node, depth, field, parent):
        self.node = node 
        self.depth = depth
        self.field = field
        self.parent = parent # parent physical state
        self.dirty = False
        self.redop = 0
        self.valid_instances = set()
        self.reduction_instances = set()

    def initialize_physical_state(self, inst):
        if inst.is_virtual():
            assert not isinstance(inst, Instance)
        if inst.redop <> 0:
            self.reduction_instances.add(inst)
            self.redop = inst.redop
        else:
            self.valid_instances.add(inst)

    def move_to(self, target, clear_state, top):
        assert self.field == target.field
        target.dirty = self.dirty
        target.redop = self.redop
        # If this is the top, pull all the valid instances down too
        if top:
            target.valid_instances = self.find_valid_instances()
        elif clear_state:
            target.valid_instances = self.valid_instances
        else:
            target.valid_instances = self.valid_instances.copy()
        if clear_state:
            target.reduction_instances = self.reduction_instances
        else:
            target.reduction_instances = self.reduction_instances.copy()
        if clear_state:
            self.dirty = False
            self.redop = 0
            self.valid_instances = set()
            self.reduction_instances = set()

    def perform_physical_analysis(self, op, req, inst, perform_checks, register):
        assert not inst.is_virtual()
        assert req.logical_node is self.node
        if req.is_reduce():
            # It's a runtime bug if this is a non-reduction instance
            assert inst.redop <> 0
            # Add it to the list of reduction instances
            self.reduction_instances.add(inst)
            self.redop = inst.redop
        elif req.is_write_only():
            assert self.redop == 0
            assert not self.reduction_instances
            # If we are write only, we just need to close up any open children
            if not self.node.close_physical_tree(self.depth, self.field, 
                                                 None, op, perform_checks):
                return False
            # Clear out all previous valid instances make this the only one
            self.valid_instances = set()
            self.valid_instances.add(inst)
            self.dirty = True
        else:
            assert self.redop == 0
            assert not self.reduction_instances
            # Find the valid set of instances
            valid = self.find_valid_instances()
            # See if we are valid yet or not 
            if inst not in valid:
                error_str = "region requirement "+str(req.index)+" of "+str(op)
                # Not valid yet, we need to issue copies to make ourselves valid
                if not self.issue_update_copies(inst, valid, op, 
                                                perform_checks, error_str):
                    return False
            # If we are writing, close up any open children
            if req.is_write():
                # Close up the tree to our instance
                if not self.node.close_physical_tree(self.depth, self.field, 
                                                     inst, op, perform_checks):
                    return False
                # We are now the only valid copy
                self.valid_instances = set()
                self.valid_instances.add(inst)
                self.dirty = True
            else:
                self.valid_instances.add(inst)
        if register:
            # Find our preconditions for using this instance
            preconditions = inst.find_use_dependences(self.depth, self.field, 
                                                      req, perform_checks)
            if perform_checks:
                bad = check_preconditions(preconditions, op)
                if bad is not None:
                    print "ERROR: Missing use precondition for field "+str(self.field)+\
                          " of region requirement "+str(req.index)+" of "+str(op)+\
                          " (UID "+str(op.uid)+") on previous "+str(bad)
                    if self.node.state.assert_on_fail:
                        assert False
                    return False
            else:
                for other in preconditions:
                    op.physical_incoming.add(other)
                    other.physical_outgoing.add(op)
            # Record ourselves as a users for this instance
            inst.add_user(self.depth, self.field, op, req)
        return True

    def perform_fill_analysis(self, op, req, perform_checks):
        assert req.logical_node is self.node
        # Clean out our state and all our child states
        if not self.node.close_physical_tree(self.depth, self.field, 
                                             None, op, perform_checks):
            return False
        # Mark that we are dirty since we are writing
        self.dirty = True  
        self.redop = 0
        self.valid_instances = set()
        self.reduction_instances = set()
        # Make our fill instance and add it to the valid instances
        fill_inst = FillInstance(op.state, self.node, self.depth, self.field)
        self.valid_instances.add(fill_inst)
        return True

    def perform_physical_close(self, op, req, inst, perform_checks):
        if inst is None:
            # This is a read only close operation, we just have to
            # invalidate all the stuff in the sub tree, no need to 
            # invalidate anything at this node
            return self.node.close_physical_tree(self.depth, self.field,
                                                 None, op, perform_checks)
        elif inst.is_virtual():
            target = CompositeInstance(op.state, self.node, self.depth, self.field)
            # Capture down the tree first
            if not self.node.close_physical_tree(self.depth, self.field,
                                                 target, op, perform_checks):
                return False
            # Now capture locally
            already_captured = set()
            target.capture(self, already_captured)
        else:
            # Issue any local updates needed first
            if self.dirty:
                # If we are not already valid, we need to be made valid
                if inst not in self.valid_instances:
                    error_str = "region requirement "+str(req.index)+" of "+str(op)
                    if not self.issue_update_copies(inst, self.valid_instances, op,
                                                    perform_checks, error_str):
                        return False
            # close sub-tree
            if not self.node.close_physical_tree(self.depth, self.field,
                                                 inst, op, perform_checks):
                return False
            # Finally flush any outstanding reductions
            if self.reduction_instances:
                assert self.redop <> 0
                error_str = "region requirement "+str(req.index)+" of "+str(op)
                if not self.issue_update_reductions(inst, self.reduction_instances, 
                                                    op, perform_checks, error_str):
                    return False
            target = inst
        self.dirty = True
        self.redop = 0
        self.valid_instances = set()
        self.reduction_instances = set()
        self.valid_instances.add(target)
        return True

    def close_physical_tree(self, target, op, perform_checks):
        # Issue any updates from our instances
        if target is not None and not target.is_virtual() and self.dirty:
            if not self.issue_update_copies(target, self.valid_instances, 
                                            op, perform_checks, str(op)):
                return False
        # Continue down the tree 
        if not self.node.close_physical_tree(self.depth, self.field, 
                                             target, op, perform_checks):
            return False
        # If the target is a composite instance do the capture
        # otherwise flush any reductions
        if target is not None:
            if target.is_virtual():
                already_captured = set()
                target.capture(self, already_captured)
            elif self.redop <> 0:
                assert self.reduction_instances
                if not self.issue_update_reductions(target, self.reduction_instances,
                                                    op, perform_checks):
                    return False
        # Now we can reset everything since we are closed
        self.dirty = False
        self.redop = 0
        self.valid_instances = set()
        self.reduction_instances = set()
        return True

    def find_valid_instances(self):
        # We can't go up anymore when we are dirty or there is no parent
        if self.dirty or self.redop <> 0 or self.parent is None:
            # Make sure to make a copy
            result = self.valid_instances.copy()
            return result
        # Keep going up
        result = self.parent.find_valid_instances()
        # Add our instances to the set
        for inst in self.valid_instances:
            result.add(inst)
        return result

    def issue_update_copies(self, dst, valid, op, perform_checks, error_str):
        # There better be a destination
        assert dst
        # Easy out if there are no valid instances
        if not valid:
            return True
        # If we only have one instance and it is a virtual instance
        # then we need to do something special
        if len(valid) == 1 and next(iter(valid)).is_virtual():
            virtual_inst = next(iter(valid))
            return virtual_inst.issue_update_copies(dst, self.depth, self.node, op, 
                                                    perform_checks, error_str)
        # Find the destination preconditions since we will
        # need to know them no matter what
        dst_preconditions = dst.find_copy_dependences(self.depth, self.field, 
                                        self.node, False, 0, perform_checks)
        if perform_checks:
            # Find the copy that was generated by this operation
            copy = op.find_generated_copy(self.field, dst.region, dst, 0, self.node)
            if copy is None:
                print "ERROR: Missing copy operation to update "+str(dst)+\
                      " for field "+str(self.field)+" by "+error_str
                if self.node.state.assert_on_fail:
                    assert False
                return False
            assert self.field in copy.dst_fields
            copy_index = copy.dst_fields.index(self.field)
            assert copy.dsts[copy_index] is dst
            src = copy.srcs[copy_index]
            if src not in valid:
                print "ERROR: Copy operation by "+error_str+" to update "+str(dst)+\
                      " is from source "+str(src)+" which is not in the set of "+\
                      "valid instances."
                if self.node.state.assert_on_fail:
                    assert False
                return False
            # Now check for event preconditions 
            src_preconditions = src.find_copy_dependences(self.depth, self.field, 
                                              self.node, True, 0, perform_checks)
            # Have to fill in the reachable cache
            if copy.reachable_cache is None:
                copy.reachable_cache = set()
                copy.get_physical_reachable(copy.reachable_cache, False)
            bad = check_preconditions(src_preconditions, copy)
            if bad is not None:
                print "ERROR: Missing source copy precondition for "+str(copy)+\
                      " of field "+str(self.field)+" issued by "+error_str+" on "+str(bad)
                if self.node.state.assert_on_fail:
                    assert False
                return False
            bad = check_preconditions(dst_preconditions, copy)
            if bad is not None:
                print "ERROR: Missing destination copy precondition for "+str(copy)+\
                      " of field "+str(self.field)+" issued by "+error_str+\
                      " on "+str(bad)
                if self.node.state.assert_on_fail:
                    assert False
                return False
        else:
            # Figure out which instance to copy from
            if len(valid) > 1:
                print "INFO: Multiple valid instances to choose from... picking one" 
            src = next(iter(valid)) 
            src_preconditions = src.find_copy_dependences(self.depth, self.field, 
                                              self.node, True, 0, perform_checks)
            # Make a realm copy from the source to the dst for this field
            copy = dst.state.create_copy(op)
            copy.add_field(self.field.fid, src, self.field.fid, dst, 0) 
            copy.set_region(self.node)
            # Add the preconditions to the physical graph
            for src_op in src_preconditions:
                src_op.physical_outgoing.add(copy)
                copy.physical_incoming.add(src_op)
            for dst_op in dst_preconditions:
                dst_op.physical_outgoing.add(copy)
                copy.physical_incoming.add(dst_op)
        # Record the copy user
        src.add_copy_user(self.depth, self.field, self.node, copy, True, 0)
        dst.add_copy_user(self.depth, self.field, self.node, copy, False, 0)
        return True

    def issue_update_reductions(self, dst, reductions, op, perform_checks, error_str):
        # There better be a destination 
        assert dst 
        if perform_checks:
            for src in reductions:
                assert src.redop <> 0
                reduction = op.find_generated_copy(self.field, self.node, dst, src.redop)
                if reduction is None:
                    print "ERROR: Missing reduction copy operation to update "+str(dst)+\
                          " for field "+str(self.field)+" from "+str(src)+" by "+error_str
                    if self.node.state.assert_on_fail:
                        assert False
                    return False
                assert self.field in reduction.dst_fields
                reduc_index = reduction.dst_fields.index(self.field)
                assert reduction.dsts[reduc_index] is dst
                assert reduction.redops[reduc_index] == src.redop
                src = reduction.srcs[reduc_index]
                # Now check for event preconditions
                src_preconditions = src.find_copy_dependences(self.depth, self.field, 
                                                            self.node, True, 0, True) 
                # Fill in the reachable cache if necessary
                if reduction.reachable_cache is None:
                    reduction.reachable_cache = set()
                    reduction.get_physical_reachable(reduction.reachable_cache, False)
                bad = check_preconditions(src_preconditions, reduction)
                if bad is not None:
                    print "ERROR: Missing source copy precondition for reduction of field "+\
                          str(self.field)+" issued by "+erro_str+" on "+str(bad)
                    if self.node.state.assert_on_fail:
                        assert False
                    return False
                dst_preconditions = dst.find_copy_dependences(self.depth, self.field, 
                                                    self.node, False, src.redop, True)
                bad = check_preconditions(dst_preconditions, reduction)
                if bad is not None:
                    print "ERROR: Missing destination copy precondition for reduction "+\
                          "of field "+str(self.field)+" issued by "+error_str+" on "+str(bad)
                    if self.node.state.assert_on_fail:
                        assert False
                    return False
                # Register the users
                src.add_copy_user(self.depth, self.field, self.node, 
                                  reduction, True, 0) # just reading
                dst.add_copy_user(self.depth, self.field, self.node, 
                                  reduction, False, src.redop)
        else:
            # flush all the reductions to the destination 
            for src in reductions:
                assert src.redop <> 0
                # Make a realm copy from the source to the dst for this field   
                reduction = self.node.state.create_copy(op)
                reduction.set_region(self.node)
                reduction.add_field(self.field.fid, src, self.field.fid, dst, src.redop) 
                src_preconditions = src.find_copy_dependences(self.depth, self.field, 
                                                            self.node, True, 0, False)
                for src_op in src_preconditions:
                    src_op.physical_outgoing.add(reduction)
                    reduction.add_physical_incoming(src_op)
                dst_preconditions = dst.find_copy_dependences(self.depth, self.field, 
                                                    self.node, False, src.redop, False)
                for dst_op in dst_preconditions:
                    dst_op.physical_outgoing.add(reduction)
                    reduction.physical_incoming.add(dst_op)
                # Record the copy user
                src.add_copy_user(self.depth, self.field, self.node, 
                                  reduction, True, 0) # Just reading
                dst.add_copy_user(self.depth, self.field, self.node, 
                                  reduction, False, src.redop)
        return True

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
            return "NO-ACCESS"
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
        return self.get_privilege() + ' ' + self.get_coherence()

class MappingDependence(object):
    __slots__ = ['op1', 'op2', 'idx1', 'idx2', 'dtype']
    def __init__(self, op1, op2, idx1, idx2, dtype):
        self.op1 = op1
        self.op2 = op2
        self.idx1 = idx1
        self.idx2 = idx2
        self.dtype = dtype

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

    def print_dataflow_edge(self, printer, previous_pairs):
        pair = (self.op1,self.op2)
        if pair not in previous_pairs:
            printer.println(self.op1.node_name+' -> '+self.op2.node_name+
                            ' [style=solid,color=black,penwidth=2];')
            previous_pairs.add(pair)
        
class Operation(object):
    __slots__ = ['state', 'uid', 'kind', 'context', 'name', 'reqs', 'mappings', 
                 'incoming', 'outgoing', 'logical_incoming', 'logical_outgoing',
                 'physical_incoming', 'physical_outgoing', 'start_event', 
                 'finish_event', 'inter_close_ops', 'task', 'task_id', 'points', 
                 'creator', 'realm_copies', 'realm_fills', 'close_idx', 
                 'partition_kind', 'partition_node', 'node_name', 'cluster_name', 
                 'generation', 'need_logical_replay', 'reachable_cache']
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.kind = NO_OP_KIND 
        self.context = None
        self.name = None
        self.reqs = None
        self.mappings = None
        self.incoming = None # Mapping dependences
        self.outgoing = None # Mapping dependences
        self.logical_incoming = None # Operation dependences
        self.logical_outgoing = None # Operation dependences
        self.physical_incoming = set() # op/realm
        self.physical_outgoing = set() # op/realm
        self.start_event = state.get_no_event() 
        self.finish_event = state.get_no_event()
        self.inter_close_ops = None
        self.realm_copies = None
        self.realm_fills = None
        # Only valid for tasks
        self.task = None
        self.task_id = -1
        # Only valid for index tasks
        self.points = None
        # Only valid for close operations
        self.creator = None
        self.close_idx = -1
        # Only valid for pending partition operations
        self.partition_kind = None
        self.partition_node = None
        self.node_name = 'op_node_'+str(uid)
        self.cluster_name = None 
        # For traversals
        self.generation = 0
        self.need_logical_replay = False
        self.reachable_cache = None

    def is_close(self):
        return self.kind == INTER_CLOSE_OP_KIND or self.kind == POST_CLOSE_OP_KIND or \
               self.kind == READ_ONLY_CLOSE_OP_KIND

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
        if self.points is not None:
            for point in self.points.itervalues():
                point.op.set_context(context, False)
        if add:
          self.context.add_operation(self)

    def get_context(self):
        assert self.context is not None
        return self.context

    def set_op_kind(self, kind):
        if self.kind == NO_OP_KIND:
            self.kind = kind
        else:
            assert self.kind is kind

    def set_events(self, start, finish):
        if start.exists():
            start.add_outgoing_op(self)
        if finish.exists():
            finish.add_incoming_op(self)
        self.start_event = start
        self.finish_event = finish

    def set_task_id(self, task_id):
        assert self.kind == SINGLE_TASK_KIND or self.kind == INDEX_TASK_KIND
        if self.task_id <> -1:
            assert task_id == self.task_id
        else:
            self.task_id = task_id

    def set_creator(self, creator, idx):
        # Better be a close op kind
        assert self.kind == INTER_CLOSE_OP_KIND or \
            self.kind == READ_ONLY_CLOSE_OP_KIND or self.kind == POST_CLOSE_OP_KIND
        self.creator = creator
        self.close_idx = idx
        # If our parent context created us we don't need to be recorded 
        if creator is not self.context.op:
            assert self.kind <> POST_CLOSE_OP_KIND
            creator.add_close_operation(self)
        else:
            assert self.kind == POST_CLOSE_OP_KIND

    def add_close_operation(self, close):
        if self.inter_close_ops is None:
            self.inter_close_ops = set()
        self.inter_close_ops.add(close)

    def get_depth(self):
        assert self.context is not None
        return self.context.get_depth() + 1

    def get_close_operation(self, req, node, field):
        if self.inter_close_ops is None:
            return None
        for close in self.inter_close_ops:
            if close.close_idx <> req.index:
                continue
            assert len(close.reqs) == 1
            close_req = close.reqs[0]
            if close_req.logical_node is not node:
                continue
            if field not in close_req.fields:
                continue 
            return close
        return None

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
        if self.context is not None:
            point.op.set_context(self.context, False)

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
        if self.logical_incoming is None:
            self.logical_incoming = set()
        self.logical_incoming.add(dep.op1)

    def add_outgoing(self, dep):
        assert dep.op1 == self
        if self.outgoing is None:
            self.outgoing = set()
        self.outgoing.add(dep)
        if self.logical_outgoing is None:
            self.logical_outgoing = set()
        self.logical_outgoing.add(dep.op2)

    def add_realm_copy(self, copy):
        if self.realm_copies is None:
            self.realm_copies = list()
        self.realm_copies.append(copy)

    def add_realm_fill(self, fill):
        if self.realm_fills is None:
            self.realm_fills = list()
        self.realm_fills.append(fill)

    def get_logical_reachable(self, reachable, forward):
        if self in reachable:
            return 
        reachable.add(self)
        if forward:
            if self.logical_outgoing is None:
                return
            for op in self.logical_outgoing:
                op.get_logical_reachable(reachable, True)
        else:
            if self.logical_incoming is None:
                return
            for op in self.logical_incoming:
                op.get_logical_reachable(reachable, False)

    def get_physical_reachable(self, reachable, forward):
        if self in reachable:
            return 
        reachable.add(self)
        if forward:
            for op in self.physical_outgoing:
                op.get_physical_reachable(reachable, True)
        else:
            for op in self.physical_incoming:
                op.get_physical_reachable(reachable, False)

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

    def compute_physical_reachable(self):
        # We can skip some of these
        if not self.is_physical_operation() or self.kind is INDEX_TASK_KIND:
            return
        # Once we reach something that is not an event
        # then we record it and return
        def traverse_node(node, traverser):
            if not node.is_physical_operation():
                return True 
            traverser.reachable.add(node)
            return False
        if self.start_event.exists():
            traverser = EventGraphTraverser(False, True,
                self.state.get_next_traversal_generation(),
                None, traverse_node, traverse_node, traverse_node)
            traverser.reachable = self.physical_incoming
            traverser.visit_event(self.start_event)
            # Keep everything symmetric
            for other in self.physical_incoming:
                other.physical_outgoing.add(self)
        if self.finish_event.exists():
            traverser = EventGraphTraverser(True, True,
                self.state.get_next_traversal_generation(),
                None, traverse_node, traverse_node, traverse_node)
            traverser.reachable = self.physical_outgoing
            traverser.visit_event(self.finish_event)
            # Keep everything symmetric
            for other in self.physical_outgoing:
                other.physical_incoming.add(self)

    def find_generated_copy(self, field, region, dst, redop=0, intersect=None):
        if self.realm_copies is None:
            return None
        for copy in self.realm_copies:
            if region is not copy.region:
                continue
            if field not in copy.src_fields:
                continue 
            if dst not in copy.dsts:
                continue
            if redop <> 0 and redop not in copy.redops:
                continue
            if intersect is not None or copy.intersect is not None:
                if copy.intersect is not None:
                    copy_shape = copy.region.intersection(copy.intersect)
                else:
                    copy_shape = copy.region.get_shape()
                base_shape = region.get_shape()
                if intersect is not None:
                    base_shape = region.intersection(intersect)
                else:
                    base_shape = region.get_shape()
                one = copy_shape - base_shape
                if not one.empty():
                    continue
                two = base_shape - copy_shape
                if not two.empty():
                    continue
            # Record that we analyzed this copy
            copy.analyzed = True
            return copy
        return None

    def find_generated_copy_across(self, src_field, dst_field, region, 
                                   src_inst, dst_inst, redop=0, intersect=None):
        if self.realm_copies is None:
            return None
        for copy in self.realm_copies:
            if region is not copy.region:
                continue
            if src_field not in copy.src_fields:
                continue
            index = copy.src_fields.index(src_field)
            if dst_field is not copy.dst_fields[index]:
                continue
            if src_inst is not copy.srcs[index]:
                continue
            if dst_inst is not copy.dsts[index]:
                continue
            if redop <> copy.redops[index]:
                continue
            if intersect is not None or copy.intersect is not None:
                if copy.intersect is not None:
                    copy_shape = copy.region.get_shape()
                else:
                    copy_shape = copy.region.intersection(copy.intersect)
                base_shape = region.get_shape()
                if intersect is not None:
                    base_shape =  region.intersection(intersect)
                else:
                    base_shape = region.get_shape()
                one = copy_shape - base_shape
                if not one.empty():
                    continue
                two = base_shape - copy_shape
                if not two.empty():
                    continue
            # Record that we analyzed the copy
            copy.analyzed = True
            return copy
        return None

    def find_generated_fill(self, field, region, dst, intersect=None):
        if self.realm_fills is None:
            return None
        for fill in self.realm_fills:
            if region is not fill.region:
                continue
            if field not in fill.fields:
                continue
            if dst not in fill.dsts:
                continue
            if fill.intersect is not intersect:
                continue
            # Record that we analyzed this fill
            fill.analyzed = True
            return fill
        return None

    def perform_cycle_check(self):
        def traverse_node(node, traverser):
            if node is traverser.origin:
                traverser.cycle = True
                print "CYCLE DETECTED!"
                for n in traverser.stack:
                    print str(n)
                print str(node)
                return False
            if traverser.cycle:
                return False
            traverser.stack.append(node)
            return True
        def post_traverse(node, traverser):
            assert traverser.stack
            traverser.stack.pop()
        traverser = EventGraphTraverser(False, True,
            self.state.get_next_traversal_generation(),
            None, traverse_node, traverse_node, traverse_node,
            None, post_traverse, post_traverse, post_traverse)
        traverser.origin = self
        traverser.cycle = False
        traverser.stack = list()
        traverser.visit_event(self.start_event)
        return traverser.cycle

    def analyze_logical_requirement(self, index, projecting, perform_checks):
        # Special out for no access
        assert index in self.reqs
        req = self.reqs[index]
        if req.priv is NO_ACCESS:
            return True
        # Destination requirements for copies are a little weird because
        # they actually need to behave like READ_WRITE privileges
        if self.kind == COPY_OP_KIND and len(self.reqs)/2 <= index:
            if req.priv == REDUCE:
                copy_reduce = True
                req.priv = READ_WRITE
            else:
                copy_reduce = False
        else:
            copy_reduce = False
        # Compute the analysis path
        path = list()
        req.logical_node.compute_path(path, req.parent)
        assert not not path
        # TODO: check restricted coherence too
        # Now do the traversal for each of the fields
        for field in req.fields:
            # Keep track of the previous dependences so we can 
            # use them for adding/checking dependences on close operations
            previous_deps = list()
            if not req.parent.perform_logical_analysis(0, path, self, 
                    req, field, projecting, previous_deps, perform_checks):
                return False
            # If we are projecting we have to iterate all the points
            # and walk their paths, still doing the checking for ourself
            if projecting:
                assert self.points
                for point in self.points.itervalues():
                    assert index in point.op.reqs
                    point_req = point.op.reqs[index]
                    point_path = list()
                    point_req.logical_node.compute_path(point_path, req.logical_node)
                    point_deps = previous_deps
                    if not req.logical_node.perform_logical_analysis(0, point_path,
                          self, point_req, field, False, point_deps, perform_checks):
                        return False
        # Restore the privileges if necessary
        if copy_reduce:
            req.priv = REDUCE
        return True

    def analyze_logical_fence(self, perform_checks):
        if perform_checks:
            # TODO: Figure out how to turn this back on 
            print "WARNING: Logical fence analysis is currently disabled"
            return True
        for index,req in self.context.op.reqs.iteritems():
            for field in req.fields:
                if not req.logical_node.perform_logical_fence(self, field, perform_checks):
                    return False
        return True

    def perform_logical_analysis(self, perform_checks):
        if self.kind == DELETION_OP_KIND and not perform_checks:
            # TODO: fix this
            print "WARNING: Legion Spy doesn't really know how to do logical "+\
                  "analysis for deletion operations at the moment. They might "+\
                  "look a little weird in the dataflow graphs. Try to ignore them."
        # We need a context to do this
        assert self.context is not None
        # See if there is a fence in place for this context
        if self.context.current_fence is not None:
            if perform_checks:
                if self.context.current_fence not in self.logical_incoming: 
                    print "ERROR: missing logical fence dependence between "+\
                          str(self.context.current_fence)+" and "+str(self)
                    if self.state.assert_on_fail:
                        assert False
                    return False
            else:
                dep = MappingDependence(self.context.current_fence, self, 0, 0,
                                        TRUE_DEPENDENCE)
                self.context.current_fence.add_outgoing(dep)
                self.add_incoming(dep)
        if self.reqs is None:
            # If this is a fence, check or record dependences on everything from
            # either the begining or from the previous fence
            if self.kind == FENCE_OP_KIND:
                # Record dependences on all the users in the region tree 
                if not self.analyze_logical_fence(perform_checks):
                    return False
                # Finally record ourselves as the next fence
                self.context.current_fence = self
            return True
        assert not self.need_logical_replay
        projecting = self.kind is INDEX_TASK_KIND
        replay_regions = list()
        for idx in self.reqs.iterkeys():
            if not self.analyze_logical_requirement(idx, projecting, perform_checks):
                # We might have failed because we have a replay region
                if self.need_logical_replay:
                    replay_regions.append(idx)
                    self.need_logical_replay = False
                else: # We really did fail
                    return False
        # If we had any replay regions, analyze them now
        for idx in replay_regions:
            if not self.analyze_logical_requirement(idx, projecting, perform_checks):
                if self.need_logical_replay:
                    print "ERROR: Replay failed! This is really bad! "+\
                          "Region requirement "+str(idx)+" of "+str(self)+\
                          "failed to replay successfully. This is most likely "+\
                          "a conceptual bug in the analysis and not an "+\
                          "implementation bug."
                    if self.state.assert_on_fail:
                        assert False
                return False
        return True

    def has_mapping_dependence(self, req, prev_op, prev_req, dtype, field):
        for dep in self.incoming:
            if dep.op1 is not prev_op:
                continue
            if dep.idx1 is not prev_req.index:
                continue
            if dep.idx2 is not req.index:
                continue
            if dep.dtype is not dtype:
                continue
            # We found a good mapping dependence, so all is good
            return True
        # No need to look for it transitively since this analysis should exactly
        # match the analysis done by the runtime
        # Issue the error and return false
        print "ERROR: Missing mapping dependence on "+str(field)+" between region "+\
              "requirement "+str(prev_req.index)+" of "+str(prev_op)+" and region "+\
              "requriement "+str(req.index)+" of "+str(self)
        if self.state.assert_on_fail:
            assert False
        return False

    def analyze_previous_interference(self, next_op, next_req, reachable):
        if not self.reqs:
            # Check to see if this is a fence operation
            if self.kind == FENCE_OP_KIND:
                # If we've got a fence operation, it should have a transitive
                # dependence on all operations that came before it
                if not self in reachable:
                    print "ERROR: Failed logical sanity check. No mapping dependence "+\
                          "between previous "+str(self)+" and later "+str(next_op)
                    if self.state.assert_on_fail:
                        assert False
                    return False
            return True
        for req in self.reqs.itervalues():
            # Check to see if there is any overlap in fields or regions 
            if len(set(req.fields) & set(next_req.fields)) == 0:
                continue # No overlapping fields so we can keep going
            # Check to see if they are in different region trees in 
            # which case there can be no aliasing
            if req.tid <> next_req.tid:
                continue
            if not self.state.is_aliased(req.index_node, next_req.index_node):
                continue
            dep_type = compute_dependence_type(req, next_req) 
            if dep_type == NO_DEPENDENCE:
                continue
            # Otherwise they do interfere, check to see if this node is
            # in the previous set, if not, that is not good
            if self not in reachable:
                print "ERROR: Failed logical sanity check. No mapping dependence "+\
                      "path exists between region requirement "+str(req.index)+\
                      " of "+str(self)+" and region requirement "+str(next_req.index)+\
                      " of "+str(next_op)
                print "  First Requirement:"
                req.print_requirement()
                print "  Second Requirement:"
                next_req.print_requirement()
                if self.state.assert_on_fail:
                    assert False
                return False
        return True

    def analyze_logical_interference(self, prev_op, reachable):
        for req in self.reqs.itervalues():  
            if not prev_op.analyze_previous_interference(self, req, reachable):
                return False
        return True

    def find_mapping(self, index, parent_node):
        assert index in self.mappings
        # Tell our context that we are using these instances, this is 
        # necessary for handling virtual mappings correctly
        result = self.mappings[index]
        assert self.context is not None
        for fid,inst in result.iteritems():
            if inst.is_virtual():
                continue
            self.context.record_instance_use(inst, fid, parent_node) 
        return result

    def analyze_physical_requirement(self, depth, index, req, perform_checks):
        if req.is_no_access():
            return True
        mappings = self.find_mapping(index, req.parent)
        for field in req.fields:
            # Find the instance that we chose to map this field to
            if field.fid not in mappings:
                print "Missing mapping decision for field "+str(field)+" of "+\
                      "requirement requirement "+str(index)+" of "+str(self)
                print "This is a runtime logging bug, please report it."
            assert field.fid in mappings
            inst = mappings[field.fid]
            if inst.is_virtual():
                # In the case of virtual mappings we don't have to
                # do any analysis here since we're just passing in the state
                continue
            if not req.logical_node.perform_physical_analysis(depth, field, self, 
                                                      req, inst, perform_checks):
                return False
        return True

    def analyze_copy_requirements(self, depth, src_index, src_req, 
                                  dst_index, dst_req, perform_checks):
        assert len(src_req.fields) == 1
        assert len(dst_req.fields) == 1
        src_field = next(iter(src_req.fields))
        dst_field = next(iter(dst_req.fields))
        src_mappings = self.find_mapping(src_index, src_req.parent)
        dst_mappings = self.find_mapping(dst_index, dst_req.parent)
        assert src_field.fid in src_mappings
        assert dst_field.fid in dst_mappings
        src_inst = src_mappings[src_field.fid]
        dst_inst = dst_mappings[dst_field.fid]
        assert not dst_inst.is_virtual()
        is_reduce = dst_req.is_reduce()
        # Switch this to read-write privileges and then switch it back after we are done
        # The runtime does this too so that its analysis is correct
        if is_reduce:
            dst_req.priv = READ_WRITE
        # Analyze the source and destination regions but don't register yet
        if not src_inst.is_virtual() and not src_req.logical_node.perform_physical_analysis(
                          depth, src_field, self, src_req, src_inst, perform_checks, False):
            return False
        if not dst_req.logical_node.perform_physical_analysis(depth, dst_field, self,
                                                  dst_req, dst_inst, perform_checks, False):
            return False
        # Now we issue the copy across
        # See if we are doing a reduction or a normal copy
        if is_reduce:
            assert dst_req.redop <> 0
            # Reduction case
            if src_inst.is_virtual():
                # This is a runtime bug, there should never be any reductions across
                assert False
                return False
            else:
                # Normal reduction, find the source and destination dependences
                src_preconditions = src_inst.find_use_dependences(depth, src_field, 
                                                                  src_req, perform_checks)
                dst_preconditions = dst_inst.find_use_dependences(depth, dst_field, 
                                                                  dst_req, perform_checks)
                if perform_checks:
                    reduction = self.find_generated_copy_across(src_field, dst_field,
                                  dst_req.logical_node, src_inst, dst_inst, dst_req.redop)
                    if reduction is None:
                        print "ERROR: Missing reduction across operation from field "+\
                              str(src_field)+" to field "+str(dst_field)+" between region "+\
                              "requirements "+str(src_index)+" and "+str(dst_index)+" of "+\
                              str(self)
                        if self.state.assert_on_fail:
                            assert False
                        return False
                    # Have to fill out the reachable cache
                    if reduction.reachable_cache is None:
                        reduction.reachable_cache = set()
                        reduction.get_physical_reachable(reduction.reachable_cache, False)
                    bad = check_preconditions(src_preconditions, reduction)
                    if bad is not None:
                        print "ERROR: Missing source precondition for reduction across "+\
                              "from field "+str(src_field)+" to field "+str(dst_field)+\
                              "between region requirements "+str(src_index)+" and "+\
                              str(dst_index)+" of "+str(self)
                        if self.state.assert_on_fail:
                            assert False
                        return False
                    bad = check_preconditions(dst_preconditions, reduction)
                    if bad is not None:
                        print "ERROR: Missing destination precondition for reduction "+\
                              "across from field "+str(src_field)+" to field "+\
                              str(dst_field)+"between region requirements "+str(src_index)+\
                              " and "+str(dst_index)+" of "+str(self)
                        if self.state.assert_on_fail:
                            assert False
                        return False
                else:
                    # Otherwise make the copy across and record the dependences  
                    reduction = self.state.create_copy(self)
                    reduction.set_region(dst_req.logical_node)
                    reduction.add_field(src_field.fid, src_inst, 
                                        dst_field.fid, dst_inst, dst_req.redop)
                    for src in src_preconditions:
                        src.physical_outgoing.add(reduction)
                        reduction.physical_incoming.add(src)
                    for dst in dst_preconditions:
                        dst.physical_outgoing.add(reduction)
                        reduction.physical_incoming.add(dst)
                # Record the copy users
                src_inst.add_copy_user(depth, src_field, src_req.logical_node, 
                                       reduction, True, 0)
                dst_inst.add_copy_user(depth, dst_field, dst_req.logical_node, 
                                       reduction, False, dst_req.redop)
            # Done with the analysis, swith the privilege back
            dst_req.priv = REDUCE
        else:
            # Normal copy case
            if dst_inst.is_virtual():
                error_str = "source field "+str(src_field)+" and destination field "+\
                            str(dst_field)+" of region requirements "+src(src_index)+\
                            " and "+str(dst_index)+" of "+str(self)
                return src_inst.issue_copies_across(dst_inst, depth, dst_field, 
                              dst_region.logical_node, self, perform_checks, error_str)
            else:
                # Normal copy
                src_preconditions = src_inst.find_use_dependences(depth, src_field, 
                                                                  src_req, perform_checks)
                dst_preconditions = dst_inst.find_use_dependences(depth, dst_field, 
                                                                  dst_req, perform_checks)
                if perform_checks:
                    copy = self.find_generated_copy_across(src_field, dst_field,
                                          dst_req.logical_node, src_inst, dst_inst)
                    if copy is None:
                        print "ERROR: Missing copy across operation from field "+\
                              str(src_field)+" to field "+str(dst_field)+" between region "+\
                              "requirements "+str(src_index)+" and "+str(dst_index)+" of "+\
                              str(self)
                        if self.state.assert_on_fail:
                            assert False
                        return False
                    # Have to fill in the copy reachable cache
                    if copy.reachable_cache is None:
                        copy.reachable_cache = set()
                        copy.get_physical_reachable(copy.reachable_cache, False)
                    bad = check_preconditions(src_preconditions, copy)
                    if bad is not None:
                        print "ERROR: Missing source precondition for copy across from "+\
                              "field "+str(src_field)+" to field "+str(dst_field)+\
                              "between region requirements "+str(src_index)+" and "+\
                              str(dst_index)+" of "+str(self)
                        if self.state.assert_on_fail:
                            assert False
                        return False
                    bad = check_preconditions(dst_preconditions, copy)
                    if bad is not None:
                        print "ERROR: Missing destination precondition for copy across "+\
                              "from field "+str(src_field)+" to field "+str(dst_field)+\
                              "between region requirements "+str(src_index)+" and "+\
                              str(dst_index)+" of "+str(self)
                        return False
                else:
                    # Otherwise make the copy across and record the dependences
                    copy = self.state.create_copy(self)
                    copy.set_region(dst_req.logical_node)
                    copy.add_field(src_field.fid, src_inst, dst_field.fid, dst_inst, 0)
                    for src in src_preconditions:
                        src.physical_outgoing.add(copy)
                        copy.physical_incoming.add(src)
                    for dst in dst_preconditions:
                        dst.physical_outgoing.add(copy)
                        copy.physical_incoming.add(dst)
                # Record the copy users
                src_inst.add_copy_user(depth, src_field, src_req.logical_node, 
                                       copy, True, 0)
                dst_inst.add_copy_user(depth, dst_field, dst_req.logical_node, 
                                       copy, False, 0)
        return True

    def analyze_fill_requirement(self, depth, index, req, perform_checks):
        for field in req.fields:
            if not req.logical_node.perform_fill_analysis(depth, field, self,
                                                      req, perform_checks):
                return False
        return True

    def perform_op_physical_analysis(self, depth, perform_checks):
        # Handle special cases first
        # Do any of our close operations before ourself
        if self.inter_close_ops:
            assert not self.is_close()
            prefix = ''
            for idx in range(depth):
                prefix += '  '
            for close in self.inter_close_ops:
                print prefix+"Performing physical dependence analysis for "+\
                      str(close)+" generated by "+str(self)
                if not close.perform_physical_close_analysis(depth, perform_checks):
                    return False
        # If we are an index space task, only do our points
        if self.kind == INDEX_TASK_KIND:
            assert self.points is not None
            for point in self.points.itervalues():
                if not point.op.perform_op_physical_analysis(depth, perform_checks):
                    return False
            return True
        prefix = ''
        for idx in range(depth):
            prefix += '  '
        print prefix+"Performing physical dependence analysis for %s..." % str(self)
        # If this is a close operation itself do the close analysis
        if self.is_close():
            return self.perform_physical_close_analysis(depth, perform_checks)
        # Do traversals for all of our region requirements and map our instances
        if not self.reqs:
            # Go down the task tree if we are a task
            if self.kind == SINGLE_TASK_KIND:
                assert self.reachable_cache is None
                self.reachable_cache = set()
                self.get_physical_reachable(self.reachable_cache, False)
                task = self.state.get_task(self)
                if not task.perform_task_physical_analysis(perform_checks):
                    return False
                self.reachable_cache = None
            return True
        # As a small performance optimization get all our reachable operations
        # now if we are going to be doing checks so we don't have to repeate BFS
        assert self.reachable_cache is None
        self.reachable_cache = set()
        self.get_physical_reachable(self.reachable_cache, False)
        # We have special cases here for copy operations and fill operations
        if self.kind == COPY_OP_KIND:
            num_reqs = len(self.reqs)
            assert num_reqs % 2 == 0
            num_copies = num_reqs / 2
            for idx in range(num_copies):
                if not self.analyze_copy_requirements(depth, idx, self.reqs[idx], 
                        idx+num_copies, self.reqs[idx+num_copies], perform_checks):
                    return False
        elif self.kind == FILL_OP_KIND:
            for index,req in self.reqs.iteritems():
                if not self.analyze_fill_requirement(depth, index, req,
                                                     perform_checks):
                    return False
        else:
            for index,req in self.reqs.iteritems():
                if not self.analyze_physical_requirement(depth, index, req, 
                                                         perform_checks):
                    return False
            # If we are a task we need to go down the task tree
            if self.kind == SINGLE_TASK_KIND:
                if self.task is not None:
                    if not self.task.perform_task_physical_analysis(perform_checks):
                        return False
        self.check_for_unanalyzed_realm_ops(perform_checks)
        # Clean up our reachable cache
        self.reachable_cache = None
        return True

    def perform_physical_close_analysis(self, depth, perform_checks):
        assert 0 in self.reqs
        req = self.reqs[0]
        if self.kind == READ_ONLY_CLOSE_OP_KIND:
            # If this is a read-only close, we just have to invalidate the tree
            for field in req.fields:
                if not req.logical_node.perform_physical_close(depth, field, self,
                                                      req, None, perform_checks):
                    return False
            return True
        elif self.kind == POST_CLOSE_OP_KIND:
            # We find our target instances from our parent task
            assert self.context
            parent_op = self.context.op
            assert self.close_idx in parent_op.mappings
            mappings = parent_op.mappings[self.close_idx]
        else:
            # This is the common path
            mappings = self.find_mapping(0, req.parent)
        for field in req.fields:
            if field.fid not in mappings:
                print "Missing mapping decision for field "+str(field)+" of "+\
                      "requirement requirement "+str(index)+" of "+str(self)
                print "This is a runtime logging bug, please report it."
            assert field.fid in mappings
            inst = mappings[field.fid]
            # If this is a virtual mapping then we can skip it if we
            # are a post close operation because this is the end of a context
            if inst.is_virtual() and self.kind == POST_CLOSE_OP_KIND:
                continue
            if not req.logical_node.perform_physical_close(depth, field, self,
                                                   req, inst, perform_checks):
                return False
        self.check_for_unanalyzed_realm_ops(perform_checks)
        return True

    def check_for_unanalyzed_realm_ops(self, perform_checks):
        if self.realm_copies:
            count = 0
            for copy in self.realm_copies:
                if not copy.analyzed:
                    count += 1
                # Clear out the reachable caches to help the garbage collector
                copy.reachable_cache = None
            if perform_checks and count > 0:
                print "WARNING: "+str(self)+" generated "+str(count)+\
                      " unnecessary Realm copies"
                for copy in self.realm_copies:
                    if not copy.analyzed:
                        print '    '+str(copy)+' was unnecessary'
        if self.realm_fills:
            count = 0
            for fill in self.realm_fills:
                if not fill.analyzed:
                    count += 1
                # Clear out the reachable caches to help the garbage collector
                fill.reachable_cache = None
            if perform_checks and count > 0:
                print "WARNING: "+str(self)+" generated "+str(count)+\
                      " unnecessary Realm fills"
                for fill in self.realm_fills:
                    if not fill.analyzed:
                        print '    '+str(fill)+' was unnecessary'

    def get_color(self):
        return {
            NO_OP_KIND : "white",
            SINGLE_TASK_KIND : "lightskyblue",
            INDEX_TASK_KIND : "mediumslateblue",
            MAP_OP_KIND : "mediumseagreen",
            INTER_CLOSE_OP_KIND : "orangered", 
            READ_ONLY_CLOSE_OP_KIND : "darkgreen",
            POST_CLOSE_OP_KIND : "darkslateblue",
            FENCE_OP_KIND : "darkorchid2",
            COPY_OP_KIND : "darkgoldenrod3",
            FILL_OP_KIND : "darkorange1",
            ACQUIRE_OP_KIND : "darkolivegreen",
            RELEASE_OP_KIND : "darksalmon",
            DELETION_OP_KIND : "dodgerblue3",
            DEP_PART_OP_KIND : "steelblue",
            PENDING_PART_OP_KIND : "honeydew",
            }[self.kind]

    def print_base_node(self, printer, dataflow):
        title = str(self)+' (UID: '+str(self.uid)+')'
        if self.task is not None and self.task.point.dim > 0:
            title += ' Point: ' + self.task.point.to_string()
        label = printer.generate_html_op_label(title, self.reqs, self.mappings,
                                       self.get_color(), self.state.detailed_graphs)
        printer.println(self.node_name+' [label=<'+label+'>,fontsize=14,'+\
                'fontcolor=black,shape=record,penwidth=0];')

    def print_dataflow_node(self, printer):
        # Print any close operations that we have, then print ourself 
        if self.inter_close_ops:
            for close in self.inter_close_ops:
                close.print_dataflow_node(printer)
        self.print_base_node(printer, True) 

    def print_incoming_dataflow_edges(self, printer, previous):
        if self.incoming is None:
            return
        if self.inter_close_ops:
            for close in self.inter_close_ops:
                close.print_incoming_dataflow_edges(printer, previous)
        for dep in self.incoming:
            dep.print_dataflow_edge(printer, previous)

    def print_event_node(self, printer):
        self.print_base_node(printer, False)

    def print_event_graph(self, printer, elevate, all_nodes, top):
        # Do any of our close operations too
        if self.inter_close_ops:
            for close in self.inter_close_ops:
                close.print_event_graph(printer, elevate, all_nodes, False)
        # Handle index space operations specially, everything
        # else is the same
        if self.kind is INDEX_TASK_KIND:
            assert self.points is not None
            for point in self.points.itervalues():
                point.op.print_event_graph(printer, elevate, all_nodes, False)
            return
        # If this is a single task, recurse and generate our subgraph first
        if self.kind is SINGLE_TASK_KIND:
            # Get our corresponding task
            task = self.state.get_task(self.uid)   
            task.print_event_graph_context(printer, elevate, all_nodes, top)
        # Look through all our generated realm operations and emit them
        if self.realm_copies:
            for copy in self.realm_copies:
                if copy not in elevate:
                    #elevate[copy] = copy.get_event_context()
                    elevate[copy] = copy.get_context()
        if self.realm_fills:
            for fill in self.realm_fills:
                if fill not in elevate:
                    #elevate[fill] = fill.get_event_context()
                    elevate[copy] = copy.get_context()
        if self.is_physical_operation():
            # Finally put ourselves in the set if we are a physical operation
            assert self.context is not None
            elevate[self] = self.context

    def is_realm_operation(self):
        return False

    def is_physical_operation(self):
        if self.is_close():
            return False
        if self.kind is COPY_OP_KIND:
            return False
        if self.kind is FILL_OP_KIND:
            return False
        if self.kind is FENCE_OP_KIND:
            return False
        if self.kind is DELETION_OP_KIND:
            return False
        return True

    def print_incoming_event_edges(self, printer):
        if self.cluster_name is not None:
            for src in self.physical_incoming:
                if src.cluster_name is not None:
                    printer.println(src.node_name+' -> '+self.node_name+
                            ' [ltail='+src.cluster_name+',lhead='+
                            self.cluster_name+',style=solid,color=black,'+
                            'penwidth=2];')
                else:
                    printer.println(src.node_name+' -> '+dst.node_name+
                            ' [lhead='+self.cluster_name+',style=solid,'+
                            'color=black,penwidth=2];')
        else:
            for src in self.physical_incoming:
                if src.cluster_name is not None:
                    printer.println(src.node_name+' -> '+self.node_name+
                            ' [ltail='+src.cluster_name+',style=solid,'+
                            'color=black,penwidth=2];')
                else:
                    printer.println(src.node_name+' -> '+self.node_name+
                            ' [style=solid,color=black,penwidth=2];')

class Task(object):
    __slots__ = ['state', 'op', 'point', 'operations', 'depth', 
                 'current_fence', 'used_instances', 'virtual_indexes']
    def __init__(self, state, op):
        self.state = state
        self.op = op
        self.op.task = self
        self.point = Point(0) 
        self.operations = list()
        self.depth = None
        self.current_fence = None
        self.used_instances = None
        self.virtual_indexes = None

    def __str__(self):
        return str(self.op)

    __repr__ = __str__

    def add_operation(self, operation):
        self.operations.append(operation)

    def set_point(self, point):
        self.point = point

    def get_parent_context(self):
        assert self.op.context is not None
        return self.op.context

    def get_depth(self):
        if self.depth is None:
            self.depth = self.op.get_depth()
        return self.depth

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

    def perform_logical_dependence_analysis(self, perform_checks):
        # If we don't have any operations we are done
        if not self.operations:
            return True
        # If this is the top-level task's context, we can skip it
        # since we know there is only one task in it
        if self.depth == 0:
            assert len(self.operations) == 1
            return True
        print 'Performing logical dependence analysis for %s...' % str(self)
        if self.op.state.verbose:
            print '  Analyzing %d operations...' % len(self.operations)
        # Iterate over all the operations in order and
        # have them perform their analysis
        success = True
        for op in self.operations:
            if not op.perform_logical_analysis(perform_checks):
                success = False
                break
        # Reset the logical state when we are done
        self.op.state.reset_logical_state()
        print "Pass" if success else "FAIL"
        return success
    
    def perform_logical_sanity_analysis(self):
        # Run the old version of the checks that
        # is more of a sanity check on our algorithm that
        # doesn't depend on our implementation but doesn't
        # really tell us what it means if something goes wrong
        if not self.operations or len(self.operations) < 2:
            return True
        print 'Performing logical sanity analysis for %s...' % str(self)
        # Iterate over all operations from 1 to N and check all their
        # dependences against all the previous operations in the context
        for idx in range(1, len(self.operations)):
            # Find all the backwards reachable operations
            current_op = self.operations[idx]
            # No need to do anything if there are no region requirements
            if not current_op.reqs and current_op.kind <> FENCE_OP_KIND:
                continue
            reachable = set()
            current_op.get_logical_reachable(reachable, False) 
            # Do something special for fence operations
            if current_op.kind == FENCE_OP_KIND: # special path for fences
                for prev in range(idx):
                    if not prev in reachable:
                        print "ERROR: Failed logical sanity check. No mapping "+\
                              "dependence between previous "+str(prev)+" and "+\
                              "later "+str(current_op)
                        if self.op.state.assert_on_fail:
                            assert False
                        return False
            else: # The normal path
                for prev in range(idx):
                    if not current_op.analyze_logical_interference(
                                  self.operations[prev], reachable):
                        print "FAIL"
                        return False
        print "Pass"
        return True

    def record_instance_use(self, inst, fid, node):
        depth = self.get_depth()
        # If this is the top-level context there is nothing to do
        if depth == 0:
            return
        assert self.used_instances is not None
        # If we've already handled it then we are done
        if (inst,fid) in self.used_instances:
            return
        # If we have virtual mappings we have check to see if this is
        # a new field in which case we have to copy in the users from
        # the next outer depth, but first we have to tell our context
        # that we are doing this too to handle nested virtual mappings
        # correctly. :)
        if self.virtual_indexes:
            field = node.field_space.get_field(fid)
            for idx in self.virtual_indexes:
                assert idx in self.op.reqs
                req = self.op.reqs[idx]
                if req.logical_node is not node:
                    continue
                if field not in req.fields:
                    continue
                # We found a virtual instance for a virtual mapped requirement
                assert self.op.context is not None
                # First tell our parent that we are using the instance
                self.op.context.record_instance_use(inst, fid, req.parent)
                # Clone the user list 
                parent_depth = depth - 1
                inst.clone_users(field, parent_depth, depth) 
                # Then add our own user at our parent's depth
                inst.add_user(parent_depth, field, self.op, req)
                # We are done
                break
        # Record that we handled this field for this instance
        self.used_instances.add((inst,fid))

    def perform_task_physical_analysis(self, perform_checks):
        if not self.operations:
            return True
        depth = self.get_depth()
        assert self.used_instances is None
        self.used_instances = set()
        # Initialize our regions at our depth
        assert self.virtual_indexes is None
        if self.op.reqs: 
            for idx,req in self.op.reqs.iteritems():
                assert idx in self.op.mappings
                mappings = self.op.mappings[idx]
                for field in req.fields:
                    assert field.fid in mappings
                    inst = mappings[field.fid]
                    if inst.is_virtual():
                        # Only need to do this if we are not the root
                        if depth > 0:
                            # If you ever hit this assertion it is indicative of a
                            # runtime bug because the runtime should never allow 
                            # a virtual instance to be made for a region requirement
                            # that only has reduction privileges
                            assert not req.is_reduce()
                            # Make a virtual instance and then copy it over
                            req.logical_node.perform_physical_close(depth-1, field,
                                                          self.op, req, inst, False)
                            # We only need to save read-write privileges to be copied
                            # back out, read-only privileges means that the copies
                            # only flow in and not back out
                            if req.is_write():
                                if not self.virtual_indexes:
                                    self.virtual_indexes = set()
                                self.virtual_indexes.add(idx)
                                clear_state = True
                            else:
                                clear_state = False
                            req.logical_node.copy_physical_state(depth-1, depth, 
                                                                 field, clear_state)
                    else:
                        req.logical_node.initialize_physical_state(depth, field, inst)
        success = True
        for op in self.operations:
            if not op.perform_op_physical_analysis(depth, perform_checks): 
                success = False
                break
        if success and self.virtual_indexes:
            for idx in self.virtual_indexes:
                assert idx in self.op.reqs
                req = self.op.reqs[idx]
                for field in req.fields:
                    req.logical_node.perform_physical_close(depth, field, self.op,
                                             req, Instance(self.op.state, 0), False)
                    req.logical_node.copy_physical_state(depth, depth-1, field, True)
        self.virtual_indexes = None
        # Reset any physical user lists at our depth for instances we used
        for inst,fid in self.used_instances:
            inst.reset_physical_users(depth);
        # Reset any physical states that we 
        # Always need to clear out the physical state on the way out
        self.op.state.reset_physical_state(depth)
        return success

    def print_dataflow_graph(self, path, simplify_graphs):
        if len(self.operations) < 2:
            return 0
        name = str(self)
        filename = 'dataflow_'+name.replace(' ', '_')+'_'+str(self.op.uid)
        printer = GraphPrinter(path,filename)
        # First emit the nodes
        for op in self.operations:
            op.print_dataflow_node(printer)
        # Simplify our graph if necessary
        if simplify_graphs:
            print "Simplifying dataflow graph for "+str(self)+"..."
            all_ops = list()
            for op in self.operations:
                # Add any close operations first
                if op.inter_close_ops:
                    for close in op.inter_close_ops:
                        all_ops.append(close)
                # Then add the operation itself
                all_ops.append(op)
            # Now traverse the list in reverse order
            while all_ops:
                src = all_ops.pop()
                if src.logical_outgoing is None:
                    continue
                actual_out = src.logical_outgoing.copy()
                diff = False
                for next_vert in src.logical_outgoing:
                    if not next_vert in actual_out:
                        continue
                    reachable = set()
                    next_vert.get_logical_reachable(reachable, True)
                    # See which edges we can remove
                    to_remove = list()
                    for other in actual_out:
                        if other == next_vert:
                            continue
                        if other in reachable:
                            to_remove.append(other)
                    del reachable
                    if len(to_remove) > 0:
                        diff = True
                        for rem in to_remove:
                            actual_out.remove(rem)
                            rem.logical_incoming.remove(src)
                    del to_remove
                if diff:
                    src.logical_outgoing = actual_out
                for dst in actual_out:
                    printer.println(src.node_name+' -> '+dst.node_name+
                                    ' [style=solid,color=black,penwidth=2];')
            print "Done"
        else:
            previous_pairs = set()
            for op in self.operations:
                op.print_incoming_dataflow_edges(printer, previous_pairs)
        printer.print_pdf_after_close(False)
        # We printed our dataflow graph
        return 1

    def print_event_graph_context(self, printer, elevate, all_nodes, top):
        if not self.operations:
            return 
        if not top:
            # Start the cluster 
            title = str(self)+' (UID: '+str(self.op.uid)+')'
            if self.point.dim > 0:
                title += ' Point: ' + self.point.to_string()
            label = printer.generate_html_op_label(title, self.op.reqs,
                                                   self.op.mappings,
                                                   self.op.get_color(), 
                                                   self.op.state.detailed_graphs)
            self.op.cluster_name = printer.start_new_cluster(label)
            # Make an invisible node for this cluster
            printer.println(self.op.node_name + ' [shape=point,style=invis];')
        # Generate the sub-graph
        for op in self.operations:
            op.print_event_graph(printer, elevate, all_nodes, False)
        # Find our local nodes
        local_nodes = list()
        for node,context in elevate.iteritems():
            if context is self:
                local_nodes.append(node)
                node.print_event_node(printer)
                all_nodes.add(node)
        # Hold off printing the edges until the very end
        # Remove our nodes from elevate
        for node in local_nodes:
            del elevate[node] 
        if not top:
            # End the cluster
            printer.end_this_cluster()

class InstanceUser(object):
    __slots__ = ['op', 'region', 'priv', 'coher', 'redop', 'shape']
    def __init__(self, op, region, priv, coher, redop, shape=None):
        assert isinstance(region, LogicalRegion)
        self.op = op
        self.region = region
        self.priv = priv
        self.coher = coher
        self.redop = redop
        self.shape = shape

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
          
class Instance(object):
    __slots__ = ['state', 'handle', 'memory', 'region', 'fields', 'redop', 'depth_users']
    def __init__(self, state, handle):
        self.state = state
        self.handle = handle
        self.memory = None
        self.region = None
        self.fields = None
        self.redop = 0
        self.depth_users = dict() # map to dict for each depth

    def __str__(self):
        #return "Instance %s in %s" % (hex(self.handle), str(self.memory))
        if self.is_virtual():
            return "Virtual Instance"
        else:
            return "Instance "+hex(self.handle)

    __repr__ = __str__

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

    def is_composite(self):
        return False

    def get_users(self, depth):
        if depth not in self.depth_users:
            self.depth_users[depth] = dict()
        return self.depth_users[depth]

    def clone_users(self, field, src_depth, dst_depth):
        src_users = self.get_users(src_depth)
        if field in src_users:
            dst_users = self.get_users(dst_depth)
            # Make this a copy, we don't want the lists aliased
            dst_users[field] = copy.copy(src_users[field])

    def reset_physical_users(self, depth):
        if depth in self.depth_users:
            self.depth_users[depth] = dict()

    def find_use_dependences(self, depth, field, req, precise):
        assert not self.is_virtual()
        users = self.get_users(depth)
        result = set()
        if field not in users:
            return result
        if precise:
            points = req.logical_node.get_shape().copy()
        for user in reversed(users[field]):
            if user.region.intersects(req.logical_node):
                if precise:
                    overlap = user.shape & points
                    if overlap.empty():
                        continue
                dep = compute_dependence_type(user, req)
                if dep == TRUE_DEPENDENCE or dep == ANTI_DEPENDENCE:
                    result.add(user.op)
                    # We only dominate and can remove points if the 
                    # the previous was an exclusive writer
                    if precise and user.is_write() and user.is_exclusive():
                        points -= user.shape
                        if points.empty():
                            break
        return result

    def find_copy_dependences(self, depth, field, region, reading, 
                              redop, precise, intersect=None):
        assert not self.is_virtual()
        users = self.get_users(depth)
        result = set()
        if field not in users:
            return result
        if reading:
            assert redop == 0
            inst = InstanceUser(None, region, READ_ONLY, EXCLUSIVE, 0)
        elif redop <> 0:
            inst = InstanceUser(None, region, REDUCE, EXCLUSIVE, redop)
        else:
            inst = InstanceUser(None, region, READ_WRITE, EXCLUSIVE, 0)
        if precise:
            if intersect is None:
                points = region.get_shape().copy()
            else:
                points = region.get_shape() & intersect.get_shape()
        for user in reversed(users[field]):
            if user.region.intersects(region):
                if precise:
                    overlap = user.shape & points
                    if overlap.empty():
                        continue
                dep = compute_dependence_type(user, inst)
                if dep == TRUE_DEPENDENCE or dep == ANTI_DEPENDENCE:
                    result.add(user.op)
                    # We only dominate and can remove points if the 
                    # previous was an exclusive writer
                    if precise and user.is_write() and user.is_exclusive():
                        points -= user.shape
                        if points.empty():
                            break
        return result

    def add_user(self, depth, field, op, req):
        assert not self.is_virtual()
        users = self.get_users(depth)
        if field not in users:
            users[field] = list()
        users[field].append(InstanceUser(op, req.logical_node, 
                req.priv, req.coher, req.redop, req.logical_node.get_shape()))

    def add_copy_user(self, depth, field, region, op, reading, redop, intersect=None):
        assert not self.is_virtual()
        users = self.get_users(depth)
        if field not in users:
            users[field] = list()
        if intersect is None:
            shape = region.get_shape()
        else:
            shape = region.get_shape() & intersect.get_shape()
        if reading:
            assert redop == 0
            users[field].append(InstanceUser(op, region,
                                             READ_ONLY, EXCLUSIVE, 0, shape))
        elif redop <> 0:
            users[field].append(InstanceUser(op, region,
                                             REDUCE, EXCLUSIVE, redop, shape))
        else:
            users[field].append(InstanceUser(op, region,
                                             READ_WRITE, EXCLUSIVE, 0, shape))

class FillInstance(object):
    __slots__ = ['state', 'region', 'depth', 'field']
    def __init__(self, state, region, depth, field):
        self.state = state
        self.region = region 
        self.depth = depth
        self.field = field

    def __str__(self):
        print "Fill Instance of "+str(self.field)+" at "+str(self.region)

    __repr__ = __str__
      
    def find_use_dependences(self, depth, field, req, precise):
        assert False

    def find_copy_dependences(self, depth, field, region, reading, redop, precise):
        assert False

    def add_user(self, depth, field, op, req):
        assert False

    def add_copy_user(self, depth, field, op, reading, redop):
        assert False

    def is_virtual(self):
        return True

    def is_composite(self):
        return False

    def issue_update_copies(self, dst, dst_depth, region, op, perform_checks, error_str):
        # This is just a special case of issue copies across 
        return self.issue_copies_across(dst, dst_depth, self.field, region, op, 
                                        perform_checks, error_str, False)

    def issue_copies_across(self, dst, dst_depth, dst_field, region, op, 
                            perform_checks, error_str, actually_across=True):
        # Find the destination preconditions
        if self.region is not region:
            preconditions = dst.find_copy_dependences(dst_depth, dst_field, region, 
                                              False, 0, perform_checks, self.region)
        else:
            preconditions = dst.find_copy_dependences(dst_depth, dst_field, region, 
                                              False, 0, perform_checks)
        if perform_checks:
            if self.region is not region:
                fill = op.find_generated_fill(dst_field, region, dst, self.region)
            else:
                fill = op.find_generated_fill(dst_field, region, dst)
            if fill is None:
                if actually_across:
                    print "ERROR: Unable to find fill across operation generated to "+\
                          str(dst)+" for field "+str(self.field)+" by "+error_str
                    if self.state.assert_on_fail:
                        assert False
                else:
                    print "ERROR: Unable to find fill operation generated to "+str(dst)+\
                          " for field "+str(self.field)+" by "+error_str
                    if self.state.assert_on_fail:
                        assert False
                return False
            if fill.reachable_cache is None:
                fill.reachable_cache = set()
                fill.get_physical_reachable(fill.reachable_cache, False)
            bad = check_preconditions(preconditions, fill)
            if bad is not None:
                if actually_across:
                    print "ERROR: Missing fill across precondition to "+str(dst)+\
                          " for field "+str(self.field)+" issued by "+error_str+\
                          " on "+str(bad)
                    if self.state.assert_on_fail:
                        assert False
                else:
                    print "ERROR: Missing fill precondition to "+str(dst)+" for fill "+\
                          "of field "+str(self.field)+" issued by "+error_str+\
                          " on "+str(bad)
                    if self.state.assert_on_fail:
                        assert False
                return False
        else:
            fill = self.state.create_fill(op)
            fill.add_field(dst_field.fid, dst)    
            for pre in preconditions:
                pre.physical_outgoing.add(fill)
                fill.physical_incoming.add(pre)
        dst.add_copy_user(dst_depth, dst_field, False, 0)
        return True

class CompositeNode(object):
    __slots__ = ['owner', 'node', 'parent', 'valid_instances', 'dirty', 'children']
    def __init__(self, owner, node, parent):
        self.owner = owner # composite instance
        self.node = node # logical node
        self.parent = parent # parent composite node
        self.children = set() # child composite nodes
        self.dirty = False
        self.valid_instances = None
        # Add ourselves to our parent's set of children
        if parent is not None:
            self.parent.children.add(self)

    def is_complete(self):
        if self.node.is_complete() and len(self.children) == self.node.get_num_children():
            return True
        return False

    def capture(self, state, is_root, already_captured, skip_instances):
        inst_capture = False
        if not skip_instances:
            # Capture any instances
            # If we are the root, get all the valid instances,
            # otherwise the local ones will suffice
            self.valid_instances = set()
            self.dirty = state.dirty
            if is_root:
                valid = state.find_valid_instances()
            else:
                valid = state.valid_instances
            for inst in valid:
                inst_capture = True
                if inst.is_composite():
                    # For composite instances, we have make an inline composite instance
                    inline_composite = inst.create_inline_composite(already_captured)
                    self.valid_instances.add(inline_composite)
                else:
                    self.valid_instances.add(inst)
        # Always add any reductions to our parent
        for reduction in state.reduction_instances:
            self.owner.reductions.add(reduction)
        # Mark that we've been captured
        already_captured.add(self.node)
        return inst_capture

    def capture_inline(self, target, already_captured):
        changed = False
        # Capture down the tree first, then do ourselves
        for child in self.children:
            if child.capture_inline(target, already_captured):
                changed = True
        # Now see if we need to capture ourself
        if self.node not in already_captured:
            copy = target.get_node(self.node)
            copy.dirty = self.dirty
            for inst in self.valid_instances:
                if copy.valid_instances is None:
                    copy.valid_instances = set()
                if inst.is_composite():
                    inline_composite = inst.create_inline_composite(already_captured) 
                    copy.valid_instances.add(inline_composite)
                    if inline_composite is not inst:
                        changed = True
                else:
                    copy.valid_instances.add(inst)
            # Add ourselves to the already captured
            already_captured.add(self.node)
        else:
            changed = True
        return changed

    def find_valid_instances(self, valid):
        # Go up the tree if necessary
        if not self.dirty and self.parent is not None:
            self.parent.find_valid_instances(valid)
        if self.valid_instances:
            for inst in self.valid_instances:
                valid.add(inst)

    def issue_update_copies(self, dst, dst_depth, dst_field, region, op, perform_checks, 
                            error_str, actually_across, need_check = True):
        if need_check:
            # Figure out how many children dominate the target region
            # Keep going down if we there is exaclty one
            dominating_children = list()
            for child in self.children:
                if child.node.dominates(region):
                    dominating_children.append(child)
            if len(dominating_children) == 1:
                return dominating_children[0].issue_update_copies(dst, dst_depth, 
                    dst_field, region, op, perform_checks, error_str, actually_across)
        # If we need check (e.g. haven't initialized data) or we are dirty
        # then we have to issue copies from this level
        if need_check or self.dirty:
            if need_check:
                local_valid = set()
                self.find_valid_instances(local_valid)
            else:
                local_valid = self.valid_instances 
            # Issue update copies from our valid instances if there are
            # valid instances and the destination is not already one of them
            if local_valid and dst not in local_valid:
                # Handle the virtual case
                if len(local_valid) == 1 and \
                    next(iter(local_valid)).is_virtual():
                    virtual_inst = next(iter(local_valid))
                    if not virtual_inst.issue_copies_across(dst, dst_depth, dst_field, 
                              region, op, perform_checks, error_str, actually_across):
                        return False
                else:
                    if self.node is not region:
                        dst_preconditions = dst.find_copy_dependences(dst_depth, 
                            dst_field, region, False, 0, perform_checks, self.node)
                    else:
                        dst_preconditions = dst.find_copy_dependences(dst_depth,
                            dst_field, region, False, 0, perform_checks)
                    if perform_checks:
                        # Find the copy
                        if self.node is not region:
                            copy = op.find_generated_copy(self.owner.field, 
                                                          region, dst, 0, self.node)
                        else:
                            copy = op.find_generated_copy(self.owner.field,
                                                          region, dst)
                        if copy is None:
                            if actually_across:
                                print "ERROR: Missing intersection copy across from "+\
                                      "composite instance to update "+str(dst)+\
                                      " for field "+str(dst_field)+" by "+error_str
                                if self.owner.state.assert_on_fail:
                                    assert False
                            else:
                                print "ERROR: Missing intersection copy from "+\
                                      "composite instance to update "+str(dst)+\
                                      " for field "+str(dst_field)+" by "+error_str
                                if self.owner.state.assert_on_fail:
                                    assert False
                            return False
                        src = copy.find_src_inst(self.owner.field)
                        if self.node is not region:
                            src_preconditions = src.find_copy_dependences(self.owner.depth,
                                self.owner.field, region, True, 0, perform_checks, self.node)
                        else:
                            src_preconditions = src.find_copy_dependences(self.owner.depth,
                                self.owner.field, region, True, 0, perform_checks)
                        if copy.reachable_cache is None:
                            copy.reachable_cache = set()
                            copy.get_physical_reachable(copy.reachable_cache, False)
                        bad = check_preconditions(src_preconditions, copy)
                        if bad is not None:
                            if actually_across:
                                print "ERROR: Missing source precondition on "+\
                                      str(bad)+" for "+str(copy)+" across from "+\
                                      "composite instance to update "+str(dst)+\
                                      " for field "+str(dst_field)+" by "+error_str
                                if self.owner.state.assert_on_fail:
                                    assert False
                            else:
                                print "ERROR: Missing source precondition on "+\
                                      str(bad)+" for "+str(copy)+" from composite "+\
                                      "instance to update "+str(dst)+" for field "+\
                                      str(dst_field)+" by "+error_str
                                if self.owner.state.assert_on_fail:
                                    assert False
                            return False
                        bad = check_preconditions(dst_preconditions, copy)
                        if bad is not None:
                            if actually_across:
                                print "ERROR: Missing destination precondition on "+\
                                      str(bad)+" for "+str(copy)+"  across from "+\
                                      "composite instance to update "+str(dst)+\
                                      " for field "+str(dst_field)+" by "+error_str
                                if self.owner.state.assert_on_fail:
                                    assert False
                            else:
                                print "ERROR: Missing destination precondition on "+\
                                      str(bad)+" for "+str(copy)+" from composite "+\
                                      "instance to update "+str(dst)+" for field "+\
                                      str(dst_field)+" by "+error_str
                                if self.owner.state.assert_on_fail:
                                    assert False
                            return False
                    else:
                        # Figure out which instance to copy from
                        if len(valid) > 1:
                            print "INFO: Multiple valid instances to choose from in "+\
                                  "composite instance... picking one"
                        src = next(iter(valid))
                        if self.node is not region:
                            src_preconditions = src.find_copy_dependences(self.owner.depth,
                                self.owner.field, region, True, 0, perform_checks, self.node)
                        else:
                            src_preconditions = src.find_copy_dependences(self.owner.depth,
                                self.owner.field, region, True, 0, perform_checks)
                        # Make a realm copy from the source to the destination
                        copy = self.owner.state.create_copy(op)
                        copy.set_region(region)
                        copy.set_intersect(self.node)
                        copy.add_field(self.owner.field.fid, src, 
                                       self.owner.field.fid, dst, 0)
                        for src_op in src_preconditions:
                            src_op.physical_outgoing.add(copy)
                            copy.physical_incoming.add(src_op)
                        for dst_op in dst_preconditions:
                            dst_op.physical_outgoing.add(copy)
                            copy.physical_incoming.add(dst_op)
                    # Record the copy user
                    if self.node is not region:
                        src.add_copy_user(self.owner.depth, self.owner.field, region, 
                                          copy, True, 0, self.node)
                        dst.add_copy_user(dst_depth, dst_field, region, 
                                          copy, False, 0, self.node)
                    else:
                        src.add_copy_user(self.owner.depth, self.owner.field, region, 
                                          copy, True, 0)
                        dst.add_copy_user(dst_depth, dst_field, region, copy, False, 0)
        # Now we can recurse down the tree from this point, no need for the check
        for child in self.children:
            # Check for intersection before going down
            if not region.intersects(child.node):
                continue
            if not child.issue_update_copies(dst, dst_depth, dst_field, region, op, 
                                perform_checks, error_str, actually_across, False):
                return False
        return True

class CompositeInstance(object):
    __slots__ = ['state', 'root', 'depth', 'field', 'nodes', 
                 'reductions', 'captured', 'complete']
    def __init__(self, state, root, depth, field):
        self.state = state
        self.root = root
        self.depth = depth
        self.field = field
        self.nodes = dict()
        self.reductions = set()
        self.complete = set() # complete nodes
        self.captured = set()

    def __str__(self):
        print "Composite Instance of "+str(self.field)+" at "+str(self.root)

    __repr__ = __str__

    def get_node(self, node):
        if node in self.nodes:
            return self.nodes[node]
        if node is self.root:
            result = CompositeNode(self, node, None)
        else:
            parent_node = self.get_node(node.parent)
            result = CompositeNode(self, node, parent_node)
        self.nodes[node] = result
        return result

    def capture(self, state, already_captured):
        # Do the capture if we are dirty or have reductions or we are the root
        if state.dirty or state.redop <> 0 or state.node is self.root:
            new_state = self.get_node(state.node)
            # See if we can avoid capturing the instances
            already_complete = False 
            # If we are complete and we captured all our children
            # then we can skip capturing instances here
            if state.node.is_complete() and state.node.has_all_children():
                already_complete = True
                for child in state.node.children.itervalues():
                    if not child in self.captured:
                        already_complete = False
                        break
            # We can also skip if one of our children is complete
            if not already_complete: 
                for child in state.node.children.itervalues():
                    if child in self.complete:
                        already_complete = True
                        break
            inst_capture = new_state.capture(state, state.node is self.root, 
                                             already_captured, already_complete)
            if inst_capture or already_complete:
                self.captured.add(state.node) 
                if state.node.is_complete():
                    self.complete.add(state.node)

    def create_inline_composite(self, already_captured):
        result = CompositeInstance(self.state, self.root, self.depth, self.field) 
        # Keep track of whether anything changed when doing the
        # transformation, if not, we can just return this instance
        if self.nodes[self.root].capture_inline(result, already_captured):
            # Did change so use the result
            result.reductions = self.reductions # No need to copy since it is immutable
            return result
        # Otherwise nothing changed so we can return ourself
        return self
            
    def find_use_dependences(self, depth, field, req, precise):
        assert False

    def find_copy_dependences(self, depth, field, region, reading, redop, precise):
        assert False

    def add_user(self, depth, field, op, req):
        assert False

    def add_copy_user(self, depth, field, op, reading, redop):
        assert False

    def is_virtual(self):
        return True

    def is_composite(self):
        return True

    def issue_update_copies(self, dst, dst_depth, region, op, perform_checks, error_str):
        # This is actually just a special case of issuing copies across 
        return self.issue_copies_across(dst, dst_depth, self.field, region, op, 
                                        perform_checks, error_str, False)

    def issue_copies_across(self, dst, dst_depth, dst_field, region, op, 
                            perform_checks, error_str, actually_across=True):
        # First issue copies from anything in our tree
        assert self.root in self.nodes
        if not self.nodes[self.root].issue_update_copies(dst, dst_depth, dst_field, 
                            region, op, perform_checks, error_str, actually_across):
            return False
        # If we have any reductions issue those now
        if self.reductions:
            if perform_checks:
                for reduction_inst in self.reductions:
                    assert reduction_inst.redop <> 0
                    # Check to see if it intersects with the region 
                    if region.intersects(reduction_inst.region):
                        if reduction_inst.region is not region:
                            reduction = op.find_generated_copy_across(self.field, 
                                          dst_field, region, reduction_inst, dst,
                                          reduction_inst.redop, reduction_inst.region)
                        else:
                            reduction = op.find_generated_copy_across(self.field,
                                          dst_field, region, reduction_inst, dst,
                                          reduction_inst.redop)
                        if reduction is None:
                            if actually_across:
                                print "ERROR: Missing reduction across copy operation "+\
                                      "from composite instance to update "+str(dst)+\
                                      " for field "+str(self.field)+" by "+error_str
                                if self.state.assert_on_fail:
                                    assert False
                            else:
                                print "ERROR: Missing reduction copy operation from "+\
                                      "composite instance to update "+str(dst)+" for "+\
                                      "field "+str(self.field)+" by "+error_str
                                if self.state.assert_on_fail:
                                    assert False
                            return False
                        if reduction_inst.region is not region:
                            src_preconditions = reduction_inst.find_copy_dependences(
                                self.depth, self.field, region, True, 0, 
                                perform_checks, reduction_inst.region)
                        else:
                            src_preconditions = reduction_inst.find_copy_dependences(
                                self.depth, self.field, region, True, 0, perform_checks)
                        if reduction.reachable_cache is None:
                            reduction.reachable_cache = set()
                            reduction.get_physical_reachable(
                                            reduction.reachable_cache, False)
                        bad = check_preconditions(src_preconditions, reduction)
                        if bad is not None:
                            if actually_across:
                                print "ERROR: Missing source reduction precondition for "+\
                                      "reduction across from composite instance to "+\
                                      str(dst)+" for field "+str(self.field)+" by "+error_str
                                if self.state.assert_on_fail:
                                    assert False
                            else:
                                print "ERROR: Missing source reduction precondition for "+\
                                      "reduction from composite instance to "+str(dst)+\
                                      "for field "+str(self.field)+" by "+error_str
                                if self.state.assert_on_fail:
                                    assert False
                            return False
                        if reduction_inst.region is not region:
                            dst_preconditions = dst.find_copy_dependences(dst_depth, 
                                dst_field, region, False, reduction_inst.redop,
                                perform_checks, reduction_inst.region)
                        else:
                            dst_preconditions = dst.find_copy_dependences(dst_depth,
                                dst_field, region, False, reduction_inst.redop, 
                                perform_checks)
                        bad = check_preconditions(dst_preconditions, reduction)
                        if bad is not None:
                            if actually_across:
                                print "ERROR: Missing destination reduction precondition "+\
                                      "for reduction across from composite instance to "+\
                                      str(dst)+" for field "+str(dst_field)+" by "+error_str
                                if self.state.assert_on_fail:
                                    assert False
                            else:
                                print "ERROR: Missing destination reduction precondition "+\
                                      "for reduction from composite instance to "+str(dst)+\
                                      "for field "+str(dst_field)+" by "+error_str
                                if self.state.assert_on_fail:
                                    assert False
                            return False
                        if reduction_inst.region is not region:
                            reduction_inst.add_copy_user(self.depth, self.field, region, 
                                reduction, True, 0, reduction_inst.region)
                            dst.add_copy_user(dst_depth, dst_field, region, reduction, False, 
                                            reduction_inst.redop, reduction_inst.region)
                        else:
                            reduction_isnt.add_copy_user(self.depth, self.field, region, 
                                reduction, True, 0)
                            dst.add_copy_user(dst_depth, dst_field, region, reduction, False,
                                              reduction_inst.redop)
            else:
                for reduction_inst in self.reductions:
                    assert reduction_inst.redop <> 0
                    if region.intersects(reduction_inst.region):
                        # Make a reduction copy 
                        reduction = self.state.create_copy(op)
                        reduction.add_field(self.field.fid, reduction_inst, 
                                            dst_field.fid, dst, reduction_inst.redop)
                        reduction.set_region(region)
                        if reduction_inst.region is not region:
                            reduction.set_intersection(reduction_inst.region)
                            src_preconditions = reduction_inst.find_copy_dependences(
                                self.depth, self.field, region, True, 0,
                                perform_checks, reduction_inst.region)
                            dst_preconditions = dst.find_copy_dependences(dst_depth,
                                dst_field, region, False, reduction_inst.redop,
                                perform_checks, reduction_inst.region)
                        else:
                            src_preconditions = reduction_inst.find_copy_dependences(
                                self.depth, self.field, region, True, 0, perform_checks)
                            dst_preconditions = dst.find_copy_dependences(dst_depth,
                                dst_field, region, False, reduction_inst.redop, 
                                perform_checks)
                        for src_op in src_preconditions:
                            src_op.physical_outgoing.add(reduction)
                            reduction.physical_incoming.add(src_op)
                        for dst_op in dst_preconditions:
                            dst_op.physical_outgoing.add(reduction)
                            reduction.physical_incoming.add(dst_op)
                        if reduction_inst.region is not region:
                            reduction_inst.add_copy_user(self.depth, self.field, 
                                region, reduction, True, 0, reduction_inst.region)
                            dst.add_copy_user(dst_depth, dst_field, region, reduction, 
                                False, reduction_inst.redop, reduction_inst.region)
                        else:
                            reduction_inst.add_copy_user(self.depth, self.field, 
                                region, reduction, True, 0)
                            dst.add_copy_user(dst_depth, dst_field, region, 
                                reduction, False, reduction_inst.redop)
        return True

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
    __slots__ = ['state', 'handle', 'phase_barrier', 'incoming', 'outgoing',
                 'incoming_ops', 'outgoing_ops', 'incoming_fills', 'outgoing_fills',
                 'incoming_copies', 'outgoing_copies', 'generation']
    def __init__(self, state, handle):
        self.state = state
        self.handle = handle
        self.phase_barrier = False
        self.incoming = None
        self.outgoing = None
        self.incoming_ops = None
        self.outgoing_ops = None
        self.incoming_fills = None
        self.outgoing_fills = None
        self.incoming_copies = None
        self.outgoing_copies = None
        # For traversals
        self.generation = 0

    def exists(self):
        return self.handle.uid > 0

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

    def add_incoming_op(self, op):
        if self.incoming_ops is None:
            self.incoming_ops = set()
        self.incoming_ops.add(op)

    def add_outgoing_op(self, op):
        if self.outgoing_ops is None:
            self.outgoing_ops = set()
        self.outgoing_ops.add(op)

    def add_incoming_fill(self, fill):
        if self.incoming_fills is None:
            self.incoming_fills = set()
        self.incoming_fills.add(fill)

    def add_outgoing_fill(self, fill):
        if self.outgoing_fills is None:
            self.outgoing_fills = set()
        self.outgoing_fills.add(fill)

    def add_incoming_copy(self, copy):
        if self.incoming_copies is None:
            self.incoming_copies = set()
        self.incoming_copies.add(copy)

    def add_outgoing_copy(self, copy):
        if self.outgoing_copies is None:
            self.outgoing_copies = set()
        self.outgoing_copies.add(copy)

class RealmBase(object):
    __slots__ = ['state', 'realm_num', 'creator', 'region', 'intersect', 'start_event', 
                 'finish_event', 'physical_incoming', 'physical_outgoing', 'generation', 
                 'event_context', 'analyzed', 'cluster_name', 'reachable_cache']
    def __init__(self, state, realm_num):
        self.state = state
        self.realm_num = realm_num
        self.creator = None
        self.region = None
        self.intersect = None
        self.physical_incoming = set()
        self.physical_outgoing = set()
        self.start_event = state.get_no_event()
        self.finish_event = state.get_no_event()
        self.generation = 0
        self.event_context = None
        self.analyzed = False
        self.cluster_name = None # always none
        self.reachable_cache = None

    def set_region(self, region):
        self.region = region

    def set_intersect(self, intersect):
        assert self.region is not None
        if intersect is not self.region:
            self.intersect = intersect

    def is_realm_operation(self):
        return True

    def is_physical_operation(self):
        return True

    def get_context(self):
        assert self.creator is not None
        return self.creator.get_context()

    def compute_physical_reachable(self):
        # Once we reach something that is not an event
        # then we record it and return
        def traverse_node(node, traverser):
            if not node.is_physical_operation():
                return True
            traverser.reachable.add(node)
            return False
        if self.start_event.exists():
            traverser = EventGraphTraverser(False, True,
                self.state.get_next_traversal_generation(),
                None, traverse_node, traverse_node, traverse_node)
            traverser.reachable = self.physical_incoming
            traverser.visit_event(self.start_event)
            # Keep everything symmetric
            for other in self.physical_incoming:
                other.physical_outgoing.add(self)
        if self.finish_event.exists():
            traverser = EventGraphTraverser(True, True,
                self.state.get_next_traversal_generation(),
                None, traverse_node, traverse_node, traverse_node)
            traverser.reachable = self.physical_outgoing
            traverser.visit_event(self.finish_event)
            # Keep everything symmetric
            for other in self.physical_outgoing:
                other.physical_incoming.add(self)

    def perform_cycle_check(self):
        def traverse_node(node, traverser):
            if node is traverser.origin:
                traverser.cycle = True
                print "CYCLE DETECTED!"
                for n in traverser.stack:
                    print str(n)
                print str(node)
                return False
            if traverser.cycle:
                return False
            traverser.stack.append(node)
            return True
        def post_traverse(node, traverser):
            assert traverser.stack
            traverser.stack.pop()
        traverser = EventGraphTraverser(False, True,
            self.state.get_next_traversal_generation(),
            None, traverse_node, traverse_node, traverse_node,
            None, post_traverse, post_traverse, post_traverse)
        traverser.origin = self
        traverser.cycle = False
        traverser.stack = list()
        traverser.visit_event(self.start_event)
        return traverser.cycle

    def get_physical_reachable(self, reachable, forward):
        if self in reachable:
            return 
        reachable.add(self)
        if forward:
            for op in self.physical_outgoing:
                op.get_physical_reachable(reachable, True)
        else:
            for op in self.physical_incoming:
                op.get_physical_reachable(reachable, False)

    def get_event_context(self):
        if self.event_context is not None:
            return self.event_context
        # Find all the preceding and postceding operations and then
        # find their common ancestor in the task hierarchy
        def traverse_op(node, traverser):
            traverser.ops.add(node)
            return False
        op_finder = EventGraphTraverser(True, True, 
            self.state.get_next_traversal_generation(), None, traverse_op)
        op_finder.ops = set()
        if self.finish_event.exists():
            op_finder.visit_event(self.finish_event)
        if self.start_event.exists():
            op_finder.forwards = False
            op_finder.visit_event(self.start_event)
        if op_finder.ops:
            result = None
            for op in op_finder.ops:
                assert op.context is not None
                if result is not None:
                    if op.context is not result:
                        # We have to do a merge
                        result_depth = result.get_depth()
                        other = op.context
                        other_depth = other.get_depth()
                        while result_depth > other_depth:
                            result = result.get_parent_context()
                            result_depth -= 1
                        while other_depth > result_depth:
                            other = other.get_parent_context()
                            other_depth -= 1
                        # As long as they are not the same keep going up
                        while other is not result:
                            result = result.get_parent_context()
                            other = other.get_parent_context()
                else:
                    result = op.context
        else:
            # If it is empty put ourselves in the same context as our creator
            result = self.get_context()
        assert result is not None
        self.event_context = result
        return result

    def print_incoming_event_edges(self, printer):
        for src in self.physical_incoming:
            if src.cluster_name is not None:
                printer.println(src.node_name+' -> '+self.node_name+
                            ' [ltail='+src.cluster_name+',style=solid,'+
                            'color=black,penwidth=2];')
            else:
                printer.println(src.node_name+' -> '+self.node_name+
                        ' [style=solid,color=black,penwidth=2];')


class RealmCopy(RealmBase):
    __slots__ = ['start_event', 'finish_event', 'src_fields', 'dst_fields', 
                 'srcs', 'dsts', 'redops', 'node_name']
    def __init__(self, state, finish, realm_num):
        RealmBase.__init__(self, state, realm_num)
        self.finish_event = finish
        if finish.exists():
            finish.add_incoming_copy(self)
        self.src_fields = list()
        self.dst_fields = list()
        self.srcs = list()
        self.dsts = list()
        self.redops = list()
        self.node_name = 'realm_copy_'+str(realm_num)

    def __str__(self):
        return "Realm Copy ("+str(self.realm_num)+")"

    __repr__ = __str__

    def set_start(self, start):
        self.start_event = start
        if start.exists:
            start.add_outgoing_copy(self) 

    def set_creator(self, creator):
        assert self.creator is None
        self.creator = creator
        self.creator.add_realm_copy(self)

    def add_field(self, src_fid, src, dst_fid, dst, redop):
        assert self.region is not None
        # Always get the fields from the source and destination regions
        # which is especially important for handling cross-region copies
        src_field = src.region.field_space.get_field(src_fid)
        dst_field = dst.region.field_space.get_field(dst_fid)
        self.src_fields.append(src_field)
        self.dst_fields.append(dst_field)
        self.srcs.append(src)
        self.dsts.append(dst)
        self.redops.append(redop)

    def find_src_inst(self, src_field):
        assert len(self.src_fields) == len(self.srcs)
        for idx in range(len(self.src_fields)):
            if src_field == self.src_fields[idx]:
                return self.srcs[idx]
        assert False

    def print_event_node(self, printer):
        if self.state.detailed_graphs:
            label = "Realm Copy ("+str(self.realm_num)+") of "+str(self.region)
        else:
            label = "Realm Copy of "+str(self.region)
        if self.intersect is not None:
            label += " (intersect with "+str(self.intersect)+")"
        if self.creator is not None:
            label += " generated by "+str(self.creator)
        lines = [[{ "label" : label, "colspan" : 3 }]]
        if self.state.detailed_graphs:
            num_fields = len(self.src_fields)
            first_field = True
            for fidx in range(num_fields):
                src_field = self.src_fields[fidx]
                dst_field = self.dst_fields[fidx]
                src_inst = self.srcs[fidx]
                dst_inst = self.dsts[fidx]
                redop = self.redops[fidx]
                line = []
                if src_field == dst_field:
                    if redop <> 0:
                        line.append(str(src_field)+' Redop='+str(redop))
                    else:
                        line.append(str(src_field))
                else:
                    if redop <> 0:
                        line.append(str(src_field)+':'+str(dst_field)+' Redop='+str(redop))
                    else:
                        line.append(str(src_field)+':'+str(dst_field))
                line.append(str(src_inst)+':'+str(dst_inst))
                if first_field:
                    line.insert(0, {"label" : "Fields",
                                    "rowspan" : num_fields})
                    first_field = False
                lines.append(line)
        color = 'darkgoldenrod1'
        for redop in self.redops:
            if redop <> 0:
                color = 'tomato'
                break
        size = 14
        label = '<table border="0" cellborder="1" cellspacing="0" cellpadding="3" bgcolor="%s">' % color + \
                "".join([printer.wrap_with_trtd(line) for line in lines]) + '</table>'
        printer.println(self.node_name+' [label=<'+label+'>,fontsize='+str(size)+\
                ',fontcolor=black,shape=record,penwidth=0];')


class RealmFill(RealmBase):
    __slots__ = ['fields', 'dsts', 'node_name']
    def __init__(self, state, finish, realm_num):
        RealmBase.__init__(self, state, realm_num)
        self.finish_event = finish
        if finish.exists():
            finish.add_incoming_fill(self)
        self.fields = list()
        self.dsts = list()
        self.node_name = 'realm_fill_'+str(realm_num)

    def __str__(self):
        return "Realm Fill ("+str(self.realm_num)+")"

    __repr__ = __str__

    def set_start(self, start):
        self.start_event = start
        if start.exists():
            start.add_outgoing_fill(self)

    def set_creator(self, creator):
        assert self.creator is None
        self.creator = creator
        self.creator.add_realm_fill(self)

    def add_field(self, fid, dst):
        assert self.region is not None
        field = dst.region.field_space.get_field(fid)
        self.fields.append(field)
        self.dsts.append(dst)

    def print_event_node(self, printer):
        if self.state.detailed_graphs:
            label = "Realm Fill ("+str(self.realm_num)+") of "+str(self.region)
        else:
            label = "Realm Fill of "+str(self.region)
        if self.intersect is not None:
            label += " (intersect with "+str(self.intersect)+")"
        if self.creator is not None:
            label += " generated by "+str(self.creator)
        lines = [[{ "label" : label, "colspan" : 3 }]]
        if self.state.detailed_graphs:
            num_fields = len(self.fields)
            first_field = True
            for fidx in range(num_fields):
                dst_field = self.fields[fidx]
                dst_inst = self.dsts[fidx]
                line = []
                line.append(str(dst_field))
                line.append(str(dst_inst))
                if first_field:
                    line.insert(0, {"label" : "Fields",
                                    "rowspan" : num_fields})
                    first_field = False
                lines.append(line)
        color = 'chartreuse'
        size = 14
        label = '<table border="0" cellborder="1" cellspacing="0" cellpadding="3" bgcolor="%s">' % color + \
                "".join([printer.wrap_with_trtd(line) for line in lines]) + '</table>'
        printer.println(self.node_name+' [label=<'+label+'>,fontsize='+str(size)+\
                ',fontcolor=black,shape=record,penwidth=0];')


class EventGraphTraverser(object):
    def __init__(self, forwards, use_gen, generation,
                 event_fn = None, op_fn = None,
                 copy_fn = None, fill_fn = None,
                 post_event_fn = None, post_op_fn = None,
                 post_copy_fn = None, post_fill_fn = None):
        self.forwards = forwards
        self.use_gen = use_gen
        self.generation = generation
        self.event_fn = event_fn
        self.op_fn = op_fn
        self.copy_fn = copy_fn
        self.fill_fn = fill_fn
        self.post_event_fn = post_event_fn
        self.post_op_fn = post_op_fn
        self.post_copy_fn = post_copy_fn
        self.post_fill_fn = post_fill_fn

    def visit_event(self, node):
        if self.use_gen:
            if node.generation == self.generation:
                return
            else:
                node.generation = self.generation
        do_next = True
        if self.event_fn is not None:
            do_next = self.event_fn(node, self)
        if not do_next:
            return
        if self.forwards:
            if node.outgoing is not None:
                for event in node.outgoing:
                    self.visit_event(event)
            if node.outgoing_ops is not None:
                for op in node.outgoing_ops:
                    self.visit_op(op)
            if node.outgoing_fills is not None:
                for fill in node.outgoing_fills:
                    self.visit_fill(fill)
            if node.outgoing_copies is not None:
                for copy in node.outgoing_copies:
                    self.visit_copy(copy)
        else:
            if node.incoming is not None:
                for event in node.incoming:
                    self.visit_event(event)
            if node.incoming_ops is not None:
                for op in node.incoming_ops:
                    self.visit_op(op)
            if node.incoming_fills is not None:
                for fill in node.incoming_fills:
                    self.visit_fill(fill)
            if node.incoming_copies is not None:
                for copy in node.incoming_copies:
                    self.visit_copy(copy)
        if self.post_event_fn is not None:
            self.post_event_fn(node, self)

    def visit_op(self, node):
        if self.use_gen:
            if node.generation == self.generation:
                return
            else:
                node.generation = self.generation
        do_next = True
        if self.op_fn is not None:
            do_next = self.op_fn(node, self)
        if not do_next:
            return
        if self.forwards:
            if node.finish_event.exists():
                self.visit_event(node.finish_event)
        else:
            if node.start_event.exists():
                self.visit_event(node.start_event)
        if self.post_op_fn is not None:
            self.post_op_fn(node, self)

    def visit_fill(self, node):
        if self.use_gen:
            if node.generation == self.generation:
                return
            else:
                node.generation = self.generation
        do_next = True
        if self.fill_fn is not None:
            do_next = self.fill_fn(node, self)
        if not do_next:
            return
        if self.forwards:
            if node.finish_event.exists():
                self.visit_event(node.finish_event)
        else:
            if node.start_event.exists():
                self.visit_event(node.start_event)
        if self.post_fill_fn is not None:
            self.post_fill_fn(node, self)

    def visit_copy(self, node):
        if self.use_gen:
            if node.generation == self.generation:
                return
            else:
                node.generation = self.generation
        do_next = True
        if self.copy_fn is not None:
            do_next = self.copy_fn(node, self)
        if not do_next:
            return
        if self.forwards:
            if node.finish_event.exists():
                self.visit_event(node.finish_event)
        else:
            if node.start_event.exists():
                self.visit_event(node.start_event)
        if self.post_copy_fn is not None:
            self.post_copy_fn(node, self)
        

class GraphPrinter(object):
    __slots__ = ['name', 'filename', 'out', 'depth', 'next_cluster_id']
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
        self.next_cluster_id = 0

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
                print "Simpliyfing dot file "+dot_file+" with tred..."
                tred = subprocess.Popen(['tred', dot_file], stdout=subprocess.PIPE)
                print "Done"
                print "Invoking dot to generate file "+pdf_file+"..."
                dot = subprocess.Popen(['dot', '-Tpdf', '-o', pdf_file], stdin=tred.stdout)
                print "Done"
                if dot.wait() != 0:
                    raise Exception('DOT failed')
            else:
                print "Invoking dot to generate file "+pdf_file+"..."
                subprocess.check_call(['dot', '-Tpdf', '-o', pdf_file, dot_file])
        except:
            print "WARNING: DOT failure, image for graph "+str(self.name)+" not generated"
            subprocess.call(['rm', '-f', 'core', pdf_file])

    def up(self):
        assert self.depth > 0
        self.depth = self.depth-1

    def down(self):
        self.depth = self.depth+1

    def start_new_cluster(self, label=None):
        cluster_name = 'cluster_'+str(self.next_cluster_id)
        self.next_cluster_id += 1
        self.println('subgraph ' + cluster_name)
        self.println('{')
        self.down()
        if label is not None:
            self.println('label=<'+label+'>;')
        return cluster_name

    def end_this_cluster(self):
        self.up()
        self.println('}')

    def println(self,string):
        for i in range(self.depth):
            self.out.write('  ')
        self.out.write(string)
        self.out.write('\n')

    def wrap_with_trtd(self, labels):
        line = "<tr>"
        for label in labels:
            if isinstance(label, str):
                l = label
                rowspan = 1
                colspan = 1
            else:
                l = label["label"]
                rowspan = label["rowspan"] if "rowspan" in label else 1
                colspan = label["colspan"] if "colspan" in label else 1
            line = line + ('<td colspan="%d" rowspan="%d">%s</td>' % (colspan, rowspan, l))
        line = line + "</tr>"
        return line

    def generate_html_op_label(self, title, requirements, mappings, color, detailed):
        lines = list()
        lines.append([{"label" : title, "colspan" : 3}])       
        if requirements is not None:
            for i in range(len(requirements)):
                req = requirements[i]
                region_name = str(req.logical_node)
                line = [str(i), region_name, req.get_privilege_and_coherence()]
                lines.append(line)
                if detailed and mappings is not None and i in mappings:
                    first_field = True
                    for f in req.fields:
                        line = []
                        if first_field:
                            line.append({"label" : "Fields", "rowspan" : len(req.fields)})
                            first_field = False
                        line.append(str(f))
                        line.append(str(mappings[i][f.fid]))
                        lines.append(line)
        return '<table border="0" cellborder="1" cellpadding="3" cellspacing="0" bgcolor="%s">' % color + \
              "".join([self.wrap_with_trtd(line) for line in lines]) + '</table>'


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
index_space_point_pat    = re.compile(
    prefix+"Index Space Point (?P<uid>[0-9a-f]+) (?P<dim>[0-9]+) (?P<p1>[0-9]+) "+
            "(?P<p2>[0-9]+) (?P<p3>[0-9]+)")
index_space_rect_pat     = re.compile(
    prefix+"Index Space Rect (?P<uid>[0-9a-f]+) (?P<dim>[0-9]+) (?P<lo1>[0-9]+) "+
           "(?P<lo2>[0-9]+) (?P<lo3>[0-9]+) (?P<hi1>[0-9]+) (?P<hi2>[0-9]+) "+
           "(?P<hi3>[0-9]+)")
empty_index_space_pat    = re.compile(
    prefix+"Empty Index Space (?P<uid>[0-9a-f]+)")
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
    prefix+"Close Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<is_inter>[0-1]) "+
           "(?P<is_read_only>[0-1])")
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
realm_copy_intersect_pat= re.compile(
    prefix+"Copy Intersect (?P<id>[0-9a-f]+) (?P<gen>[0-9]+) (?P<reg>[0-1]+) "+
           "(?P<index>[0-9a-f]+) (?P<field>[0-9]+) (?P<tid>[0-9]+)")
realm_fill_pat          = re.compile(
    prefix+"Fill Events (?P<uid>[0-9]+) (?P<ispace>[0-9]+) (?P<fspace>[0-9]+) "+
           "(?P<tid>[0-9]+) (?P<preid>[0-9a-f]+) (?P<pregen>[0-9]+) "+
           "(?P<postid>[0-9a-f]+) (?P<postgen>[0-9]+)")
realm_fill_field_pat    = re.compile(
    prefix+"Fill Field (?P<id>[0-9a-f]+) (?P<gen>[0-9]+) (?P<fid>[0-9]+) "+
           "(?P<dstid>[0-9a-f]+)")
realm_fill_intersect_pat= re.compile(
    prefix+"Fill Intersect (?P<id>[0-9a-f]+) (?P<gen>[0-9]+) (?P<reg>[0-1]+) "+
           "(?P<index>[0-9a-f]+) (?P<field>[0-9]+) (?P<tid>[0-9]+)")
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
        assert e2.exists()
        if e1.exists():
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
        copy.set_start(e1)
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
    m = realm_copy_intersect_pat.match(line)
    if m is not None:
        e = state.get_event(int(m.group('id'),16),int(m.group('gen')))
        copy = state.get_realm_copy(e)
        is_region = True if int(m.group('reg')) == 1 else False
        if is_region:
            copy.set_intersect(state.get_region(int(m.group('index'),16),
              int(m.group('field')), int(m.group('tid'))))
        else:
            copy.set_intersect(state.get_partition(int(m.group('index'),16),
              int(m.group('field')), int(m.group('tid'))))
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
    m = realm_fill_intersect_pat.match(line)
    if m is not None:
        e = state.get_event(int(m.group('id'),16),int(m.group('gen')))
        fill = state.get_realm_fill(e)
        is_region = True if int(m.group('reg')) == 1 else False
        if is_region:
            fill.set_intersect(state.get_region(int(m.group('index'),16),
              int(m.group('field')), int(m.group('tid'))))
        else:
            fill.set_intersect(state.get_partition(int(m.group('index'),16),
              int(m.group('field')), int(m.group('tid'))))
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
        # Save the top-level uid
        state.top_level_uid = int(m.group('uid'))
        return True
    m = single_task_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(SINGLE_TASK_KIND)
        op.set_name(m.group('name'))
        op.set_task_id(int(m.group('tid')))
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
        op.set_context(context)
        return True
    m = close_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        inter = True if int(m.group('is_inter')) == 1 else False
        if inter:
            read_only = True if int(m.group('is_read_only')) == 1 else False
            if read_only:
                op.set_op_kind(READ_ONLY_CLOSE_OP_KIND)
                op.set_name("Read Only Close Op "+m.group('uid'))
            else:
                op.set_op_kind(INTER_CLOSE_OP_KIND)
                op.set_name("Inter Close Op "+m.group('uid'))
        else:
            op.set_op_kind(POST_CLOSE_OP_KIND)
            op.set_name("Post Close Op "+m.group('uid'))
        
        context = state.get_task(int(m.group('ctx')))
        # Only add this to the context if it not an intermediate
        # close operation, otherwise add it to the context like normal
        # because it as an actual operation
        op.set_context(context, not inter)
        return True
    m = close_creator_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        creator = state.get_operation(int(m.group('cuid')))
        op.set_creator(creator, int(m.group('idx')))
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
        index_point.vals[0] = int(m.group('val1'))
        if dim > 1:
            index_point.vals[1] = int(m.group('val2'))
            if dim > 2:
                index_point.vals[2] = int(m.group('val3'))
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
        color.vals[0] = int(m.group('val1'))
        if dim > 1:
            color.vals[1] = int(m.group('val2'))
            if dim > 2:
                color.vals[2] = int(m.group('val3'))
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
        color.vals[0] = int(m.group('val1'))
        if dim > 1:
            color.vals[1] = int(m.group('val2'))
            if dim > 2:
                color.vals[2] = int(m.group('val3'))
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
    m = index_space_point_pat.match(line)
    if m is not None:
        index_space = state.get_index_space(int(m.group('uid'),16)) 
        dim = int(m.group('dim'))
        point = Point(dim)
        point.vals[0] = int(m.group('p1'))
        if dim >= 2:
            point.vals[1] = int(m.group('p2'))
            if dim >= 3:
                point.vals[2] = int(m.group('p3'))
        index_space.add_point(point)
        return True
    m = index_space_rect_pat.match(line)
    if m is not None:
        index_space = state.get_index_space(int(m.group('uid'),16))
        dim = int(m.group('dim'))
        lo = Point(dim)
        hi = Point(dim)
        lo.vals[0] = int(m.group('lo1'))
        hi.vals[0] = int(m.group('hi1'))
        if dim >= 2:
            lo.vals[1] = int(m.group('lo2'))
            hi.vals[1] = int(m.group('hi2'))
            if dim >= 3:
                lo.vals[2] = int(m.group('lo3'))
                hi.vals[2] = int(m.group('hi3'))
        index_space.add_rect(Rect(lo, hi))
        return True
    m = empty_index_space_pat.match(line)
    if m is not None:
        index_space = state.get_index_space(int(m.group('uid'),16))
        index_space.set_empty()
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
                 'ops', 'tasks', 'task_names', 'has_mapping_deps', 'instances', 'events', 
                 'copies', 'fills', 'phase_barriers', 'no_event', 'slice_index', 
                 'slice_slice', 'point_slice', 'next_generation', 'next_realm_num',
                 'detailed_graphs', 'assert_on_fail']
    def __init__(self, verbose, details, fail):
        self.verbose = verbose
        self.detailed_graphs = details
        self.assert_on_fail = fail
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
        # For physical traversals
        self.next_generation = 1
        self.next_realm_num = 1

    def get_next_traversal_generation(self):
        result = self.next_generation
        self.next_generation += 1
        return result

    def parse_log_file(self, file_name):
        print 'Reading log file %s...' % file_name
        try:
            log = open(file_name, 'r')
        except:
            print 'ERROR: Unable to find file '+file_name
            print 'Legion Spy will now exit'
            sys.exit(1)
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

    def post_parse(self, simplify_graphs, need_physical):
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
                task.op.set_name(self.task_names[task.op.task_id])
        # Assign the depth of the top context
        op = self.get_operation(self.top_level_uid)
        assert op.context is not None
        op.context.depth = 0
        # Compute the physical reachable
        for op in self.ops.itervalues():
            op.compute_physical_reachable()
        for copy in self.copies.itervalues():
            copy.compute_physical_reachable()
        for fill in self.fills.itervalues():
            fill.compute_physical_reachable()
        if need_physical and simplify_graphs:
            self.simplify_physical_graph() 
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
            for event in self.events.itervalues():
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

    def simplify_physical_graph(self):
        print "Simplifying event graph..."
        def traverse_node(node, traverser):
            if node not in traverser.order:
                traverser.order.append(node)
            return True
        # Build a topological order of everything 
        topological_sorter = EventGraphTraverser(False, True,
            self.get_next_traversal_generation(), None,
            traverse_node, traverse_node, traverse_node)
        topological_sorter.order = list()
        # Traverse all the sinks
        for op in self.ops.itervalues():
            if not op.physical_outgoing:
                topological_sorter.visit_op(op)
        for copy in self.ops.itervalues():
            if not copy.physical_outgoing:
                topological_sorter.visit_copy(op)
        for fill in self.fills.itervalues():
            if not fill.physical_outgoing:
                toplogical_sorter.visit_fill(fill)
        # Now that we have everything sorter based on topology
        # Do the simplification in reverse order
        for src in topological_sorter.order:
            if src.physical_outgoing is None:
                continue
            actual_out = src.physical_outgoing.copy()
            diff = False
            for next_vert in src.physical_outgoing:
                if not next_vert in actual_out:
                    continue
                reachable = set()
                next_vert.get_physical_reachable(reachable, True)
                # See which edges we can remove
                to_remove = list()
                for other in actual_out:
                    if other == next_vert:
                        continue
                    if other in reachable:
                        to_remove.append(other)
                del reachable
                if len(to_remove) > 0:
                    diff = True
                    for rem in to_remove:
                        actual_out.remove(rem)
                        rem.physical_incoming.remove(src)
                del to_remove
            if diff:
                src.physical_outgoing = actual_out
        print "Done"

    def alias_points(self, p1, p2):
        # These two tasks are aliased so merge them together 
        # Merge the operations first 
        p1.op.merge(p2.op)
        self.ops[p2.op.uid] = p1.op
        # Now merge the tasks and delete the other task
        p1.merge(p2)
        del self.tasks[p2.op]

    def perform_logical_analysis(self, perform_checks, sanity_checks):
        # Run the full analysis first, this will confirm that
        # the runtime did what we thought it should do
        for task in self.tasks.itervalues():
            # If we're only performing checks then we might break out early
            if not task.perform_logical_dependence_analysis(perform_checks):
                return False
            # If we're doing full on sanity checks, run them now
            if perform_checks and sanity_checks:
                if not task.perform_logical_sanity_analysis():
                    return False 
        return True

    def perform_physical_analysis(self, perform_checks, sanity_checks):
        assert self.top_level_uid is not None
        top_task = self.get_task(self.top_level_uid)
        # Perform the physical analysis on all the operations in program order
        if not top_task.perform_task_physical_analysis(perform_checks):
            print "FAIL"
            return
        print "Pass"
        #if sanity_checks:
            # Run dataflow checks

            # Run race detection

    def perform_cycle_checks(self):
        for op in self.ops.itervalues(): 
            if op.perform_cycle_check():
                return
        for copy in self.copies.itervalues():
            if copy.perform_cycle_check():
                return
        for fill in self.fills.itervalues():
            if fill.perform_cycle_check():
                return
        print "No cycles detected"

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
        total_dataflow_graphs = 0
        for task in self.tasks.itervalues():
            total_dataflow_graphs += task.print_dataflow_graph(path, simplify_graphs)
        if self.verbose:
            print "Made "+str(total_dataflow_graphs)+" dataflow graphs"

    def make_event_graph(self, path):
        # we print these recursively so we can see the hierarchy
        assert self.top_level_uid is not None
        op = self.get_operation(self.top_level_uid)
        file_name = 'event_graph_'+str(op).replace(' ','_')+'_'+str(op.uid)
        printer = GraphPrinter(path, file_name)
        elevate = dict()
        all_nodes = set()
        op.print_event_graph(printer, elevate, all_nodes, True) 
        # Now print the edges at the very end
        for node in all_nodes:
            node.print_incoming_event_edges(printer) 
        printer.print_pdf_after_close(False)

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
        result = RealmCopy(self, event, self.next_realm_num)
        self.next_realm_num += 1
        self.copies[event] = result
        return result

    def get_realm_fill(self, event):
        if event in self.fills:
            return self.fills[event]
        result = RealmFill(self, event, self.next_realm_num)
        self.next_realm_num += 1
        self.fills[event] = result
        return result

    def create_copy(self, creator):
        result = RealmCopy(self, self.no_event, self.next_realm_num)
        self.next_realm_num += 1
        result.set_creator(creator)
        return result

    def create_fill(self, creator):
        result = RealmFill(self, self.no_event, self.next_realm_num)
        self.next_realm_num += 1
        result.set_creator(creator)
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

    def reset_physical_state(self, depth):
        for region in self.regions.itervalues():
            region.reset_physical_state(depth)
        for partition in self.partitions.itervalues():
            partition.reset_physical_state(depth)
        # Definitely run the garbage collector here
        gc.collect()

def usage():
    print "Usage: "+sys.argv[0]+" [-l -p -c -r -m -d -e -z -s -k -u -v -a] <file_name(s)>+"
    print "  -l : perform logical checks"
    print "  -p : perform physical checks"
    print "  -c : check for cycles"
    print "  -r : make region tree graphs"
    print "  -m : make machine graphs"
    print "  -d : make dataflow graphs"
    print "  -e : make event graphs"
    print "  -z : add extra detail to graphs"
    print "  -s : perform sanity checking analysis (old Legion Spy analysis)"
    print "  -k : keep temporary files"
    print "  -u : keep graphs unsimplified (maintains all redundant edges)"
    print "  -v : verbose"
    print "  -a : assert on analysis failure (useful for self-debugging)"
    sys.exit(1)

def main(temp_dir):
    if len(sys.argv) < 2:
        usage()
    try:
        opts, args = getopt(sys.argv[1:],'lpcrmdeszkuva')
    except GetoptError as err:
        print "ERROR: "+str(err)
        sys.exit(1)
    opts = dict(opts)
    if len(args) <> 1:
        usage()
    file_names = args

    logical_checks = False
    physical_checks = False
    cycle_checks = False
    region_tree_graphs = False
    machine_graphs = False
    dataflow_graphs = False
    event_graphs = False
    detailed_graphs = False
    sanity_checks = False
    keep_temp_files = False
    simplify_graphs = True # Note that this defaults to true
    verbose = False
    assert_on_fail = False
    for opt in opts:
        if opt == '-l':
            logical_checks = True
            continue
        if opt == '-p':
            physical_checks = True
            continue
        if opt == '-c':
            cycle_checks = True
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
        if opt == '-z':
            detailed_graphs = True
            continue
        if opt == '-s':
            sanity_checks = True
            continue
        if opt == '-k':
            keep_temp_files = True
            continue
        if opt == '-u':
            simplify_graphs = False
            continue
        if opt == '-v':
            verbose = True
            continue
        if opt == '-a':
            assert_on_fail = True
            continue

    state = State(verbose, detailed_graphs, assert_on_fail)
    total_matches = 0 
    for file_name in file_names:
        total_matches += state.parse_log_file(file_name)
    if verbose:
        print 'Matched %d lines across all files.' % total_matches
    if total_matches == 0:
        print 'No matches found! Exiting...'
        return
    logical_enabled,physical_enabled = state.post_parse(simplify_graphs, 
                                          physical_checks or event_graphs)
    if logical_checks and not logical_enabled:
        print "WARNING: Requested logical analysis but logging information is "+\
              "missing. Please compile the runtime with -DLEGION_SPY to enable "+\
              "validation of the runtime. Disabling logical checks." 
        logical_checks = False
    if physical_checks and not physical_enabled:
        print "WARNING: Requested physical analysis but logging information is "+\
              "missing. Please compile the runtime with -DLEGION_SPY to enable "+\
              "validation of the runtime. Disabling physical checks."
        physical_checks = False
    if physical_checks and sanity_checks and not logical_enabled:
        print "WARNING: Requested sanity checks for physical analysis but "+\
              "logging information of logical analysis is missing. Please "+\
              "compile the runtime with -DLEGION_SPY to enable validation "+\
              "of the runtime. Disabling sanity checks."
        sanity_checks = False
    if cycle_checks and not physical_enabled:
        print "WARNING: Requested cycle checks but logging information is "+\
              "missing. Please compile the runtime with -DLEGION_SPY to enable "+\
              "validation of the runtime. Disabling cycle checks."
        cycle_checks = False
    # If we are doing logical checks or the user asked for the dataflow
    # graph but we don't have any logical data then perform the logical analysis
    need_logical = dataflow_graphs and not logical_enabled
    if logical_checks or need_logical:
        if need_logical:
            print "INFO: No logical dependence data was found so we are running "+\
                  "logical analysis to show the dataflow graphs that the runtime "+\
                  "should compute. These are not the actual dataflow graphs computed."
        print "Performing logical analysis..."
        state.perform_logical_analysis(logical_checks, sanity_checks)
    # If we are doing physical checks or the user asked for the event
    # graph but we don't have any logical data then perform the physical analysis
    need_physical = event_graphs and not physical_enabled
    if physical_checks or need_physical:
        if need_physical:
            print "INFO: No physical dependence data was found so we are running "+\
                  "physical analysis to show the event graph that the runtime "+\
                  "should compute. This is not the actual event graph computed."
        print "Performing physical analysis..."
        state.perform_physical_analysis(physical_checks, sanity_checks)
    if cycle_checks:
        print "Performing cycle checks..."
        state.perform_cycle_checks()
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
        state.make_event_graph(temp_dir)

    print 'Legion Spy analysis complete.  Exiting...'
    if keep_temp_files:
        try:
            subprocess.check_call('cp '+temp_dir+'* .',shell=True)
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

