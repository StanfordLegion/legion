#!/usr/bin/env python3

# Copyright 2022 Stanford University, NVIDIA Corporation
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

# mypy: ignore-errors

# TODO LIST
# Support predication for physical analysis

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import array
import collections
import copy
import gc
import itertools
import os
import random
import re
import shutil
import string
import struct
import subprocess
import sys
import tempfile

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

NO_ACCESS     = 0x00000000
READ_ONLY     = 0x00000001
READ_WRITE    = 0x00000007
WRITE_ONLY    = 0x10000002
WRITE_DISCARD = 0x10000007
REDUCE        = 0x00000004

EXCLUSIVE = 0
ATOMIC = 1
SIMULTANEOUS = 2
RELAXED = 3

NO_OP_KIND = 0
SINGLE_TASK_KIND = 1
INDEX_TASK_KIND = 2
MAP_OP_KIND = 3
INTER_CLOSE_OP_KIND = 4
#READ_ONLY_CLOSE_OP_KIND = 5
POST_CLOSE_OP_KIND = 6
#OPEN_OP_KIND = 7
#ADVANCE_OP_KIND = 8
FENCE_OP_KIND = 9
COPY_OP_KIND = 10 
FILL_OP_KIND = 11
ACQUIRE_OP_KIND = 12 
RELEASE_OP_KIND = 13
DELETION_OP_KIND = 14
ATTACH_OP_KIND = 15
DETACH_OP_KIND = 16
DEP_PART_OP_KIND = 17
PENDING_PART_OP_KIND = 18
DYNAMIC_COLLECTIVE_OP_KIND = 19
TRACE_OP_KIND = 20
TIMING_OP_KIND = 21
ALL_REDUCE_OP_KIND = 22
PREDICATE_OP_KIND = 23
MUST_EPOCH_OP_KIND = 24
CREATION_OP_KIND = 25
TUNABLE_OP_KIND = 26

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
"Read Only Close Op",
"Post Close Op",
"Open Op",
"Advance Op",
"Fence Op",
"Copy Op",
"Fill Op",
"Acquire Op",
"Release Op",
"Deletion Op",
"Attach Op",
"Detach Op",
"Dependent Partition Op",
"Pending Partition Op",
"Dynamic Collective Op",
"Trace Op",
"Timing Op",
"Reduce Op",
"Predicate Op",
"Must Epoch Op",
"Creation Op",
]

INDEX_SPACE_EXPR = 0
UNION_EXPR = 1
INTERSECT_EXPR = 2
DIFFERENCE_EXPR = 3,

# Helper methods for python 2/3 foolishness
def iteritems(obj):
    return obj.items() if sys.version_info > (3,) else obj.viewitems()

def iterkeys(obj):
    return obj.keys() if sys.version_info > (3,) else obj.viewkeys()

def itervalues(obj):
    return obj.values() if sys.version_info > (3,) else obj.viewvalues()

try:
    xrange # Python 2
except NameError:
    xrange = range # Python 3

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
    to_traverse = collections.deque()
    to_traverse.append(op)
    backward_reachable = set()
    for pre in preconditions:
        # If we've already seen it traverse backwards
        if pre in backward_reachable:
            continue
        # Keep BFS-ing until we find the node or run out of
        # nodes to traverse
        found = False
        while to_traverse:
            node = to_traverse.popleft()
            for prev in node.physical_incoming:
                if prev not in backward_reachable:
                    backward_reachable.add(prev)
                    to_traverse.append(prev)
                    # If it's what we're looking for then record that
                    if not found and pre is prev:
                        found = True
                        # Don't break, still need to add the other
                        # backward reachable nodes from this node
            if found:
                break
        if not found:
            return pre
    return None

# Borrowed from stack overflow 3173320
def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '|'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    if sys.version_info > (3,):
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r', flush=True)
    else:
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
        sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()

class Point(object):
    __slots__ = ['dim', 'vals', 'shape', 'index_set']
    def __init__(self, dim):
        self.dim = dim
        self.vals = array.array('l', (0,)*dim)
        self.shape = None
        self.index_set = None

    def mk_string(self, start, delim, end):
        return '%s%s%s' % (start, delim.join(map(str, self.vals)), end)

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
        for i in xrange(self.dim):
            if self.vals[i] != point.vals[i]:
                return False
        return True

    def copy(self):
        result = Point(self.dim)
        for i in xrange(self.dim):
            result.vals[i] = self.vals[i]
        result.shape = self.shape
        result.index_set = self.index_set
        return result;

class Rect(object):
    __slots__ = ['dim', 'lo', 'hi']
    def __init__(self, lo, hi):
        assert lo.dim == hi.dim
        self.dim = lo.dim
        self.lo = lo
        self.hi = hi

    def __str__(self):
        return '%s-%s' % (self.lo, self.hi)

    __repr__ = __str__

    def __hash__(self):
        return hash(str(self))

    def empty(self):
        for i in xrange(self.dim):
            if self.lo.vals[i] > self.hi.vals[i]:
                return True
        return False

    def volume(self):
        if self.empty():
            return 0
        result = 1
        for i in xrange(self.dim):
            result *= (self.hi.vals[i] - self.lo.vals[i] + 1)
        return result

    def dominates(self, rect):
        for i in xrange(self.dim):
            if rect.lo.vals[i] < self.lo.vals[i] or self.hi.vals[i] < rect.hi.vals[i]:
                return False
        return True

    def intersects(self, rect):
        # If they overlap in all dimensions then they intersect
        assert rect.dim == self.dim
        for i in xrange(self.dim):
            if rect.hi.vals[i] < self.lo.vals[i] or self.hi.vals[i] < rect.lo.vals[i]:
                return False
        return True

    def intersection(self, rect):
        lo = Point(self.dim)
        hi = Point(self.dim)
        for i in xrange(self.dim):
            lo.vals[i] = max(self.lo.vals[i], rect.lo.vals[i])
            hi.vals[i] = min(self.hi.vals[i], rect.hi.vals[i])
        return Rect(lo, hi)

    def contains_point(self, point):
        for i in xrange(point.dim):
            x = point.vals[i]
            if x < self.lo.vals[i] or self.hi.vals[i] < x:
                return False
        return True

    def iterator(self):
        if self.dim == 1:
            for x in xrange(self.lo.vals[0], self.hi.vals[0]+1):
                point = Point(1)
                point.vals[0] = x
                yield point
        elif self.dim == 2:
            for x in xrange(self.lo.vals[0], self.hi.vals[0]+1):
                for y in xrange(self.lo.vals[1], self.hi.vals[1]+1):
                    point = Point(2)
                    point.vals[0] = x
                    point.vals[1] = y
                    yield point
        elif self.dim == 3:
            for x in xrange(self.lo.vals[0], self.hi.vals[0]+1):
                for y in xrange(self.lo.vals[1], self.hi.vals[1]+1):
                    for z in xrange(self.lo.vals[2], self.hi.vals[2]+1):
                        point = Point(3)
                        point.vals[0] = x
                        point.vals[1] = y
                        point.vals[2] = z
                        yield point
        elif self.dim == 4:
            for x in xrange(self.lo.vals[0], self.hi.vals[0]+1):
                for y in xrange(self.lo.vals[1], self.hi.vals[1]+1):
                    for z in xrange(self.lo.vals[2], self.hi.vals[2]+1):
                        for w in xrange(self.lo.vals[3], self.hi.vals[3]+1):
                            point = Point(4)
                            point.vals[0] = x
                            point.vals[1] = y
                            point.vals[2] = z
                            point.vals[3] = w
                            yield point
        elif self.dim == 5:
            for x in xrange(self.lo.vals[0], self.hi.vals[0]+1):
                for y in xrange(self.lo.vals[1], self.hi.vals[1]+1):
                    for z in xrange(self.lo.vals[2], self.hi.vals[2]+1):
                        for w in xrange(self.lo.vals[3], self.hi.vals[3]+1):
                            for v in xrange(self.lo.vals[4], self.hi.vals[4]+1):
                                point = Point(5)
                                point.vals[0] = x
                                point.vals[1] = y
                                point.vals[2] = z
                                point.vals[3] = w
                                point.vals[4] = v
                                yield point
        elif self.dim == 6:
            for x in xrange(self.lo.vals[0], self.hi.vals[0]+1):
                for y in xrange(self.lo.vals[1], self.hi.vals[1]+1):
                    for z in xrange(self.lo.vals[2], self.hi.vals[2]+1):
                        for w in xrange(self.lo.vals[3], self.hi.vals[3]+1):
                            for v in xrange(self.lo.vals[4], self.hi.vals[4]+1):
                                for u in xrange(self.lo.vals[5], self.hi.vals[5]+1):
                                    point = Point(6)
                                    point.vals[0] = x
                                    point.vals[1] = y
                                    point.vals[2] = z
                                    point.vals[3] = w
                                    point.vals[4] = v
                                    point.vals[5] = u
                                    yield point
        elif self.dim == 7:
            for x in xrange(self.lo.vals[0], self.hi.vals[0]+1):
                for y in xrange(self.lo.vals[1], self.hi.vals[1]+1):
                    for z in xrange(self.lo.vals[2], self.hi.vals[2]+1):
                        for w in xrange(self.lo.vals[3], self.hi.vals[3]+1):
                            for v in xrange(self.lo.vals[4], self.hi.vals[4]+1):
                                for u in xrange(self.lo.vals[5], self.hi.vals[5]+1):
                                    for t in xrange(self.lo.vals[6], self.hi.vals[6]+1):
                                        point = Point(7)
                                        point.vals[0] = x
                                        point.vals[1] = y
                                        point.vals[2] = z
                                        point.vals[3] = w
                                        point.vals[4] = v
                                        point.vals[5] = u
                                        point.vals[6] = t
                                        yield point
        elif self.dim == 8:
            for x in xrange(self.lo.vals[0], self.hi.vals[0]+1):
                for y in xrange(self.lo.vals[1], self.hi.vals[1]+1):
                    for z in xrange(self.lo.vals[2], self.hi.vals[2]+1):
                        for w in xrange(self.lo.vals[3], self.hi.vals[3]+1):
                            for v in xrange(self.lo.vals[4], self.hi.vals[4]+1):
                                for u in xrange(self.lo.vals[5], self.hi.vals[5]+1):
                                    for t in xrange(self.lo.vals[6], self.hi.vals[6]+1):
                                        for s in xrange(self.lo.vals[7], self.hi.vals[7]+1):
                                            point = Point(8)
                                            point.vals[0] = x
                                            point.vals[1] = y
                                            point.vals[2] = z
                                            point.vals[3] = w
                                            point.vals[4] = v
                                            point.vals[5] = u
                                            point.vals[6] = t
                                            point.vals[7] = s
                                            yield point
        elif self.dim == 9:
            for x in xrange(self.lo.vals[0], self.hi.vals[0]+1):
                for y in xrange(self.lo.vals[1], self.hi.vals[1]+1):
                    for z in xrange(self.lo.vals[2], self.hi.vals[2]+1):
                        for w in xrange(self.lo.vals[3], self.hi.vals[3]+1):
                            for v in xrange(self.lo.vals[4], self.hi.vals[4]+1):
                                for u in xrange(self.lo.vals[5], self.hi.vals[5]+1):
                                    for t in xrange(self.lo.vals[6], self.hi.vals[6]+1):
                                        for s in xrange(self.lo.vals[7], self.hi.vals[7]+1):
                                            for r in xrange(self.lo.vals[8], self.hi.vals[8]+1):
                                                point = Point(9)
                                                point.vals[0] = x
                                                point.vals[1] = y
                                                point.vals[2] = z
                                                point.vals[3] = w
                                                point.vals[4] = v
                                                point.vals[5] = u
                                                point.vals[6] = t
                                                point.vals[7] = s
                                                point.vals[8] = r
                                                yield point
        else:
            raise NotImplementedError("Need support for more dimensions")

    def copy(self):
        return Rect(self.lo.copy(), self.hi.copy())

# A Collection of points and rectangles
class Shape(object):
    __slots__ = ['points', 'rects']
    def __init__(self):
        self.points = set()
        self.rects = set()

    @property
    def dense(self):
        # pretty sure that realm is making sure these are
        # not overlapping so as long as we only have one
        # of either then we're dense, otherwise we're sparse
        return (len(self.points)+len(self.rects)) == 1

    @property
    def bounds(self):
        lo = None
        hi = None
        for point in self.points:
            if lo is None:
                lo = point.copy()
            else:
                for d in xrange(point.dim):
                    if point.vals[d] < lo.vals[d]:
                        lo.vals[d] = point.vals[d]
            if hi is None:
                hi = point.copy()
            else:
                for d in xrange(point.dim):
                    if point.vals[d] > hi.vals[d]:
                        hi.vals[d] = point.vals[d]
        for rect in self.rects:
            if lo is None:
                lo = rect.lo.copy()
            else:
                for d in xrange(rect.lo.dim):
                    if rect.lo.vals[d] < lo.vals[d]:
                        lo.vals[d] = rect.lo.vals[d]
            if hi is None:
                hi = rect.hi.copy()
            else:
                for d in xrange(rect.hi.dim):
                    if rect.hi.vals[d] > hi.vals[d]:
                        hi.vals[d] = rect.hi.vals[d]
        return (lo,hi)

    def get_dim(self):
        if self.points:
            return next(iter(self.points)).dim
        else:
            assert self.rects
            return next(iter(self.rects)).dim

    def __str__(self):
        return ' '.join(map(str, itertools.chain(self.points, self.rects)))

    __repr__ = __str__

    def add_point(self, point):
        self.points.add(point)

    def add_rect(self, rect):
        self.rects.add(rect)

    def empty(self):
        return not self.points and not self.rects

    def has_point(self, point):
        for rect in self.rects:
            if rect.contains_point(point):
                return True
        for p in self.points:
            if p == point:
                return True
        return False

    def volume(self):
        result = len(self.points)
        for rect in self.rects:
            result += rect.volume()
        return result

    def copy(self):
        result = Shape()
        for point in self.points:
            result.points.add(point)
        for rect in self.rects:
            result.rects.add(rect)
        return result

    def dominates(self, other):
        if len(self.rects) == 1 and len(self.points) == 0:
            our_rect = next(iter(self.rects)) 
            for point in other.points:
                if not our_rect.contains_point(point):
                    return False
            for rect in other.rects:
                if not our_rect.dominates(rect):
                    return False
            return True
        else:
            diff = other - self
            return diff.empty()

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
            for point in other.points:
                if rect.contains_point(point):
                    self.points.add(point)
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

    @classmethod
    def rect_sub_helper(cls, rect, other, to_add, dim=0):
        # We are guaranteed to intersect but not dominate
        # We have fast, explicit methods for dimensions 1-3 that will
        # not use the recursive method for better performance. For all
        # dimensions larger than that we use a recursive method that 
        # will make a bunch of intermediate Python objects that will 
        # need to garbage collected. It's still asymptotically fast 
        # but the constant factors are significant (> 50% slower)
        #print('%s - %s' % (rect, other))
        if rect.lo.dim == 1:
            # 3 cases: 2 edges, 1 center 
            if other.lo.vals[0] <= rect.lo.vals[0]:
                # Lower edge
                lo = Point(1)
                lo.vals[0] = other.hi.vals[0]+1
                to_add.append(Rect(lo,rect.hi))
            elif other.hi.vals[0] >= rect.hi.vals[0]:
                # Upper edge
                hi = Point(1)
                hi.vals[0] = other.lo.vals[0]-1
                to_add.append(Rect(rect.lo,hi))
            else:
                # Center
                hi = Point(1)
                hi.vals[0] = other.lo.vals[0]-1
                lo = Point(1)
                lo.vals[0] = other.hi.vals[0]+1
                to_add.append(Rect(rect.lo,hi))
                to_add.append(Rect(lo,rect.hi))
        elif rect.lo.dim == 2:
            # 9 cases: 4 corners, 4 edges, 1 center
            if other.lo.vals[0] <= rect.lo.vals[0]:
                if other.lo.vals[1] <= rect.lo.vals[1]:
                    # Lower-left corner
                    #print("Lower left")
                    if other.hi.vals[0] >= rect.hi.vals[0]:
                        # Just have above
                        lo = Point(2)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = other.hi.vals[1]+1
                        to_add.append(Rect(lo,rect.hi))
                    elif other.hi.vals[1] >= rect.hi.vals[1]:
                        # Just have to the right
                        lo = Point(2)
                        lo.vals[0] = other.hi.vals[0]+1
                        lo.vals[1] = rect.lo.vals[1]
                        to_add.append(Rect(lo,rect.hi))
                    else:
                        # Have both above and to the right
                        lo = Point(2)
                        lo.vals[0] = other.hi.vals[0]+1
                        lo.vals[1] = rect.lo.vals[1]
                        to_add.append(Rect(lo,rect.hi))
                        lo = Point(2)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = other.hi.vals[1]+1
                        hi = Point(2)
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        to_add.append(Rect(lo,hi))
                elif other.hi.vals[1] >= rect.hi.vals[1]:
                    # Upper-left corner (can't overlap lower-level corner)
                    #print("Upper left")
                    if other.hi.vals[0] >= rect.hi.vals[0]:
                        # Overlap with upper right,  only have below
                        hi = Point(2)
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        to_add.append(Rect(rect.lo,hi))
                    else:
                        # Have below and to the right
                        lo = Point(2)
                        lo.vals[0] = other.hi.vals[0]+1
                        lo.vals[1] = rect.lo.vals[1]
                        to_add.append(Rect(lo,rect.hi))
                        hi = Point(2)
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        to_add.append(Rect(rect.lo,hi))
                else:
                    # Left edge but neither of the two left corners
                    #print("Left edge")
                    lo = Point(2)
                    lo.vals[0] = rect.lo.vals[0]
                    lo.vals[1] = other.hi.vals[1]+1
                    to_add.append(Rect(lo,rect.hi))
                    hi = Point(2)
                    hi.vals[0] = rect.hi.vals[0]
                    hi.vals[1] = other.lo.vals[1]-1
                    to_add.append(Rect(rect.lo,hi))
                    # Check to see if we have an overlap with left edge
                    if other.hi.vals[0] < rect.hi.vals[0]:
                        # No overlap with right edge
                        lo = Point(2)
                        hi = Point(2)
                        lo.vals[0] = other.hi.vals[0]+1
                        lo.vals[1] = other.lo.vals[1]
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = other.hi.vals[1]
                        to_add.append(Rect(lo,hi))
            elif other.hi.vals[0] >= rect.hi.vals[0]:
                if other.lo.vals[1] <= rect.lo.vals[1]:
                    #print("Lower right")
                    # Lower-right corner (can't overlap with any left corners)
                    hi = Point(2)
                    hi.vals[0] = other.lo.vals[0]-1
                    hi.vals[1] = rect.hi.vals[1]
                    to_add.append(Rect(rect.lo,hi))
                    if other.hi.vals[1] < rect.hi.vals[1]:
                        # No overlap with top-right corner 
                        lo = Point(2)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = other.hi.vals[1]+1
                        to_add.append(Rect(lo,rect.hi))
                elif other.hi.vals[1] >= rect.hi.vals[1]:
                    #print("Upper right")
                    # Upper-right corner (can't overlap with any other corners)
                    hi = Point(2)
                    hi.vals[0] = other.lo.vals[0]-1
                    hi.vals[1] = rect.hi.vals[1]
                    to_add.append(Rect(rect.lo,hi))
                    lo = Point(2)
                    lo.vals[0] = other.lo.vals[0]
                    lo.vals[1] = rect.lo.vals[1]
                    hi = Point(2)
                    hi.vals[0] = rect.hi.vals[0]
                    hi.vals[1] = other.lo.vals[1]-1
                    to_add.append(Rect(lo,hi))
                else:
                    #print("Right edge")
                    # Right edge (no overlap with left edge)
                    hi = Point(2)
                    hi.vals[0] = rect.hi.vals[0]
                    hi.vals[1] = other.lo.vals[1]-1
                    to_add.append(Rect(rect.lo,hi))
                    lo = Point(2)
                    lo.vals[0] = rect.lo.vals[0]
                    lo.vals[1] = other.hi.vals[1]+1
                    to_add.append(Rect(lo,rect.hi))
                    lo = Point(2)
                    hi = Point(2)
                    lo.vals[0] = rect.lo.vals[0]
                    lo.vals[1] = other.lo.vals[1]
                    hi.vals[0] = other.lo.vals[0]-1
                    hi.vals[1] = other.hi.vals[1]
                    to_add.append(Rect(lo,hi))
            else:
                if other.lo.vals[1] <= rect.lo.vals[1]:
                    #print("Bottom edge")
                    # Bottom edge
                    hi = Point(2)
                    hi.vals[0] = other.lo.vals[0]-1
                    hi.vals[1] = rect.hi.vals[1]
                    to_add.append(Rect(rect.lo,hi))
                    lo = Point(2)
                    lo.vals[0] = other.hi.vals[0]+1
                    lo.vals[1] = rect.lo.vals[1]
                    to_add.append(Rect(lo,rect.hi))
                    # See if it intersects with top edge
                    if other.hi.vals[1] < rect.hi.vals[1]:
                        # Doesn't intersect
                        lo = Point(2)
                        hi = Point(2)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = other.hi.vals[1]+1
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        to_add.append(Rect(lo,hi))
                elif other.hi.vals[1] >= rect.hi.vals[1]:
                    #print("Top edge")
                    # Top edge (can't overlap with bottom edge)
                    hi = Point(2)
                    hi.vals[0] = other.lo.vals[0]-1
                    hi.vals[1] = rect.hi.vals[1]
                    to_add.append(Rect(rect.lo,hi))
                    lo = Point(2)
                    lo.vals[0] = other.hi.vals[0]+1
                    lo.vals[1] = rect.lo.vals[1]
                    to_add.append(Rect(lo,rect.hi))
                    lo = Point(2)
                    hi = Point(2)
                    lo.vals[0] = other.lo.vals[0]
                    lo.vals[1] = rect.lo.vals[1]
                    hi.vals[0] = other.hi.vals[0]
                    hi.vals[1] = other.lo.vals[1]-1
                    to_add.append(Rect(lo,hi))
                else:
                    #print("Center")
                    # Center (with no overlaps)
                    hi = Point(2)
                    hi.vals[0] = other.lo.vals[0]-1
                    hi.vals[1] = rect.hi.vals[1]
                    to_add.append(Rect(rect.lo,hi))
                    lo = Point(2)
                    lo.vals[0] = other.hi.vals[0]+1
                    lo.vals[1] = rect.lo.vals[1]
                    to_add.append(Rect(lo,rect.hi))
                    lo = Point(2)
                    hi = Point(2)
                    lo.vals[0] = other.lo.vals[0]
                    lo.vals[1] = rect.lo.vals[1]
                    hi.vals[0] = other.hi.vals[0]
                    hi.vals[1] = other.lo.vals[1]-1
                    to_add.append(Rect(lo,hi))
                    lo = Point(2)
                    hi = Point(2)
                    lo.vals[0] = other.lo.vals[0]
                    lo.vals[1] = other.hi.vals[1]+1
                    hi.vals[0] = other.hi.vals[0]
                    hi.vals[1] = rect.hi.vals[1]
                    to_add.append(Rect(lo,hi))
        elif rect.lo.dim == 3:
            # 27 cases: 6 faces, 8 corners, 12 edges, 1 center
            if other.lo.vals[0] <= rect.lo.vals[0]:
                if other.lo.vals[1] <= rect.lo.vals[1]:
                    if other.lo.vals[2] <= rect.lo.vals[2]:
                        # Front lower-left corner
                        #print("Front lower left corner")
                        if other.hi.vals[0] >= rect.hi.vals[0]:
                            # Front lower-right corner
                            if other.hi.vals[1] >= rect.hi.vals[1]:
                                # Overlaps with front face
                                if other.hi.vals[2] >= rect.hi.vals[2]:
                                    assert False # overlap all should never happen
                                else:
                                    # Overlaps with just front face
                                    lo = Point(3)
                                    lo.vals[0] = rect.lo.vals[0]
                                    lo.vals[1] = rect.lo.vals[1]
                                    lo.vals[2] = other.hi.vals[2]+1
                                    to_add.append(Rect(lo,rect.hi))
                            else:
                                # Overlaps with front lower corners
                                if other.hi.vals[2] >= rect.hi.vals[2]:
                                    # Overlaps with all lower corners
                                    lo = Point(3)
                                    lo.vals[0] = rect.lo.vals[0]
                                    lo.vals[1] = other.hi.vals[1]+1
                                    lo.vals[2] = rect.lo.vals[2]
                                    to_add.append(Rect(lo,rect.hi))
                                else:
                                    # Only overlaps with front lower corners 
                                    lo = Point(3)
                                    lo.vals[0] = rect.lo.vals[0]
                                    lo.vals[1] = rect.lo.vals[1]
                                    lo.vals[2] = other.hi.vals[2]+1
                                    to_add.append(Rect(lo,rect.hi))
                                    lo = Point(3)
                                    hi = Point(3)
                                    lo.vals[0] = rect.lo.vals[0]
                                    lo.vals[1] = other.hi.vals[1]+1
                                    lo.vals[2] = rect.lo.vals[2]
                                    hi.vals[0] = rect.hi.vals[0]
                                    hi.vals[1] = rect.hi.vals[1]
                                    hi.vals[2] = other.hi.vals[2]
                                    to_add.append(Rect(lo,hi))
                        else:
                            # No overlap with front right corner
                            if other.hi.vals[1] >= rect.hi.vals[1]:
                                # Overlaps with front left corners
                                if other.hi.vals[2] >= rect.hi.vals[2]:
                                    # Overlaps with all left corners
                                    lo = Point(3)
                                    lo.vals[0] = other.hi.vals[0]+1
                                    lo.vals[1] = rect.lo.vals[1]
                                    lo.vals[2] = rect.lo.vals[2]
                                    to_add.append(Rect(lo,rect.hi))
                                else:
                                    # Only overlaps with front left corners
                                    lo = Point(3)
                                    lo.vals[0] = other.hi.vals[0]+1
                                    lo.vals[1] = rect.lo.vals[1]
                                    lo.vals[2] = rect.lo.vals[2]
                                    to_add.append(Rect(lo,rect.hi))
                                    lo = Point(3)
                                    hi = Point(3)
                                    lo.vals[0] = rect.lo.vals[0]
                                    lo.vals[1] = rect.lo.vals[1]
                                    lo.vals[2] = other.hi.vals[2]+1
                                    hi.vals[0] = other.hi.vals[0]
                                    hi.vals[1] = rect.hi.vals[1]
                                    hi.vals[2] = rect.hi.vals[2]
                                    to_add.append(Rect(lo,hi))
                            else:
                                # Overlaps with front lower-left
                                if other.hi.vals[2] >= rect.hi.vals[2]:
                                    # Overlaps front and back lower-left corners
                                    lo = Point(3)
                                    lo.vals[0] = other.hi.vals[0]+1
                                    lo.vals[1] = rect.lo.vals[1]
                                    lo.vals[2] = rect.lo.vals[2]
                                    to_add.append(Rect(lo,rect.hi))
                                    lo = Point(3)
                                    hi = Point(3)
                                    lo.vals[0] = rect.lo.vals[0]
                                    lo.vals[1] = other.hi.vals[1]+1
                                    lo.vals[2] = rect.lo.vals[2]
                                    hi.vals[0] = other.hi.vals[0]
                                    hi.vals[1] = rect.hi.vals[1]
                                    hi.vals[2] = rect.hi.vals[2]
                                    to_add.append(Rect(lo,hi))
                                else:
                                    # Overlaps with just front lower-left
                                    lo = Point(3)
                                    lo.vals[0] = other.hi.vals[0]+1
                                    lo.vals[1] = rect.lo.vals[1]
                                    lo.vals[2] = rect.lo.vals[2]
                                    to_add.append(Rect(lo,rect.hi))
                                    lo = Point(3)
                                    hi = Point(3)
                                    lo.vals[0] = rect.lo.vals[0]
                                    lo.vals[1] = rect.lo.vals[1]
                                    lo.vals[2] = other.hi.vals[2]+1
                                    hi.vals[0] = other.hi.vals[0]
                                    hi.vals[1] = other.hi.vals[1]
                                    hi.vals[2] = rect.hi.vals[2]
                                    to_add.append(Rect(lo,hi))
                                    lo = Point(3)
                                    hi = Point(3)
                                    lo.vals[0] = rect.lo.vals[0]
                                    lo.vals[1] = other.hi.vals[1]+1
                                    lo.vals[2] = rect.lo.vals[2]
                                    hi.vals[0] = other.hi.vals[0]
                                    hi.vals[1] = rect.hi.vals[1]
                                    hi.vals[2] = rect.hi.vals[2]
                                    to_add.append(Rect(lo,hi))
                    elif other.hi.vals[2] >= rect.hi.vals[2]:
                        #print("Back lower left corner")
                        # Back lower-left corner (no overlap with any front corners)
                        hi = Point(3)
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = other.lo.vals[2]-1
                        to_add.append(Rect(rect.lo,hi))
                        if other.hi.vals[0] >= rect.hi.vals[0]:
                            # Overlap with back lower-right
                            if other.hi.vals[1] >= rect.hi.vals[1]:
                                # Overlap with back face so we're done
                                pass
                            else:
                                # Overlap with back lower corners
                                lo = Point(3)
                                lo.vals[0] = rect.lo.vals[0]
                                lo.vals[1] = other.hi.vals[1]+1
                                lo.vals[2] = other.lo.vals[2]
                                to_add.append(Rect(lo,rect.hi))
                        else:
                            # No overlaps with back right corners 
                            if other.hi.vals[1] >= rect.hi.vals[1]:
                                # Overlap with back upper-left
                                lo = Point(3)
                                lo.vals[0] = other.hi.vals[0]+1
                                lo.vals[1] = rect.lo.vals[1]
                                lo.vals[2] = other.lo.vals[2]
                                to_add.append(Rect(lo,rect.hi))
                            else:
                                # Just back lower-left corner
                                lo = Point(3)
                                lo.vals[0] = other.hi.vals[0]+1
                                lo.vals[1] = rect.lo.vals[1]
                                lo.vals[2] = other.lo.vals[2]
                                to_add.append(Rect(lo,rect.hi))
                                lo = Point(3)
                                hi = Point(3)
                                lo.vals[0] = rect.lo.vals[0]
                                lo.vals[1] = other.hi.vals[1]+1
                                lo.vals[2] = other.lo.vals[2]
                                hi.vals[0] = other.hi.vals[0]
                                hi.vals[1] = rect.hi.vals[1]
                                hi.vals[2] = rect.hi.vals[2]
                                to_add.append(Rect(lo,hi))
                    else:
                        #print("Lower left z edge")
                        # Lower-left z edge
                        hi = Point(3)
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = other.lo.vals[2]-1
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = other.hi.vals[2]+1
                        to_add.append(Rect(lo,rect.hi))
                        if other.hi.vals[0] >= rect.hi.vals[0]:
                            if other.hi.vals[1] >= rect.hi.vals[1]:
                                # cutting plane through z so we're done
                                pass
                            else:
                                # only bottom half cutting plane
                                lo = Point(3)
                                hi = Point(3)
                                lo.vals[0] = rect.lo.vals[0]
                                lo.vals[1] = other.hi.vals[1]+1
                                lo.vals[2] = other.lo.vals[2]
                                hi.vals[0] = rect.hi.vals[0]
                                hi.vals[1] = rect.hi.vals[1]
                                hi.vals[2] = other.hi.vals[2]
                                to_add.append(Rect(lo,hi))
                        else:
                            if other.hi.vals[1] >= rect.hi.vals[1]:
                                # only left half cutting plane
                                lo = Point(3)
                                hi = Point(3)
                                lo.vals[0] = other.hi.vals[0]+1
                                lo.vals[1] = rect.lo.vals[1]
                                lo.vals[2] = other.lo.vals[2]
                                hi.vals[0] = rect.hi.vals[0]
                                hi.vals[1] = rect.hi.vals[1]
                                hi.vals[2] = other.hi.vals[2]
                                to_add.append(Rect(lo,hi))
                            else:
                                # only lower-left z edge
                                lo = Point(3)
                                hi = Point(3)
                                lo.vals[0] = rect.lo.vals[0]
                                lo.vals[1] = other.hi.vals[1]+1
                                lo.vals[2] = other.lo.vals[2]
                                hi.vals[0] = rect.hi.vals[0]
                                hi.vals[1] = rect.hi.vals[1]
                                hi.vals[2] = other.hi.vals[2]
                                to_add.append(Rect(lo,hi))
                                lo = Point(3)
                                hi = Point(3)
                                lo.vals[0] = other.hi.vals[0]+1
                                lo.vals[1] = rect.lo.vals[1]
                                lo.vals[2] = other.lo.vals[2]
                                hi.vals[0] = rect.hi.vals[0]
                                hi.vals[1] = other.hi.vals[1]
                                hi.vals[2] = other.hi.vals[2]
                                to_add.append(Rect(lo,hi))
                elif other.hi.vals[1] >= rect.hi.vals[1]:
                    if other.lo.vals[2] <= rect.lo.vals[2]:
                        #print("Front upper left corner")
                        # Front upper-left corner 
                        # No overlap with any lower corners
                        if other.hi.vals[0] >= rect.hi.vals[0]:
                            if other.hi.vals[2] >= rect.hi.vals[2]:
                                # Overlap with whole top
                                hi = Point(3)
                                hi.vals[0] = rect.hi.vals[0]
                                hi.vals[1] = other.lo.vals[1]-1
                                hi.vals[2] = rect.hi.vals[2]
                                to_add.append(Rect(rect.lo,hi))
                            else:
                                # Overlap with front upper corners
                                hi = Point(3)
                                hi.vals[0] = rect.hi.vals[0]
                                hi.vals[1] = other.lo.vals[1]-1
                                hi.vals[2] = rect.hi.vals[2]
                                to_add.append(Rect(rect.lo,hi))
                                lo = Point(3)
                                lo.vals[0] = rect.lo.vals[0]
                                lo.vals[1] = other.lo.vals[1]
                                lo.vals[2] = other.hi.vals[2]+1
                                to_add.append(Rect(lo,rect.hi))
                        else:
                            if other.hi.vals[2] >= rect.hi.vals[2]:
                                # Overlap with left upper corners 
                                hi = Point(3)
                                hi.vals[0] = rect.hi.vals[0]
                                hi.vals[1] = other.lo.vals[1]-1
                                hi.vals[2] = rect.hi.vals[2]
                                to_add.append(Rect(rect.lo,hi))
                                lo = Point(3)
                                lo.vals[0] = other.hi.vals[0]+1
                                lo.vals[1] = other.lo.vals[1]
                                lo.vals[2] = rect.lo.vals[2]
                                to_add.append(Rect(lo,rect.hi))
                            else:
                                # Overlap with just front upper-left
                                hi = Point(3)
                                hi.vals[0] = rect.hi.vals[0]
                                hi.vals[1] = other.lo.vals[1]-1
                                hi.vals[2] = rect.hi.vals[2]
                                to_add.append(Rect(rect.lo,hi))
                                lo = Point(3)
                                lo.vals[0] = other.hi.vals[0]+1
                                lo.vals[1] = other.lo.vals[1]
                                lo.vals[2] = rect.lo.vals[2]
                                to_add.append(Rect(lo,rect.hi))
                                lo = Point(3)
                                hi = Point(3)
                                lo.vals[0] = rect.lo.vals[0]
                                lo.vals[1] = other.lo.vals[1]
                                lo.vals[2] = other.hi.vals[2]+1
                                hi.vals[0] = other.hi.vals[0]
                                hi.vals[1] = rect.hi.vals[1]
                                hi.vals[2] = rect.hi.vals[2]
                                to_add.append(Rect(lo,hi))
                    elif other.hi.vals[2] >= rect.hi.vals[2]:
                        #print("Back upper left corner")
                        # Back upper-left corner (can overlap with back upper right)
                        hi = Point(3)
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = other.lo.vals[2]-1
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = other.lo.vals[2]
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        if other.hi.vals[0] < rect.hi.vals[0]:
                            # No overlap with back upper right corner
                            lo = Point(3)
                            lo.vals[0] = other.hi.vals[0]+1
                            lo.vals[1] = other.lo.vals[1]
                            lo.vals[2] = other.lo.vals[2]
                            to_add.append(Rect(lo,rect.hi))
                    else:
                        #print("Upper left z edge")
                        # Upper-left z edge
                        # No overlap with lower-left z edge
                        hi = Point(3)
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = other.lo.vals[2]-1
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = other.hi.vals[2]+1
                        to_add.append(Rect(lo,rect.hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = other.lo.vals[2]
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        hi.vals[2] = other.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        if other.hi.vals[0] >= rect.hi.vals[0]:
                            # Overlap with upper-right z edge so we are done 
                            pass
                        else:
                            # Just upper-left z edge
                            lo = Point(3)
                            hi = Point(3)
                            lo.vals[0] = other.hi.vals[0]+1
                            lo.vals[1] = other.lo.vals[1]
                            lo.vals[2] = other.lo.vals[2]
                            hi.vals[0] = rect.hi.vals[0]
                            hi.vals[1] = rect.hi.vals[1]
                            hi.vals[2] = other.hi.vals[2]
                            to_add.append(Rect(lo,hi))
                else:
                    if other.lo.vals[2] <= rect.lo.vals[2]:
                        #print("Front left y edge")
                        # Front-left y edge
                        hi = Point(3)
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = other.hi.vals[1]+1
                        lo.vals[2] = rect.lo.vals[2]
                        to_add.append(Rect(lo,rect.hi))
                        if other.hi.vals[0] >= rect.hi.vals[0]:
                            if other.hi.vals[2] >= rect.hi.vals[2]:
                                # Cutting plane, so we are done
                                pass
                            else:
                                # Only cuts front y edges
                                lo = Point(3)
                                hi = Point(3)
                                lo.vals[0] = rect.lo.vals[0]
                                lo.vals[1] = other.lo.vals[1]
                                lo.vals[2] = other.hi.vals[2]+1
                                hi.vals[0] = rect.hi.vals[0]
                                hi.vals[1] = other.hi.vals[1]
                                hi.vals[2] = rect.hi.vals[2]
                                to_add.append(Rect(lo,hi))
                        else:
                            if other.hi.vals[2] >= rect.hi.vals[2]:
                                # Cuts left y edges
                                lo = Point(3)
                                hi = Point(3)
                                lo.vals[0] = other.hi.vals[0]+1
                                lo.vals[1] = other.lo.vals[1]
                                lo.vals[2] = rect.lo.vals[2]
                                hi.vals[0] = rect.hi.vals[0]
                                hi.vals[1] = other.hi.vals[1]
                                hi.vals[2] = rect.hi.vals[2]
                                to_add.append(Rect(lo,hi))
                            else:
                                # Only cuts front left y edge
                                lo = Point(3)
                                hi = Point(3)
                                lo.vals[0] = rect.lo.vals[0]
                                lo.vals[1] = other.lo.vals[1]
                                lo.vals[2] = other.hi.vals[2]+1
                                hi.vals[0] = rect.hi.vals[0]
                                hi.vals[1] = other.hi.vals[1]
                                hi.vals[2] = rect.hi.vals[2]
                                to_add.append(Rect(lo,hi))
                                lo = Point(3)
                                hi = Point(3)
                                lo.vals[0] = other.hi.vals[0]+1
                                lo.vals[1] = other.lo.vals[1]
                                lo.vals[2] = rect.lo.vals[2]
                                hi.vals[0] = rect.hi.vals[0]
                                hi.vals[1] = other.hi.vals[1]
                                hi.vals[2] = other.hi.vals[2]
                                to_add.append(Rect(lo,hi))
                    elif other.hi.vals[2] >= rect.hi.vals[2]:
                        #print("Back left y edge")
                        # Back-left y edge, no intersect with front-left
                        hi = Point(3)
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = other.hi.vals[1]+1
                        lo.vals[2] = rect.lo.vals[2]
                        to_add.append(Rect(lo,rect.hi))
                        if other.hi.vals[0] >= rect.hi.vals[0]:
                            # Interset with both back edges
                            lo = Point(3)
                            hi = Point(3)
                            lo.vals[0] = rect.lo.vals[0]
                            lo.vals[1] = other.lo.vals[1]
                            lo.vals[2] = rect.lo.vals[2]
                            hi.vals[0] = rect.hi.vals[0]
                            hi.vals[1] = other.hi.vals[1]
                            hi.vals[2] = other.lo.vals[2]-1
                            to_add.append(Rect(lo,hi))
                        else:
                            # Just back-left y edge (same as above + 1 more)
                            lo = Point(3)
                            hi = Point(3)
                            lo.vals[0] = rect.lo.vals[0]
                            lo.vals[1] = other.lo.vals[1]
                            lo.vals[2] = rect.lo.vals[2]
                            hi.vals[0] = rect.hi.vals[0]
                            hi.vals[1] = other.hi.vals[1]
                            hi.vals[2] = other.lo.vals[2]-1
                            to_add.append(Rect(lo,hi))
                            lo = Point(3)
                            hi = Point(3)
                            lo.vals[0] = other.hi.vals[0]+1
                            lo.vals[1] = other.lo.vals[1]
                            lo.vals[2] = other.lo.vals[2]
                            hi.vals[0] = rect.hi.vals[0]
                            hi.vals[1] = other.hi.vals[1]
                            hi.vals[2] = rect.hi.vals[2]
                            to_add.append(Rect(lo,hi))
                    else: 
                        #print("Left face")
                        # Left face
                        hi = Point(3)
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = other.lo.vals[2]-1
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = other.hi.vals[2]+1
                        to_add.append(Rect(lo,rect.hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = other.lo.vals[2]
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        hi.vals[2] = other.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = other.hi.vals[1]+1
                        lo.vals[2] = other.lo.vals[2]
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = other.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        # See if we intersect with right face
                        if other.hi.vals[0] < rect.hi.vals[0]:
                            # No intersect so one more rectangle
                            lo = Point(3)
                            hi = Point(3)
                            lo.vals[0] = other.hi.vals[0]+1
                            lo.vals[1] = other.lo.vals[1]
                            lo.vals[2] = other.lo.vals[2]
                            hi.vals[0] = rect.hi.vals[0]
                            hi.vals[1] = other.hi.vals[1]
                            hi.vals[2] = other.hi.vals[2]
                            to_add.append(Rect(lo,hi))
            elif other.hi.vals[0] >= rect.hi.vals[0]:
                if other.lo.vals[1] <= rect.lo.vals[1]:
                    if other.lo.vals[2] <= rect.lo.vals[2]:
                        #print("Front lower-right corner")
                        # Front lower-right corner
                        hi = Point(3)
                        hi.vals[0] = other.lo.vals[0]-1
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(rect.lo,hi))
                        # No overlap with front lower-left corner
                        if other.hi.vals[1] >= rect.hi.vals[1]:
                            if other.hi.vals[2] >= rect.hi.vals[2]:
                                # Overlap with full right face so we are done
                                pass 
                            else:
                                # Just front right corners
                                lo = Point(3)
                                lo.vals[0] = other.lo.vals[0]
                                lo.vals[1] = rect.lo.vals[1]
                                lo.vals[2] = other.hi.vals[2]+1
                                to_add.append(Rect(lo,rect.hi))
                        else:
                            if other.hi.vals[2] >= rect.hi.vals[2]:
                                # Lower-right corners only
                                lo = Point(3)
                                lo.vals[0] = other.lo.vals[0]
                                lo.vals[1] = other.hi.vals[1]+1
                                lo.vals[2] = rect.lo.vals[2]
                                to_add.append(Rect(lo,rect.hi))
                            else:
                                # Just front lower-right corner
                                lo = Point(3)
                                lo.vals[0] = other.lo.vals[0]
                                lo.vals[1] = other.hi.vals[1]+1
                                lo.vals[2] = rect.lo.vals[2]
                                to_add.append(Rect(lo,rect.hi))
                                lo = Point(3)
                                hi = Point(3)
                                lo.vals[0] = other.lo.vals[0]
                                lo.vals[1] = rect.lo.vals[1]
                                lo.vals[2] = other.hi.vals[2]+1
                                hi.vals[0] = rect.hi.vals[0]
                                hi.vals[1] = other.hi.vals[1]
                                hi.vals[2] = rect.hi.vals[2]
                                to_add.append(Rect(lo,hi))
                    elif other.hi.vals[2] >= rect.hi.vals[2]:
                        #print("Back lower-right corner")
                        # Back lower-right corner
                        # No intersection with front lower-right
                        # of back lower-left
                        hi = Point(3)
                        hi.vals[0] = other.lo.vals[0]-1
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = other.lo.vals[2]-1
                        to_add.append(Rect(lo,hi))
                        if other.hi.vals[1] < rect.hi.vals[1]:
                            # No intersect with back upper right
                            lo = Point(3)
                            lo.vals[0] = other.lo.vals[0]
                            lo.vals[1] = other.hi.vals[1]+1
                            lo.vals[2] = other.lo.vals[2]
                            to_add.append(Rect(lo,rect.hi))
                    else:
                        #print("Lower right z edge")
                        # Lower-right z edge
                        # No intersect with lower-left z-edge
                        hi = Point(3)
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = other.lo.vals[2]-1
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = other.hi.vals[2]+1
                        to_add.append(Rect(lo,rect.hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = other.lo.vals[2]
                        hi.vals[0] = other.lo.vals[0]-1
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = other.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        # See if we intersect the upper right z edge
                        if other.hi.vals[1] < rect.hi.vals[1]:
                            # No intersect so we have another box
                            lo = Point(3)
                            hi = Point(3)
                            lo.vals[0] = other.lo.vals[0]
                            lo.vals[1] = other.hi.vals[1]+1
                            lo.vals[2] = other.lo.vals[2]
                            hi.vals[0] = rect.hi.vals[0]
                            hi.vals[1] = rect.hi.vals[1]
                            hi.vals[2] = other.hi.vals[2]
                            to_add.append(Rect(lo,hi))
                elif other.hi.vals[1] >= rect.hi.vals[1]:
                    if other.lo.vals[2] <= rect.lo.vals[2]:
                        #print("Front upper right corner")
                        # Front upper-right corner
                        # No intersect with other front corners
                        hi = Point(3)
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = other.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = other.lo.vals[0]-1
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        # See if we intersect the back-right corner 
                        if other.hi.vals[2] < rect.hi.vals[2]:
                            # no intersect, so we have an extra rectangle
                            lo = Point(3)
                            lo.vals[0] = other.lo.vals[0]
                            lo.vals[1] = other.lo.vals[1]
                            lo.vals[2] = other.hi.vals[2]+1
                            to_add.append(Rect(lo,rect.hi))
                    elif other.hi.vals[2] >= rect.hi.vals[2]:
                        #print("Back upper right corner")
                        # Back upper-right corner
                        # No intersect with any other corners
                        hi = Point(3)
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = other.lo.vals[2]-1
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = other.lo.vals[2]
                        hi.vals[0] = other.lo.vals[0]-1
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = other.lo.vals[2]
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                    else:
                        #print("Upper right z edge")
                        # Upper-right z edge
                        # No intersect with other z edges
                        hi = Point(3)
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = other.lo.vals[2]-1
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = other.hi.vals[2]+1
                        to_add.append(Rect(lo,rect.hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = other.lo.vals[2]
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        hi.vals[2] = other.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = other.lo.vals[1]
                        lo.vals[2] = other.lo.vals[2]
                        hi.vals[0] = other.lo.vals[0]-1
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = other.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                else:
                    if other.lo.vals[2] <= rect.lo.vals[2]:
                        #print("Front right y edge")
                        # Front-right y edge
                        # No intersect with front-left y edge
                        hi = Point(3)
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = other.hi.vals[1]+1
                        lo.vals[2] = rect.lo.vals[2]
                        to_add.append(Rect(lo,rect.hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = other.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = other.lo.vals[0]-1
                        hi.vals[1] = other.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        # See if it intersects with back-right y edge
                        if other.hi.vals[2] < rect.hi.vals[2]:
                            # No intersect so another rectangle
                            lo = Point(3)
                            hi = Point(3)
                            lo.vals[0] = other.lo.vals[0]
                            lo.vals[1] = other.lo.vals[1]
                            lo.vals[2] = other.hi.vals[2]+1
                            hi.vals[0] = rect.hi.vals[0]
                            hi.vals[1] = other.hi.vals[1]
                            hi.vals[2] = rect.hi.vals[2]
                            to_add.append(Rect(lo,hi))
                    elif other.hi.vals[2] >= rect.hi.vals[2]:
                        #print("Back right y edge")
                        # Back-right y edge
                        # No intesect with other y edges
                        hi = Point(3)
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = other.hi.vals[1]+1
                        lo.vals[2] = rect.lo.vals[2]
                        to_add.append(Rect(lo,rect.hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = other.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = other.lo.vals[0]-1
                        hi.vals[1] = other.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = other.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = other.hi.vals[1]
                        hi.vals[2] = other.lo.vals[2]-1
                        to_add.append(Rect(lo,hi))
                    else:
                        #print("Right face")
                        # Right face
                        # No intersect with left face
                        hi = Point(3)
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = other.lo.vals[2]-1
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = other.hi.vals[2]+1
                        to_add.append(Rect(lo,rect.hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = other.lo.vals[2]
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        hi.vals[2] = other.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = other.hi.vals[1]+1
                        lo.vals[2] = other.lo.vals[2]
                        hi.vals[0] = rect.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = other.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = rect.lo.vals[0]
                        lo.vals[1] = other.lo.vals[1]
                        lo.vals[2] = other.lo.vals[2]
                        hi.vals[0] = other.lo.vals[0]-1
                        hi.vals[1] = other.hi.vals[1]
                        hi.vals[2] = other.hi.vals[2]
                        to_add.append(Rect(lo,hi))
            else:
                if other.lo.vals[1] <= rect.lo.vals[1]:
                    if other.lo.vals[2] <= rect.lo.vals[2]:
                        #print("Front bottom x edge")
                        # Front bottom x edge
                        hi = Point(3)
                        hi.vals[0] = other.lo.vals[0]-1
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = other.hi.vals[0]+1
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        to_add.append(Rect(lo,rect.hi))
                        if other.hi.vals[1] >= rect.hi.vals[1]:
                            if other.hi.vals[2] >= rect.hi.vals[2]:
                                # Cutting plane so we are done
                                pass
                            else:
                                # Intersects with top x edge but not back
                                lo = Point(3)
                                hi = Point(3)
                                lo.vals[0] = other.lo.vals[0]
                                lo.vals[1] = rect.lo.vals[1]
                                lo.vals[2] = other.hi.vals[2]+1
                                hi.vals[0] = other.hi.vals[0]
                                hi.vals[1] = rect.hi.vals[1]
                                hi.vals[2] = rect.hi.vals[2]
                                to_add.append(Rect(lo,hi))
                        else:
                            if other.hi.vals[2] >= rect.hi.vals[2]:
                                # Intersects with back x edge
                                lo = Point(3)
                                hi = Point(3)
                                lo.vals[0] = other.lo.vals[0]
                                lo.vals[1] = other.hi.vals[1]+1
                                lo.vals[2] = rect.lo.vals[2]
                                hi.vals[0] = other.hi.vals[0]
                                hi.vals[1] = rect.hi.vals[1]
                                hi.vals[2] = rect.hi.vals[2]
                                to_add.append(Rect(lo,hi))
                            else:
                                # No intersection with any other edges
                                lo = Point(3)
                                hi = Point(3)
                                lo.vals[0] = other.lo.vals[0]
                                lo.vals[1] = other.hi.vals[1]+1
                                lo.vals[2] = rect.lo.vals[2]
                                hi.vals[0] = other.hi.vals[0]
                                hi.vals[1] = rect.hi.vals[1]
                                hi.vals[2] = rect.hi.vals[2]
                                to_add.append(Rect(lo,hi))
                                lo = Point(3)
                                hi = Point(3)
                                lo.vals[0] = other.lo.vals[0]
                                lo.vals[1] = rect.lo.vals[1]
                                lo.vals[2] = other.hi.vals[2]+1
                                hi.vals[0] = other.hi.vals[0]
                                hi.vals[1] = other.hi.vals[1]
                                hi.vals[2] = rect.hi.vals[2]
                                to_add.append(Rect(lo,hi))
                    elif other.hi.vals[2] >= rect.hi.vals[2]:
                        #print("Back bottom x edge")
                        # Back bottom x edge
                        # No intersection with front bottom x edge
                        hi = Point(3)
                        hi.vals[0] = other.lo.vals[0]-1
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = other.hi.vals[0]+1
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        to_add.append(Rect(lo,rect.hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = other.lo.vals[2]-1
                        to_add.append(Rect(lo,hi))
                        if other.hi.vals[1] < rect.hi.vals[1]:
                            # No intersect so we have another rect to add
                            lo = Point(3)
                            hi = Point(3)
                            lo.vals[0] = other.lo.vals[0]
                            lo.vals[1] = other.hi.vals[1]+1
                            lo.vals[2] = other.lo.vals[2]
                            hi.vals[0] = other.hi.vals[0]
                            hi.vals[1] = rect.hi.vals[1]
                            hi.vals[2] = rect.hi.vals[2]
                            to_add.append(Rect(lo,hi))
                    else:
                        #print("Bottom face")
                        # Bottom face
                        hi = Point(3)
                        hi.vals[0] = other.lo.vals[0]-1
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = other.hi.vals[0]+1
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        to_add.append(Rect(lo,rect.hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = other.lo.vals[2]-1
                        to_add.append(Rect(lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = other.hi.vals[2]+1
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        # See if it intersects with the top face
                        if other.hi.vals[1] < rect.hi.vals[1]:
                            # No intersect so we have another rect
                            lo = Point(3)
                            hi = Point(3)
                            lo.vals[0] = other.lo.vals[0]
                            lo.vals[1] = other.hi.vals[1]+1
                            lo.vals[2] = other.lo.vals[2]
                            hi.vals[0] = other.hi.vals[0]
                            hi.vals[1] = rect.hi.vals[1]
                            hi.vals[2] = other.hi.vals[2]
                            to_add.append(Rect(lo,hi))
                elif other.hi.vals[1] >= rect.hi.vals[1]:
                    if other.lo.vals[2] <= rect.lo.vals[2]:
                        #print("Front top x edge")
                        # Front top x edge
                        # No overlap with bottom x edges
                        hi = Point(3)
                        hi.vals[0] = other.lo.vals[0]-1
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = other.hi.vals[0]+1
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        to_add.append(Rect(lo,rect.hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        # See if we intersect with the back top x edge
                        if other.hi.vals[2] < rect.hi.vals[2]:
                            # No intersect so there is another rect
                            lo = Point(3)
                            hi = Point(3)
                            lo.vals[0] = other.lo.vals[0]
                            lo.vals[1] = other.lo.vals[1]
                            lo.vals[2] = other.hi.vals[2]+1
                            hi.vals[0] = other.hi.vals[0]
                            hi.vals[1] = rect.hi.vals[1]
                            hi.vals[2] = rect.hi.vals[2]
                            to_add.append(Rect(lo,hi))
                    elif other.hi.vals[2] >= rect.hi.vals[2]:
                        #print("Back top x edge")
                        # Back top x edge
                        # Cannot intersect with any other edges
                        hi = Point(3)
                        hi.vals[0] = other.lo.vals[0]-1
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = other.hi.vals[0]+1
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        to_add.append(Rect(lo,rect.hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = other.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = other.lo.vals[2]-1
                        to_add.append(Rect(lo,hi))
                    else:
                        #print("Top face")
                        # Top face
                        # No intersect with bottom face
                        hi = Point(3)
                        hi.vals[0] = other.lo.vals[0]-1
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = other.hi.vals[0]+1
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        to_add.append(Rect(lo,rect.hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = other.lo.vals[2]-1
                        to_add.append(Rect(lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = other.hi.vals[2]+1
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = other.lo.vals[2]
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        hi.vals[2] = other.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                else:
                    if other.lo.vals[2] <= rect.lo.vals[2]:
                        #print("Front face")
                        # Front face
                        hi = Point(3)
                        hi.vals[0] = other.lo.vals[0]-1
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = other.hi.vals[0]+1
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        to_add.append(Rect(lo,rect.hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = other.hi.vals[1]+1
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        # See if it intersects with the back face
                        if other.hi.vals[2] < rect.hi.vals[2]:
                            # No intersect with the back so add another rect
                            lo = Point(3)
                            hi = Point(3)
                            lo.vals[0] = other.lo.vals[0]
                            lo.vals[1] = other.lo.vals[1]
                            lo.vals[2] = other.hi.vals[2]+1
                            hi.vals[0] = other.hi.vals[0]
                            hi.vals[1] = other.hi.vals[1]
                            hi.vals[2] = rect.hi.vals[2]
                            to_add.append(Rect(lo,hi))
                    elif other.hi.vals[2] >= rect.hi.vals[2]:
                        #print("Back face")
                        # Back face
                        # No intersect with front face
                        hi = Point(3)
                        hi.vals[0] = other.lo.vals[0]-1
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = other.hi.vals[0]+1
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        to_add.append(Rect(lo,rect.hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = other.hi.vals[1]+1
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = other.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = other.hi.vals[1]
                        hi.vals[2] = other.lo.vals[2]-1
                        to_add.append(Rect(lo,hi))
                    else:
                        #print("Center")
                        # Center
                        hi = Point(3)
                        hi.vals[0] = other.lo.vals[0]-1
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(rect.lo,hi))
                        lo = Point(3)
                        lo.vals[0] = other.hi.vals[0]+1
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        to_add.append(Rect(lo,rect.hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = rect.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = other.lo.vals[1]-1
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = other.hi.vals[1]+1
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = rect.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = other.lo.vals[1]
                        lo.vals[2] = rect.lo.vals[2]
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = other.hi.vals[1]
                        hi.vals[2] = other.lo.vals[2]-1
                        to_add.append(Rect(lo,hi))
                        lo = Point(3)
                        hi = Point(3)
                        lo.vals[0] = other.lo.vals[0]
                        lo.vals[1] = other.lo.vals[1]
                        lo.vals[2] = other.hi.vals[2]+1
                        hi.vals[0] = other.hi.vals[0]
                        hi.vals[1] = other.hi.vals[1]
                        hi.vals[2] = rect.hi.vals[2]
                        to_add.append(Rect(lo,hi))
        else:
            # This is the recursive method for dimensions >3
            if dim == rect.lo.dim:
                # Base case, check to see if the rectangle is contained or not
                # rect should either be totally separate or totally contained
                if other.intersects(rect):
                    assert other.dominates(rect)
                else:
                    to_add.append(rect)
            else:
                # Figure out how to break this rectangle along this dimension
                if other.lo.vals[dim] <= rect.lo.vals[dim]:
                    if other.hi.vals[dim] < rect.hi.vals[dim]:
                        # Dominate lower edge, two outputs
                        hi = Point(rect.hi.dim)
                        for d in xrange(rect.hi.dim):
                            if d == dim:
                                hi.vals[d] = other.hi.vals[d]
                            else:
                                hi.vals[d] = rect.hi.vals[d]
                        cls.rect_sub_helper(Rect(rect.lo,hi), other, to_add, dim+1)
                        lo = Point(rect.lo.dim)
                        for d in xrange(rect.lo.dim):
                            if d == dim:
                                lo.vals[d] = other.hi.vals[d]+1
                            else:
                                lo.vals[d] = rect.lo.vals[d]
                        cls.rect_sub_helper(Rect(lo,rect.hi), other, to_add, dim+1)
                    else:
                        # Dominate both edges, one output
                        cls.rect_sub_helper(rect, other, to_add, dim+1)
                elif other.hi.vals[dim] >= rect.hi.vals[dim]:
                    # Dominate upper edge, two outputs
                    hi = Point(rect.hi.dim)
                    for d in xrange(rect.hi.dim):
                        if d == dim:
                            hi.vals[d] = other.lo.vals[d]-1
                        else:
                            hi.vals[d] = rect.hi.vals[d]
                    cls.rect_sub_helper(Rect(rect.lo,hi), other, to_add, dim+1)
                    lo = Point(rect.lo.dim)
                    for d in xrange(rect.lo.dim):
                        if d == dim:
                            lo.vals[d] = other.lo.vals[d]
                        else:
                            lo.vals[d] = rect.lo.vals[d]
                    cls.rect_sub_helper(Rect(lo,rect.hi), other, to_add, dim+1)
                else:
                    # No domination, three outputs
                    hi = Point(rect.hi.dim) 
                    for d in xrange(rect.hi.dim):
                        if d == dim:
                            hi.vals[d] = other.lo.vals[d]-1
                        else:
                            hi.vals[d] = rect.hi.vals[d]
                    cls.rect_sub_helper(Rect(rect.lo,hi), other, to_add, dim+1)
                    lo = Point(rect.lo.dim)
                    hi = Point(rect.hi.dim)
                    for d in xrange(rect.lo.dim):
                        if d == dim:
                            lo.vals[d] = other.lo.vals[d]
                            hi.vals[d] = other.hi.vals[d]
                        else:
                            lo.vals[d] = rect.lo.vals[d]
                            hi.vals[d] = rect.hi.vals[d]
                    cls.rect_sub_helper(Rect(lo,hi), other, to_add, dim+1)
                    lo = Point(rect.lo.dim)
                    for d in xrange(rect.lo.dim):
                        if d == dim:
                            lo.vals[d] = other.hi.vals[d]+1
                        else:
                            lo.vals[d] = rect.lo.vals[d]
                    cls.rect_sub_helper(Rect(lo,rect.hi), other, to_add, dim+1)

    def __isub__(self, other):
        for orect in other.rects:
            to_remove = list()
            to_add = list()
            for rect in self.rects:
                if rect.intersects(orect):
                    to_remove.append(rect)
                    # See if it dominates:
                    if orect.dominates(rect):
                        continue
                    else:
                        self.rect_sub_helper(rect, orect, to_add)
            if to_remove:
                for rect in to_remove:
                    self.rects.remove(rect)
            if to_add:
                for rect in to_add:
                    self.rects.add(rect)
        for point in other.points:
            prect = Rect(point,point)
            to_remove = list()
            to_add = list()
            for rect in self.rects:
                if rect.intersects(prect):
                    to_remove.append(rect)
                    # See if it dominates:
                    if prect.dominates(rect):
                        continue
                    else:
                        self.rect_sub_helper(rect, prect, to_add)
            if to_remove:
                for rect in to_remove:
                    self.rects.remove(rect)
            if to_add:
                for rect in to_add:
                    self.rects.add(rect)
        # Now remove any of our points
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

class PointSet(object):
    __slots__ = ['points', 'space_string']
    def __init__(self):
        self.points = set()
        self.space_string = None

    def __str__(self):
        return ' '.join(map(str, self.points))

    __repr__ = __str__

    def add_point(self, point):
        self.points.add(point)

    def empty(self):
        return not self.points

    def has_point(self, point):
        for p in self.points:
            if p == point:
                return True
        return False

    def copy(self):
        result = PointSet()
        for point in self.points:
            result.points.add(point)
        return result

    def __contains__(self, point):
        if not self.points:
            return False
        return point in self.points

    def __len__(self):
        return len(self.points)

    def __nonzero__(self):
        return len(self.points) > 0

    def __eq__(self, other):
        if len(self.points) != len(other.points):
            return False
        for point in self.points:
            if point not in other.points:
                return False
        return True

    def __neq__(self, other):
        return not self == other

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
            to_remove.append(point)
        for point in to_remove:
            self.points.remove(point)
        return self

    # Set union
    def __or__(self, other):
        result = self.copy()
        result |= other
        return result

    def __ior__(self, other):
        for point in other.points:
            self.points.add(point)
        return self

    # Set substraction
    def __sub__(self, other):
        result = self.copy()
        result -= other
        return result

    def __isub__(self, other):
        for point in other.points:
            if point in self.points:
                self.points.remove(point)
        return self

    def iterator(self):
        for p in self.points:
            yield p

    def point_space_graphviz_string(self):
        if self.space_string != None:
            return self.space_string
        # Easy case if this empty
        if len(self) == 0:
            self.space_string = 'Empty'
            return self.space_string
        # Check to see if there are any index spaces which
        # appear in all the points. If so, check to see if 
        # our set of points is equivalent to the point set
        # for any of those index spaces
        spaces_for_all_points = None
        for point in self.iterator():
            if spaces_for_all_points is None:
                spaces_for_all_points = point.index_set
            else:
                # Note that this makes a new set
                spaces_for_all_points = spaces_for_all_points & point.index_set
            if not spaces_for_all_points:
                break
        if spaces_for_all_points:
            same_spaces = None
            for space in spaces_for_all_points:
                # Check to see if the points are the same
                other_points = space.get_point_set()
                if len(self) != len(other_points):
                    continue
                same = True
                for point in self.iterator():
                    if not other_points.has_point(point):
                        same = False
                        break
                if same:
                    # It's possible for there to be multiple spaces with 
                    # the same shape but different names
                    if same_spaces is None:
                        same_spaces = list()
                    same_spaces.append(space)
            if same_spaces:
                for space in same_spaces:
                    if self.space_string is None:
                        self.space_string = str(space)
                    else:
                        self.space_string += ' u ' + str(space)
        # If we didn't find an index space to represent this
        # set of points then we express this as a union of intersections
        if self.space_string is None:
            for point in self.iterator():
                point_str = None
                # Only report the bottom-most index spaces for 
                # this particular point
                bottom_spaces = set()
                parent_spaces = set()
                for space in point.index_set:
                    if space in parent_spaces:
                        continue
                    bottom_spaces.add(space)
                    if space.parent:
                        parent = space.parent.parent
                        while True:
                            if parent in parent_spaces:
                                break
                            parent_spaces.add(parent)
                            if parent in bottom_spaces:
                                bottom_spaces.remove(parent)
                            if parent.parent:
                                parent = parent.parent.parent
                            else:
                                break
                # Should have at least one bottom space
                assert bottom_spaces
                for space in bottom_spaces:
                    if point_str is None:
                        point_str = '(' + str(space)
                    else:
                        point_str += ' ^ ' + str(space)
                point_str += ')'
                if self.space_string is None:
                    self.space_string = point_str
                else:
                    # Put each point on it's own line
                    self.space_string += ' u <br/>' + point_str
        assert self.space_string is not None
        self.space_string = self.space_string.replace('<','&lt;').replace('>','&gt;').replace('&','&amp;')
        return self.space_string

class NodeSet(object):
    __slots__ = ['impl', 'bound', 'sparse']
    def __init__(self, bound):
        self.impl = None
        self.bound = bound + 31 // 32
        assert self.bound > 0
        self.sparse = True

    def densify(self):
        assert self.impl
        assert self.sparse
        # Cross-over to dense
        new_impl = array.array('I', (0,) * self.bound)
        for node in self.impl:
            index = node >> 5
            offset = node & 31
            mask = 1 << offset
            new_impl[index] |= mask
        self.impl = new_impl
        self.sparse = False

    def add(self, node):
        assert node < 32 * self.bound
        if self.impl is None:
            self.impl = set()
            self.impl.add(node)
        elif self.sparse:
            self.impl.add(node)
            # Check for cross-over to dense
            if len(self.impl) > 32:
                self.densify()
        else:
            index = node >> 5
            offset = node & 31
            mask = 1 << offset
            self.impl[index] |= mask

    def union(self, other):
        assert other.bound == self.bound
        if other.impl is None:
            return
        if self.impl is None:
            if other.sparse:
                self.impl = other.impl.copy()
            else:
                self.impl = array.array('I', other.impl)
                self.sparse = False
        elif self.sparse:
            if other.sparse:
                for node in other.impl:
                    self.impl.add(node)
                if len(self.impl) > 32:
                    self.densify()
            else:
                new_impl = array.array('I', other.impl)
                for node in self.impl:
                    index = node >> 5
                    offset = node & 31
                    mask = 1 << offset
                    new_impl[index] |= mask
                self.impl = new_impl
                self.sparse = False
        else:
            if other.sparse:
                for node in other.impl:
                    index = node >> 5
                    offset = node & 31
                    mask = 1 << offset
                    self.impl[index] |= mask
            else:
                for index in xrange(self.bound):
                    self.impl[index] |= other.impl[index]

    def contains(self, node):
        assert node < 32 * self.bound
        if self.impl is None:
            return False
        elif self.sparse:
            return node in self.impl
        else:
            index = node >> 5
            offset = node & 31
            mask = 1 << offset
            return self.impl[index] & mask != 0

class Processor(object):
    __slots__ = ['state', 'uid', 'kind_num', 'kind', 'mem_latency', 
                 'mem_bandwidth', 'node_name']
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.kind_num = None
        self.kind = "Unknown"
        self.mem_latency = dict()
        self.mem_bandwidth = dict()
        self.node_name = 'proc_node_%s' % uid

    def __str__(self):
        return self.kind + " Processor " + hex(self.uid)

    def set_kind(self, kind_num, kind):
        self.kind_num = kind_num
        self.kind = kind

    def add_memory(self, mem, bandwidth, latency):
        self.mem_latency[mem] = latency
        self.mem_bandwidth[mem] = bandwidth

    def print_node(self, printer):
        label = self
        printer.println(
            '%s [label="%s",shape=oval,fontsize=14,fontcolor=black,fontname="Helvetica"];' %
            (self.node_name, label))

    def print_mem_edges(self, printer):
        for mem,band in iteritems(self.mem_bandwidth):
            label = 'bandwidth=%s,latency=%s' % (band, self.mem_latency[mem])
            printer.println(
                '%s -> %s [label="%s",style=solid,color=black,penwidth=2];' %
                (self.node_name, mem.node_name, label))

class Memory(object):
    __slots__ = ['state', 'uid', 'kind_num', 'kind', 'capacity', 'proc_latency',
                 'proc_bandwidth', 'mem_latency', 'mem_bandwidth', 'node_name']
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.kind_num = None
        self.kind = "Unknown"
        self.capacity = -1
        self.proc_latency = dict()
        self.proc_bandwidth = dict()
        self.mem_latency = dict()
        self.mem_bandwidth = dict()
        self.node_name = 'mem_node_%s' % uid

    def __str__(self):
        return "%s Memory %s" % (self.kind, hex(self.uid))

    __repr__ = __str__

    def set_kind(self, kind_num, kind):
        self.kind_num = kind_num
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
        label = self
        printer.println(
            '%s [label="%s",shape=box,fontsize=14,fontcolor=black,fontname="Helvetica"];' %
            (self.node_name, label))

    def print_mem_edges(self, printer):
        for mem,band in iteritems(self.mem_bandwidth):
            label = 'bandwidth=%s,latency=%s' % (band, self.mem_latency[mem])
            printer.println(
                '%s -> %s [label="%s",style=solid,color=black,penwidth=2];' %
                (self.node_name, mem.node_name, label))

class IndexExpr(object):
    __slot__ = ['state', 'expr_id', 'kind', 'base', 'point_set', 'node_name', 'expr_str']
    def __init__(self, state, expr_id):
        self.state = state
        self.expr_id = expr_id
        self.kind = None
        self.base = None
        self.point_set = None
        self.node_name = None
        self.expr_str = None

    def set_index_space(self, index_space):
        if self.kind is None:
            self.kind = INDEX_SPACE_EXPR
            self.base = index_space
        elif self.kind == INDEX_SPACE_EXPR:
            # If they aren't the same index space then
            # check that they have the same set of points
            if self.base is not index_space:
                one = self.base.get_point_set()
                two = index_space.get_point_set()
                diff = one - two
                assert len(one) == len(two) and len(diff) == 0

    def get_index_space(self):
        if self.kind == INDEX_SPACE_EXPR:
            return self.base
        else:
            return None

    def add_union_expr(self, union_expr):
        if self.kind != UNION_EXPR:
            assert self.kind is None
            self.kind = UNION_EXPR
            assert self.base is None
            self.base = set()
        self.base.add(union_expr)

    def add_intersect_expr(self, inter_expr):
        if self.kind != INTERSECT_EXPR:
            assert self.kind is None
            self.kind = INTERSECT_EXPR
            assert self.base is None
            self.base = set()
        self.base.add(inter_expr)

    def set_diff_expr(self, left, right):
        assert self.kind is None
        self.kind = DIFFERENCE_EXPR
        self.base = list()
        self.base.append(left)
        self.base.append(right)

    def get_point_set(self):
        if self.point_set is None:
            assert self.kind is not None
            if self.kind == INDEX_SPACE_EXPR:
                self.point_set = self.base.get_point_set().copy()
            elif self.kind == UNION_EXPR:
                for expr in self.base:
                    if self.point_set is None:
                        self.point_set = expr.get_point_set().copy()
                    else:
                        self.point_set |= expr.get_point_set()
            elif self.kind == INTERSECT_EXPR:
                for expr in self.base:
                    if self.point_set is None:
                        self.point_set = expr.get_point_set().copy()
                    else:
                        self.point_set &= expr.get_point_set()
            else:
                assert self.kind == DIFFERENCE_EXPR
                for expr in self.base:
                    if self.point_set is None:
                        self.point_set = expr.get_point_set().copy()
                    else:
                        self.point_set -= expr.get_point_set()
        return self.point_set

    def point_space_graphviz_string(self):
        if self.kind == INDEX_SPACE_EXPR:
            return str(self)
        else:
            point_set = self.get_point_set()
            assert point_set
            return point_set.point_space_graphviz_string()

    def __str__(self):
        if self.expr_str is not None:
            return self.expr_str
        assert self.kind is not None
        if self.kind == INDEX_SPACE_EXPR:
            self.expr_str = str(self.base)
        elif self.kind == UNION_EXPR:
            for expr in self.base:
                if self.expr_str is None:
                    self.expr_str = '(' + str(expr)
                else:
                    self.expr_str += ' u ' + str(expr)
            self.expr_str += ')'
        elif self.kind == INTERSECT_EXPR:
            for expr in self.base:
                if self.expr_str is None:
                    self.expr_str = '(' + str(expr)
                else:
                    self.expr_str += ' ^ ' + str(expr)
            self.expr_str += ')'
        else:
            assert self.kind == DIFFERENCE_EXPR
            self.expr_str = '(' + str(self.base[0]) + ' - ' + str(self.base[1]) + ')'
        assert self.expr_str
        return self.expr_str

    __repr__ = __str__

class IndexSpace(object):
    __slots__ = ['state', 'uid', 'parent', 'color', 'children', 
                 'instances', 'name', 'independent_children',
                 'depth', 'shape', 'point_set', 'node_name', 'owner',
                 'intersections', 'dominated', 'expr', 'provenance']
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
        self.point_set = None
        self.node_name = 'index_space_node_%s' % uid
        self.intersections = dict() 
        self.dominated = dict()
        self.expr = None
        self.owner = None
        self.provenance = None

    def set_name(self, name):
        self.name = name

    def set_expr(self, expr):
        assert self.expr is None or self.expr == expr
        self.expr = expr

    def set_parent(self, parent, color):
        self.parent = parent
        self.color = color
        self.parent.add_child(self)

    def get_index_space(self):
        return self

    def update_depth(self, parent_depth):
        self.depth = parent_depth + 1
        for child in itervalues(self.children):
            child.update_depth(self.depth)

    def add_child(self, child):
        self.children[child.color] = child

    def add_instance(self, tid, region):
        self.instances[tid] = region

    def add_point(self, point):
        if self.shape is None:
            self.shape = Shape()
            self.shape.add_point(point.copy())
        else:
            update = Shape()
            update.add_point(point.copy())
            update -= self.shape
            if not update.empty():
                self.shape |= update

    def add_rect(self, rect):
        if self.shape is None:
            self.shape = Shape()
            self.shape.add_rect(rect.copy())
        else:
            update = Shape()
            update.add_rect(rect.copy())
            update -= self.shape
            if not update.empty():
                self.shape |= update

    def update_index_sets(self, index_sets, done, total_spaces):
        done += 1
        if self.shape is None or self.shape.empty():
            if not self.state.verbose:
                print_progress_bar(done, total_spaces, length=50)
            return done
        if self.state.verbose:
            print('    Reducing index sub-space %s' % self)
        local_points = self.shape.copy()
        new_sets = dict()
        del_sets = list()
        for shape,index_set in iteritems(index_sets):
            intersect = local_points & shape
            # No overlap so keep going
            if intersect.empty():
                continue
            # Now see if the intersect split the initial shape
            residual = shape - intersect
            local_points -= intersect
            if residual.empty():
                index_set.add(self)
                if local_points.empty():
                    # Both empty means that this is an exact match
                    break
                # Else we keep going 
            else:
                # Need to split
                new_sets[residual] = index_set.copy()
                index_set.add(self)
                new_sets[intersect] = index_set
                del_sets.append(shape)
                if local_points.empty():
                    # We were dominated so we're done now
                    break
        # Remove the deleted entries
        for key in del_sets:
            del index_sets[key]
        # Add the new entries
        for shape,index_set in iteritems(new_sets):
            index_sets[shape] = index_set
        # If we had left over points, add them as a new set
        if not local_points.empty():
            index_set = set()
            index_set.add(self)
            index_sets[local_points] = index_set
        if not self.state.verbose:
            print_progress_bar(done, total_spaces, length=50)
        # Traverse the children
        for child in itervalues(self.children):
            done = child.update_index_sets(index_sets, done, total_spaces)
        return done

    def add_refined_point(self, point):
        if self.point_set is None:
            self.point_set = PointSet()
        self.point_set.add_point(point.copy())

    def set_empty(self):
        assert self.shape is None or self.shape.empty()
        self.shape = Shape()
        self.point_set = PointSet()

    def __str__(self):
        if self.name is None:
            if self.parent is None:
                return "Index Space %s" % self.uid
            else:
                return "Index Subspace %s" % self.uid
        else:
          return '%s (%s)' % (self.name, self.uid)

    __repr__ = __str__

    @property
    def html_safe_name(self):
        name = str(self)
        provenance = self.get_provenance()
        if provenance is not None and len(provenance) > 0:
            name = name + " [" + provenance + "]"
        return name.replace('<','&lt;').replace('>','&gt;').replace('&','&amp;')

    def check_partition_properties(self):
        # Check all the partitions
        for child in itervalues(self.children):
            child.check_partition_properties()

    def compute_reduced_shapes(self, dim_sets, done, total_spaces):
        if self.shape is None or self.shape.empty():
            if not self.state.verbose:
                print_progress_bar(done+1, total_spaces, length=50)
            return done + 1
        if self.state.verbose:
            print('Reducing %s ...' % self)
        if self.shape.get_dim() not in dim_sets:
            dim_sets[self.shape.get_dim()] = dict()
        return self.update_index_sets(dim_sets[self.shape.get_dim()], done, total_spaces)

    def are_all_children_disjoint(self):
        return False

    def are_children_disjoint(self, c1, c2):
        if c1.intersects(c2):
            return False
        return True

    def get_shape(self):
        if self.shape is None:
            print("No shape for %s" % self)
            return
        assert self.shape is not None
        return self.shape

    def get_point_set(self):
        if self.point_set is None:
            self.point_set = PointSet()
        return self.point_set

    def intersection(self, other):
        if self is other:
            return self.get_point_set()
        if other in self.intersections:
            return self.intersections[other]
        intersection = self.get_point_set() & other.get_point_set()
        if intersection.empty():
            self.intersections[other] = None
            return None
        self.intersections[other] = intersection
        return intersection

    def intersects(self, other):
        if self is other:
            return True
        return self.intersection(other) is not None

    def dominates(self, other):
        if self is other:
            return True
        if other in self.dominated:
            return self.dominated[other]
        non_dominated = other.get_point_set() - self.get_point_set()
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

    def get_provenance(self):
        if self.parent is not None:
            return self.parent.provenance
        else:
            return self.provenance

    def print_graph(self, printer):
        label = self.html_safe_name
        if self.parent is not None:
            color = None
            for c, child in iteritems(self.parent.children):
                if child == self:
                    color = c
                    break
            assert color is not None
            label += ' (color: %s)' % color
        if self.owner is not None:
            label += '\nOwner Node: ' + str(self.owner)
        if self.shape is None or self.shape.empty():
            label += '\nEmpty Bounds'
        else:
            if self.shape.dense:
                label += '\nDense Bounds: '
            else:
                label += '\nSparse Bounds: '
            lo,hi = self.shape.bounds
            label += lo.to_string()+' - '+hi.to_string()
        printer.println('%s [label="%s",shape=plaintext,fontsize=14,fontcolor=black,fontname="Helvetica"];' %
                        (self.node_name, label))
        # print links to children
        for child in itervalues(self.children):
            child.print_link_to_parent(printer, self.node_name)
        for child in itervalues(self.children):
            child.print_graph(printer)

    def print_tree(self):
        if self.depth == 0:
            print("---------------------------------------")
            print(self)
        else:
            prefix = ''
            for i in xrange(self.depth):
                prefix += '  '
            print('%s%s Color: %s' % (prefix, self, self.color.to_string()))
        for child in itervalues(self.children):
            child.print_tree()
        if self.depth == 0:
            print("---------------------------------------")

class IndexPartition(object):
    __slots__ = ['state', 'uid', 'parent', 'color', 'children', 'instances', 
                 'disjoint', 'complete', 'name', 'depth', 'shape', 'point_set',
                 'node_name', 'intersections', 'dominated', 'owner', 'provenance']
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
        self.point_set = None
        self.node_name = 'index_part_node_%s' % uid
        self.intersections = dict()
        self.dominated = dict()
        self.owner = None
        self.provenance = None

    def set_parent(self, parent, color):
        self.parent = parent
        self.color = color
        self.parent.add_child(self)

    def update_depth(self, parent_depth):
        self.depth = parent_depth + 1
        for child in itervalues(self.children):
            child.update_depth(self.depth)

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
            return "Index Partition: %s" % self.uid
        else:
            return '%s (%s)' % (self.name, self.uid)

    __repr__ = __str__

    @property
    def html_safe_name(self):
        name = str(self)
        if self.provenance is not None and len(self.provenance) > 0:
            name = name + ' [' + self.provenance + ']'
        return name.replace('<','&lt;').replace('>','&gt;').replace('&','&amp;')

    def check_partition_properties(self):
        # Check for dominance of children by parent
        for child in itervalues(self.children):
            if not self.parent.dominates(child):
                print(('WARNING: child %s is not dominated by parent %s in %s. '+
                      'This is definitely an application bug.') %
                      (child, self.parent, self))
                if self.state.assert_on_warning:
                    assert False
            # Recurse down the tree too
            child.check_partition_properties()
        # Check disjointness
        # We always compute disjointness and fill it in if
        # it wasnt' computed previously
        previous = PointSet()
        aliased = False
        for child in itervalues(self.children):
            child_shape = child.get_point_set()
            if not (child_shape & previous).empty():
                aliased = True
                if self.disjoint:
                    print(('ERROR: %s was labelled disjoint '+
                            'but there are overlapping children. This '+
                            'is definitely an application bug.') % self)
                    if self.state.assert_on_error:
                        assert False
                break
            previous |= child_shape
        if self.disjoint is not None:
            if not self.disjoint and not aliased:
                print(('WARNING: %s was labelled aliased '+
                        'but there are no overlapping children. This '+
                        'could lead to a performance bug.') % self)
        else:
            self.disjoint = not aliased
        if self.complete is not None:
            total = PointSet()
            for child in itervalues(self.children):
                total |= child.get_point_set()
            if self.complete:
                if len(total) != len(self.parent.get_point_set()):
                    print(('ERROR: %s was labelled complete '+
                            'but there are missing points. This '+
                            'is definitely an application bug.') % self)
                    if self.state.assert_on_error:
                        assert False
            else:
                if len(total) == len(self.parent.get_point_set()):
                    print(('WARNING: %s was labelled incomplete '+
                            'but actually covers all points. This '+
                            'could lead to a performance bug.') % self)

    def update_index_sets(self, index_sets, done, total_spaces):
        for child in itervalues(self.children):
            done = child.update_index_sets(index_sets, done, total_spaces)
        return done

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
            for child in itervalues(self.children):
                if self.shape is None:
                    self.shape = child.get_shape().copy()
                else:
                    self.shape |= child.get_shape()
        return self.shape

    def get_point_set(self):
        if self.point_set is None:
            for child in itervalues(self.children):
                if self.point_set is None:
                    self.point_set = child.get_point_set().copy()
                else:
                    self.point_set |= child.get_point_set()
            if self.point_set is None:
                self.point_set = PointSet() 
        return self.point_set

    def intersection(self, other):
        if self is other:
            return self.get_point_set()
        if other in self.intersections:
            return self.intersections[other]
        intersection = self.get_point_set() & other.get_point_set()
        if intersection.empty():
            self.intersections[other] = None
            return None 
        self.intersections[other] = intersection
        return intersection

    def intersects(self, other):
        if self is other:
            return True
        return self.intersection(other) is not None

    def dominates(self, other):
        if self is other:
            return True
        if other in self.dominated:
            return self.dominated[other]
        non_dominated = other.get_point_set() - self.get_point_set()
        if non_dominated.empty():
            self.dominated[other] = True
            return True
        else:
            self.dominated[other] = False
            return False

    def is_complete(self):
        if self.complete is None:
            # Figure out if this partition is complete or not 
            self.complete = (self.parent.get_point_set() - 
                             self.get_point_set()).empty()
        return self.complete

    def get_num_children(self):
        return len(self.children)

    def print_link_to_parent(self, printer, parent):
        printer.println(parent+' -> '+ self.node_name+
                ' [style=dotted,color=black,penwidth=2];')

    def print_graph(self, printer):
        label = self.html_safe_name
        color = None
        for c,child in iteritems(self.parent.children):
            if child == self:
                color = c
                break
        assert color is not None
        label += ' (color: %s)' % color
        if self.owner is not None:
            label += '\nOwner Node: ' + str(self.owner)
        label += '\nDisjoint=%s, Complete=%s' % (self.disjoint, self.is_complete())
        printer.println(
            '%s [label="%s",shape=plaintext,fontsize=14,fontcolor=black,fontname="times italic"];' %
            (self.node_name, label))
        # print links to children
        for child in itervalues(self.children):
            child.print_link_to_parent(printer, self.node_name)
        for child in itervalues(self.children):
            child.print_graph(printer)

    def print_tree(self):
        prefix = '  ' * self.depth
        print('%s%s Color: %s' % (prefix, self, self.color.to_string()))
        for child in itervalues(self.children):
            child.print_tree()

class Field(object):
    __slots__ = ['space', 'fid', 'size', 'name', 'provenance']
    def __init__(self, space, fid):
        self.space = space
        self.fid = fid
        self.size = None
        self.name = None
        self.provenance = None

    def set_name(self, name):
        self.name = name

    def __str__(self):
        if self.name is None:
            return "Field "+str(self.fid)
        else:
            return self.name + ' (' + str(self.fid) + ')'

    @property
    def html_safe_name(self):
        name = str(self)
        if self.provenance is not None and len(self.provenance) > 0:
            name = name + ' [' + self.provenance + ']'
        return name.replace('<','&lt;').replace('>','&gt;').replace('&','&amp;')

    __repr__ = __str__

class FieldSpace(object):
    __slots__ = ['state', 'uid', 'name', 'fields', 'node_name', 'owner', 'provenance']
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.name = None
        self.fields = dict()
        self.node_name = 'field_space_node_'+str(uid)
        self.owner = None
        self.provenance = None

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

    @property
    def html_safe_name(self):
        name = str(self)
        if self.provenance is not None and len(self.provenance) > 0:
            name = name + ' [' + self.provenance + ']'
        return name.replace('<','&lt;').replace('>','&gt;').replace('&','&amp;')

    def print_graph(self, printer):
        label = self.html_safe_name
        if self.owner is not None:
            label += '\nOwner Node: ' + str(self.owner)
        printer.println(self.node_name+' [label="'+label+
                '",shape=plaintext,fontsize=14,'+
                'fontcolor=black,fontname="Helvetica"];')

        for fid,field in iteritems(self.fields):
            field_id = "field_node_"+str(self.uid)+"_"+str(fid)
            provenance = field.provenance
            if provenance is not None and len(provenance) == 0:
                provenance = None
            if field.name is not None:
                if provenance is not None:
                    field_name = field.name + ' [' + provenance + '] (FID: ' + str(fid) + ')'
                else:
                    field_name = field.name + ' (FID: ' + str(fid) + ')'
            else:
                if provenance is not None:
                    field_name = 'FID: ' + str(fid) + ' [' + provenance + ']'
                else:
                    field_name = 'FID: ' + str(fid)
            printer.println(field_id+' [label="'+field_name+
                    '",shape=plaintext,fontsize=14,'+
                    'fontcolor=black,fontname="Helvetica"]')
            printer.println(self.node_name+' -> '+ field_id+
                    " [style=dotted,color=black,penwidth=2];")

class LogicalRegion(object):
    __slots__ = ['state', 'index_space', 'field_space', 'tree_id', 'children',
                 'name', 'parent', 'logical_state', 'verification_state', 
                 'node_name', 'has_named_children', 'owner', 'provenance']
    def __init__(self, state, iid, fid, tid):
        self.state = state
        self.index_space = iid
        self.field_space = fid
        self.tree_id = tid
        self.children = dict() 
        self.name = None
        self.parent = None
        self.logical_state = dict()
        self.verification_state = dict() # only for top-level regions
        self.index_space.add_instance(self.tree_id, self)
        self.node_name = 'region_node_'+str(self.index_space.uid)+\
            '_'+str(self.field_space.uid)+'_'+str(self.tree_id)
        self.has_named_children = False
        self.owner = None
        self.provenance = None

    def set_name(self, name):
        self.name = name

    def set_parent(self, parent):
        self.parent = parent
        self.parent.add_child(self.index_space.color, self)

    def add_child(self, color, child):
        self.children[color] = child 

    def has_all_children(self):
        return len(self.children) == len(self.index_space.children)

    def get_index_node(self):
        return self.index_space

    def update_parent(self):
        if not self.parent and self.index_space.parent is not None:
            self.set_parent(self.state.get_partition(
                self.index_space.parent.uid, self.field_space.uid, self.tree_id))

    @property
    def html_safe_name(self):
        name = str(self)
        if self.provenance is not None and len(self.provenance) > 0:
            name = name + ' [' + self.provenance + ']'
        return name.replace('<','&lt;').replace('>','&gt;').replace('&','&amp;')

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

    def get_point_set(self):
        return self.index_space.get_point_set()

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

    def reset_verification_state(self, depth):
        # Should be top-level region
        assert not self.parent
        if self.verification_state and depth in self.verification_state:
            self.verification_state[depth] = dict()

    def compute_path(self, path, target):
        if self is not target:
            assert self.parent is not None
            self.parent.compute_path(path, target)
        path.append(self)

    def perform_logical_analysis(self, depth, path, op, req, field, open_local, 
                                 unopened, advance, closed, prev, aliased, checks):
        assert self is path[depth]
        if field not in self.logical_state:
            self.logical_state[field] = LogicalState(self, field)
        arrived = (depth+1) == len(path)
        next_child = path[depth+1] if not arrived else None
        result,next_open,next_unopened,next_advance,next_closed = \
            self.logical_state[field].perform_logical_analysis(op, req, next_child, 
                            open_local, unopened, advance, closed, prev, aliased, checks)
        if not result:
            return False
        if not arrived:
            return path[depth+1].perform_logical_analysis(depth+1, path, op, req, 
                        field, next_open, next_unopened, next_advance, next_closed, 
                        prev, aliased, checks)
        return True

    def register_logical_user(self, op, req, field):
        if field not in self.logical_state:
            self.logical_state[field] = LogicalState(self, field)
        self.logical_state[field].register_logical_user(op, req)

    def perform_logical_deletion(self, depth, path, op, req, field, closed, prev, checks):
        assert self is path[depth]
        if field not in self.logical_state:
            return True
        arrived = (depth+1) == len(path)
        force_close = (depth+1) < len(path)
        next_child = path[depth+1] if not arrived else None
        result,next_closed = self.logical_state[field].perform_logical_deletion(op, req, 
                                            next_child, closed, prev, checks, force_close)
        if not result:
            return False
        if not arrived:
            return path[depth+1].perform_logical_deletion(depth+1, path, op, req, field,
                                                          next_closed, prev, checks)
        elif not checks:
            # Do all the invalidations and record any dependences
            self.perform_deletion_invalidation(op, req, field)
        return True

    def perform_deletion_invalidation(self, op, req, field):
        if field not in self.logical_state:
            return
        self.logical_state[field].perform_deletion_invalidation(op, req)
        for child in itervalues(self.children):
            child.perform_deletion_invalidation(op, req, field)

    def close_logical_tree(self, field, closed_users):
        if field not in self.logical_state:
            return
        self.logical_state[field].close_logical_tree(closed_users)

    def get_verification_state(self, depth, field, point):
        # Should always be at the root
        assert not self.parent
        if depth not in self.verification_state:
            self.verification_state[depth] = dict()
        field_point_dict = self.verification_state[depth]
        key = (field,point)
        if key not in field_point_dict:
            result = EquivalenceSet(self, depth, field, point)
            field_point_dict[key] = result
            return result
        return field_point_dict[key]

    def initialize_verification_state(self, depth, field, inst, restricted, point_set=None):
        if point_set is None:
            # First get the point set
            self.initialize_verification_state(depth, field, inst, restricted, self.get_point_set())
        elif self.parent:
            # Recurse up the tree to the root
            self.parent.parent.initialize_verification_state(depth, field, inst, restricted, point_set)
        else:
            # Then do the actual work 
            for point in point_set.iterator():
                state = self.get_verification_state(depth, field, point)
                state.initialize_verification_state(inst, restricted)

    def compute_current_version_numbers(self, depth, field, op, tree, point_set = None):
        if point_set is None:
            # First get the point set
            self.compute_current_version_numbers(depth, field, op, tree, 
                                                 self.get_point_set())
        elif self.parent:
            # Recurse up the tree to the root
            self.parent.parent.compute_current_version_numbers(depth, field, op,
                                                               tree, point_set)
        else:
            # Do the actual work
            for point in point_set.iterator():
                state = self.get_verification_state(depth, field, point)
                op.record_current_version(point, field, tree, state.version_number)

    def perform_fill_verification(self, depth, field, op, req, point_set=None):
        if point_set is None:
            # First get the point set
            return self.perform_fill_verification(depth, field, op, req,
                                                  self.get_point_set())
        elif self.parent:
            # Recurse up the tree to the root
            return self.parent.parent.perform_fill_verification(depth, field, op, 
                                                                req, point_set)
        else:
            # Do the actual work
            for point in point_set.iterator():
                state = self.get_verification_state(depth, field, point)
                if not state.perform_fill_verification(op, req):
                    return False
            return True

    def add_restriction(self, depth, field, op, req, inst, perform_checks, point_set=None):
        if point_set is None:
            # First get the point set
            return self.add_restriction(depth, field, op, req, inst,
                                        perform_checks, self.get_point_set())
        elif self.parent:
            # Recurse up the tree to the root
            return self.parent.parent.add_restriction(depth, field, op, req, inst,
                                                      perform_checks, point_set)
        else:
            # Do the actual work
            for point in point_set.iterator():
                state = self.get_verification_state(depth, field, point)
                if not state.add_restriction(op, req, inst):
                    return False
            return True

    def remove_restriction(self, depth, field, op, req, inst, perform_checks, point_set=None):
        if point_set is None:
            # First get the point set
            return self.remove_restriction(depth, field, op, req, inst, 
                                           perform_checks, self.get_point_set())
        elif self.parent:
            # Recurse up the tree to the root
            return self.parent.parent.remove_restriction(depth, field, op, req, inst,
                                                         perform_checks, point_set)
        else:
            # Do the actual work
            for point in point_set.iterator():
                state = self.get_verification_state(depth, field, point)
                if not state.remove_restriction(op, req, inst):
                    return False
            return True

    def perform_physical_verification(self, depth, field, op, req, inst, perform_checks, 
                                      perform_registration, point_set = None,
                                      version_numbers = None):
        if point_set is None:
            # First get the point set
            return self.perform_physical_verification(depth, field, op, req, inst,
              perform_checks, perform_registration, self.get_point_set(), version_numbers)
        elif self.parent:
            # Recurse up the tree to the root
            return self.parent.parent.perform_physical_verification(depth, field, op, req,
                   inst, perform_checks, perform_registration, point_set, version_numbers)
        else:
            # Do the actual work
            for point in point_set.iterator():
                state = self.get_verification_state(depth, field, point)
                if not state.perform_physical_verification(op, req, inst, 
                        perform_checks, perform_registration):
                    return False
                # Record the version numbers if necessary
                if version_numbers is not None:
                    version_numbers[point] = state.version_number
            return True

    def perform_verification_registration(self, depth, field, op, req, inst, 
                                          perform_checks, point_set=None): 
        if point_set is None:
            # First get the point set
            return self.perform_verification_registration(depth, field, op, req, inst,
                    perform_checks, self.get_point_set())
        elif self.parent:
            # Recurse up the tree to the root
            return self.parent.parent.perform_verification_registration(depth, field, 
                    op, req, inst, perform_checks, point_set)
        else:
            # Do the actual work
            for point in point_set.iterator():
                state = self.get_verification_state(depth, field, point)
                if not state.perform_verification_registration(op, req, inst, perform_checks):
                    return False
            return True

    def perform_copy_across_verification(self, op, redop, perform_checks, point_set,
                                         src_depth, src_field, src_req, src_inst,
                                         dst_depth, dst_field, dst_req, dst_inst,
                                         dst_versions):
        # Recurse up the tree until we get to the root
        if self.parent:
            return self.parent.parent.perform_copy_across_verification(op, redop,
                perform_checks, point_set, src_depth, src_field, src_req, src_inst,
                dst_depth, dst_field, dst_req, dst_inst, dst_versions)
        # Do the actual work
        for point in point_set.iterator():
            state = self.get_verification_state(src_depth, src_field, point)
            assert point in dst_versions
            if not state.perform_copy_across_verification(op, redop, 
                  perform_checks, src_depth, src_field, src_req, src_inst,
                  dst_depth, dst_field, dst_req, dst_inst, dst_versions[point]):
                return False
        return True

    def perform_indirect_copy_verification(self, op, redop, perform_checks, copies,
                                           point_set, depth, field, req, inst, versions):
        # Recurse up the tree until we get to the root
        if self.parent:
            return self.parent.parent.perform_indirect_copy_verification(op, redop,
                    perform_checks, copies, point_set, depth, field, req, inst, versions)
        # Do the actual work
        for point in point_set.iterator():
            state = self.get_verification_state(depth, field, point)
            if not state.perform_indirect_copy_verification(op, redop, perform_checks,
                    copies, depth, field, req, inst, versions):
                return False
        return True

    def mark_named_children(self):
        if self.name is not None:
            self.has_named_children = True
        for child in itervalues(self.children):
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
        provenance = self.provenance
        if provenance is not None and len(provenance) == 0:
            provenance = None
        if self.name is not None:
            if provenance is not None:
                label = self.name+' [' + provenance + '] ('+self.gen_id()+')'
            else:
                label = self.name+' ('+self.gen_id()+')'
        else:
            if self.parent is None:
                if provenance is not None:
                    label = 'Region ('+self.gen_id()+') [' + provenance + ']'
                else:
                    label = 'Region ('+self.gen_id()+')'
            else:
                assert provenance is None
                label = 'Subregion ('+self.gen_id()+')'
        if self.owner is not None:
            label += '\nOwner Node: ' + str(self.owner)
        shape = self.get_shape()
        if shape is None or shape.empty():
            label += '\nEmpty Bounds'
        else:
            if shape.dense:
                label += '\nDense Bounds: '
            else:
                label += '\nSparse Bounds: '
            lo,hi = shape.bounds
            label += lo.to_string()+' - '+hi.to_string()
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
            for color,child in iteritems(self.index_space.children):
                if color not in self.children:
                    self.state.get_partition(child.uid, self.field_space.uid, self.tree_id)
        # print links to children
        for child in itervalues(self.children):
            if not simplify_graph or child.has_named_children:
                child.print_link_to_parent(printer, self.node_name)
        for child in itervalues(self.children):
            child.print_graph(printer, simplify_graph)

    def print_tree(self):
        if self.index_space.depth == 0:
            print("---------------------------------------")
            print(self)
        else:
            prefix = ''
            for i in xrange(self.index_space.depth):
                prefix += '  '
            print('%s%s Color: %s' % (prefix, self, self.index_space.color.to_string()))
        for child in itervalues(self.children):
            child.print_tree()
        if self.index_space.depth == 0:
            print("---------------------------------------")

class LogicalPartition(object):
    __slots__ = ['state', 'index_partition', 'field_space', 'tree_id', 'children',
                 'name', 'parent', 'logical_state', 'node_name', 'has_named_children']
    def __init__(self, state, iid, fid, tid):
        self.state = state
        self.index_partition = iid
        self.field_space = fid
        self.tree_id = tid
        self.children = dict()
        self.name = None 
        self.parent = None
        self.logical_state = dict()
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

    def get_index_node(self):
        return self.index_partition

    def update_parent(self):
        if not self.parent:
            assert self.index_partition.parent
            self.set_parent(self.state.get_region(self.index_partition.parent.uid,
                                                  self.field_space.uid, self.tree_id))

    @property
    def html_safe_name(self):
        return str(self).replace('<','&lt;').replace('>','&gt;').replace('&','&amp;')

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

    def get_point_set(self):
        return self.index_partition.get_point_set()

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

    def compute_path(self, path, target):
        if self is not target:
            assert self.parent is not None
            self.parent.compute_path(path, target)
        path.append(self)

    def perform_logical_analysis(self, depth, path, op, req, field, open_local, 
                                  unopened, advance, closed, prev, aliased, checks):
        assert self is path[depth]
        if field not in self.logical_state:
            self.logical_state[field] = LogicalState(self, field)
        arrived = (depth+1) == len(path)
        next_child = path[depth+1] if not arrived else None
        result,next_open,next_unopened,next_advance,next_closed = \
          self.logical_state[field].perform_logical_analysis(op, req, next_child, 
                          open_local, unopened, advance, closed, prev, aliased, checks)
        if not result:
            return False
        if not arrived:
            return path[depth+1].perform_logical_analysis(depth+1, path, op, req, 
                                    field, next_open, next_unopened, next_advance, 
                                    next_closed, prev, aliased, checks)
        return True

    def register_logical_user(self, op, req, field):
        if field not in self.logical_state:
            self.logical_state[field] = LogicalState(self, field)
        self.logical_state[field].register_logical_user(op, req)

    def perform_logical_deletion(self, depth, path, op, req, field, closed, prev, checks):
        assert self is path[depth]
        if field not in self.logical_state:
            return True
        arrived = (depth+1) == len(path)
        force_close = (depth+1) < len(path) 
        next_child = path[depth+1] if not arrived else None
        result,next_closed = self.logical_state[field].perform_logical_deletion(op, req, 
                                            next_child, closed, prev, checks, force_close)
        if not result:
            return False
        if not arrived:
            return path[depth+1].perform_logical_deletion(depth+1, path, op, req, field,
                                                          next_closed, prev, checks)
        elif not checks:
            # Do all the invalidations and record and dependences
            self.perform_deletion_invalidation(op, req, field)
        return True

    def perform_deletion_invalidation(self, op, req, field):
        if field not in self.logical_state:
            return
        self.logical_state[field].perform_deletion_invalidation(op, req)
        for child in itervalues(self.children):
            child.perform_deletion_invalidation(op, req, field)

    def close_logical_tree(self, field, closed_users):
        if field not in self.logical_state:
            return
        self.logical_state[field].close_logical_tree(closed_users)

    def compute_current_version_numbers(self, depth, field, op, tree):
        self.parent.compute_current_version_numbers(depth, field, op, 
                                                    tree, self.get_point_set())

    def perform_physical_verification(self, depth, field, op, req, inst, perform_checks,
                                      perform_registration):
        return self.parent.perform_physical_verification(depth, field, op, req, inst,
                perform_checks, perform_registration, self.get_point_set())

    def mark_named_children(self):
        if self.name is not None:
            self.has_named_children = True
        for child in itervalues(self.children):
            has_named_children = child.mark_named_children()
            self.has_named_children = self.has_named_children or has_named_children
        return self.has_named_children

    def print_link_to_parent(self, printer, parent):
        printer.println(parent+' -> '+ self.node_name+
                ' [style=dotted,color=black,penwidth=2];')

    def gen_id(self):
        return 'part: '+str(self.index_partition.uid)+','+\
                'field: '+str(self.field_space.uid)+','+\
                'tree: '+str(self.tree_id)

    def print_node(self, printer):
        if self.name is not None:
            label = self.name+' ('+self.gen_id() +')'
        else:
            label = 'partition ('+self.gen_id() +')'
        label += '\nDisjoint=' + ('True' if self.index_partition.disjoint else 'False')
        label += ', Complete=' + ('True' if self.index_partition.is_complete() else 'False')
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
            for color,child in iteritems(self.index_partition.children):
                if color not in self.children:
                    self.state.get_region(child.uid,
                        self.field_space.uid, self.tree_id)
        # print links to children
        for child in itervalues(self.children):
            child.print_link_to_parent(printer, self.node_name)
            if simplify_graph and not child.has_named_children:
                child.print_node(printer)
        for child in itervalues(self.children):
            child.print_graph(printer, simplify_graph)

    def print_tree(self):
        prefix = ''
        for i in xrange(self.index_partition.depth):
            prefix += '  '
        print('%s%s Color: %s' % (prefix, self, self.index_partition.color.to_string()))
        for child in itervalues(self.children):
            child.print_tree()

class LogicalState(object):
    __slots__ = ['node', 'field', 'open_children', 'open_redop',
                 'current_epoch_users', 'previous_epoch_users', 
                 'current_redop', 'dirty_below', 'projection_mode',
                 'projection_epoch']
    def __init__(self, node, field):
        self.node = node
        self.field = field
        self.open_children = dict() # Map from children to the mode they are open in
        self.open_redop = dict() # for open children reductions
        self.current_epoch_users = list()
        self.previous_epoch_users = list()
        self.current_redop = 0 # for reductions being done at this node
        self.dirty_below = False
        self.projection_mode = OPEN_NONE 
        self.projection_epoch = list()

    def perform_logical_analysis(self, op, req, next_child, open_local, unopened, advance, 
                                 closed, previous_deps, aliased_children, perform_checks):
        # At most one of these should be true, they can both be false
        assert not open_local or not unopened
        arrived = next_child is None
        # Figure out if we need to check close operations or not
        if unopened:
            if arrived and req.is_projection():
                passed,closed = self.siphon_logical_projection(op, req, closed,
                                                    previous_deps, perform_checks)
                if not passed:
                    return (False,None,None,None,closed)
            else:
                passed,closed = self.siphon_logical_children(op, req, next_child, 
                            closed, previous_deps, aliased_children, perform_checks)
                if not passed:
                    return (False,None,None,None,closed)
        # Perform any open operations if necessary
        if next_child or req.is_projection():
            next_open = self.open_state(op, req, next_child)
        # If we have an advance op, perform its analysis
        # otherwise see if we need to make one
        if advance:
            # Mark that we are now dirty below
            assert not self.dirty_below
            self.dirty_below = True
        elif req.has_write() and not self.dirty_below and \
                          (not arrived or (req.is_projection() and
                            (not req.is_reg or req.projection_function.depth > 0))):
            # Mark that we need to record this dirty below from now on
            advance = True
            # Mark that we are dirty below
            if not arrived or req.is_projection():
                self.dirty_below = True
        # Now do our analysis to figure out who we need to wait on locally
        if not self.perform_epoch_analysis(op, req, perform_checks,
                                           arrived, previous_deps):
            return (False,None,None,None,closed)
        if arrived: 
            # Add ourselves as the current user
            self.register_logical_user(op, req)
            # Record if we have outstanding reductions
            if req.redop != 0:
                self.current_redop = req.redop                
            return (True,None,None,None,closed)
        else:
            next_unopened = unopened and not next_open 
            return (True,next_open and unopened,next_unopened,advance,closed)

    def register_logical_user(self, op, req):
        self.current_epoch_users.append((op,req))

    def perform_logical_deletion(self, op, req, next_child, already_closed, 
                                 previous_deps, perform_checks, force_close):
        arrived = next_child is None
        if not arrived:
            result,already_closed = self.siphon_logical_deletion(op, req, next_child, 
                            already_closed, previous_deps, perform_checks, force_close)
            if not result:
                return False,already_closed
            if not self.perform_epoch_analysis(op, req, perform_checks, 
                                               arrived, previous_deps):  
                return False,already_closed
        return True,already_closed

    def perform_deletion_invalidation(self, op, req):
        dummy_previous = list()
        self.perform_epoch_analysis(op, req, False, False, dummy_previous)
        self.open_children = dict()
        self.open_redop = dict()
        self.current_redop = 0
        self.dirty_below = False
        self.projection_mode = OPEN_NONE
        self.projection_epoch = list()

    def open_state(self, op, req, next_child):
        # Then figure out how to open our child if necessary  
        if next_child is not None:
            assert self.projection_mode == OPEN_NONE
            next_open = next_child not in self.open_children
            if req.is_read_only():
                if next_child in self.open_children:
                    # Only need to change it if it was in reduce mode
                    if self.open_children[next_child] == OPEN_SINGLE_REDUCE:
                        self.open_children[next_child] = OPEN_READ_WRITE
                        del self.open_redop[next_child]
                    elif self.open_children[next_child] == OPEN_MULTI_REDUCE:
                        self.open_children[next_child] = OPEN_READ_ONLY
                        del self.open_redop[next_child]
                else:
                    self.open_children[next_child] = OPEN_READ_ONLY
            elif req.is_reduce():
                # See how many interfering children there are in the same mode
                other_children = set()
                for child,redop in iteritems(self.open_redop):
                    if redop != req.redop:
                        continue
                    if self.node.are_children_disjoint(child,next_child):
                        continue
                    if child is next_child:
                        continue
                    other_children.add(child)
                if other_children:
                    # Everyone is in multi-reduce mode
                    self.open_children[next_child] = OPEN_MULTI_REDUCE
                    for child in other_children:
                        assert self.open_children[child] == OPEN_SINGLE_REDUCE or \
                            self.open_children[child] == OPEN_MULTI_REDUCE
                        self.open_children[child] = OPEN_MULTI_REDUCE
                else:
                    # Just single reduce mode
                    self.open_children[next_child] = OPEN_SINGLE_REDUCE
                assert req.redop != 0
                self.open_redop[next_child] = req.redop
            else:
                # Normal read-write case is easy
                self.open_children[next_child] = OPEN_READ_WRITE
        else:
            assert req.is_projection()
            assert not self.open_children
            next_open = False
            if req.is_read_only():
                if self.projection_mode == OPEN_NONE:
                    self.projection_mode = OPEN_READ_ONLY
                else:
                    assert self.projection_mode == OPEN_READ_WRITE or \
                           self.projection_mode == OPEN_READ_ONLY
            elif req.is_reduce() and self.projection_mode == OPEN_NONE:
                self.projection_mode = OPEN_MULTI_REDUCE
                self.current_redop = req.redop
            else:
                self.projection_mode = OPEN_READ_WRITE
            self.projection_epoch.append((req.projection_function, 
                                          op.get_index_launch_shape()))
        return next_open

    # Maybe not the most intuitive name for a method but it aligns with the runtime
    def siphon_logical_children(self, op, req, next_child, already_closed,
                                previous_deps, aliased_children, perform_checks):
        closed = False
        # First see if we have any reductions to flush
        if self.projection_mode != OPEN_NONE:
            assert not self.open_children # shouldn't have any open children
            empty_children_to_close = dict()
            if self.projection_mode == OPEN_READ_ONLY:
                # We need a close if req is not read only or there is a next child
                if not req.is_read_only() or next_child:
                    closed = True
                    if not self.perform_close_operation(empty_children_to_close,
                                            True, False, op, req, already_closed, 
                                            previous_deps, perform_checks):
                        return False,closed
                    self.projection_mode = OPEN_NONE
            elif self.projection_mode == OPEN_READ_WRITE:
                # We close this no matter what
                closed = True
                if not self.perform_close_operation(empty_children_to_close,
                                            False, False, op, req, already_closed,
                                            previous_deps, perform_checks):
                    return False,closed
                self.projection_mode = OPEN_NONE
            else:
                assert self.projection_mode == OPEN_MULTI_REDUCE
                assert self.current_redop != 0
                if not req.is_reduce() or next_child or \
                            req.redop != self.current_redop: 
                    closed = True
                    if not self.perform_close_operation(empty_children_to_close,
                                            False, False, op, req, already_closed,
                                            previous_deps, perform_checks):
                        return False,closed
                    self.projection_mode = OPEN_NONE
        elif self.current_redop != 0 and self.current_redop != req.redop:
            children_to_close = set()
            # Flushing reductions close all children no matter what
            for child,open_mode in iteritems(self.open_children):
                children_to_close.add(child)
            # If we are flushing reductions we do a close no matter what
            closed = True
            if not self.perform_close_operation(children_to_close, False, False,
                                                op, req, already_closed,
                                                previous_deps, perform_checks):
                return False,closed
        elif next_child is None or not self.node.are_all_children_disjoint():
            # Figure out which children we need to do closes for
            need_close = False
            need_read_only_close = False
            upgrade_child = False
            depth = self.node.get_index_node().depth
            # Not flushing reductions so we can take the normal path
            for child,open_mode in iteritems(self.open_children):
                if open_mode == OPEN_READ_ONLY:
                    # Both read-only we can keep going
                    if req.is_read_only():
                        continue
                    if next_child is not None:
                        # Same child we can keep going
                        if next_child is child and not depth in aliased_children:
                            upgrade_child = True # not read-only requires upgrade 
                            continue
                        # Disjoint children then we can keep going
                        if self.node.are_children_disjoint(child, next_child):
                            continue
                    # Otherwise, we have to read-only close this child
                    # Keep going to see if we need a real close
                    need_read_only_close = True
                elif open_mode == OPEN_READ_WRITE:
                    if next_child is not None:
                        # Same child we can skip this
                        if next_child is child and not depth in aliased_children:
                            continue
                        # If we are disjoint we can keep going
                        if self.node.are_children_disjoint(child, next_child):
                            continue
                    # Otherwise we do an actual close here
                    need_close = True
                    break
                elif open_mode == OPEN_SINGLE_REDUCE:
                    # If they are the same reduction operator we can skip
                    if req.redop == self.open_redop[child]:
                        continue
                    if next_child is not None:
                        # Same child we can skip this
                        if next_child is child and not depth in aliased_children:
                            upgrade_child = True
                            continue
                        # If we are disjoint we can keep going
                        if self.node.are_children_disjoint(child, next_child):
                            continue
                    # Otherwise we need to close this child
                    need_close = True
                    break
                elif open_mode == OPEN_MULTI_REDUCE:
                    # If they are the same reduction operator we can skip
                    if req.redop == self.open_redop[child]:
                        continue
                    # Otherwise we definitely need the close operation
                    need_close = True
                    break
                else:
                    assert False # Should never get here
            if need_close:
                # Figure out the children to close
                # Full closes have to close everybody 
                children_to_close = dict() 
                # If we're going to do a write discard then
                # this can be a read only close, but only if
                # the operation is not predicated and it dominates
                overwrite = req.priv == WRITE_DISCARD and not op.predicate and \
                               req.logical_node.dominates(self.node)
                for child,open_mode in iteritems(self.open_children):
                    if open_mode == OPEN_READ_ONLY:                
                        children_to_close[child] = False
                    elif open_mode == OPEN_READ_WRITE:
                        # Can leave open if closer is a reader
                        children_to_close[child] = req.is_read_only()
                    elif open_mode == OPEN_SINGLE_REDUCE:
                        children_to_close[child] = False
                    elif open_mode == OPEN_MULTI_REDUCE:
                        children_to_close[child] = False
                    else:
                        assert False
                closed = True
                if not self.perform_close_operation(children_to_close, False,
                                            overwrite, op, req, already_closed,
                                            previous_deps, perform_checks):
                    return False,closed
                # No upgrades if we closed it
                upgrade_child = False
            elif need_read_only_close:
                children_to_read_close = dict()
                # Read only closes can close specific children
                for child,open_mode in iteritems(self.open_children):
                    # Only read-only matters for read-only closes
                    if open_mode != OPEN_READ_ONLY:
                        continue
                    if child is next_child:
                        # If we can keep it open we will, but not
                        # if it is going to have aliased children
                        if depth not in aliased_children:
                          continue   
                        # Otherwise we're going to close it so no more upgrades
                        upgrade_child = False
                    elif next_child is not None and \
                            self.node.are_children_disjoint(child, next_child):
                        continue
                    children_to_read_close[child] = False
                closed = True
                if not self.perform_close_operation(children_to_read_close,
                                        True, False, op, req, already_closed, 
                                        previous_deps, perform_checks):
                    return False,closed
            if upgrade_child:
                assert next_child
                assert next_child in self.open_children
                self.open_children[next_child] = OPEN_READ_WRITE
        else:
            # All children are disjoint so no closes necessary
            # See if we need to upgrade our child
            if next_child and next_child in self.open_children:
                if self.open_children[next_child] == OPEN_READ_ONLY:
                    if not req.is_read_only():
                        self.open_children[next_child] = OPEN_READ_WRITE
                elif self.open_children[next_child] == OPEN_SINGLE_REDUCE:
                    if not req.is_reduce() or req.redop != self.open_redop[next_child]:
                        self.open_children[next_child] = OPEN_READ_WRITE
                else: # Should be in read-write mode
                    assert self.open_children[next_child] == OPEN_READ_WRITE
        return True,closed

    def siphon_logical_projection(self, op, req, already_closed,
                                  previous_deps, perform_checks):
        assert req.is_projection()
        closed = False
        if self.projection_mode != OPEN_NONE:
            assert not self.open_children # Should not have any open children
            empty_children_to_close = dict()
            # Only get to do a disjoint close if we're either going to be
            # in read-write shallow mode
            disjoint_close = isinstance(self.node, LogicalPartition) and \
                self.node.are_all_children_disjoint() and \
                (req.is_read_only() or req.projection_function.depth == 0)
            if self.projection_mode == OPEN_READ_ONLY:
                # No need to do anything here, we're just 
                # going to change projection modes
                pass
            elif self.projection_mode == OPEN_READ_WRITE:
                # Figure out if we are in shallow disjoint mode or not
                # We can still be in shallow disjoint if we are reading
                # no matter what the projection depth
                shallow_disjoint = req.projection_function.depth == 0 or req.is_read_only()
                # Quick pass if we are disjoint shallow and have a
                # read-only user, then there is nothing we need to do
                same_func_and_shape = True
                current_shape = op.get_index_launch_shape()
                for func,shape in self.projection_epoch:
                    # We can stay in shallow disjoint mode if the
                    # next user is going to be reading only
                    if shallow_disjoint and func.depth != 0:
                        shallow_disjoint = False
                        if not same_func_and_shape:
                            break
                    if same_func_and_shape:
                        if func is not req.projection_function:
                            same_func_and_shape = False
                            if not shallow_disjoint:
                                break
                        elif shape.get_dim() != current_shape.get_dim():
                            # Different dimensions can't be compared
                            same_func_and_shape = False
                            if not shallow_disjoint:
                                break
                        elif not shape.dominates(current_shape):
                            same_func_and_shape = False
                            if not shallow_disjoint:
                                break
                # If we can't do either of these then we have to close
                # Also have to close if this is a reduction and not shallow disjoint
                if not shallow_disjoint and (not same_func_and_shape or req.is_reduce()):
                    closed = True
                    if not self.perform_close_operation(empty_children_to_close, False, 
                                        False, op, req, already_closed, previous_deps, 
                                        perform_checks, disjoint_close):
                        return False,closed
                    if disjoint_close:
                        self.projection_mode = OPEN_READ_WRITE
                    else:
                        self.projection_mode = OPEN_NONE
            else:
                assert self.projection_mode == OPEN_MULTI_REDUCE
                assert self.current_redop != 0
                if self.current_redop != req.redop:
                    
                    if disjoint_close:
                        self.projection_mode = OPEN_READ_WRITE
                    else:
                        closed = True
                        if not self.perform_close_operation(empty_children_to_close, 
                                        False, False, op, req, already_closed,
                                        previous_deps, perform_checks, disjoint_close):
                            return False,closed
                        self.projection_mode = OPEN_NONE
        elif self.current_redop != 0 and self.current_redop != req.redop:
            children_to_close = set()
            # Flushing reductions close all children no matter what
            for child,open_mode in iteritems(self.open_children):
                children_to_close.add(child)
            # If we are flushing reductions we do a close no matter what
            closed = True
            if not self.perform_close_operation(children_to_close, False, False, 
                        op, req, already_closed, previous_deps, perform_checks):
                return False,closed
        else:
            # Close all open children
            children_to_close = dict()
            children_to_read_close = dict()
            for child,open_mode in iteritems(self.open_children):
                if open_mode == OPEN_READ_ONLY:
                    children_to_read_close[child] = False
                else:
                    children_to_close[child] = False
            if children_to_close:
                closed = True
                if not self.perform_close_operation(children_to_close, False, False,
                            op, req, already_closed, previous_deps, perform_checks):
                    return False,closed
            if children_to_read_close:
                closed = True
                if not self.perform_close_operation(children_to_read_close, True, 
                        False, op, req, already_closed, previous_deps, perform_checks):
                    return False,closed
            # Can now reset this
            self.open_children = dict()
        return True,closed

    def siphon_logical_deletion(self, op, req, next_child, closed, 
                                previous_deps, perform_checks, force_close):
        # If our child is not open, then we are done
        if next_child not in self.open_children:
            return True,closed
        # See which mode it is open in 
        open_mode = self.open_children[next_child]
        child_to_close = dict()
        child_to_close[next_child] = False # permit leave open
        if open_mode == OPEN_READ_ONLY:
            # If it is open read-only, there is nothing to do
            del self.open_children[next_child]
        elif open_mode == OPEN_READ_WRITE:
            if force_close:
                closed = True
                if self.perform_close_operation(child_to_close, False, False, op, 
                                        req, closed, previous_deps, perform_checks):
                    return False,closed
        elif open_mode == OPEN_SINGLE_REDUCE:
            if force_close: 
                closed = True
                if not self.perform_close_operation(child_to_close, False, False, op, 
                                            req, closed, previous_deps, perform_checks):
                    return False,closed
            else:
                # Update the state to read-write
                self.open_children[next_child] = OPEN_READ_WRITE
        elif open_mode == OPEN_MULTI_REDUCE:
            closed = True
            if not self.perform_close_operation(child_to_close, False, False, op, req, 
                                                closed, previous_deps, perform_checks):
                return False,closed
        else:
            assert False # should never get here
        return True,closed

    def find_close_operation(self, op, req, read_only, perform_checks, error_str):
        close = op.get_close_operation(req, self.node, self.field, read_only)
        if close is None:
            if perform_checks:
                print(("ERROR: %s (UID=%s) failed to generate "+
                      "a close operation for field %s of region "+
                      "requirement %s at %s%s") %
                      (op, str(op.uid), self.field, req.index, self.node, error_str))
            else:
                print(("ERROR: %s (UID=%s) failed to generate "+
                      "a close operation that we normally would have expected. This "+
                      "is likely a runtime bug. Re-run with logical checks "+
                      "to confirm.") % (op, str(op.uid)))
            if self.node.state.bad_graph_on_error:
                self.node.state.dump_bad_graph(op.context, req.logical_node.tree_id, self.field)
            if self.node.state.assert_on_error:
                assert False
        return close

    def perform_close_checks(self, close, closed_users, op, req, 
                             previous_deps, error_str, perform_checks):
        assert 0 in close.reqs
        close_req = close.reqs[0]
        # Check for dependences against all the closed users
        # as well as any users in the previous set
        for prev_op,prev_req in closed_users:
            # Check for replays
            if prev_op is op:
                # If it is a previous registration of ourself, skip it
                # This will only happen during replays
                if prev_req.index == req.index:
                    continue
                assert False
            if perform_checks:
                if not close.has_mapping_dependence(close_req, prev_op, prev_req,
                                      ANTI_DEPENDENCE if prev_req.is_read_only()
                                      else TRUE_DEPENDENCE, self.field):
                    print(("ERROR: close operation %s generated by "+
                          "field %s of region requirement "+
                          "%s of %s failed to find a "+
                          "mapping dependence on previous operation "+
                          "%s in sub-tree being closed%s") %
                          (close, self.field, req.index, op, prev_op, error_str))
                    if self.node.state.assert_on_error:
                        assert False
                    return False
            else:
                # Not performing checks so record the mapping dependence
                dep_type = compute_dependence_type(prev_req, close_req)
                dep = MappingDependence(prev_op, close, prev_req.index,
                                        close_req.index, dep_type)
                prev_op.add_outgoing(dep)
                close.add_incoming(dep)
        for prev_op,prev_req in previous_deps:
            # Check for replays
            if prev_op is op:
                # If it is a previous registration of ourself, skip it
                # This will only happen during replays
                if prev_req.index == req.index:
                    continue
                assert False
            if perform_checks:
                if not close.has_mapping_dependence(close_req, prev_op, prev_req,
                                      ANTI_DEPENDENCE if prev_req.is_read_only()
                                      else TRUE_DEPENDENCE, self.field):
                    print(("ERROR: close operation %s generated by "+
                          "field %s of region requirement "+
                          "%s of %s failed to find a "+
                          "mapping dependence on previous operation "+
                          "%s from higher in the region tree") %
                          (close, self.field, req.index, op, prev_op))
                    if self.node.state.assert_on_error:
                        assert False
                    return False
            else:
                # Not performing checks so record the mapping dependence
                dep_type = compute_dependence_type(prev_req, close_req)
                dep = MappingDependence(prev_op, close, prev_req.index,
                                        close_req.index, dep_type)
                prev_op.add_outgoing(dep)
                close.add_incoming(dep)
        return True

    def record_close_dependences(self, close, closed_users, op, req, previous_deps):
        assert 0 in close.reqs
        close_req = close.reqs[0]
        for prev_op,prev_req in closed_users:
            # Can skip it if it is the close creator
            if prev_op is close.creator:
                continue
            if op != prev_op or req.index != prev_req.index:
                dep = MappingDependence(prev_op, close, prev_req.index,
                                        close_req.index, TRUE_DEPENDENCE)
                prev_op.add_outgoing(dep)
                close.add_incoming(dep)
        for prev_op,prev_req in previous_deps:
            # Can skip it if it is the close creator
            if prev_op is close.creator:
                continue
            if op != prev_op or req.index != prev_req.index:
                dep = MappingDependence(prev_op, close, prev_req.index,
                                        close_req.index, TRUE_DEPENDENCE)
                prev_op.add_outgoing(dep)
                close.add_incoming(dep)

    def perform_close_operation(self, children_to_close, read_only_close, 
                                overwriting_close, op, req, already_closed,
                                previous_deps, perform_checks, disjoint_close = False):
        error_str = ' for read-only close operation' if read_only_close \
            else ' for normal close operation'
        # Find the close operation first
        if not already_closed:
            close = self.find_close_operation(op, req, 
                    read_only_close or overwriting_close, perform_checks, error_str)
            if not close:
                return False
        for child in children_to_close:
            closed_users = list()
            # Close the child tree
            child.close_logical_tree(self.field, closed_users)
            # Perform any checks
            if not already_closed:
                if perform_checks:
                    if not self.perform_close_checks(close, closed_users, op, req, 
                                        previous_deps, error_str, perform_checks):
                        return False
                else:
                    self.record_close_dependences(close, closed_users, op, req,
                                                  previous_deps)
            del self.open_children[child]
            # Remove it from the list of open reductions to if it is there
            if child in self.open_redop:
                del self.open_redop[child]
        # Perform the epoch analysis for the close operation 
        if not already_closed:
            assert 0 in close.reqs
            if not self.perform_epoch_analysis(close, close.reqs[0], 
                                               perform_checks, True, None, op):
                return False
            # Record the close operation in the current epoch
            self.register_logical_user(close, close.reqs[0])
        # See if we still have any dirty children below
        # Read only close operations don't update the dirty fields
        if not read_only_close and self.dirty_below:
            # If we're doing a disjoint close then we get to
            # leave the dirty fields in place
            still_dirty = disjoint_close 
            if not still_dirty:
                for privilege in itervalues(self.open_children):
                    if privilege == READ_WRITE or privilege == REDUCE or \
                        privilege == WRITE_DISCARD:
                          still_dirty = True
                          break
            if not still_dirty:
                self.dirty_below = False
        return True

    def close_logical_tree(self, closed_users):
        # Save the closed users and then close the subtrees
        closed_users += self.current_epoch_users
        self.current_epoch_users = list()
        self.previous_epoch_users = list()
        for child in self.open_children:
            child.close_logical_tree(self.field, closed_users)
        self.open_children = dict()
        self.open_redop = dict()
        self.current_redop = 0
        self.dirty_below = False
        self.projection_mode = OPEN_NONE
        self.projection_epoch = list()
        
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
                # If it is a previous registration of ourself, skip it
                # This will only happen during replays
                dominates = False
                continue
            # Close operations from the same creator can never depend on eachother
            if op.is_close() and prev_op.is_close() and op.creator is prev_op.creator:
                continue
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
                    # If it is a previous registration of ourself, skip it
                    # This will only happen during replays
                    if prev_req.index == req.index:
                        continue
                    assert False
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

class DataflowTraverser(object):
    def __init__(self, state, dst_depth, dst_field, dst_req, dst_inst,
                 op, src_req, src_version, dst_version, error_str):
        self.state = state
        self.target = dst_inst
        self.src_tree = state.tree.tree_id
        self.dst_tree = dst_req.tid
        self.point = state.point
        self.op = op
        self.src_field = state.field
        self.dst_field = dst_field
        self.src_depth = state.depth
        self.dst_depth = dst_depth
        self.src_req = src_req
        self.dst_req = dst_req
        self.src_version = src_version
        self.dst_version = dst_version
        self.error_str = error_str
        # Across is either different fields or same field in different trees
        self.across = self.src_field.fid != self.dst_field.fid or \
                        self.src_tree != self.dst_tree
        # There is an implicit assumption here that if we did a close
        # to flush a bunch of reductions then copies will always come
        # from the newly created instance and not from an composite
        # instance that also buffered the reductions
        if state.pending_reductions and state.is_initialized():
            # If it's already in the set of valid instances we don't need reductions
            if dst_inst in state.valid_instances:
                self.found_dataflow_path = True
                self.needs_reductions = False
            elif dst_inst in state.previous_instances:
                self.found_dataflow_path = True
                self.needs_reductions = True
            else:
                self.found_dataflow_path = False
                self.needs_reductions = True
        else:
            assert dst_inst not in state.valid_instances
            self.found_dataflow_path = not state.is_initialized()
            self.needs_reductions = False
        if not self.found_dataflow_path or self.needs_reductions:
            self.dataflow_stack = list()
            self.dataflow_stack.append(dst_inst)    
            self.dataflow_copy = list()
        else:
            self.dataflow_stack = None
        self.observed_reductions = dict()
        self.reductions_to_perform = dict()
        self.failed_analysis = False

    def visit_node(self, node, eq_key): 
        if isinstance(node, Operation):
            pass 
        elif isinstance(node, RealmCopy):
            if not self.visit_copy(node, eq_key):
                return False
        elif isinstance(node, RealmFill):
            if not self.visit_fill(node, eq_key):
                return False
        elif isinstance(node, RealmDeppart):
            pass
        else:
            assert False # should never get here
        return True

    def post_visit_node(self, node, eq_key):
        if isinstance(node, RealmCopy):
            self.post_visit_copy(node, eq_key)

    def run(self, first, eq_key):
        # Do this with DFS since we care about paths
        nodes = list()
        nodes.append((first,True))
        while nodes:
            node,first_pass = nodes[-1]
            if first_pass:
                if node.version_numbers and eq_key in node.version_numbers and \
                        node.version_numbers[eq_key] != self.state.version_number:
                    # We can't traverse this node if it's from a previous version number
                    # because that is not the same value of the equivalence class
                    # Skip this check on the first node though for things like copy across
                    nodes.pop()
                    continue
                if not self.visit_node(node, eq_key):
                    nodes.pop()
                    continue
                eq_privileges = node.get_equivalence_privileges()
                privilege = eq_privileges[eq_key]
                # We can't traverse past any operation that writes this field 
                # unless this is the first operation which we're trying to
                # traverse backwards from
                if privilege == READ_ONLY or node is first:
                    # Check to see if the version number is the same, if this
                    # is an operation from a previous version then we can't traverse it
                    if node.eq_incoming and eq_key in node.eq_incoming:
                        incoming = node.eq_incoming[eq_key]
                        if incoming:
                            # Record that we haven't run the post-visit method yet
                            nodes[-1] = (node,False)
                            # Add these nodes to the stack
                            for next_node in incoming:
                                nodes.append((next_node,True))
                            # Can't run the post visit method yet
                            continue
            self.post_visit_node(node, eq_key)
            nodes.pop()
            # See if we are done
            if self.failed_analysis or self.verified(eq_key):
                break
        # Unwind the stack in case we finished early
        while nodes:
            node,first_pass = nodes.pop()
            # Skip any nodes that we finished early for
            if first_pass:
                continue
            # Run the post visit method for any ndoes on the stack
            self.post_visit_node(node, eq_key)

    def visit_copy(self, copy, eq_key):
        # We should never traverse through indirection copies here
        if copy.indirections is not None:
            return False
        # Check to see if this is a reduction copy or not
        if 0 in copy.redops:
            # Normal copy
            # See if we need to do the dataflow check
            # and the copy has our field
            if self.dst_field in copy.dst_fields and \
                    copy.dsts[copy.dst_fields.index(self.dst_field)] is self.dataflow_stack[-1] and \
                    self.src_field is copy.src_fields[copy.dst_fields.index(self.dst_field)]:
                # Traverse the dataflow path
                src = copy.srcs[copy.dst_fields.index(self.dst_field)]
                # See if the source is a valid instance or a
                # previous instance in the presence of pending reductions
                if self.state.pending_reductions and self.needs_reductions:
                    # We have pending reductions, see if we went through any to find this
                    # valid instance. If we did then check in previous instances otherwise
                    # we can look directly in the valid instances
                    if src in self.state.valid_instances:
                        self.found_dataflow_path = True
                        # No longer need reductions since we found a direct 
                        # path to a valid instance
                        self.needs_reductions = False
                    elif src in self.state.previous_instances:
                        self.found_dataflow_path = True
                elif src in self.state.valid_instances:
                    self.found_dataflow_path = True
                # Continue the traversal if we're not done
                if not self.verified(eq_key, last=False):
                    # Push it on the stack and continue traversal
                    self.dataflow_stack.append(src)
                    self.dataflow_copy.append(True)
                    return True
                elif self.found_dataflow_path:
                    # If we just finished finding it do the analysis now
                    self.perform_copy_analysis(copy, src, self.dataflow_stack[-1], eq_key)
            else:
                # Always traverse through non-dataflow copies
                self.dataflow_copy.append(False)
                return True
        elif self.needs_reductions:
            # Reduction copy
            red_target = self.dataflow_stack[-1]
            if self.dst_field in copy.dst_fields and \
                    copy.dsts[copy.dst_fields.index(self.dst_field)] is red_target and \
                    self.src_field is copy.src_fields[copy.dst_fields.index(self.dst_field)]:
                src = copy.srcs[copy.dst_fields.index(self.dst_field)]
                if src.redop != 0:
                    if src not in self.state.pending_reductions:
                        return False
                    if src in self.observed_reductions:
                        assert self.observed_reductions[src] is not copy
                        print("ERROR: Duplicate application of reductions by copies "+
                                str(copy)+" and "+str(self.observed_reductions[src])+
                                " from reduction instance "+str(src)+ " for op "+
                                self.error_str)
                        if self.op.state.eq_graph_on_error:
                            self.op.state.dump_eq_graph(eq_key)
                        if self.op.state.assert_on_error:
                            assert False
                        return False
                    else:
                        self.observed_reductions[src] = copy
                        if not red_target in self.reductions_to_perform:
                            self.reductions_to_perform[red_target] = list()
                        self.reductions_to_perform[red_target].append(src)
                        # Keep going as long as we haven't found the dataflow path
                        # or there are more reductions to find
                        return not self.verified(eq_key, last=False)
        return False

    def post_visit_copy(self, copy, eq_key):
        if self.failed_analysis:
            if self.dataflow_copy[-1]:
                self.dataflow_stack.pop()
            self.dataflow_copy.pop()
            return
        if 0 in copy.redops:
            # Normal copy, definitely do the analysis if we found the path
            if self.found_dataflow_path and self.dataflow_copy[-1]: 
                assert len(self.dataflow_stack) > 1
                src = self.dataflow_stack[-1]
                dst = self.dataflow_stack[-2]
                # Check to see if we have any reductions to perform
                if src in self.reductions_to_perform:
                    # Do these in the reverse order of how they were added
                    for red_src in reversed(self.reductions_to_perform[src]):
                        reduction = self.observed_reductions[red_src]
                        self.perform_copy_analysis(reduction, red_src, src, eq_key)
                        if self.failed_analysis and self.op.state.assert_on_error:
                            assert False
                    del self.reductions_to_perform[src]
                # Perform the copy analysis
                self.perform_copy_analysis(copy, src, dst, eq_key)
            # Only pop off our instance if this wasn't a reduction copy
            if self.dataflow_copy:
                if self.dataflow_copy[-1]:
                    self.dataflow_stack.pop()
                self.dataflow_copy.pop()

    def perform_copy_analysis(self, copy, src, dst, eq_key):
        # If we've already traversed this then we can skip the verification
        if copy.record_version_number(self.state):
            return
        if self.across:
            copy.record_across_version_number(self.point, self.dst_field,
                                              self.dst_tree, self.dst_version)
        src_preconditions = src.find_verification_copy_dependences(self.src_depth,
                                        self.src_field, self.point, self.op, 
                                        self.src_req.index, True, 0, self.src_version)
        bad = check_preconditions(src_preconditions, copy)
        if bad is not None:
            print("ERROR: Missing source precondition for "+str(copy)+
                " on field "+str(self.src_field)+" for op "+self.error_str+
                " on "+str(bad))
            if self.op.state.eq_graph_on_error:
                self.op.state.dump_eq_graph(eq_key)
            self.failed_analysis = True
            if self.op.state.assert_on_error:
                assert False
            return
        dst_preconditions = dst.find_verification_copy_dependences(self.dst_depth,
                            self.dst_field, self.point, self.op, self.dst_req.index, 
                            False, src.redop, self.dst_version)
        bad = check_preconditions(dst_preconditions, copy)
        if bad is not None:
            print("ERROR: Missing destination precondition for "+str(copy)+
                " on field "+str(self.dst_field)+" for op "+self.error_str+
                " on "+str(bad))
            if self.op.state.eq_graph_on_error:
                self.op.state.dump_eq_graph(eq_key)
            self.failed_analysis = True
            if self.op.state.assert_on_error:
                assert False
            return
        src.add_verification_copy_user(self.src_depth, self.src_field, self.point, 
                               copy, self.src_req.index, True, 0, self.src_version)
        dst.add_verification_copy_user(self.dst_depth, self.dst_field, self.point,
                       copy, self.dst_req.index, False, src.redop, self.dst_version)

    def visit_fill(self, fill, eq_key):
        # See if this fill is for the current target
        if not self.found_dataflow_path and self.dataflow_stack and \
              self.dst_field in fill.fields and \
              fill.dsts[fill.fields.index(self.dst_field)] is self.dataflow_stack[-1]:
            # If we don't have a pending fill, then this isn't right
            if not self.state.pending_fill:
                return False
            self.found_dataflow_path = True
            # If we've already traversed this then we can skip the verification
            if fill.record_version_number(self.state):
                return False
            assert self.state.fill_op is fill.fill_op or fill.fill_op.replayed
            if self.across:
                fill.record_across_version_number(self.point, self.dst_field,
                                                  self.dst_tree, self.dst_version)
            dst = fill.dsts[fill.fields.index(self.dst_field)]
            preconditions = dst.find_verification_copy_dependences(self.dst_depth,
                            self.dst_field, self.point, self.op, self.dst_req.index, 
                            False, 0, self.dst_version)
            bad = check_preconditions(preconditions, fill)
            if bad is not None:
                print("ERROR: Missing destination precondition for "+
                    str(fill)+" on field "+str(self.dst_field)+" for op "+
                    self.error_str+" on "+str(bad))
                self.failed_analysis = True
                if self.op.state.eq_graph_on_error:
                    self.op.state.dump_eq_graph(eq_key)
                if self.op.state.assert_on_error:
                    assert False
                return False
            dst.add_verification_copy_user(self.dst_depth, self.dst_field, self.point,
                                 fill, self.dst_req.index, False, 0, self.dst_version)
        # We should never traverse backwards through a fill
        return False

    def verified(self, eq_key, last = False):
        if self.failed_analysis:
            if last and self.op.state.assert_on_error:
                assert False
            return False
        # If we didn't have a dataflow path then we're done
        if not self.found_dataflow_path:
            if last:
                print("ERROR: No dataflow path found to update field "+
                        str(self.dst_field)+" of instance "+str(self.target)+
                        " of region requirement "+str(self.dst_req.index)+
                        " of "+str(self.op))
                if self.op.state.eq_graph_on_error:
                    self.op.state.dump_eq_graph(eq_key)
                if self.op.state.assert_on_error:
                    assert False
            return False
        # See if we saw all the needed reductions
        if self.needs_reductions:
            if len(self.state.pending_reductions) != len(self.observed_reductions):
                if last:
                    print("ERROR: Missing reductions to apply to field "+
                            str(self.dst_field)+" of instance "+str(self.target)+
                            " of region requirement "+str(self.dst_req.index)+
                            " of "+str(self.op))
                    if self.op.state.eq_graph_on_error:
                        self.op.state.dump_eq_graph(eq_key)
                    if self.op.state.assert_on_error:
                        assert False
                return False
            elif last:
                # If this is the last check, replay any reductions for the target
                if self.target in self.reductions_to_perform:
                    # Do these in the reverse order of how they were added
                    for src in reversed(self.reductions_to_perform[self.target]):
                        reduction = self.observed_reductions[src]
                        self.perform_copy_analysis(reduction, src, self.target, eq_key)
                        if self.failed_analysis:
                            if self.op.state.assert_on_error:
                                assert False
                            return False
        return True

    def verify(self, op, restricted = False):
        src_key = (self.point, self.src_field, self.src_tree)
        dst_key = (self.point, self.dst_field, self.dst_tree)
        # The verification key is the src_key unless otherwise specified
        ver_key = src_key
        # Copies are a little weird in that they don't actually
        # depend on their region requirements so we just need
        # to traverse from their finish event
        if op.kind == COPY_OP_KIND:
            # Find the latest copies that we generated
            if op.realm_copies:
                # If we are across, we start by visiting the last
                # copies because they are the across ones, otherwise
                # we just traverse them
                if self.across:
                    ver_key = dst_key
                    for copy in op.realm_copies:
                        # Skip non-across copies
                        if not copy.is_across():
                            continue
                        eq_privileges = copy.get_equivalence_privileges()
                        if src_key in eq_privileges and dst_key in eq_privileges:
                            self.run(copy, dst_key)
                else:
                    for copy in op.realm_copies:
                        if copy.is_across():
                            # For across, we still might need to traverse
                            # their incoming because of how the transitive
                            # reduction deduplicates dependences
                            for node in copy.physical_incoming:
                                # Skip anything we're going to traverse anyway
                                if node in op.realm_copies or \
                                        (op.realm_fills and node in op.realm_fills):
                                    continue
                                eq_privileges = node.get_equivalence_privileges()
                                if src_key in eq_privileges:
                                    self.run(node, src_key)
                        else:
                            eq_privileges = copy.get_equivalence_privileges()
                            if src_key in eq_privileges:
                                self.run(copy, src_key)
            # Only need to traverse fills directly for across cases as the 
            # non-accross ones will be traverse by the normal copy traversasl
            if op.realm_fills:
                if self.across:
                    ver_key = dst_key
                    for fill in op.realm_fills:
                        # Skip non-across fills
                        if not fill.is_across():
                            continue
                        eq_privileges = fill.get_equivalence_privileges()
                        if src_key not in eq_privileges and dst_key in eq_privileges:
                            self.run(fill, dst_key)
                else:
                    for fill in op.realm_fills:
                        # Skip across fills
                        if fill.is_across():
                            continue
                        eq_privileges = fill.get_equivalence_privileges()
                        if src_key in eq_privileges:
                            self.run(fill, src_key)
        elif op.kind == INTER_CLOSE_OP_KIND or op.kind == POST_CLOSE_OP_KIND:
            # Close operations are similar to copies in that they don't
            # wait for data to be ready before starting, so we can't
            # start at their node. However, inter close ops are even
            # more different in that they don't wait at the end, so
            # just traverse over all their realm events
            if op.realm_copies:
                for copy in op.realm_copies:
                    eq_privileges = copy.get_equivalence_privileges()
                    if src_key in eq_privileges:
                        self.run(copy, src_key)
            if op.realm_fills:
                for fill in op.realm_fills:
                    eq_privileges = fill.get_equivalence_privileges()
                    if src_key in eq_privileges:
                        self.run(fill, src_key)
        elif restricted:
            assert not self.across
            # If this is restricted, do the traversal from the copies
            # themselves since they might have occurred after the op
            if op.realm_copies:
                for copy in op.realm_copies:
                    eq_privileges = copy.get_equivalence_privileges()
                    if src_key not in eq_privileges:
                        continue
                    # Only look at these if the destination is correct
                    if self.target in copy.dsts and \
                            self.dst_tree == copy.dst_tree_id and \
                            self.dst_field in copy.dst_fields:
                        self.run(copy, src_key)
        else:
            # Traverse the node and then see if we satisfied everything
            self.run(op, src_key)
        return self.verified(ver_key, True)

class EquivalenceSet(object):
    __slots__ = ['tree', 'depth', 'field', 'point', 'valid_instances', 
                 'previous_instances', 'pending_reductions', 
                 'pending_fill', 'fill_op', 'version_number',
                 'restricted_inst']
    def __init__(self, tree, depth, field, point):
        self.tree = tree
        self.depth = depth
        self.field = field
        self.point = point
        # State machine is write -> reduce -> read
        self.valid_instances = set()
        self.previous_instances = set()
        # Reductions of different kinds must be kept in order
        self.pending_reductions = list()
        self.pending_fill = False 
        self.fill_op = None
        self.version_number = 0
        self.restricted_inst = None
        
    def is_initialized(self):
        return self.version_number > 0

    def reset(self):
        self.version_number += 1
        self.pending_fill = False
        self.fill_op = None
        self.previous_instances = set()
        self.valid_instances = set()
        self.pending_reductions = list()
        # Doesn't change restricted inst

    def initialize_verification_state(self, inst, restricted):
        self.valid_instances.add(inst)
        self.version_number = 1
        if restricted:
            self.restricted_inst = inst

    def perform_fill_verification(self, op, req):
        # Fills clear everything out so we are just done
        self.reset()
        self.pending_fill = True
        assert op.kind == FILL_OP_KIND
        self.fill_op = op
        return True

    def add_restriction(self, op, req, inst):
        assert inst in self.valid_instances
        self.reset()
        self.valid_instances.add(inst)
        self.restricted_inst = inst
        return True

    def remove_restriction(self, op, req, filter_inst):
        self.restricted_inst = None
        if filter_inst is not None and filter_inst in self.valid_instances:
            self.valid_instances.remove(filter_inst)
        return True

    def perform_physical_verification(self, op, req, inst, perform_checks, 
                                      perform_registration):
        assert not inst.is_virtual()
        if req.is_reduce():
            assert inst.redop != 0
            # Move valid instances back to previous instances
            # and add ourselves to the reduction instances
            # See if we are the first reduction
            # The first reduction also bumps the version number
            if self.valid_instances:
                self.previous_instances = self.valid_instances
                self.valid_instances = set()
                self.version_number += 1
                # Clear our old stale reductions
                self.pending_reductions = list()
            # Check to see if this instance is already in the list
            # of reduction instances, also check for the ABA problem
            # The instance could also be the restricted instance
            found = inst is self.restricted_inst
            for prev in self.pending_reductions:
                if prev is not inst:
                    if found and prev.redop != inst.redop:
                        print("ERROR: Reduction ABA violation for field "+
                                str(self.field)+" of region requirement "+
                                str(req.index)+" of "+str(op)+" (UID "+
                                str(op.uid)+") between reduction operators "+
                                str(inst.redop)+" and "+str(prev.redop))
                        if self.tree.state.eq_graph_on_error:
                            self.tree.state.dump_eq_graph(
                                    (self.point, self.field, self.tree.tree_id))
                        if self.tree.state.assert_on_error:
                            assert False
                        return False
                else:
                    found = True # Keep going to for ABA
            if not found:
                if not self.issue_reduction_initialization(inst, op, req, perform_checks):
                    return False
                self.pending_reductions.append(inst)
        elif req.is_write_only():
            assert inst.redop == 0
            # We overwrite everything else
            self.reset()
            self.valid_instances.add(inst)
        else:
            # See if we need to do anything to bring this up to date
            if inst not in self.valid_instances:
                # Find or make copies to bring this up to date
                error_str = "region requirement "+str(req.index)+" of "+str(op)
                if not self.issue_update_copies(inst, op, req, perform_checks, error_str):
                    return False
            # Now that it is up to date, we can update the instance sets
            if req.is_write():
                # We overwrite everything else
                # Unless we are a close operation in which case we
                # aren't really making a new version, we're just 
                # flushing everything to a common instance which 
                # makes a new valid instance but doesn't invalidate
                # any of the other data that already exists
                if op.kind != INTER_CLOSE_OP_KIND:
                    self.reset()
                self.valid_instances.add(inst)
            else:
                assert req.is_read_only()
                # Just have to add ourselves to the list of valid instances
                # Only do this if we had valid data to begin with
                if self.is_initialized():
                    self.valid_instances.add(inst)
        # Finally perform our registrations
        if perform_registration and not self.perform_verification_registration(op, 
                                                        req, inst, perform_checks):
            return False
        return True

    def perform_copy(self, src, dst, op, req):
        copy = op.find_or_create_copy(req, self.field, src, dst)
        # Record this point for the copy operation so it renders properly
        copy.record_version_number(self)
        # Update the source instance
        src_preconditions = src.find_verification_copy_dependences(self.depth, 
              self.field, self.point, op, req.index, True, 0, self.version_number)
        for pre in src_preconditions:
            pre.physical_outgoing.add(copy)
            copy.physical_incoming.add(pre)
        src.add_verification_copy_user(self.depth, self.field, self.point,
                                       copy, req.index, True, 0, self.version_number)
        # Then do the destination instance
        dst_preconditions = dst.find_verification_copy_dependences(self.depth, 
            self.field, self.point, op, req.index, False, src.redop, self.version_number)
        for pre in dst_preconditions:
            pre.physical_outgoing.add(copy)
            copy.physical_incoming.add(pre)
        dst.add_verification_copy_user(self.depth, self.field, self.point,
                             copy, req.index, False, src.redop, self.version_number)

    def issue_update_copies(self, inst, op, req, perform_checks, 
                            error_str, restricted = False):
        assert inst not in self.valid_instances
        # If we're performing checks, we just need to check everything
        if perform_checks:
            # Special case for the first access with uninitialized state
            if not self.is_initialized():
                return True
            traverser = DataflowTraverser(self, self.depth, self.field, req, inst, 
                         op, req, self.version_number, self.version_number, error_str)
            return traverser.verify(op, restricted)
        # First see if we have a valid instance we can copy from  
        if self.valid_instances:
            src = next(iter(self.valid_instances))
            self.perform_copy(src, inst, op, req)
        # If we have a fill operation, we can just do that
        elif self.pending_fill:
            fill = op.find_or_create_fill(req, self.field, inst, self.fill_op)
            # Record this point for the copy operation so it renders properly
            fill.record_version_number(self)
            preconditions = inst.find_verification_copy_dependences(self.depth, 
                self.field, self.point, op, req.index, False, 0, self.version_number)
            for pre in preconditions:
                pre.physical_outgoing.add(fill)
                fill.physical_incoming.add(pre)
            inst.add_verification_copy_user(self.depth, self.field, 
                self.point, fill, req.index, False, 0, self.version_number)
        # Otherwise we'll only have previous instances if there are reductions
        elif self.previous_instances and inst not in self.previous_instances:
            assert self.pending_reductions
            src = next(iter(self.previous_instances)) 
            self.perform_copy(src, inst, op, req)
        # If there are any pending reductions, we must issue those now too
        if self.pending_reductions:
            return self.issue_update_reductions(inst, op, req, False, error_str)
        return True

    def issue_reduction_initialization(self, inst, op, req, perform_checks):
        if perform_checks:
            key = (self.point, self.field, self.tree.tree_id)
            if op.eq_incoming and key in op.eq_incoming:
                incoming = op.eq_incoming[key]
                if incoming:
                    for fill in incoming:
                        if not isinstance(fill, RealmFill):
                            continue
                        if inst not in fill.dsts:
                            continue
                        if fill.record_version_number(self):
                            continue
                        preconditions = inst.find_verification_copy_dependences(
                                self.depth, self.field, self.point, op, req.index,
                                False, 0, self.version_number)
                        bad = check_preconditions(preconditions, fill)
                        if bad is not None:
                            print("ERROR: Missing destination precondition for "+str(fill)+
                                    " on field "+str(self.field)+" for region "+
                                    "requirement "+str(req.index)+" of "+str(op)+
                                    " (UID "+str(op.uid)+")")
                            if self.tree.state.eq_graph_on_error:
                                self.tree.state.dump_eq_graph(
                                        (self.point, self.field, self.tree.tree_id))
                            if self.tree.state.assert_on_error:
                                assert False
                            return False
                        inst.add_verification_copy_user(self.depth, self.field,
                                self.point, fill, req.index, False, 0, self.version_number)
                        return True
            print("ERROR: Missing fill to initialize reduction instance "+str(inst)+
                    " for field "+str(self.field)+" of region requirement "+
                    str(req.index)+" of "+str(op)+" (UID "+str(op.uid)+")")
            if self.tree.state.eq_graph_on_error:
                self.tree.state.dump_eq_graph((self.point, self.field, self.tree.tree_id))
            if self.tree.state.assert_on_error:
                assert False
            return False
        else:
            fill = op.find_or_create_fill(req, self.field, inst, None)
            # Record this point for the copy operation so it renders properly
            fill.record_version_number(self)
            preconditions = inst.find_verification_copy_dependences(self.depth, 
                self.field, self.point, op, req.index, False, 0, self.version_number)
            for pre in preconditions:
                pre.physical_outgoing.add(fill)
                fill.physical_incoming.add(pre)
            inst.add_verification_copy_user(self.depth, self.field, 
                self.point, fill, req.index, False, 0, self.version_number)
            return True

    def issue_update_reductions(self, inst, op, req, perform_checks, 
                                error_str, restricted = False):
        assert self.pending_reductions
        if perform_checks:
            traverser = DataflowTraverser(self, self.depth, self.field, req, inst, 
                          op, req, self.version_number, self.version_number, error_str)
            return traverser.verify(op, restricted)
        # Make sure a reduction was issued from each reduction instance
        for reduction_inst in self.pending_reductions:
            self.perform_copy(reduction_inst, inst, op, req)
        return True

    def perform_verification_registration(self, op, req, inst, perform_checks):
        preconditions = inst.find_verification_use_dependences(self.depth, 
                                          self.field, self.point, op, req)
        if perform_checks:
            bad = check_preconditions(preconditions, op)
            if bad is not None:
                print("ERROR: Missing use precondition for field "+str(self.field)+
                      " of region requirement "+str(req.index)+" of "+str(op)+
                      " (UID "+str(op.uid)+") on previous "+str(bad))
                if self.tree.state.eq_graph_on_error:
                    self.tree.state.dump_eq_graph((self.point, self.field, self.tree.tree_id))
                if self.tree.state.assert_on_error:
                    assert False
                return False
        else:
            for other in preconditions:
                op.physical_incoming.add(other)
                other.physical_outgoing.add(op)
        # Record ourselves as a user for this instance
        inst.add_verification_user(self.depth, self.field, self.point, 
                                   op, req, self.version_number)
        # If we are restricted and we're not read-only we have to issue
        # copies back to the restricted instance
        if self.restricted_inst is not None:
            if inst is not self.restricted_inst and req.priv != READ_ONLY:
                error_str = "restricted region requirement "+\
                        str(req.index)+" of "+str(op)
                # We need to issue a copy or a reduction back to the 
                # restricted instance in order to have the proper semantics
                if inst.redop != 0:
                    # Have to perform a reduction back
                    if not self.issue_update_reductions(self.restricted_inst, op, req,
                                                        perform_checks, error_str, True):
                        return False
                else:
                    # Perform a normal copy back
                    if not self.issue_update_copies(self.restricted_inst, op, req, 
                                                    perform_checks, error_str, True):
                        return False
            # Restrictions always overwrite everything when they are done
            self.reset()
            self.valid_instances.add(self.restricted_inst)
        return True

    def perform_copy_across_verification(self, op, redop, perform_checks,
                                         src_depth, src_field, src_req, src_inst,
                                         dst_depth, dst_field, dst_req, dst_inst,
                                         dst_version_number):
        # If we're performing checks we do that now
        # Otherwise we'll fall through and build the graph ourselves
        if perform_checks:
            if src_inst.is_virtual():
                assert redop == 0
                error_str = "copy across from "+str(src_req.index)+" to "+\
                    str(dst_req.index)+" of "+str(op)
                traverser = DataflowTraverser(self, dst_depth, 
                    dst_field, dst_req, dst_inst, op, src_req, 
                    self.version_number, dst_version_number, error_str)
                return traverser.verify(op)
            else:
                # Just have to find the copy operation and check it
                # as all the other copies have already been checked
                copy = op.find_verification_copy_across(src_field, dst_field,
                                      self.point, src_inst, dst_inst, redop)
                if copy is None:
                    print("ERROR: Missing copy across operation from field "+
                        str(src_field)+" to field "+str(dst_field)+" between "+
                        "region requirements "+str(src_req.index)+" and "+
                        str(dst_req.index)+" of "+str(op))
                    if op.state.eq_graph_on_error:
                        op.state.dump_eq_graph((self.point, self.field, self.tree.tree_id))
                    if op.state.assert_on_error:
                        assert False
                    return False
                # If we already preformed the verification then we can skip this
                if copy.record_version_number(self):
                    return True
                copy.record_across_version_number(self.point, dst_field, 
                                                  dst_req.tid, dst_version_number)
                src_preconditions = src_inst.find_verification_copy_dependences(
                                    src_depth, src_field, self.point, op, 
                                    src_req.index, True, 0, self.version_number)
                bad = check_preconditions(src_preconditions, copy)
                if bad is not None:
                    print("ERROR: Missing source precondition for "+str(copy)+
                          " on field "+str(src_field)+" for "+str(op)+
                          " on "+str(bad))
                    if op.state.eq_graph_on_error:
                        op.state.dump_eq_graph((self.point, src_field, self.tree.tree_id))
                    if op.state.assert_on_error:
                        assert False
                    return False
                dst_preconditions = dst_inst.find_verification_copy_dependences(
                                    dst_depth, dst_field, self.point, op, 
                                    dst_req.index, False, redop, dst_version_number)
                bad = check_preconditions(dst_preconditions, copy)
                if bad is not None:
                    print("ERROR: Missing destination precondition for "+str(copy)+
                          " on field "+str(dst_field)+" for "+str(op)+
                          " on "+str(bad))
                    if op.state.eq_graph_on_error:
                        op.state.dump_eq_graph((self.point, dst_field, self.tree.tree_id))
                    if op.state.assert_on_error:
                        assert False
                    return False
                # Then we can do the registration ourselves
                src_inst.add_verification_copy_user(src_depth, src_field, self.point,
                                    copy, src_req.index, True, 0, self.version_number)
                dst_inst.add_verification_copy_user(dst_depth, dst_field, self.point,
                                copy, dst_req.index, False, redop, self.version_number)
                return True
        # If we already have a src_inst it should be in the valid set
        if not src_inst.is_virtual():
            assert src_inst in self.valid_instances
            self.perform_copy_across(src_inst, dst_inst, op, redop, src_depth, 
                src_field, src_req, dst_depth, dst_field, dst_req, dst_version_number)
            return True
        # Otherwise we can do our normal copy routine from this state 
        if self.valid_instances:
            src = next(iter(self.valid_instances))
            self.perform_copy_across(src, dst_inst, op, redop, src_depth, src_field, 
                      src_req, dst_depth, dst_field, dst_req, dst_version_number)
        elif self.pending_fill:
            # Should be no reductions here
            assert redop == 0
            fill = op.find_or_create_fill(dst_req, dst_field, dst_inst, self.fill_op)
            # Record this point for the copy operation so it renders properly
            fill.record_version_number(self)
            preconditions = dst_inst.find_verification_copy_dependences(
                                dst_depth, dst_field, self.point, op, 
                                dst_req.index, False, redop, dst_version_number)
            for pre in preconditions:
                pre.physical_outgoing.add(fill)
                fill.physical_incoming.add(pre)
            dst_inst.add_verification_copy_user(dst_depth, dst_field, 
                self.point, fill, dst_req.index, False, redop, self.version_number)
        elif self.previous_instances:
            assert redop == 0
            assert self.pending_reductions
            src = next(iter(self.previous_instances))
            self.perform_copy_across(src, dst_inst, op, redop, src_depth, src_field, 
                        src_req, dst_depth, dst_field, dst_req, dst_version_number)
        if self.pending_reductions:
            assert redop == 0
            for reduction_inst in self.pending_reductions:
                self.perform_copy_across(reduction_inst, dst_inst, op, 
                    reduction_inst.redop, src_depth, src_field, src_req,
                    dst_depth, dst_field, dst_req, dst_version)
        return True

    def perform_copy_across(self, src_inst, dst_inst, op, redop,
                            src_depth, src_field, src_req,
                            dst_depth, dst_field, dst_req, dst_version):
        copy = op.find_or_create_copy_across(src_inst, src_field, src_req,
                                             dst_inst, dst_field, dst_req, redop)
        # Record this point for the copy operation so it renders properly
        copy.record_version_number(self)
        src_preconditions = src_inst.find_verification_copy_dependences(
                            src_depth, src_field, self.point, op, 
                            src_req.index, True, 0, self.version_number)
        for pre in src_preconditions:
            pre.physical_outgoing.add(copy)
            copy.physical_incoming.add(pre)
        src_inst.add_verification_copy_user(src_depth, src_field, self.point,
                            copy, src_req.index, True, 0, self.version_number)
        dst_preconditions = dst_inst.find_verification_copy_dependences(
                            dst_depth, dst_field, self.point, op, 
                            dst_req.index, False, redop, dst_version)
        for pre in dst_preconditions:
            pre.physical_outgoing.add(copy)
            copy.physical_incoming.add(pre)
        dst_inst.add_verification_copy_user(dst_depth, dst_field, self.point,
                              copy, dst_req.index, False, redop, dst_version)

    def perform_indirect_copy_verification(self, op, redop, perform_checks,
                                copies, depth, field, req, inst, versions):
        preconditions = inst.find_verification_copy_dependences(depth,
                field, self.point, op, req.index, req.is_read_only(),
                redop if req.is_reduce() else 0, versions)
        if perform_checks:
            # First check that all the copies have the appropriate dependences
            for copy in copies:
                bad = check_preconditions(preconditions, copy)
                if bad is not None:
                    # Handle a special case here of collective scatter copies
                    # They look racy to Legion Spy but they are what the user
                    # controls so it's up to the user to specify them
                    if isinstance(bad,RealmCopy) and \
                            copy.creator.index_owner is bad.creator.index_owner:
                        continue
                    print("ERROR: Missing indirect precondition for "+str(copy)+
                          " on field "+str(field)+" for "+str(op)+" on "+str(bad))
                    if op.state.eq_graph_on_error:
                        op.state.dump_eq_graph((self.point, self.field, self.tree.tree_id))
                    if op.state.assert_on_error:
                        assert False
                    return False
        else:
            # We're not verifying, so just do the analysis to register this copy
            for copy in copies:
                for pre in preconditions:
                    pre.physical_outgoing.add(copy)
                    copy.physical_incoming.add(pre)
        # Then we can register all the copies as users of the instance
        for copy in copies:
            inst.add_verification_copy_user(depth, field, self.point, copy, req.index,
                    req.is_read_only(), 0 if req.is_read_only() else redop, versions)
        return True


class Requirement(object):
    __slots__ = ['state', 'index', 'is_reg', 'index_node', 'field_space', 'tid',
                 'logical_node', 'priv', 'coher', 'redop', 'fields', 'parent',
                 'projection_function']
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
        self.projection_function = None

    def print_requirement(self):
        if self.is_reg:
            print("        Logical Region Requirement ("+hex(self.index_node)+","+
                  str(self.field_space)+","+str(self.tid)+")")
        else:
            print("        Logical Partition Requirement ("+str(self.index_node)+","+
                  str(self.field_space)+","+str(self.tid)+")")
        field_str = "          Fields: "
        first = True
        for f in self.fields:
            if first:
                field_str = field_str+str(f)
                first = False
            else:
                field_str = field_str+", "+str(f)
        print(field_str)
        print("        Privilege: "+self.get_privilege())
        print("        Coherence: "+self.get_coherence())

    def add_field(self, fid):
        field = self.field_space.get_field(fid)
        self.fields.append(field)

    def set_projection_function(self, proj_func):
        if self.projection_function:
            assert proj_func is self.projection_function
        else:
            self.projection_function = proj_func
        # Special case here for depth 0 and is_reg
        if self.projection_function.depth == 0 and self.is_reg:
            self.projection_function = None

    def is_no_access(self):
        return self.priv == NO_ACCESS

    def is_read_only(self):
        return self.priv == READ_ONLY

    def has_write(self):
        return (self.priv == READ_WRITE) or (self.priv == REDUCE) or \
                (self.priv == WRITE_DISCARD) or (self.priv == WRITE_ONLY)

    def is_write(self):
        return (self.priv == READ_WRITE) or (self.priv == WRITE_DISCARD) or \
                (self.priv == WRITE_ONLY)

    def is_read_write(self):
        return self.priv == READ_WRITE

    def is_write_only(self):
        return self.priv == WRITE_DISCARD or self.priv == WRITE_ONLY

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

    def is_projection(self):
        return self.projection_function is not None

    def to_string(self):
        if self.is_reg:
            print("Region Requirement for ("+hex(self.ispace)+","+
                  str(self.fspace)+","+str(self.tid)+")")
        else:
            print("Partition Requirement for ("+str(self.ispace)+","+
                  str(self.fspace)+","+str(self.tid)+")")
        print("    Privilege: "+self.get_privilege())
        print("    Coherence: "+self.get_coherence())
        print("    Fields: "+" ".join(self.get_field_names()))

    def get_privilege(self):
        if self.priv == NO_ACCESS:
            return "NO-ACCESS"
        elif self.priv == READ_ONLY:
            return "READ-ONLY"
        elif self.priv == READ_WRITE:
            return "READ-WRITE"
        elif self.priv == WRITE_DISCARD:
            return "WRITE-DISCARD"
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

    def __str__(self):
        return "index %d of %s and index %d of %s (type: %s)" % \
                (self.idx1, str(self.op1),
                 self.idx2, str(self.op2),
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
                 'fully_logged', 'incoming', 'outgoing', 'logical_incoming', 
                 'logical_outgoing', 'physical_incoming', 'physical_outgoing', 
                 'copy_kind', 'context_index', 'collective_src', 'collective_dst',
                 'collective_copies', 'eq_incoming', 'eq_outgoing', 'eq_privileges',
                 'start_event', 'finish_event', 'inter_close_ops', 'inlined',
                 'summary_op', 'task', 'task_id', 'predicate', 'predicate_result',
                 'futures', 'index_owner', 'points', 'index_point', 'launch_shape',
                 'creator', 'realm_copies', 'realm_fills', 'realm_depparts', 
                 'version_numbers', 'internal_idx', 'partition_kind', 'partition_node', 
                 'node_name', 'cluster_name', 'generation', 'transitive_warning_issued',
                 'arrival_barriers', 'wait_barriers', 'created_futures', 'used_futures', 
                 'intra_space_dependences', 'merged', "replayed", "restricted", "provenance"]
                  # If you add a field here, you must update the merge method
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.kind = NO_OP_KIND 
        self.context = None
        self.context_index = None
        self.name = None
        self.reqs = None
        self.mappings = None
        # If we didn't do detailed logging just assumed we are fully logged
        self.fully_logged = not self.state.detailed_logging
        self.incoming = None # Mapping dependences
        self.outgoing = None # Mapping dependences
        self.logical_incoming = None # Operation dependences
        self.logical_outgoing = None # Operation dependences
        self.physical_incoming = set() # op/realm
        self.physical_outgoing = set() # op/realm
        self.eq_incoming = None
        self.eq_outgoing = None
        self.eq_privileges = None
        self.start_event = state.get_no_event() 
        self.finish_event = state.get_no_event()
        self.inter_close_ops = None
        self.summary_op = None
        self.realm_copies = None
        self.realm_depparts = None
        self.realm_fills = None
        self.version_numbers = None
        self.predicate = None
        self.predicate_result = True
        self.futures = None
        # Only valid for tasks
        self.task = None
        self.task_id = -1
        self.index_owner = None
        self.inlined = False
        # Only valid for copies
        self.copy_kind = None
        self.collective_src = None
        self.collective_dst = None
        self.collective_copies = None
        # Only valid for index operations 
        self.points = None
        self.launch_shape = None
        # Only valid for point operations
        self.index_point = None
        # Only valid for internal operations (e.g. open, close, advance)
        self.creator = None
        self.internal_idx = -1
        # Only valid for pending partition operations
        self.partition_kind = None
        self.partition_node = None
        self.node_name = 'op_node_'+str(uid)
        self.cluster_name = None 
        # For traversals
        self.generation = 0
        self.transitive_warning_issued = False
        # Phase barrier information
        self.arrival_barriers = None
        self.wait_barriers = None
        # Future information
        self.created_futures = None
        self.used_futures = None
        # Intra-space dependences
        self.intra_space_dependences = None
        # Check if this operation was merged
        self.merged = False
        # Check if this operation was physical replayed
        self.replayed = False
        # For attach ops only - whether we should add a restriction
        self.restricted = False
        # Provenance string from the application
        self.provenance = None

    def is_close(self):
        return self.kind == INTER_CLOSE_OP_KIND or self.kind == POST_CLOSE_OP_KIND

    def is_internal(self):
        return self.is_close()

    def set_name(self, name):
        self.name = name
        if self.kind != SINGLE_TASK_KIND and self.kind != INDEX_TASK_KIND:
            self.name += " "+str(self.uid)
        if self.points is not None:
            for point in itervalues(self.points):
                point.set_name(name)

    def get_provenance(self):
        if  self.index_owner is not None:
            return self.index_owner.get_provenance()
        return self.provenance

    def __str__(self):
        if self.name is None:
            return OpNames[self.kind] + " " + str(self.uid)
        else:
            return self.name

    __repr__ = __str__

    def set_context(self, context, index=None):
        self.context = context
        # Recurse for any inter close operations
        if self.inter_close_ops:
            for close in self.inter_close_ops:
                close.set_context(context)
        # Also recurse for any points we have
        if self.points is not None:
            if self.kind == INDEX_TASK_KIND:
                for point in itervalues(self.points):
                    point.op.set_context(context)
            else:
                for point in itervalues(self.points):
                    point.set_context(context)
        # Finaly recurse for any summary operations
        if self.summary_op is not None and self.summary_op != self:
            self.summary_op.set_context(context)
        if index is not None:
            self.context.add_operation(self, index)

    def get_context(self):
        assert self.context is not None
        return self.context

    def set_op_kind(self, kind):
        if self.kind == NO_OP_KIND:
            self.kind = kind
        else:
            assert self.kind is kind
        if self.points:
            for point in itervalues(self.points):
                point.set_op_kind(kind)

    def set_events(self, start, finish):
        if start.exists():
            start.add_outgoing_op(self)
        if finish.exists():
            finish.add_incoming_op(self)
        self.start_event = start
        self.finish_event = finish
        # We know that once we've seen this then all the logging 
        # statements for the operation are done being performed
        self.fully_logged = True

    def set_task_id(self, task_id):
        assert self.kind == SINGLE_TASK_KIND or self.kind == INDEX_TASK_KIND
        if self.task_id != -1:
            assert task_id == self.task_id
        else:
            self.task_id = task_id

    def set_creator(self, creator, idx):
        # Better be an internal op kind
        assert self.kind == INTER_CLOSE_OP_KIND or \
            self.kind == POST_CLOSE_OP_KIND
        self.creator = creator
        self.internal_idx = idx
        # If our parent context created us we don't need to be recorded 
        if creator is not self.context.op:
            assert self.kind != POST_CLOSE_OP_KIND
            creator.add_close_operation(self)
        else:
            assert self.kind == POST_CLOSE_OP_KIND

    def add_launch_rect(self, rect):
        if self.launch_shape is None:
            self.launch_shape = Shape()
        self.launch_shape.add_rect(rect)

    def set_predicate(self, pred):
        self.predicate = pred
        if self.logical_incoming is None:
            self.logical_incoming = set()
        self.logical_incoming.add(pred)
        if pred.logical_outgoing is None:
            pred.logical_outgoing = set()
        pred.logical_outgoing.add(self)

    def set_replayed(self):
        self.replayed = True
        # Once we see this then we know all the logging operations
        # for the operation have been performed
        self.fully_logged = True

    def get_index_launch_shape(self):
        assert self.launch_shape
        return self.launch_shape

    def add_close_operation(self, close):
        if self.inter_close_ops is None:
            self.inter_close_ops = list()
        self.inter_close_ops.append(close)

    def set_summary_operation(self, summary):
        self.summary_op = summary

    def get_depth(self):
        assert self.context is not None
        return self.context.get_depth() + 1

    def get_logical_op(self):
        return self

    def get_close_operation(self, req, node, field, read_only):
        if self.inter_close_ops is None:
            return None
        for close in self.inter_close_ops:
            #if close.internal_idx != req.index:
                #continue
            assert len(close.reqs) == 1
            close_req = close.reqs[0]
            if close_req.logical_node is not node:
                continue
            if field not in close_req.fields:
                continue 
            if not read_only and close.kind != INTER_CLOSE_OP_KIND:
                continue
            return close
        return None

    def set_pending_partition_info(self, node, kind):
        assert self.kind == PENDING_PART_OP_KIND
        self.partition_node = node
        self.partition_kind = kind

    def set_index_owner(self, owner, point):
        assert not self.index_owner
        self.index_owner = owner
        assert not self.index_point
        self.index_point = point

    def add_point_task(self, point):
        assert self.kind == INDEX_TASK_KIND
        # Initialize if necessary
        if self.points is None:
            self.points = dict()
        point.op.set_name(self.name)
        index_point = point.point
        point.op.set_index_owner(self, index_point)
        if index_point in self.points:
            self.points[index_point] = self.state.alias_index_points(point,
                                                  self.points[index_point])
        else:
            self.points[index_point] = point
        if self.context is not None:
            self.points[index_point].op.set_context(self.context)

    def add_point_op(self, op, point):
        op.kind = self.kind
        op.set_index_owner(self, point)
        # Initialize if necessary
        if self.points is None:
            self.points = dict()
        assert point not in self.points
        self.points[point] = op
        if self.context is not None:
            op.set_context(self.context)

    def add_requirement(self, requirement):
        if self.reqs is None:
            self.reqs = dict()
        assert requirement.index not in self.reqs
        self.reqs[requirement.index] = requirement

    def add_requirement_field(self, index, fid):
        assert self.reqs is not None
        assert index in self.reqs
        self.reqs[index].add_field(fid)

    def set_projection_function(self, index, proj_func):
        assert self.reqs is not None
        assert index in self.reqs
        self.reqs[index].set_projection_function(proj_func)

    def add_mapping_decision(self, index, fid, inst):
        if self.mappings is None:
            self.mappings = dict()
        if index not in self.mappings:
            self.mappings[index] = dict()
        self.mappings[index][fid] = inst

    def set_predicate_result(self, result):
        self.predicate_result = result
        # If we predicated this false then we don't expect
        # to see any additional logging for it
        if not result:
            self.fully_logged = True

    def add_future(self, future):
        if not self.futures:
            self.futures = set()
        self.futures.add(future)

    def update_instance_uses(self):
        if self.mappings:
            for mapping in itervalues(self.mappings):
                unique_insts = set()
                for inst in itervalues(mapping):
                    unique_insts.add(inst) 
                for inst in unique_insts:
                    inst.increment_use_count() 

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

    def add_equivalence_incoming(self, eq, src):
        assert eq in self.eq_privileges
        if self.eq_incoming is None:
            self.eq_incoming = dict()
        if eq not in self.eq_incoming:
            self.eq_incoming[eq] = set()
        self.eq_incoming[eq].add(src)

    def add_equivalence_outgoing(self, eq, dst):
        assert eq in self.eq_privileges
        if self.eq_outgoing is None:
            self.eq_outgoing = dict()
        if eq not in self.eq_outgoing:
            self.eq_outgoing[eq] = set()
        self.eq_outgoing[eq].add(dst)

    def add_realm_copy(self, copy):
        if self.realm_copies is None:
            self.realm_copies = list()
        self.realm_copies.append(copy)

    def add_realm_fill(self, fill):
        if self.realm_fills is None:
            self.realm_fills = list()
        self.realm_fills.append(fill)

    def add_realm_deppart(self, deppart):
        if self.realm_depparts is None:
            self.realm_depparts = list()
        self.realm_depparts.append(deppart)

    def add_arrival_barrier(self, bar):
        if not self.arrival_barriers:
            self.arrival_barriers = list()
        self.arrival_barriers.append(bar)

    def add_wait_barrier(self, bar):
        if not self.wait_barriers:
            self.wait_barriers = list()
        self.wait_barriers.append(bar)

    def add_created_future(self, future):
        if not self.created_futures:
            self.created_futures = set()
        self.created_futures.add(future)

    def add_used_future(self, future):
        if not self.used_futures:
            self.used_futures = set()
        self.used_futures.add(future)

    def get_point_task(self, point):
        assert self.kind == INDEX_TASK_KIND
        assert point in self.points
        return self.points[point]

    def get_equivalence_privileges(self):
        if self.eq_privileges is None:
            self.eq_privileges = dict()
            if self.reqs is not None:
                for req in itervalues(self.reqs):
                    point_set = req.index_node.get_point_set()
                    for point in point_set.iterator():
                        for field in req.fields:
                            # Point,Field,Tree ID
                            key = (point,field,req.tid)
                            if key not in self.eq_privileges:
                                self.eq_privileges[key] = req.priv
                            elif self.launch_shape is None:
                                # If we have aliased region requirements
                                # then they shouldn't interfere with each other
                                # However, some privileges can appear to interfere
                                # for projection requirements which ultimately
                                assert self.eq_privileges[key] >= req.priv 
        return self.eq_privileges

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

    def add_intra_space_dependence(self, dep):
        if self.intra_space_dependences is None:
            self.intra_space_dependences = set()
        self.intra_space_dependences.add(dep)

    def merge(self, other):
        if self.kind == NO_OP_KIND:
            self.kind = other.kind
        elif other.kind != NO_OP_KIND:
            assert self.kind == other.kind
        if self.context is None:
            self.context = other.context
        elif other.context is not None:
            assert self.context == other.context
        if self.name is None:
            self.name = other.name
        if self.provenance is None:
            self.provenance = other.provenance
        elif other.provenance is not None:
            assert self.provenance == other.provenance
        self.fully_logged = self.fully_logged or other.fully_logged
        if not self.reqs:
            self.reqs = other.reqs
        elif other.reqs:
            for idx,req in iteritems(other.reqs):
                assert idx not in self.reqs
                self.reqs[idx] = req
        if not self.mappings:
            self.mappings = other.mappings
        elif other.mappings:
            for idx,mapping in iteritems(other.mappings):
                assert idx not in self.mappings
                self.mappings[idx] = mapping
        if not self.start_event.exists():
            self.start_event = other.start_event
            if self.start_event.exists():
                self.start_event.update_outgoing_op(other, self)
        else:
            assert not other.start_event.exists()
        if not self.finish_event.exists():
            self.finish_event = other.finish_event
            if self.finish_event.exists():
                self.finish_event.update_incoming_op(other, self)
        else:
            assert not other.finish_event.exists() 
        if not self.inter_close_ops:
            self.inter_close_ops = other.inter_close_ops
        else:
            assert not other.inter_close_ops
        if not self.realm_copies:
            self.realm_copies = other.realm_copies
            if self.realm_copies:
                for copy in self.realm_copies:
                    copy.update_creator(self)
        else:
            assert not other.realm_copies
        if not self.realm_fills:
            self.realm_fills = other.realm_fills
            if self.realm_fills:
                for fill in self.realm_fills:
                    fill.update_creator(self)
        else:
            assert not other.realm_fills
        if not self.realm_depparts:
            self.realm_depparts = other.realm_depparts
            if self.realm_depparts:
                for deppart in self.realm_depparts:
                    deppart.update_creator(self)
        else:
            assert not other.realm_deppart
        if self.task_id == -1:
            self.task_id = other.task_id
        elif other.task_id != -1:
            assert self.task_id == other.task_id
        # Should only be called on point tasks
        assert not self.points
        assert not other.points
        other.merged = True
        self.replayed = self.replayed or other.replayed

    def record_current_version(self, point, field, tree, version_number):
        if not self.version_numbers:
            self.version_numbers = dict()
        eq_key = (point, field, tree)
        assert eq_key in self.eq_privileges
        if eq_key in self.version_numbers:
            assert self.version_numbers[eq_key] == version_number
        else:
            self.version_numbers[eq_key] = version_number

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
            traverser = EventGraphTraverser(forwards=False, use_gen=True,
                generation=self.state.get_next_traversal_generation(),
                op_fn=traverse_node, copy_fn=traverse_node, 
                fill_fn=traverse_node, deppart_fn=traverse_node)
            traverser.reachable = self.physical_incoming
            traverser.run(self.start_event)
            # Keep everything symmetric
            for other in self.physical_incoming:
                other.physical_outgoing.add(self)
        if self.finish_event.exists():
            traverser = EventGraphTraverser(forwards=True, use_gen=True,
                generation=self.state.get_next_traversal_generation(),
                op_fn=traverse_node, copy_fn=traverse_node, 
                fill_fn=traverse_node, deppart_fn=traverse_node)
            traverser.reachable = self.physical_outgoing
            traverser.run(self.finish_event)
            # Keep everything symmetric
            for other in self.physical_outgoing:
                other.physical_incoming.add(self)

    def find_or_create_fill(self, req, field, dst, fill_op):
        # Run through our copies and see if we can find one that matches
        if self.realm_fills:
            for fill in self.realm_fills:
                if req.tid != fill.dst_tree_id:
                    continue
                if field not in fill.fields:
                    continue
                idx = fill.fields.index(field)
                if dst is not fill.dsts[idx]:
                    continue
                return fill
        else:
            self.realm_fills = list()
        fill = self.state.create_fill(self)
        fill.set_tree_properties(None, req.field_space, req.tid)
        fill.add_field(field.fid, dst)
        self.realm_fills.append(fill)
        if fill_op is not None:
            fill.set_fill_op(fill_op)
        return fill

    def find_verification_copy_across(self, src_field, dst_field, point,
                                      src_inst, dst_inst, redop):
        if not self.realm_copies:
            return None
        for copy in self.realm_copies:
            if not copy.index_expr.get_point_set().has_point(point):
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
            if redop != copy.redops[index]:
                continue
            return copy
        return None

    def find_verification_indirection_copy(self, src_field, src_inst, src_req,
                                           dst_field, dst_inst, dst_req,
                                           src_idx_field, src_idx_inst, src_idx_req,
                                           dst_idx_field, dst_idx_inst, dst_idx_req, redop):
        if not self.realm_copies:
            return None
        for copy in self.realm_copies:
            if copy.indirections is None:
                continue 
            if src_field not in copy.src_fields:
                continue
            index = copy.src_fields.index(src_field)
            if src_idx_req is not None:
                if copy.srcs[index] is not None:
                    continue
                src_index = copy.src_indirections[index] 
                if not copy.indirections.has_group_instance(src_index,
                            src_inst, src_req.logical_node.index_space):
                    continue
                if not copy.indirections.has_indirect_instance(src_index,
                                            src_idx_inst, src_idx_field):
                    continue
                src_copy_space = src_idx_req.logical_node.index_space
            else:
                if src_inst is not copy.srcs[index]:
                    continue
                src_copy_space = src_req.logical_node.index_space
            if dst_field is not copy.dst_fields[index]:
                continue
            if dst_idx_req is not None:
                if copy.dsts[index] is not None:
                    continue
                dst_index = copy.dst_indirections[index]
                if not copy.indirections.has_group_instance(dst_index,
                            dst_inst, dst_req.logical_node.index_space):
                    continue
                if not copy.indirections.has_indirect_instance(dst_index,
                                            dst_idx_inst, dst_idx_field):
                    continue
                dst_copy_space = dst_idx_req.logical_node.index_space
            else:
                if dst_inst is not copy.dsts[index]:
                    continue
                dst_copy_space = dst_req.logical_node.index_space
            # Check that the copy domain is equal to the intersection
            # of the source and destination copy expressions
            if src_copy_space is dst_copy_space:
                if copy.index_expr.get_index_space() is not src_copy_space and \
                    copy.index_expr.get_point_set() != src_copy_space.get_point_set():
                    continue
            else:
                point_set = src_copy_space.get_point_set() & dst_copy_space.get_point_set()
                if point_set != copy.index_expr.get_point_set():
                    continue
            if redop != copy.redops[index]:
                continue
            return copy
        return None

    def find_or_create_copy(self, req, field, src, dst):
        # Run through our copies and see if we can find one that matches
        if self.realm_copies:
            for copy in self.realm_copies:
                if req.tid != copy.src_tree_id or req.tid != copy.dst_tree_id:
                    continue
                if field not in copy.src_fields:
                    continue
                idx = copy.src_fields.index(field)
                if field.fid != copy.dst_fields[idx]:
                    continue
                if src is not copy.srcs[idx]:
                    continue
                if dst is not copy.dsts[idx]:
                    continue
                return copy
        else:
            self.realm_copies = list()
        # If we get here we have to make our copy
        copy = self.state.create_copy(self)
        copy.set_tree_properties(None, req.tid, req.tid)
        copy.add_field(field.fid, src, field.fid, dst, src.redop)
        self.realm_copies.append(copy)
        return copy

    def find_or_create_copy_across(self, src_inst, src_field, src_req,
                                   dst_inst, dst_field, dst_req, redop):
        if self.realm_copies:
            for copy in self.realm_copies:
                if src_req.tid != copy.src_tree_id or dst_req.tid != copy.dst_tree_id:
                    continue
                if src_field.fid not in copy.src_fields:
                    continue
                idx = copy.src_fields.index(src_field.fid)
                if dst_field.fid != copy.dst_fields[idx]:
                    continue
                if src_inst is not copy.srcs[idx]:
                    continue
                if dst_inst is not copy.dsts[idx]:
                    continue
                return copy
        else:
            self.realm_copies = list()
        # If we get here we have to make our own copy
        copy = self.state.create_copy(self)
        copy.set_tree_properties(None, src_req.tid, dst_req.tid)
        copy.add_field(src_field.fid, src_inst, dst_field.fid, dst_inst, redop)
        self.realm_copies.append(copy)
        return copy

    def find_or_create_indirection_copy(self, src_field, src_inst, src_req,
                                        dst_field, dst_inst, dst_req,
                                        src_idx_field, src_idx_inst, src_idx_req,
                                        dst_idx_field, dst_idx_inst, dst_idx_req, redop):
        assert src_idx_field is not None or dst_idx_field is not None
        if self.realm_copies:
            for copy in self.realm_copies:
                if copy.indirections is None:
                    continue 
                if src_field not in copy.src_fields:
                    continue
                index = copy.src_fields.index(src_field)
                if src_idx_req is not None:
                    if copy.srcs[index] is not None:
                        continue
                    src_index = copy.src_indirections[index] 
                    if not copy.indirections.has_group_instance(src_index,
                                src_inst, src_req.logical_node.index_space):
                        continue
                    if not copy.indirections.has_indirect_instance(src_index,
                                                src_idx_inst, src_idx_field):
                        continue
                    src_copy_space = src_idx_req.logical_node.index_space
                else:
                    if src_inst is not copy.srcs[index]:
                        continue
                    src_copy_space = src_req.logical_node.index_space
                if dst_field is not copy.dst_fields[index]:
                    continue
                if dst_idx_req is not None:
                    if copy.dsts[index] is not None:
                        continue
                    dst_index = copy.dst_indirections[index]
                    if not copy.indirections.has_group_instance(dst_index,
                                dst_inst, dst_req.logical_node.index_space):
                        continue
                    if not copy.indirections.has_indirect_instance(dst_index,
                                                dst_idx_inst, dst_idx_field):
                        continue
                    dst_copy_space = dst_idx_req.logical_node.index_space
                else:
                    if dst_inst is not copy.dsts[index]:
                        continue
                    dst_copy_space = dst_req.logical_node.index_space
                # Check that the copy domain is equal to the intersection
                # of the source and destination copy expressions
                if src_copy_space is dst_copy_space:
                    if copy.index_expr.get_index_space() is not src_copy_space and \
                        copy.index_expr.get_point_set() != src_copy_space.get_point_set():
                        continue
                else:
                    point_set = src_copy_space.get_point_set() & dst_copy_space.get_point_set()
                    if point_set != copy.index_expr.get_point_set():
                        continue
                if redop != copy.redops[index]:
                    continue
                return copy
        else:
            self.realm_copies = list()
        # If we get here we have to make our own copy
        copy = self.state.create_copy(self)       
        # Only need to fill in indirections here with local instances
        # If there are collective things we'll update them later when
        # we exchange the copies between the points of the index owner
        index_expr = None
        if src_idx_req is not None:
            index_expr = src_idx_req.logical_node.index_space
        else:
            index_expr = dst_idx_req.logical_node.index_space
        indirections = self.state.create_indirections()
        index = 0
        if src_idx_req is not None:
            indirections.add_indirect_instance(index, src_idx_inst, src_idx_field.fid)
            indirections.add_group_instance(index, src_inst, 
                            src_req.logical_node.index_space)
            src_index = index
            src = None
            index += 1
        else:
            src_index = -1
            src = src_inst
        if dst_idx_req is not None:
            indirections.add_indirect_instance(index, dst_idx_inst, dst_idx_field.fid)
            indirections.add_group_instance(index, dst_inst,
                            dst_req.logical_node.index_space)
            dst_index = index
            dst = None
        else:
            dst_index = -1
            dst = dst_inst
        copy.set_indirection_properties(index_expr, indirections)
        copy.add_indirect_field(src_field.fid, src, src_index, 
                            dst_field.fid, dst, dst_index, redop) 
        self.realm_copies.append(copy)
        return copy

    def perform_cycle_check(self, cycle_detector):
        return cycle_detector.check_for_cycles(self, self.physical_outgoing) 

    def is_interfering_index_space_launch(self):
        if self.reqs is None or self.points is None:
            return False
        all_reqs = list()
        # Find all non-projection requirements, and ensure that they are
        # compatible with themselves (as they will be used by all point tasks)
        for req in itervalues(self.reqs):
            if not req.is_projection():
                if len(self.points) > 1:
                    dep_type = compute_dependence_type(req, req)
                    if dep_type == TRUE_DEPENDENCE or dep_type == ANTI_DEPENDENCE:
                        print(("Non index region requirement %d of index space "
                               "operation %s is self interfering in %s") %
                               (req.index,str(self),str(self.context)))
                        return True
        if self.kind == INDEX_TASK_KIND:
            for point_task in itervalues(self.points):
                for req in itervalues(point_task.op.reqs):
                    all_reqs.append((req,point_task.op))
        else:
            for point in itervalues(self.points):
                for req in itervalues(point.reqs):
                    all_reqs.append((req,point))
        order_points = False
        # All requirements should be non interfering
        for idx1 in xrange(0, len(all_reqs)):
            req1,op1 = all_reqs[idx1]
            if req1.is_no_access():
                continue
            for idx2 in xrange(idx1+1, len(all_reqs)):
                req2,op2 = all_reqs[idx2]
                if req2.is_no_access():
                    continue
                if req1.parent.tree_id != req2.parent.tree_id:
                    continue
                # No interference if the fields are disjoint
                fields_disjoint = True
                for field in req1.fields:
                    if field in req2.fields:
                        fields_disjoint = False
                        break
                if fields_disjoint:
                    continue 
                # Check for interference at a common ancestor
                aliased,ancestor = self.state.has_aliased_ancestor_tree_only(
                    req1.logical_node.get_index_node(),
                    req2.logical_node.get_index_node())
                if aliased:
                    assert ancestor
                    dep_type = compute_dependence_type(req1, req2)
                    if dep_type != NO_DEPENDENCE:
                        # Check for invertible projection functions which can 
                        # the runtime knows how to handle their dependences
                        if req1.index == req2.index and \
                                self.reqs[req1.index].projection_function is not None and \
                                self.reqs[req1.index].projection_function.invertible:
                            # Check to make sure we find a dependence going one way or
                            # the other between the two operations
                            if op1.intra_space_dependences is not None and \
                                    op2.index_point in op1.intra_space_dependences:
                                order_points = True
                                continue
                            if op2.intra_space_dependences is not None and \
                                    op1.index_point in op2.intra_space_dependences:
                                order_points = True
                                continue
                            print(("Missing intra space dependence between requirements "+
                                   "%d and %d of points %s and %s of %s in %s") %
                                   (req1.index,req2.index,str(op1),str(op2),
                                       str(self),str(self.context)))
                        else:
                            # The normal bad path
                            print(("Region requirements %d and %d of operation %s "+
                                   "are interfering in %s") %
                                   (req1.index,req2.index,str(self),str(self.context)))
                        # Everything here fails
                        return True
        new_points = collections.OrderedDict()
        if order_points:
            # If we have intra-space dependences between points then put
            # them in an ordered dictionary so that whenever we iterate
            # over them we do them in an order consistent with their deps
            # This should only happen for index task kinds currently
            if self.kind == INDEX_TASK_KIND:
                satisfied_deps = set()
                remaining_tasks = dict()
                for point,task in iteritems(self.points):
                    if task.op.intra_space_dependences is None:
                        new_points[point] = task
                        satisfied_deps.add(point)
                    else:
                        remaining_tasks[point] = task
                while remaining_tasks:
                    next_remaining = dict()
                    for point,task in iteritems(remaining_tasks):
                        satisfied = True
                        for dep in task.op.intra_space_dependences:
                            if dep not in satisfied_deps:
                                satisfied = False
                                break
                        if satisfied:
                            new_points[point] = task
                            satisfied_deps.add(point)
                        else:
                            next_remaining[point] = task
                    remaining_tasks = next_remaining
            else:
                satisfied_deps = set()
                remaining_ops = dict()
                for point,op in iteritems(self.points):
                    if op.intra_space_dependences is None:
                        new_points[point] = op
                        satisfied_deps.add(point)
                    else:
                        remaining_ops[point] = op
                while remaining_ops:
                    next_remaining = dict()
                    for point,op in iteritems(remaining_ops):
                        satisfied = True
                        for dep in op.intra_space_dependences:
                            if dep not in satisfied_deps:
                                satisfied = False
                                break
                        if satisfied:
                            new_points[point] = op 
                            satisfied_deps.add(point)
                        else:
                            next_remaining[point] = task
                    remaining_ops = next_remaining
        else:
            # Let's put things in order by their UID
            if self.kind == INDEX_TASK_KIND:
                for point,task in sorted(iteritems(self.points), key=lambda x: x[1].op.uid):
                    new_points[point] = task
            else:
                for point,op in sorted(iteritems(self.points), key=lambda x: x[1].uid):
                    new_points[point] = op
        assert len(new_points) == len(self.points)
        self.points = new_points
        return False

    def analyze_logical_requirement(self, index, perform_checks):
        assert index in self.reqs
        req = self.reqs[index]
        # Special out for no access
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
        # See if we are restricted in any way
        assert self.context
        # Now do the traversal for each of the fields
        for field in req.fields:
            # See if we have any aliased fields in future region requirements
            aliased_children = set()
            for idx in xrange(index+1,len(self.reqs)):
                other_req = self.reqs[idx]
                if other_req.priv is NO_ACCESS:
                    continue
                # See if the trees are the same
                if req.parent.tree_id != other_req.parent.tree_id:
                    continue
                # See if they have the same field
                if field not in other_req.fields:
                    continue
                # Now see if they have a common aliased ancestor
                aliased,ancestor = self.state.has_aliased_ancestor_tree_only(
                    req.logical_node.get_index_node(), 
                    other_req.logical_node.get_index_node())
                if aliased:
                    assert ancestor
                    dep_type = compute_dependence_type(req, other_req)
                    if dep_type != NO_DEPENDENCE:
                        # Only report this at least one is not a projection requirement
                        if req.projection_function is None or \
                            other_req.projection_function is None:
                            print(("Region requirements %d and %d of operation %s "+
                                   "are interfering in %s") % 
                                   (index,idx,str(self),str(self.context)))
                            if self.state.assert_on_error:
                                assert False
                            return False
                    aliased_children.add(ancestor.depth) 
            # Keep track of the previous dependences so we can 
            # use them for adding/checking dependences on close operations
            previous_deps = list()
            if not req.parent.perform_logical_analysis(0, path, self, req, field,
                                        False, True, False, False, previous_deps,
                                        aliased_children, perform_checks):
                return False
        # Restore the privileges if necessary
        if copy_reduce:
            req.priv = REDUCE
        return True

    def analyze_logical_fence(self, perform_checks):
        # Find all the operations since the previous fence and then make sure
        # we either depend on them directly or we have a transitive dependence
        start_index = 0 if self.context.current_fence is None else \
            self.context.operations.index(self.context.current_fence)
        stop_index = self.context.operations.index(self)
        for index in xrange(start_index, stop_index):
            prev_op = self.context.operations[index]
            if prev_op.replayed:
                continue
            if perform_checks:
                found = False
                if self.incoming:
                    for dep in self.incoming:
                        if dep.op1 is not prev_op:
                            # If the prev op is a close op see if we have a dependence 
                            # on its creator
                            # We need this transitivity to deal with tracing properly
                            if prev_op.is_internal() and prev_op.creator is dep.op1 and \
                                prev_op.internal_idx == dep.idx1:
                                found = True
                                break
                            continue
                        found = True
                        break
                if not found and not self.has_transitive_mapping_dependence(prev_op):
                    print(("ERROR: missing logical fence dependence between %s "+
                          "(UID %s) and %s (UID %s)") % (prev_op, prev_op.uid, self, self.uid))
                    if self.state.assert_on_error:
                        assert False
                    return False 
            else:
                dep = MappingDependence(prev_op, self, 0, 0, TRUE_DEPENDENCE)
                prev_op.add_outgoing(dep)
                self.add_incoming(dep)
        return True

    def analyze_logical_deletion(self, index, perform_checks):
        assert index in self.reqs
        req = self.reqs[index]
        path = list()
        req.logical_node.compute_path(path, req.parent)
        assert not not path
        for field in req.fields:
            previous_deps = list()
            if not req.parent.perform_logical_deletion(0, path, self, req, field, 
                                            False, previous_deps, perform_checks):
                return False
        return True

    def perform_logical_analysis(self, perform_checks):
        if self.replayed:
            return True
        # We need a context to do this
        assert self.context is not None
        # If this operation was predicated false, then there is nothing to do
        if self.predicate and not self.predicate_result:
            return True
        # See if there is a fence in place for this context
        if self.context.current_fence is not None:
            if perform_checks:
                if self.logical_incoming is None or \
                        self.context.current_fence not in self.logical_incoming: 
                    print("ERROR: missing logical fence dependence between "+
                          str(self.context.current_fence)+" and "+str(self))
                    if self.state.assert_on_error:
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
        if self.kind == DELETION_OP_KIND:
            # Perform dependence analysis on the deletion region requirements
            for idx in iterkeys(self.reqs):
                if not self.analyze_logical_deletion(idx, perform_checks):
                    return False
            return True
        for idx in xrange(0,len(self.reqs)):
            if not self.analyze_logical_requirement(idx, perform_checks):
                return False
        return True

    def has_mapping_dependence(self, req, prev_op, prev_req, dtype, field):
        if self.incoming:
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
        # Look for the dependence transitively
        if self.has_transitive_mapping_dependence(prev_op):
            return True
        # No need to look for it transitively since this analysis should exactly
        # match the analysis done by the runtime
        # Issue the error and return false
        print("ERROR: Missing mapping dependence on "+str(field)+" between region "+
              "requirement "+str(prev_req.index)+" of "+str(prev_op)+" (UID "+
              str(prev_op.uid)+") and region requriement "+str(req.index)+" of "+
              str(self)+" (UID "+str(self.uid)+")")
        if self.state.bad_graph_on_error:
            self.state.dump_bad_graph(prev_op.context, req.logical_node.tree_id, field)
        if self.state.assert_on_error:
            assert False
        return False

    def has_transitive_mapping_dependence(self, prev_op):
        # A common example of imprecision is in how the runtime checks
        # for groups of open children that are aliased but non-interfering
        # for reductions. The runtime will group them all together while
        # Legion Spy is smart enough to keep them separate.
        if not self.transitive_warning_issued:
            print("INFO: Falling back to transitive mapping dependences for "+str(self)+
                  " due to either imprecision in the runtime analysis or a bug.")
            self.transitive_warning_issued = True
        next_gen = self.state.get_next_traversal_generation()
        self.generation = next_gen
        queue = collections.deque()
        queue.append(self)
        while queue:
            current = queue.popleft()
            if current is prev_op:
                return True
            if not current.logical_incoming:
                continue
            for next_op in current.logical_incoming:
                if next_op.generation == next_gen:
                    continue
                next_op.generation = next_gen
                queue.append(next_op)
        return False

    def analyze_previous_interference(self, next_op, next_req, reachable):
        if not self.reqs:
            # Check to see if this is a fence operation
            if self.kind == FENCE_OP_KIND:
                # If we've got a fence operation, it should have a transitive
                # dependence on all operations that came before it
                if not self in reachable:
                    print("ERROR: Failed logical sanity check. No mapping dependence "+
                          "between previous "+str(self)+" and later "+str(next_op))
                    if self.state.assert_on_error:
                        assert False
                    return False
            return True
        for req in itervalues(self.reqs):
            # Check to see if there is any overlap in fields or regions 
            if len(set(req.fields) & set(next_req.fields)) == 0:
                continue # No overlapping fields so we can keep going
            # Check to see if they are in different region trees in 
            # which case there can be no aliasing
            if req.tid != next_req.tid:
                continue
            if not req.index_node.intersects(next_req.index_node):
                continue
            dep_type = compute_dependence_type(req, next_req) 
            if dep_type == NO_DEPENDENCE:
                continue
            # Otherwise they do interfere, check to see if this node is
            # in the previous set, if not, that is not good
            if self not in reachable:
                print("ERROR: Failed logical sanity check. No mapping dependence "+
                      "path exists between region requirement "+str(req.index)+
                      " of "+str(self)+" and region requirement "+str(next_req.index)+
                      " of "+str(next_op))
                print("  First Requirement:")
                req.print_requirement()
                print("  Second Requirement:")
                next_req.print_requirement()
                if self.state.assert_on_error:
                    assert False
                return False
        return True

    def analyze_logical_interference(self, prev_op, reachable):
        for req in itervalues(self.reqs):  
            if not prev_op.analyze_previous_interference(self, req, reachable):
                return False
        return True

    def find_mapping(self, index):
        if self.mappings is None or index not in self.mappings:
            return None
        return self.mappings[index]

    def check_for_spurious_realm_ops(self, perform_checks):
        if not perform_checks:
            return True
        # TODO: We would want Legion Spy to check if reduction instances
        #       are initialized correctly before they are reused in a replay.
        if self.replayed:
            return True
        if self.realm_copies:
            for copy in self.realm_copies:
                if not copy.check_for_spurious_updates(copy.dst_fields, copy.dst_tree_id,
                        copy.across_version_numbers if copy.is_across() 
                        else copy.version_numbers):
                    return False
        if self.realm_fills:
            for fill in self.realm_fills:
                if not fill.check_for_spurious_updates(fill.fields, fill.dst_tree_id,
                        fill.across_version_numbers if fill.is_across()
                        else fill.version_numbers):
                    return False
        return True

    def compute_current_version_numbers(self):
        assert self.context
        # Now do all of our region requirements
        assert not self.version_numbers
        for index,req in iteritems(self.reqs):
            if self.mappings and index in self.mappings:
                mapping = self.mappings[index]
            else:
                mapping = None
            depth = self.context.find_enclosing_context_depth(req, mapping)
            for field in req.fields:
                req.logical_node.compute_current_version_numbers(depth, field, self, req.tid)

    def verify_copy_requirements(self, src_idx, src_req, dst_idx, dst_req, perform_checks):
        # If this was predicated there might not be any mappings
        if not self.mappings:
            return True
        # We always copy from the destination point set which 
        # should be smaller than the src point set
        dst_points = dst_req.logical_node.get_point_set()
        assert (dst_points - src_req.logical_node.get_point_set()).empty()
        # Get the mappings
        src_mappings = self.find_mapping(src_req.index)
        dst_mappings = self.find_mapping(dst_req.index)
        assert self.context
        src_depth = self.context.find_enclosing_context_depth(src_req, src_mappings) 
        dst_depth = self.context.find_enclosing_context_depth(dst_req, dst_mappings)
        assert len(src_req.fields) == len(dst_req.fields)
        for fidx in xrange(len(src_req.fields)):
            src_field = src_req.fields[fidx]
            dst_field = dst_req.fields[fidx]
            assert src_field.fid in src_mappings
            assert dst_field.fid in dst_mappings
            src_inst = src_mappings[src_field.fid]
            dst_inst = dst_mappings[dst_field.fid]
            assert not dst_inst.is_virtual()
            is_reduce = dst_req.is_reduce()
            # Switch this to read-write privileges and then switch it back 
            # after we are done. The runtime does this too so that its 
            # analysis is correct
            copy_redop = 0
            if is_reduce:
                copy_redop = dst_req.redop
                dst_req.redop = 0
                dst_req.priv = READ_WRITE 
            if not src_inst.is_virtual() and \
                not src_req.logical_node.perform_physical_verification(
                      src_depth, src_field, self, src_req, src_inst, 
                      perform_checks, False):
                return False
            # Record the destination version numbers
            dst_versions = dict()
            if not dst_req.logical_node.perform_physical_verification(
                      dst_depth, dst_field, self, dst_req, dst_inst,
                      perform_checks, False, None, dst_versions):
                return False
            # Now we can issue the copy across
            if is_reduce:
                # Reduction case
                assert copy_redop != 0
                if src_inst.is_virtual():
                    # This is a runtime bug, there should never be
                    # any reductions across with a virtual source instance
                    assert False
                    return False
                if not src_req.logical_node.perform_copy_across_verification(
                      self, copy_redop, perform_checks, dst_points,  
                      src_depth, src_field, src_req, src_inst,
                      dst_depth, dst_field, dst_req, dst_inst, dst_versions):
                    return False
            else:
                # Normal copy across
                if not src_req.logical_node.perform_copy_across_verification(
                      self, copy_redop, perform_checks, dst_points,
                      src_depth, src_field, src_req, src_inst,
                      dst_depth, dst_field, dst_req, dst_inst, dst_versions):
                    return False
            # Restore these when we are done
            if is_reduce:
                dst_req.priv = REDUCE
                dst_req.redop = copy_redop
        return True

    def find_collective_copies(self, copy_index, perform_checks, src_field, 
                               dst_field, src_idx_field, dst_idx_field, redop):
        assert self.kind == COPY_OP_KIND
        assert self.points is not None
        assert self.copy_kind is not None and self.copy_kind > 0
        key = (copy_index,src_field.fid,dst_field.fid,
                src_idx_field.fid if src_idx_field is not None else None,
                dst_idx_field.fid if dst_idx_field is not None else None, redop)
        if self.collective_copies is not None:
            if key in self.collective_copies:
                return self.collective_copies[key]
        else:
            self.collective_copies = dict()
        copies = set()
        if self.copy_kind == 1:
            assert src_idx_field is not None
            assert dst_idx_field is None
            assert len(self.reqs) % 3 == 0
            num_copies = len(self.reqs) // 3
            src_index = copy_index
            dst_index = copy_index + num_copies
            idx_index = copy_index + 2*num_copies
            for point in itervalues(self.points):
                src_mappings = point.find_mapping(src_index)
                dst_mappings = point.find_mapping(dst_index)
                idx_mappings = point.find_mapping(idx_index)
                if perform_checks:
                    copy = point.find_verification_indirection_copy(src_field,
                            src_mappings[src_field.fid], point.reqs[src_index],
                            dst_field, dst_mappings[dst_field.fid], point.reqs[dst_index], 
                            src_idx_field, idx_mappings[src_idx_field.fid], 
                            point.reqs[idx_index], None, None, None, redop)
                else:
                    copy = point.find_or_create_indirection_copy(src_field,
                            src_mappings[src_field.fid], point.reqs[src_index],
                            dst_field, dst_mappings[dst_field.fid], point.reqs[dst_index], 
                            src_idx_field, idx_mappings[src_idx_field.fid], 
                            point.reqs[idx_index], None, None, None, redop)
                if copy is not None:
                    copies.add(copy)
        elif self.copy_kind == 2:
            assert src_idx_field is None
            assert dst_idx_field is not None
            assert len(self.reqs) % 3 == 0
            num_copies = len(self.reqs) // 3
            src_index = copy_index
            dst_index = copy_index + num_copies
            idx_index = copy_index + 2*num_copies
            for point in itervalues(self.points):
                src_mappings = point.find_mapping(src_index)
                dst_mappings = point.find_mapping(dst_index)
                idx_mappings = point.find_mapping(idx_index)
                if perform_checks:
                    copy = point.find_verification_indirection_copy(src_field,
                            src_mappings[src_field.fid], point.reqs[src_index],
                            dst_field, dst_mappings[dst_field.fid], point.reqs[dst_index], 
                            None, None, None, dst_idx_field,
                            idx_mappings[dst_idx_field.fid], point.reqs[idx_index], redop)
                else:
                    copy = point.find_or_create_indirection_copy(src_field,
                            src_mappings[src_field.fid], point.reqs[src_index],
                            dst_field, dst_mappings[dst_field.fid], point.reqs[dst_index], 
                            None, None, None, dst_idx_field,
                            idx_mappings[dst_idx_field.fid], point.reqs[idx_index], redop)
                if copy is not None:
                    copies.add(copy)
        elif self.copy_kind == 3:
            assert src_idx_field is not None
            assert dst_idx_field is not None
            assert len(self.reqs) % 4 == 0
            num_copies = len(self.reqs) // 4
            src_index = copy_index
            dst_index = copy_index + num_copies
            src_idx_index = copy_index + 2*num_copies
            dst_idx_index = copy_idnex + 3*num_copies
            for point in itervalues(self.points):
                src_mappings = point.find_mapping(src_index)
                dst_mappings = point.find_mapping(dst_index)
                src_idx_mappings = point.find_mapping(src_idx_index)
                dst_idx_mappings = point.find_mapping(dst_idx_index)
                if perform_checks:
                    copy = point.find_verification_indirection_copy(src_field,
                            src_mappings[src_field.fid], point.reqs[src_index],
                            dst_field, dst_mappings[dst_field.fid], point.reqs[dst_index], 
                            src_idx_field, src_idx_mappings[src_idx_field.fid],
                            point.reqs[src_idx_index], dst_idx_field,
                            dst_idx_mappings[dst_idx_field.fid], 
                            point.reqs[dst_idx_index], redop)
                else:
                    copy = point.find_or_create_indirection_copy(src_field,
                            src_mappings[src_field.fid], point.reqs[src_index],
                            dst_field, dst_mappings[dst_field.fid], point.reqs[dst_index], 
                            src_idx_field, src_idx_mappings[src_idx_field.fid],
                            point.reqs[src_idx_index], dst_idx_field,
                            dst_idx_mappings[dst_idx_field.fid], 
                            point.reqs[dst_idx_index], redop)
                if copy is not None:
                    copies.add(copy)
        else:
            assert False
        # If we're constructing these copies then we need to do indirection exchanges
        if not perform_checks:
            for c1 in copies:
                for c2 in copies:
                    if c1 is c2:
                        continue
                    c1.indirections.exchange(c2.indirections)
        self.collective_copies[key] = copies
        return copies

    def verify_gather_scatter_requirements(self, copy_idx, gather, src_idx, src_req, 
                                dst_idx, dst_req, idx_idx, idx_req, perform_checks):
        # If this was predicated there might not be any mappings
        if not self.mappings:
            return True
        src_points = src_req.logical_node.get_point_set()
        dst_points = dst_req.logical_node.get_point_set()
        idx_points = idx_req.logical_node.get_point_set()
        # Get the mappings
        src_mappings = self.find_mapping(src_req.index)
        dst_mappings = self.find_mapping(dst_req.index)
        idx_mappings = self.find_mapping(idx_req.index)
        assert self.context
        src_depth = self.context.find_enclosing_context_depth(src_req, src_mappings) 
        dst_depth = self.context.find_enclosing_context_depth(dst_req, dst_mappings)
        idx_depth = self.context.find_enclosing_context_depth(idx_req, idx_mappings)
        assert len(src_req.fields) == len(dst_req.fields)
        assert len(idx_req.fields) == 1
        idx_field = idx_req.fields[0] 
        idx_inst = idx_mappings[idx_field.fid]
        assert not idx_inst.is_virtual()
        # We just need to verify this region requirement one time
        idx_versions = dict()
        if not idx_req.logical_node.perform_physical_verification(
                idx_depth, idx_field, self, idx_req, idx_inst,
                perform_checks, False, None, idx_versions):
            return False
        idx_copies = set()
        for fidx in xrange(len(src_req.fields)):
            src_field = src_req.fields[fidx]
            dst_field = dst_req.fields[fidx]
            assert src_field.fid in src_mappings
            assert dst_field.fid in dst_mappings
            src_inst = src_mappings[src_field.fid]
            dst_inst = dst_mappings[dst_field.fid]
            assert not src_inst.is_virtual()
            assert not dst_inst.is_virtual()
            is_reduce = dst_req.is_reduce()
            # Switch this to read-write privileges and then switch it back 
            # after we are done. The runtime does this too so that its 
            # analysis is correct
            copy_redop = 0
            if is_reduce:
                copy_redop = dst_req.redop
                dst_req.redop = 0
                dst_req.priv = READ_WRITE
            # Record the source version numbers
            src_versions = dict()
            if not src_req.logical_node.perform_physical_verification(
                      src_depth, src_field, self, src_req, src_inst, 
                      perform_checks, False, None, src_versions):
                return False
            # Record the destination version numbers
            dst_versions = dict()
            if not dst_req.logical_node.perform_physical_verification(
                      dst_depth, dst_field, self, dst_req, dst_inst,
                      perform_checks, False, None, dst_versions):
                return False
            if gather:
                # Check to see if the copy space is empty
                if idx_req.logical_node.index_space is not dst_req.logical_node.index_space:
                    idx_points = idx_req.logical_node.index_space.get_point_set()
                    dst_points = dst_req.logical_node.index_space.get_point_set()
                    copy_points = idx_points & dst_points
                    
                else:
                    copy_points = idx_points
                if len(copy_points) == 0:
                    continue
                local_copies = set()
                if perform_checks:
                    copy = self.find_verification_indirection_copy(src_field,
                            src_inst, src_req, dst_field, dst_inst, dst_req,
                            idx_field, idx_inst, idx_req, None, None, None, copy_redop)
                    if copy is None:
                        print("ERROR: Missing indirect copy operation from field "+
                            str(src_field)+" to field "+str(dst_field)+" with indirect "+
                            "field "+str(idx_field)+" for gather "+str(self))
                        if self.state.assert_on_error:
                            assert False
                        return False
                else:
                    copy = self.find_or_create_indirection_copy(src_field,
                            src_inst, src_req, dst_field, dst_inst, dst_req,
                            idx_field, idx_inst, idx_req, None, None, None, copy_redop)
                local_copies.add(copy)
                if self.index_owner is not None and self.index_owner.collective_src:
                    global_copies = self.index_owner.find_collective_copies(copy_idx, 
                        perform_checks, src_field, dst_field, idx_field, None, copy_redop)
                    assert copy in global_copies
                else:
                    global_copies = local_copies 
                if not src_req.logical_node.perform_indirect_copy_verification(self,
                        copy_redop, perform_checks, global_copies, src_points, 
                        src_depth, src_field, src_req, src_inst, src_versions):
                    return False
                if not dst_req.logical_node.perform_indirect_copy_verification(self,
                        copy_redop, perform_checks, local_copies, copy_points,
                        dst_depth, dst_field, dst_req, dst_inst, dst_versions):
                    return False
            else:
                # Check to see if the copy space is empty
                if idx_req.logical_node.index_space is not src_req.logical_node.index_space:
                    copy_points = idx_points & src_points
                else:
                    copy_points = idx_points
                if len(copy_points) == 0:
                    continue
                local_copies = set()
                if perform_checks:
                    copy = self.find_verification_indirection_copy(src_field,
                            src_inst, src_req, dst_field, dst_inst, dst_req,
                            None, None, None, idx_field, idx_inst, idx_req, copy_redop)
                    if copy is None:
                        print("ERROR: Missing indirect copy operation from field "+
                            str(src_field)+" to field "+str(dst_field)+" with indirect "+
                            "field "+str(idx_field)+" for scatter "+str(self))
                        if self.state.assert_on_error:
                            assert False
                        return False
                else:
                    copy = self.find_or_create_indirection_copy(src_field,
                            src_inst, src_req, dst_field, dst_inst, dst_req,
                            None, None, None, idx_field, idx_inst, idx_req, copy_redop)
                local_copies.add(copy)
                if self.index_owner is not None and self.index_owner.collective_dst:
                    global_copies = self.index_owner.find_collective_copies(copy_idx,
                        perform_checks, src_field, dst_field, None, idx_field, copy_redop)
                    assert copy in global_copies
                else:
                    global_copies = local_copies 
                if not src_req.logical_node.perform_indirect_copy_verification(self,
                        copy_redop, perform_checks, local_copies, copy_points,
                        src_depth, src_field, src_req, src_inst, src_versions):
                    return False
                if not dst_req.logical_node.perform_indirect_copy_verification(self,
                        copy_redop, perform_checks, global_copies, dst_points,
                        dst_depth, dst_field, dst_req, dst_inst, dst_versions):
                    return False
            for copy in local_copies:
                idx_copies.add(copy)
            # Restore these when we are done
            if is_reduce:
                dst_req.priv = REDUCE
                dst_req.redop = copy_redop
        if not idx_req.logical_node.perform_indirect_copy_verification(self,
                copy_redop, perform_checks, idx_copies, copy_points,
                idx_depth, idx_field, idx_req, idx_inst, idx_versions):
            return False
        return True

    def verify_indirect_requirements(self, copy_idx, src_idx, src_req, dst_idx, dst_req,
                    src_idx_idx, src_idx_req, dst_idx_idx, dst_idx_req, perform_checks):
        # If this was predicated there might not be any mappings
        if not self.mappings:
            return True
        src_idx_points = src_idx_req.logical_node.get_point_set()
        dst_idx_points = dst_idx_req.logical_node.get_point_set()
        # There should always be the same points in both sets
        assert (src_idx_points - dst_idx_points).empty()
        assert (dst_idx_points - src_idx_points).empty()
        # Get the mappings
        src_mappings = self.find_mapping(src_req.index)
        dst_mappings = self.find_mapping(dst_req.index)
        src_idx_mappings = self.find_mapping(src_idx_req.index)
        dst_idx_mappings = self.find_mapping(dst_idx_req.index)
        assert self.context
        src_depth = self.context.find_enclosing_context_depth(src_req, src_mappings) 
        dst_depth = self.context.find_enclosing_context_depth(dst_req, dst_mappings)
        src_idx_depth = self.context.find_enclosing_context_depth(src_idx_req, src_idx_mappings)
        dst_idx_depth = self.context.find_enclosing_context_depth(dst_idx_req, dst_idx_mappings)
        assert len(src_req.fields) == len(dst_req.fields)
        assert len(src_idx_req.fields) == 1
        assert len(dst_idx_req.fields) == 1
        src_idx_field = src_idx_req.fields[0] 
        src_idx_inst = src_idx_mappings[src_idx_field.fid]
        assert not src_idx_inst.is_virtual()
        dst_idx_field = dst_idx_req.fields[0] 
        dst_idx_inst = dst_idx_mappings[dst_idx_field.fid]
        assert not dst_idx_inst.is_virtual()
        # Check to see if the copy space is empty
        if src_idx_req.logical_node.index_space is not dst_idx_req.logical_node.index_space:
            copy_points = src_idx_points & dst_idx_points
        else:
            copy_points = src_idx_points
        if len(copy_points == 0):
            return True
        # We just need to verify these region requirements one time
        src_idx_versions = dict()
        if not src_idx_req.logical_node.perform_physical_verification(
                src_idx_depth, src_idx_field, self, src_idx_req, src_idx_inst,
                perform_checks, False, None, src_idx_versions):
            return False
        dst_idx_versions = dict()
        if not dst_idx_req.logical_node.perform_physical_verification(
                dst_idx_depth, dst_idx_field, self, dst_idx_req, dst_idx_inst,
                perform_checks, False, None, dst_idx_versions):
            return False
        idx_copies = set()
        for fidx in xrange(len(src_req.fields)):
            src_field = src_req.fields[fidx]
            dst_field = dst_req.fields[fidx]
            assert src_field.fid in src_mappings
            assert dst_field.fid in dst_mappings
            src_inst = src_mappings[src_field.fid]
            dst_inst = dst_mappings[dst_field.fid]
            assert not src_inst.is_virtual()
            assert not dst_inst.is_virtual()
            is_reduce = dst_req.is_reduce()
            # Switch this to read-write privileges and then switch it back 
            # after we are done. The runtime does this too so that its 
            # analysis is correct
            copy_redop = 0
            if is_reduce:
                copy_redop = dst_req.redop
                dst_req.redop = 0
                dst_req.priv = READ_WRITE
            src_versions = dict()
            if not src_req.logical_node.perform_physical_verification(
                      src_depth, src_field, self, src_req, src_inst, 
                      perform_checks, False, None, src_versions):
                return False
            dst_versions = dict()
            if not dst_req.logical_node.perform_physical_verification(
                      dst_depth, dst_field, self, dst_req, dst_inst,
                      perform_checks, False, None, dst_versions):
                return False
            local_copies = set()
            if perform_checks:
                copy = self.find_verification_indirection_copy(src_field,
                        src_inst, src_req, dst_field, dst_inst, dst_req,
                        src_idx_field, src_idx_inst, src_idx_req, 
                        dst_idx_field, dst_idx_inst, dst_idx_req, copy_redop)
                if copy is None:
                    print("ERROR: Missing indirect copy operation from field "+
                        str(src_field)+" to field "+str(dst_field)+" with indirect "+
                        "fields "+str(src_idx_field)+" and "+str(dst_idx_field)+
                        " for full indirection "+str(self))
                    if self.state.assert_on_error:
                        assert False
                    return False
            else:
                copy = self.find_or_create_indirection_copy(src_field,
                        src_inst, src_req, dst_field, dst_inst, dst_req,
                        src_idx_field, src_idx_inst, src_idx_req, 
                        dst_idx_field, dst_idx_inst, dst_idx_req, copy_redop)
            local_copies.add(copy)
            if self.collective_src or self.collective_dst:
                global_copies = self.index_owner.find_collective_copies(copy_idx,
                                            perform_checks, src_field, dst_field, 
                                            src_idx_field, dst_idx_field, copy_redop)
                assert copy in global_copies
            else:
                global_copies = local_copies
            if not src_req.logical_node.perform_indirect_copy_verification(self,
                    copy_redop, perform_checks, 
                    global_copies if self.collective_src else local_copies, 
                    src_points, src_depth, src_field, src_req, src_inst, src_versions):
                return False
            if not dst_req.logical_node.perform_indirect_copy_verification(self,
                    copy_redop, perform_checks, 
                    global_copies if self.collective_dst else local_copies,
                    dst_points, dst_depth, dst_field, dst_req, dst_inst, dst_versions):
                return False
            for copy in local_copies:
                idx_copies.add(copy)
            # Restore these when we are done
            if is_reduce:
                dst_req.priv = REDUCE
                dst_req.redop = copy_redop
        if not src_idx_req.logical_node.perform_indirect_copy_verification(self,
                copy_redop, perform_checks, idx_copies, copy_points,
                src_idx_depth, src_idx_field, src_idx_req, src_idx_inst, src_idx_versions):
            return False
        if not dst_idx_req.logical_node.perform_indirect_copy_verification(self,
                copy_redop, perform_checks, idx_copies, copy_points,
                dst_idx_depth, dst_idx_field, dst_idx_req, dst_idx_inst, dst_idx_versions):
            return False
        return True

    def verify_fill_requirement(self, index, req, perform_checks):
        assert self.context
        mappings = self.find_mapping(index)
        depth = self.context.find_enclosing_context_depth(req, mappings)
        for field in req.fields:
            if not req.logical_node.perform_fill_verification(depth, field, self, req):
                return False
            # If this field is restricted, we effectively have to fill it
            # now to get the proper semantics of seeing updates right away
            if mappings is not None and field in mappings:
                if not req.logical_node.perform_physical_verification(depth, field,
                        self, req, mappings[field], perform_checks, False):
                    return False
        return True

    def add_restriction(self, index, req, perform_checks):
        assert self.context
        assert index in self.mappings
        mappings = self.mappings[index]
        depth = self.context.find_enclosing_context_depth(req, mappings)
        for field in req.fields:
            inst = mappings[field.fid]
            assert not inst.is_virtual()
            if not req.logical_node.add_restriction(depth, field, self, req, 
                                                    inst, perform_checks):
                return False
        return True

    def remove_restriction(self, index, req, perform_checks):
        assert self.context
        assert index in self.mappings
        mappings = self.mappings[index]
        depth = self.context.find_enclosing_context_depth(req, mappings)
        perform_filter = self.kind == DETACH_OP_KIND
        for field in req.fields:
            if perform_filter:
                inst = mappings[field.fid]
            else:
                inst = None
            if not req.logical_node.remove_restriction(depth, field, self, req, 
                                                       inst, perform_checks):
                return False
        return True

    def verify_physical_requirement(self, index, req, perform_checks):
        if req.is_no_access() or len(req.fields) == 0:
            return True
        assert index in self.mappings
        mappings = self.mappings[index]
        assert self.context
        depth = self.context.find_enclosing_context_depth(req, mappings)
        # Don't do registrations for single tasks or post close ops
        # Single tasks are registered after all copies are issued
        # Post tasks never register users since they aren't necessary
        perform_registration = (self.kind != SINGLE_TASK_KIND) and \
            (self.kind != POST_CLOSE_OP_KIND) and (self.kind != INTER_CLOSE_OP_KIND)
        for field in req.fields:
            # Find the instance that we chose to map this field to
            if field.fid not in mappings:
                print("Missing mapping decision for field "+str(field)+" of "+
                      "requirement requirement "+str(index)+" of "+str(self))
                print("This is a runtime logging bug, please report it.")
            assert field.fid in mappings
            inst = mappings[field.fid]
            if inst.is_virtual():
                # In the case of virtual mappings we don't have to
                # do any analysis here since we're just passing in the state
                continue
            if not req.logical_node.perform_physical_verification(depth, field,
                    self, req, inst, perform_checks, perform_registration):
                # Switch privilege back if necessary
                if self.kind == INTER_CLOSE_OP_KIND:
                    req.priv = READ_WRITE
                return False
        return True

    def perform_verification_registration(self, index, req, perform_checks):
        assert self.kind == SINGLE_TASK_KIND
        if req.is_no_access() or len(req.fields) == 0:
            return True
        assert index in self.mappings
        mappings = self.mappings[index]
        assert self.context
        depth = self.context.find_enclosing_context_depth(req, mappings)
        for field in req.fields:
            assert field.fid in mappings
            inst = mappings[field.fid]
            # skip any virtual mappings
            if inst.is_virtual():
                continue
            if not req.logical_node.perform_verification_registration(depth, field,
                                                    self, req, inst, perform_checks):
                return False
        return True

    def perform_op_physical_verification(self, perform_checks):
        # If we were predicated false, then there is nothing to do
        if not self.predicate_result:
            return True
        prefix = ''
        if self.context:
            depth = self.context.get_depth()
            for idx in xrange(depth):
                prefix += '  '
        # If we are an index space task, only do our points
        if self.kind == INDEX_TASK_KIND:
            for point in itervalues(self.points):
                if not point.op.perform_op_physical_verification(perform_checks):
                    return False
            return True
        # Handle other index space operations too
        elif self.points: 
            for point in sorted(itervalues(self.points), key=lambda x: x.uid):
                if not point.perform_op_physical_verification(perform_checks):
                    return False
            return True
        if perform_checks:
            print((prefix+"Performing physical verification analysis "+
                         "for %s (UID %d)...") % (str(self),self.uid))
        else:
            print((prefix+"Performing physical emulation analysis "+
                         "for %s (UID %d)...") % (str(self),self.uid))
        # Handle special cases
        if self.kind == COPY_OP_KIND:
            # Compute our version numbers first
            if perform_checks:
                self.compute_current_version_numbers()
            num_reqs = len(self.reqs)
            copy_kind = self.copy_kind if self.index_owner is None \
                    else self.index_owner.copy_kind
            assert copy_kind is not None
            if copy_kind == 0:
                # Normal region-to-region copy
                assert num_reqs % 2 == 0
                num_copies = num_reqs // 2
                for idx in xrange(num_copies):
                    if not self.verify_copy_requirements(idx, self.reqs[idx],
                            idx+num_copies, self.reqs[idx+num_copies], perform_checks):
                        return False
            elif copy_kind == 1 or copy_kind == 2:
                # Gather or scatter copy
                assert num_reqs % 3 == 0
                num_copies = num_reqs // 3
                for idx in xrange(num_copies):
                    if not self.verify_gather_scatter_requirements(idx, copy_kind == 1,
                            idx, self.reqs[idx], idx+num_copies, self.reqs[idx+num_copies],
                            idx+2*num_copies, self.reqs[idx+2*num_copies], perform_checks):
                        return False
            elif copy_kind == 3:
                # Full indirection copy
                assert num_reqs % 4 == 0
                num_copies = num_reqs // 4
                for idx in xrange(num_copies):
                    if not self.verify_indirection_requirements(idx, idx, self.reqs[idx],
                            idx+num_copies, self.reqs[idx+num_copies],
                            idx+2*num_copies, self.reqs[idx+2*num_copies],
                            idx+3*num_copies, self.reqs[idx+3*num_copies], perform_checks):
                        return False
            else:
                assert False # Should never get here
        elif self.kind == FILL_OP_KIND:
            # Compute our version numbers first
            if perform_checks:
                self.compute_current_version_numbers()
            for index,req in iteritems(self.reqs):
                if not self.verify_fill_requirement(index, req, perform_checks):
                    return False
        elif self.kind == DELETION_OP_KIND:
            # Skip deletions, they only impact logical analysis
            pass
        else:
            if self.reqs:
                # Compute our version numbers first
                if perform_checks:
                    self.compute_current_version_numbers()
                for index,req, in iteritems(self.reqs):
                    if not self.verify_physical_requirement(index, req, perform_checks):
                        return False
            # Add any restrictions for different kinds of ops
            if self.kind == RELEASE_OP_KIND or \
                    (self.kind == ATTACH_OP_KIND and self.restricted):
                for index,req in iteritems(self.reqs):
                    if not self.add_restriction(index, req, perform_checks):
                        return False
            elif self.kind == ACQUIRE_OP_KIND or self.kind == DETACH_OP_KIND:
                for index,req in iteritems(self.reqs):
                    if not self.remove_restriction(index, req, perform_checks):
                        return False
                return True
            elif self.kind == SINGLE_TASK_KIND:
                # We now need to do the registration for our region
                # requirements since we didn't do it as part of the 
                # normal physical analysis
                if self.reqs:
                    for index,req, in iteritems(self.reqs):
                        if not self.perform_verification_registration(index, req, 
                                                                      perform_checks):
                            return False
                # If we are not a leaf task, go down the task tree
                if self.task is not None:
                    if not self.task.perform_task_physical_verification(perform_checks):
                        return False
        return self.check_for_spurious_realm_ops(perform_checks)

    def print_op_mapping_decisions(self, depth):
        if self.inter_close_ops:
            assert not self.is_close()
            for close in self.inter_close_ops:
                close.print_op_mapping_decisions(depth)
        # If we are an index task just do our points and return
        if self.kind == INDEX_TASK_KIND:
            assert self.points is not None
            for point in itervalues(self.points):
                point.op.print_op_mapping_decisions(depth)
            return
        # Print our mapping decisions
        prefix = ''
        for idx in xrange(depth):
            prefix += '  '
        print(prefix+'-------------------------------------------------')
        print(prefix+' Mapping Decisions for '+str(self)+' (depth='+str(depth)+')')

        if self.kind == SINGLE_TASK_KIND and self.task is not None:
            print(prefix+'  Task Mapped to ' + str(self.task.processor))
            assert self.task.variant
            print(prefix+'  Task Variant %d (inner=%s,leaf=%s)' % (self.task.variant.vid,
                    'Yes' if self.task.variant.inner else 'No',
                    'Yes' if self.task.variant.leaf else 'No'))
        if self.mappings is not None:
            for index,mappings in iteritems(self.mappings):
                assert index in self.reqs
                req = self.reqs[index]
                print(prefix+'  Region Requirement '+str(index)+' Region=('+
                      str(req.index_node)+','+str(req.field_space)+','+str(req.tid)+')')
                for fid,inst in iteritems(mappings):
                    field = req.field_space.get_field(fid)
                    print(prefix+'    '+str(field)+': '+str(inst))
        print(prefix+' End '+str(self)+' (depth='+str(depth)+')')
        print(prefix+'-------------------------------------------------')
        # If we are a single task recurse
        if self.kind == SINGLE_TASK_KIND and self.task is not None:
            self.task.print_task_mapping_decisions()

    def get_color(self):
        return {
            NO_OP_KIND : "white",
            SINGLE_TASK_KIND : "lightskyblue",
            INDEX_TASK_KIND : "mediumslateblue",
            MAP_OP_KIND : "mediumseagreen",
            INTER_CLOSE_OP_KIND : "orangered", 
            #READ_ONLY_CLOSE_OP_KIND : "darkgreen",
            POST_CLOSE_OP_KIND : "darkslateblue",
            #OPEN_OP_KIND : "royalblue",
            #ADVANCE_OP_KIND : "magenta",
            FENCE_OP_KIND : "darkorchid2",
            COPY_OP_KIND : "darkgoldenrod3",
            FILL_OP_KIND : "darkorange1",
            ACQUIRE_OP_KIND : "darkolivegreen",
            RELEASE_OP_KIND : "darksalmon",
            CREATION_OP_KIND : "forestgreen",
            DELETION_OP_KIND : "maroon",
            ATTACH_OP_KIND : "firebrick1",
            DETACH_OP_KIND : "cornflowerblue",
            DEP_PART_OP_KIND : "steelblue",
            PENDING_PART_OP_KIND : "honeydew",
            DYNAMIC_COLLECTIVE_OP_KIND : "navy",
            TRACE_OP_KIND : "springgreen",
            TIMING_OP_KIND : "turquoise",
            ALL_REDUCE_OP_KIND : "cyan",
            PREDICATE_OP_KIND : "olivedrab1",
            MUST_EPOCH_OP_KIND : "tomato",
            TUNABLE_OP_KIND : "lightcoral",
            }[self.kind]

    @property
    def html_safe_name(self):
        name = str(self)
        provenance = self.get_provenance()
        if provenance is not None and len(provenance) > 0:
            name = name + ' [' + provenance + ']'
        return name.replace('<','&lt;').replace('>','&gt;').replace('&','&amp;')

    def print_base_node(self, printer, dataflow):
        title = self.html_safe_name+' (UID: '+str(self.uid)+')'
        if self.task is not None and self.task.point.dim > 0:
            title += ' Point: ' + self.task.point.to_string()
        if self.replayed:
            title += '  (replayed)'
        label = printer.generate_html_op_label(title, self.reqs, self.mappings,
                                       self.get_color(), self.state.detailed_graphs)
        if dataflow or self.task is None or len(self.task.operations) == 0:
            # Box shape is sufficient unless we have nesting
            printer.println(self.node_name+' [label=<'+label+'>,fontsize=14,'+\
                    'fontcolor=black,shape=box,penwidth=0];')
        else:
            # For non-leaf tasks we need record shape to handle subgraphs
            printer.println(self.node_name+' [label=<'+label+'>,fontsize=14,'+\
                    'fontcolor=black,shape=record,penwidth=0];')

    def print_dataflow_node(self, printer):
        # Print any close operations that we have, then print ourself 
        if self.inter_close_ops:
            for close in self.inter_close_ops:
                close.print_dataflow_node(printer)
        self.print_base_node(printer, True) 

    def print_incoming_dataflow_edges(self, printer, previous):
        if self.inter_close_ops:
            for close in self.inter_close_ops:
                close.print_incoming_dataflow_edges(printer, previous)
        if self.incoming:
            for dep in self.incoming:
                dep.print_dataflow_edge(printer, previous)
        # Handle any phase barriers
        if self.state.detailed_graphs:
            self.print_phase_barrier_edges(printer) 

    def print_phase_barrier_edges(self, printer):
        if self.wait_barriers:
            for bar in self.wait_barriers:
                if bar.barrier_contributors:
                    for contributor in bar.barrier_contributors:
                        title = str(contributor)+' (UID: '+str(contributor.uid)+\
                                ' in context '+str(contributor.context.op)+' UID: '+\
                                str(contributor.context.op.uid)+')'
                        label = printer.generate_html_op_label(title, None, None,
                                "white", self.state.detailed_graphs)
                        node_name = contributor.node_name+'_'+self.node_name
                        printer.println(node_name+' [label=<'+label+'>,fontsize=14,'+\
                                'fontcolor=black,shape=box,penwidth=0];')
                        printer.println(node_name+' -> '+self.node_name+
                                ' [style=solid,color=black,penwidth=2];')
        if self.arrival_barriers:
            for bar in self.arrival_barriers:
                if bar.barrier_waiters:
                    for waiter in bar.barrier_waiters:
                        title = str(waiter)+' (UID: '+str(waiter.uid)+\
                                ' in context '+str(waiter.context.op)+' UID: '+\
                                str(waiter.context.op.uid)+')'
                        label = printer.generate_html_op_label(title, None, None,
                                "white", self.state.detailed_graphs)
                        node_name = waiter.node_name+'_'+self.node_name
                        printer.println(node_name+' [label=<'+label+'>,fontsize=14,'+\
                                'fontcolor=black,shape=box,penwidth=0];')
                        printer.println(self.node_name+' -> '+node_name+
                                ' [style=solid,color=black,penwidth=2];')

    def print_event_node(self, printer):
        self.print_base_node(printer, False)

    def print_event_graph(self, printer, elevate, all_nodes, top):
        # If we were predicated false then we don't get printed
        if not self.predicate_result:
            return
        # Do any of our close operations too
        if self.inter_close_ops:
            for close in self.inter_close_ops:
                close.print_event_graph(printer, elevate, all_nodes, False)
        # Handle index space operations specially, everything
        # else is the same
        if self.kind is INDEX_TASK_KIND or self.points:
            # Might have been predicated
            if self.points:
                if self.kind is INDEX_TASK_KIND:
                    for point in itervalues(self.points):
                        point.op.print_event_graph(printer, elevate, all_nodes, False)
                else:
                    for point in itervalues(self.points):
                        point.print_event_graph(printer, elevate, all_nodes, False)
            # Put any operations we generated in the elevate set
            if self.realm_copies:
                for copy in self.realm_copies:
                    elevate[copy] = copy.get_context()
            if self.realm_fills:
                for fill in self.realm_fills:
                    elevate[fill] = fill.get_context()
            if self.realm_depparts:
                for deppart in self.realm_depparts:
                    elevate[deppart] = deppart.get_context()
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
                    elevate[copy] = copy.get_context()
        if self.realm_fills:
            for fill in self.realm_fills:
                if fill not in elevate:
                    elevate[fill] = fill.get_context()
        if self.realm_depparts:
            for deppart in self.realm_depparts:
                if deppart not in elevate:
                    elevate[deppart] = deppart.get_context()
        if self.is_physical_operation():
            # Finally put ourselves in the set if we are a physical operation
            assert self.context is not None
            elevate[self] = self.context

    def is_realm_operation(self):
        return False

    def is_physical_operation(self):
        if self.is_internal():
            return False
        if self.kind is COPY_OP_KIND:
            return False
        if self.kind is FILL_OP_KIND:
            return False
        if self.kind is FENCE_OP_KIND:
            return False
        if self.kind is CREATION_OP_KIND:
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
                    printer.println(src.node_name+' -> '+self.node_name+
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

    def print_eq_node(self, printer, eq_key):
        pass

    def print_incoming_eq_edges(self, printer, eq_key):
        if self.eq_incoming is None or eq_key not in self.eq_incoming:
            return
        if self.cluster_name is not None:
            for src in self.eq_incoming[eq_key]:
                if src.cluster_name is not None:
                    printer.println(src.node_name+' -> '+self.node_name+
                            ' [ltail='+src.cluster_name+',lhead='+
                            self.cluster_name+',style=solid,color=black,'+
                            'penwidth=2];')
                else:
                    printer.println(src.node_name+' -> '+self.node_name+
                            ' [lhead='+self.cluster_name+',style=solid,'+
                            'color=black,penwidth=2];')
        else:
            for src in self.eq_incoming[eq_key]:
                if src.cluster_name is not None:
                    printer.println(src.node_name+' -> '+self.node_name+
                            ' [ltail='+src.cluster_name+',style=solid,'+
                            'color=black,penwidth=2];')
                else:
                    printer.println(src.node_name+' -> '+self.node_name+
                            ' [style=solid,color=black,penwidth=2];')

    def pack_requirement_replay_info(self, replay_file, req, mapping):
        if mapping:
            # Get the unique set of instances
            unique_insts = set()
            for inst in itervalues(mapping):
                if inst.is_virtual():
                    continue
                unique_insts.add(inst)
            replay_file.write(struct.pack('I',len(unique_insts)))
            for inst in unique_insts:
                replay_file.write(struct.pack('Q',inst.handle))
        else:
            replay_file.write(struct.pack('I',0))

    def pack_inline_replay_info(self, replay_file):
        assert self.kind == MAP_OP_KIND
        replay_file.write(struct.pack('Q',self.uid))
        assert 0 in self.reqs
        assert 0 in self.mappings
        assert len(self.mappings) == 1
        replay_file.write(struct.pack('I',1))
        self.pack_requirement_replay_info(replay_file, self.reqs[0], 
                                          self.mappings[0])

    def pack_copy_replay_info(self, replay_file):
        assert self.kind == COPY_OP_KIND
        replay_file.write(struct.pack('Q',self.uid))
        assert len(self.reqs) % 2 == 0
        half = len(self.reqs) / 2
        replay_file.write(struct.pack('I',half))
        
    def pack_close_replay_info(self, replay_file):
        assert self.kind == INTER_CLOSE_OP_KIND
        replay_file.write(struct.pack('Q',self.uid))
        assert 0 in self.reqs
        if 0 in self.mappings:
            assert len(self.mappings) == 1
            replay_file.write(struct.pack('I',1))
            self.pack_requirement_replay_info(replay_file, self.reqs[0], 
                                              self.mappings[0])
        else:
            replay_file.write(struct.pack('I',0))

    def pack_release_replay_info(self, replay_file):
        assert self.kind == RELEASE_OP_KIND
        replay_file.write(struct.pack('Q',self.uid))

class Variant(object):
    __slots__ = ['state', 'vid', 'inner', 'leaf', 'idempotent', 'name']
    def __init__(self, state, vid):
        self.state = state
        self.vid = vid
        self.inner = False
        self.leaf = False
        self.idempotent = False
        self.name = None

    def initialize(self, inner, leaf, idempotent, name):
        self.inner = inner
        self.leaf = leaf
        self.idempotent = idempotent
        self.name = name

class ProjectionFunction(object):
    __slots__ = ['state', 'pid', 'depth', 'invertible']
    def __init__(self, state, pid):
        self.state = state
        self.pid = pid
        self.depth = None
        self.invertible = None

    def set_depth(self, depth):
        if self.depth:
            assert self.depth == depth
        else:
            self.depth = depth

    def set_invertible(self, invertible):
        if self.invertible:
            assert self.invertible == invertible
        else:
            self.invertible = invertible

class Task(object):
    __slots__ = ['state', 'op', 'point', 'operations', 'depth', 
                 'current_fence', 'used_instances', 'virtual_indexes', 
                 'processor', 'priority', 'premappings', 'postmappings', 
                 'tunables', 'operation_indexes', 'close_indexes', 'variant']
                  # If you add a field here, you must update the merge method
    def __init__(self, state, op):
        self.state = state
        self.op = op
        if op is not None:
            self.op.task = self
        self.point = Point(0) 
        self.operations = list()
        self.depth = None
        self.current_fence = None
        self.used_instances = None
        self.virtual_indexes = None
        self.processor = None
        self.priority = None
        self.premappings = None
        self.postmappings = None
        self.tunables = None
        self.operation_indexes = None
        self.close_indexes = None
        self.variant = None

    def __str__(self):
        if self.op is None:
            return "Root context"
        else:
            return str(self.op)

    __repr__ = __str__

    def get_provenance(self):
        if self.op is None:
            return None
        return self.op.get_provenance()

    @property
    def html_safe_name(self):
        name = str(self)
        provenance = self.get_provenance()
        if provenance is not None and len(provenance) > 0:
            name = name + ' [' + provenance + ']'
        return name.replace('<','&lt;').replace('>','&gt;').replace('&','&amp;')

    def add_operation(self, operation, index):
        assert operation.context_index is None
        operation.context_index = index
        # In general operations will always be appended here
        # because they are logged in order, but because of 
        # unordered operations we may need to insert somewhere earlier 
        for idx in xrange(len(self.operations)-1, -1, -1):
            op = self.operations[idx]
            # Equal case can happen for must epoch operations...
            if index >= op.context_index:
                self.operations.insert(idx+1, operation)
                return
        self.operations.insert(0, operation)

    def set_point(self, point):
        self.point = point

    def set_priority(self, priority):
        assert not self.priority
        self.priority = priority
     
    def set_processor(self, processor):
        assert not self.processor
        self.processor = processor

    def set_variant(self, variant):
        assert not self.variant
        self.variant = variant

    def add_premapping(self, index):
        if not self.premappings:
            self.premappings = set()
        self.premappings.add(index)

    def add_postmapping(self, index, fid, inst):
        if not self.postmappings:
            self.postmappings = dict()
        if index not in self.postmappings:
            self.postmappings[index] = dict()
        self.postmappings[index][fid] = inst

    def update_instance_uses(self):
        if self.premappings:
            for index in self.premappings:
                assert index in self.op.mappings
                mapping = self.op.mappings[index]
                unique_insts = set()
                for inst in itervalues(mapping):
                    unique_insts.add(inst)
                for inst in unique_insts:
                    inst.increment_use_count() 
        if self.postmappings:
            for mapping in itervalues(self.postmappings):
                unique_insts = set()
                for inst in itervalues(mapping):
                    unique_insts.add(inst)
                for inst in unique_insts:
                    inst.increment_use_count()

    def add_tunable(self, index, size, value):
        if not self.tunables:
            self.tunables = dict()
        assert index not in self.tunables
        self.tunables[index] = (value,size)

    def add_operation_index(self, index, uid):
        if not self.operation_indexes:
            self.operation_indexes = dict()
        self.operation_indexes[index] = uid

    def add_close_index(self, index, uid):
        if not self.close_indexes:
            self.close_indexes = dict()
        self.close_indexes[index] = uid

    def get_parent_context(self):
        assert self.op.context is not None
        return self.op.context

    def get_depth(self):
        if self.depth is None:
            self.depth = self.op.get_depth()
        return self.depth

    def merge(self, other):
        if self.op.task_id == -1:
            self.op.task_id = other.op.task_id
        elif other.op.task_id != -1:
            assert self.op.task_id == other.op.task_id
        if self.point.dim == 0:
            self.point = other.point
        elif other.point.dim != 0:
            assert self.point == other.point
        if not self.operations:
            self.operations = other.operations
            # Update the contexts
            for op in self.operations:
                op.set_context(self)
        else:
            assert not other.operations
        if not self.processor:
            self.processor = other.processor
        else:
            assert not other.processor
        if not self.priority:
            self.priority = other.priority
        else:
            assert not other.priority
        if not self.premappings:
            self.premappings = other.premappings
        else:
            assert not other.premappings
        if not self.postmappings:
            self.postmappings = other.postmappings
        else:
            assert not other.postmappings
        if not self.tunables:
            self.tunables = other.tunables
        else:
            assert not other.tunables
        if not self.operation_indexes:
            self.operation_indexes = other.operation_indexes
        else:
            assert not other.operation_indexes
        if not self.close_indexes:
            self.close_indexes = other.close_indexes
        else:
            assert not other.close_indexes
        if not self.variant:
            self.variant = other.variant
        else:
            assert not other.variant

    def flatten_summary_operations(self):
        flattened = list()
        for op in self.operations:
            if op.summary_op == op:
                continue
            if op.summary_op is not None:
                flattened.append(op.summary_op)
            flattened.append(op)
        self.operations = flattened

    def perform_logical_dependence_analysis(self, perform_checks):
        # If we don't have any operations we are done
        if not self.operations:
            return True
        # If this is the top-level task's context, we can skip it
        # since we know there is only one task in it
        if self.depth == 0:
            assert len(self.operations) == 1
            return True
        print('Performing logical dependence analysis for %s...' % str(self))
        if self.op.state.verbose:
            print('  Analyzing %d operations...' % len(self.operations))
        # Iterate over all the operations in order and
        # have them perform their analysis
        success = True
        for op in self.operations:
            if op.inlined:
                continue
            if not op.fully_logged:
                print(('Warning: skipping logical analysis of %s because it '+
                        'was not fully logged...') % str(op))
                if op.state.assert_on_warning:
                    assert False
                continue
            if not op.perform_logical_analysis(perform_checks):
                success = False
                break
        # Reset the logical state when we are done
        self.op.state.reset_logical_state()
        print("Pass" if success else "FAIL")
        return success

    def perform_logical_sanity_analysis(self):
        # Run the old version of the checks that
        # is more of a sanity check on our algorithm that
        # doesn't depend on our implementation but doesn't
        # really tell us what it means if something goes wrong
        if not self.operations or len(self.operations) < 2:
            return True
        print('Performing logical sanity analysis for %s...' % str(self))
        # Iterate over all operations from 1 to N and check all their
        # dependences against all the previous operations in the context
        for idx in xrange(1, len(self.operations)):
            # Find all the backwards reachable operations
            current_op = self.operations[idx]
            # No need to do anything if there are no region requirements
            if not current_op.reqs and current_op.kind != FENCE_OP_KIND:
                continue
            reachable = set()
            current_op.get_logical_reachable(reachable, False) 
            # Do something special for fence operations
            if current_op.kind == FENCE_OP_KIND: # special path for fences
                for prev in xrange(idx):
                    if not prev in reachable:
                        print("ERROR: Failed logical sanity check. No mapping "+
                              "dependence between previous "+str(prev)+" and "+
                              "later "+str(current_op))
                        if self.op.state.assert_on_error:
                            assert False
                        return False
            else: # The normal path
                for prev in xrange(idx):
                    if not current_op.analyze_logical_interference(
                                  self.operations[prev], reachable):
                        print("FAIL")
                        return False
        print("Pass")
        return True

    def find_enclosing_context_depth(self, child_req, mappings):
        # Special case for the top-level task
        depth = self.get_depth()
        if depth == 0:
            return depth
        if self.op.reqs:
            # Find which region requirement privileges were derived from
            for idx,our_req in iteritems(self.op.reqs):
                if child_req.parent is not our_req.logical_node:
                    continue
                fields_good = True
                for field in child_req.fields:
                    if field not in our_req.fields:
                        fields_good = False
                        break
                if not fields_good:
                    continue
                # We'll assume the privileges are good for now
                # See if we are virtual mapped or not
                any_virtual = False
                for field in child_req.fields:
                    assert idx in self.op.mappings
                    assert field.fid in self.op.mappings[idx]
                    if self.op.mappings[idx][field.fid].is_virtual():
                        any_virtual = True
                        break
                if any_virtual:
                    assert self.op.context
                    return self.op.context.find_enclosing_context_depth(our_req, mappings)
                else:
                    if mappings:
                        for fid,inst in iteritems(mappings):
                            self.used_instances.add((inst,fid))
                    return depth
        # Trust the runtime privilege checking here
        # If we get here this is a created privilege flowing back
        # up the task tree so just give it depth zero
        return 0

    def perform_task_physical_verification(self, perform_checks):
        if not self.operations:
            return True
        # Depth is a proxy for context 
        depth = self.get_depth()
        assert self.used_instances is None
        self.used_instances = set()
        # Initialize any regions that we mapped
        if self.op.reqs:
            for idx,req in iteritems(self.op.reqs):
                # Skip any no access requirements
                if req.is_no_access() or len(req.fields) == 0:
                    continue
                assert idx in self.op.mappings
                mappings = self.op.mappings[idx]
                # If we are doing restricted analysis then add any restrictions
                # We treat all reduction instances as restricted to eagerly flush
                # back reductions to this instance for now
                add_restrictions = \
                        ((req.priv == READ_WRITE or req.priv == READ_ONLY) and \
                        req.coher == SIMULTANEOUS) or (req.priv == REDUCE)
                for field in req.fields:
                    assert field.fid in mappings
                    inst = mappings[field.fid]
                    if inst.is_virtual():
                        assert not add_restrictions # Better not be virtual if restricted
                        continue
                    req.logical_node.initialize_verification_state(depth, field, inst, 
                                                                   add_restrictions)
        success = True
        for op in self.operations:
            if op.inlined:
                continue
            if not op.fully_logged:
                print(('Warning: skipping physical verification of %s '+
                        'because it was not fully logged...') % str(op))
                if op.state.assert_on_warning:
                    assert False
                continue
            if not op.perform_op_physical_verification(perform_checks): 
                success = False
                break
        # Reset any physical user lists at our depth
        for inst,fid in self.used_instances:
            inst.reset_verification_users(depth)
        self.op.state.reset_verification_state(depth)
        return success

    def print_task_mapping_decisions(self):
        depth = self.get_depth()
        for op in self.operations:
            if not op.fully_logged:
                continue
            op.print_op_mapping_decisions(depth)

    def print_dataflow_graph(self, path, simplify_graphs, zoom_graphs):
        if len(self.operations) == 0:
            return 0
        if len(self.operations) == 1:
            op = self.operations[0]
            if not op.inter_close_ops or not op.fully_logged:
                return 0
        name = str(self)
        filename = 'dataflow_'+name.replace(' ', '_')+'_'+str(self.op.uid)
        printer = GraphPrinter(path,filename)
        # First emit the nodes
        for op in self.operations:
            if op.inlined:
                continue
            if not op.fully_logged:
                print(('Warning: skipping dataflow printing of %s because it '+
                        'was not fully logged...') % str(op))
                if op.state.assert_on_warning:
                    assert False
                continue
            op.print_dataflow_node(printer)
        # Simplify our graph if necessary
        if simplify_graphs:
            print("Simplifying dataflow graph for "+str(self)+"...")
            all_ops = list()
            for op in self.operations:
                if not op.fully_logged:
                    continue
                # Add any close operations first
                if op.inter_close_ops:
                    for close in op.inter_close_ops:
                        all_ops.append(close)
                # Then add the operation itself
                all_ops.append(op)
                # If this is an index space operation prune any
                # self-edges which could interfere with the
                # transitive reduction that we're about to do
                # These transitive edges occur when we have potentially
                # interfering and aliased projection region requirements
                # but the point tasks ultimately do not end up interfering
                # We know this is safe because it is checked in post_parse
                # by the is_interfering_index_space_launch method so these
                # cannot be actually intefering edges.
                if op.points is not None and op.logical_incoming is not None:
                    if op in op.logical_incoming:
                        op.logical_incoming.remove(op)
                        op.logical_outgoing.remove(op)
                # Print any phase barier edges now, since
                # we know they will all be printed
                if self.state.detailed_graphs:
                    op.print_phase_barrier_edges(printer)
            index_map = dict()
            reachable = dict()
            count = 0
            total_nodes = len(all_ops)
            # Now traverse the list in reverse order
            for src_index in xrange(total_nodes-1,-1,-1): 
                src = all_ops[src_index]
                count += 1 
                index_map[src] = src_index
                our_reachable = NodeSet(total_nodes)
                reachable[src] = our_reachable
                if src.logical_outgoing is None or len(src.logical_outgoing) == 0:
                    print_progress_bar(count, total_nodes, length=50)
                    continue
                # Otherwise iterate through our outgoing edges and get the set of 
                # nodes reachable from all of them
                for dst in src.logical_outgoing:
                    # Some nodes won't appear in the list of all operations
                    # such as must epoch operations which we can safely skip
                    if dst not in reachable:
                        assert dst not in all_ops
                        continue
                    our_reachable.union(reachable[dst])
                # Now see which of our nodes can be reached indirectly
                to_remove = None
                for dst in src.logical_outgoing:
                    # See comment above for why we can skip some edges
                    if dst not in index_map:
                        assert dst not in all_ops
                        continue
                    dst_index = index_map[dst]
                    if our_reachable.contains(dst_index):
                        if to_remove is None:
                            to_remove = list()
                        to_remove.append(dst)
                    else:
                        # We need to add it to our reachable set
                        our_reachable.add(dst_index)
                if to_remove:
                    for dst in to_remove:
                        src.logical_outgoing.remove(dst)
                        dst.logical_incoming.remove(src)
                # We should never remove everything
                assert len(src.logical_outgoing) > 0
                for dst in src.logical_outgoing:
                    # Skip any edges to nodes not in the reachable list
                    # (e.g. must epoch operations)
                    if dst not in reachable:
                        continue
                    printer.println(src.node_name+' -> '+dst.node_name+
                                    ' [style=solid,color=black,penwidth=2];')
                print_progress_bar(count, total_nodes, length=50)
            print("Done")
        else:
            previous_pairs = set()
            for op in self.operations:
                if op.inlined:
                    continue
                if not op.fully_logged:
                    print(('Warning: skipping dataflow printing of %s because it '+
                            'was not fully logged...') % str(op))
                    if op.state.assert_on_warning:
                        assert False
                    continue
                op.print_incoming_dataflow_edges(printer, previous_pairs)
        printer.print_pdf_after_close(False, zoom_graphs)
        # We printed our dataflow graph
        return 1   

    def print_event_graph_context(self, printer, elevate, all_nodes, top):
        if not self.operations:
            return 
        if not top:
            # Start the cluster 
            title = self.html_safe_name + ' (UID: '+str(self.op.uid)+')'
            if self.point.dim > 0:
                title += ' Point: ' + self.point.to_string()
            if self.op.replayed:
                title += '  (replayed)'
            label = printer.generate_html_op_label(title, self.op.reqs,
                                                   self.op.mappings,
                                                   self.op.get_color(), 
                                                   self.op.state.detailed_graphs)
            self.op.cluster_name = printer.start_new_cluster(label)
            # Make an invisible node for this cluster
            printer.println(self.op.node_name + ' [shape=point,style=invis];')
        # Generate the sub-graph
        for op in self.operations:
            if op.inlined:
                continue
            if not op.fully_logged:
                print(('Warning: skipping event graph printing of %s because it '+
                            'was not fully logged...') % str(op))
                if op.state.assert_on_warning:
                    assert False
                continue
            op.print_event_graph(printer, elevate, all_nodes, False)
        # Find our local nodes
        local_nodes = list()
        for node,context in iteritems(elevate):
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

    def pack_task_replay_info(self, replay_file, op_id):
        # Pack the point
        replay_file.write(struct.pack('Q', op_id))
        replay_file.write(struct.pack('i', self.point.dim))
        for idx in xrange(self.point.dim):
            replay_file.write(struct.pack('Q',self.point.vals[idx]))
        # Pack the base data
        replay_file.write(struct.pack('Q', op_id)) 
        replay_file.write(struct.pack('Q', self.processor.uid))
        replay_file.write(struct.pack('i', self.priority))
        replay_file.write(struct.pack('Q', self.variant.vid))
        # Pack premappings
        if self.premappings:
            replay_file.write(struct.pack('I',len(self.premappings)))
            for index in self.premappings:
                assert index in self.op.mappings
                replay_file.write(struct.pack('I',index))
                self.op.pack_requirement_replay_info(replay_file, 
                    self.op.reqs[index], self.op.mappings[index]) 
        else:
            replay_file.write(struct.pack('I',0))
        # Pack mappings
        if self.op.reqs:
            replay_file.write(struct.pack('I',len(self.op.reqs)))
            for index in xrange(len(self.op.reqs)):
                self.op.pack_requirement_replay_info(replay_file, self.op.reqs[index], 
                    None if index not in self.op.mappings else self.op.mappings[index])
        else:
            replay_file.write(struct.pack('I',0))
        # Pack postmappings
        if self.postmappings:
            replay_file.write(struct.pack('I',len(self.postmappings)))
            for index,mapping in self.postmappings:
                replay_file.write(struct.pack('I',index))
                self.op.pack_requirement_replay_info(replay_file,
                    self.op.reqs[index], mapping)
        else:
            replay_file.write(struct.pack('I',0))
        # Pack the tunables
        if self.tunables:
            replay_file.write(struct.pack('I',len(self.tunables)))
            for index in xrange(len(self.tunables)):
                assert index in self.tunables
                value,size = self.tunables[index]
                replay_file.write(struct.pack('I',size))
                string_length = len(value)+1 # need null termination
                replay_file.write(struct.pack('I',string_length))
                replay_file.write(struct.pack(str(string_length)+'s',value))
        else:
            replay_file.write(struct.pack('I',0))
        # Pack the operation indexes
        if self.operation_indexes:
            replay_file.write(struct.pack('I',len(self.operation_indexes)))
            for idx in xrange(len(self.operation_indexes)):
                assert idx in self.operation_indexes
                replay_file.write(struct.pack('Q',self.operation_indexes[idx]))
        else:
            replay_file.write(struct.pack('I',0))
        # Pack the close indexes
        if self.close_indexes:
            replay_file.write(struct.pack('I',len(self.close_indexes)))
            for idx in xrange(len(self.close_indexes)):
                assert idx in self.close_indexes
                replay_file.write(struct.pack('Q',self.close_indexes[idx]))
        else:
            replay_file.write(struct.pack('I',0))

class Future(object):
    __slots__ = ['state', 'iid', 'creator_uid', 'logical_creator', 
                 'physical_creators', 'point', 'user_ids',
                 'logical_users', 'physical_users']
    def __init__(self, state, iid):
        self.state = state
        self.iid = iid
        self.creator_uid = None
        # These can be different for index space operations
        self.logical_creator = None
        # Can be multiple creators for index space launches with reductions
        self.physical_creators = None
        self.point = None
        self.user_ids = None
        self.logical_users = None
        self.physical_users = None

    def set_creator(self, uid):
        self.creator_uid = uid

    def set_point(self, point):
        self.point = point

    def add_uid(self, uid):
        if not self.user_ids:
            self.user_ids = set()
        self.user_ids.add(uid)

    def update_creator_and_users(self):
        # The creator uid is always for the logical operation   
        if self.creator_uid:
            self.logical_creator = self.state.get_operation(self.creator_uid)
            assert self.logical_creator
            self.logical_creator.add_created_future(self)
            # If this is an index space operation and has a point, 
            # then get the physical creator
            self.physical_creators = set()
            if self.logical_creator.kind == INDEX_TASK_KIND:
                # Deal with predication
                if self.logical_creator.points:
                    if self.point.dim > 0:
                        self.physical_creators.add(
                                self.logical_creator.get_point_task(self.point).op)
                    else:
                        for point in itervalues(self.logical_creator.points):
                            self.physical_creators.add(point.op)
            else:
                self.physical_creators.add(self.logical_creator) 
            for creator in self.physical_creators:
                creator.add_created_future(self)
        if self.user_ids:
            # Future use is always for the physical operations   
            self.physical_users = set()
            for uid in self.user_ids:
                user = self.state.get_operation(uid)
                self.physical_users.add(user)
                user.add_used_future(self)
            self.user_ids = None
            # Now convert back to logical users
            self.logical_users = set()
            for user in self.physical_users:
                # If this was part of an index space launch then
                # the logical user is the index owner
                if user.index_owner:
                    self.logical_users.add(user.index_owner)
                    user.index_owner.add_used_future(self)
                else:
                    self.logical_users.add(user)
                    user.add_used_future(self)
        # Hook up the dependences between operations
        if self.logical_creator and self.logical_users:
            for user in self.logical_users:
                if not user.logical_incoming:
                    user.logical_incoming = set()
                user.logical_incoming.add(self.logical_creator)
                if not self.logical_creator.logical_outgoing:
                    self.logical_creator.logical_outgoing= set()
                self.logical_creator.logical_outgoing.add(user)
        if self.physical_creators and self.physical_users:
            for user in self.physical_users:
                if not user.physical_incoming:
                    user.physical_incoming = set()
                for creator in self.physical_creators:
                    user.physical_incoming.add(creator)
                    if not creator.physical_outgoing:
                        creator.physical_outgoing = set()
                    creator.physical_outgoing.add(user)

class PointUser(object):
    __slots__ = ['op', 'index', 'logical_op', 'priv', 'coher', 'redop', 'version']
    def __init__(self, op, index, priv, coher, redop, version):
        self.op = op
        self.index = index
        # The application level operation that generated this user
        if op is not None:
            self.logical_op = op.get_logical_op()
            assert isinstance(self.logical_op, Operation)
        else:
            self.logical_op = None
        self.priv = priv
        self.coher = coher
        self.redop = redop
        self.version = version

    def is_realm_op(self):
        assert self.op is not None
        return self.op.is_realm_operation()

    def is_no_access(self):
        return self.priv == NO_ACCESS

    def is_read_only(self):
        return self.priv == READ_ONLY

    def has_write(self):
        return (self.priv == READ_WRITE) or (self.priv == REDUCE) or \
                (self.priv == WRITE_DISCARD) or (self.priv == WRITE_ONLY)

    def is_write(self):
        return (self.priv == READ_WRITE) or (self.priv == WRITE_DISCARD) or \
                (self.priv == WRITE_ONLY)

    def is_read_write(self):
        return self.priv == READ_WRITE

    def is_write_only(self):
        return self.priv == WRITE_DISCARD or self.priv == WRITE_ONLY

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

class SpecializedConstraint(object):
    __slots__ = ['kind', 'redop']
    def __init__(self, kind, redop):
        self.kind = kind
        self.redop = redop

    def pack_constraint(self, replay_file):
        replay_file.write(struct.pack('I', self.kind))
        replay_file.write(struct.pack('I', self.redop))

class MemoryConstraint(object):
    __slots__ = ['kind']
    def __init__(self, kind):
        self.kind = kind

    def pack_constraint(self, replay_file):
        replay_file.write(struct.pack('I', self.kind))

class FieldConstraint(object):
    __slots__ = ['contiguous', 'inorder', 'fields', 'count']
    def __init__(self, contig, inorder, count):
        self.contiguous = contig
        self.inorder = inorder
        self.fields = list()
        self.count = count

    def add_field(self, fid):
        self.fields.append(fid)

    def pack_constraint(self, replay_file):
        replay_file.write(struct.pack('I', len(self.fields)))
        for fid in self.fields:
            replay_file.write(struct.pack('I', fid))
        replay_file.write(struct.pack('I', self.contiguous))
        replay_file.write(struct.pack('I', self.inorder))

class OrderingConstraint(object):
    __slots__ = ['contiguous', 'dimensions', 'count']
    def __init__(self, contig, count):
        self.contiguous = contig
        self.dimensions = list()
        self.count = count

    def add_dim(self, dim):
        self.dimensions.append(dim)

    def pack_constraint(self, replay_file):
        replay_file.write(struct.pack('I',len(self.dimensions)))
        for dim in self.dimensions:
            replay_file.write(struct.pack('I',dim))
        replay_file.write(struct.pack('I', self.contiguous))

class SplittingConstraint(object):
    __slots__ = ['dim', 'value', 'chunks']
    def __init__(self, dim, value, chunks):
        self.dim = dim
        self.value = value
        self.chunks = chunks

    def pack_constraint(self, replay_file):
        replay_file.write(struct.pack('I', self.dim))
        replay_file.write(struct.pack('Q', self.value))
        replay_file.write(struct.pack('I', self.chunks))

class DimensionConstraint(object):
    __slots__ = ['dim', 'eqk', 'value']
    def __init__(self, dim, eqk, value):
        self.dim = dim
        self.eqk = eqk
        self.value = value

    def pack_constraint(self, replay_file):
        replay_file.write(struct.pack('I', self.dim))
        replay_file.write(struct.pack('I', self.eqk))
        replay_file.write(struct.pack('Q', self.value))
          
class AlignmentConstraint(object):
    __slots__ = ['fid', 'eqk', 'alignment']
    def __init__(self, fid, eqk, alignment):
        self.fid = fid
        self.eqk = eqk
        self.alignment = alignment

    def pack_constraint(self, replay_file):
        replay_file.write(struct.pack('I', self.fid))
        replay_file.write(struct.pack('I', self.eqk))
        replay_file.write(struct.pack('Q', self.alignment))

class OffsetConstraint(object):
    __slots__ = ['fid', 'offset']
    def __init__(self, fid, offset):
        self.fid = fid
        self.offset = offset

    def pack_constraint(self, replay_file):
        replay_file.write(struct.pack('I', self.fid))
        replay_file.write(struct.pack('q', self.offset))

class InstanceUser(object):
    __slots__ = ['op', 'index', 'logical_op', 'region', 'priv', 'coher', 
                 'redop', 'shape', 'intersect']
    def __init__(self, op, index, region, priv, coher, redop, 
                 shape=None, intersect=None):
        # The operation that generated this user
        self.op = op
        self.index = index
        # The application level operation that generated this user
        if op is not None:
            self.logical_op = op.get_logical_op()
            assert isinstance(self.logical_op, Operation)
        else:
            self.logical_op = None
        self.region = region
        self.priv = priv
        self.coher = coher
        self.redop = redop
        self.shape = shape
        self.intersect = intersect

    def is_realm_op(self):
        assert self.op is not None
        return self.op.is_realm_operation()

    def is_no_access(self):
        return self.priv == NO_ACCESS

    def is_read_only(self):
        return self.priv == READ_ONLY

    def has_write(self):
        return (self.priv == READ_WRITE) or (self.priv == REDUCE) or \
                (self.priv == WRITE_DISCARD) or (self.priv == WRITE_ONLY)

    def is_write(self):
        return (self.priv == READ_WRITE) or (self.priv == WRITE_DISCARD) or \
                (self.priv == WRITE_ONLY)

    def is_read_write(self):
        return self.priv == READ_WRITE

    def is_write_only(self):
        return self.priv == WRITE_DISCARD or self.priv == WRITE_ONLY

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
    __slots__ = ['state', 'use_event', 'handle', 'memory', 'fields', 'redop', 
                 'index_expr', 'field_space', 'tree_id', 'verification_users', 'processor', 
                 'creator', 'uses', 'creator_regions', 'specialized_constraint',
                 'memory_constraint', 'field_constraint', 'ordering_constraint',
                 'splitting_constraints', 'dimension_constraints',
                 'alignment_constraints', 'offset_constraints']
    def __init__(self, state, use_event):
        self.state = state
        # Instances are uniquely identified by their use event since Realm
        # can recycle physical instance IDs
        self.use_event = use_event
        if self.use_event == 0:
            self.handle = 0 # Virtual Instance
        else:
            self.handle = None 
        self.memory = None
        self.index_expr = None
        self.field_space = None
        self.tree_id = None
        self.creator_regions = None # Regions contributing to upper bound
        self.fields = None
        self.redop = 0
        self.verification_users = dict()
        self.creator = None # Initially a uid, later an op after post-parsing
        self.processor = None
        self.uses = 0
        self.specialized_constraint = None
        self.memory_constraint = None
        self.field_constraint = None
        self.ordering_constraint = None
        self.splitting_constraints = None
        self.dimension_constraints = None
        self.alignment_constraints = None
        self.offset_constraints = None

    def __str__(self):
        #return "Instance %s in %s" % (hex(self.handle), str(self.memory))
        if self.is_virtual():
            return "Virtual Instance"
        else:
            return "Instance "+hex(self.handle)

    __repr__ = __str__

    def set_handle(self, handle):
        self.handle = handle

    def set_memory(self, memory):
        self.memory = memory

    def set_properties(self, index_expr, field_space, tid):
        self.index_expr = index_expr
        self.field_space = field_space
        self.tree_id = tid

    def set_redop(self, redop):
        self.redop = redop

    def add_field(self, fid):
        # We better have a region at this point
        assert self.field_space is not None
        field = self.field_space.get_field(fid)
        if self.fields is None:
            self.fields = set()
        self.fields.add(field)

    def set_creator(self, uid, proc):
        self.creator = uid
        self.processor = proc

    def update_creator(self):
        if self.creator is not None:
            op = self.state.get_operation(self.creator)
            self.creator = op

    def add_creator_region(self, region):
        if not self.creator_regions:
            self.creator_regions = set()
        self.creator_regions.add(region)

    def set_specialized_constraint(self, kind, redop):
        assert not self.specialized_constraint
        self.specialized_constraint = SpecializedConstraint(kind, redop)

    def set_memory_constraint(self, kind):
        assert not self.memory_constraint
        self.memory_constraint = MemoryConstraint(kind)

    def set_field_constraint(self, contig, inorder, fields):
        assert not self.field_constraint
        self.field_constraint = FieldConstraint(contig, inorder, fields)

    def add_field_constraint_field(self, fid):
        assert self.field_constraint
        self.field_constraint.add_field(fid)

    def set_ordering_constraint(self, contig, dims):
        assert not self.ordering_constraint
        self.ordering_constraint = OrderingConstraint(contig, dims)

    def add_ordering_constraint_dim(self, dim):
        assert self.ordering_constraint
        self.ordering_constraint.add_dim(dim)

    def add_splitting_constraint(self, dim, value, chunks):
        if not self.splitting_constraints:
            self.splitting_constraints = list()
        self.splitting_constraints.append(SplittingConstraint(dim, value, chunks))

    def add_dimension_constraint(self, dim, eqk, value):
        if not self.dimension_constraints:
            self.dimension_constraints = list()
        self.dimension_constraints.append(DimensionConstraint(dim, eqk, value))

    def add_alignment_constraint(self, fid, eqk, alignment):
        if not self.alignment_constraints:
            self.alignment_constraints = list()
        self.alignment_constraints.append(AlignmentConstraint(fid, eqk, alignment))

    def add_offset_constraint(self, fid, offset):
        if not self.offset_constraints:
            self.offset_constraints = list()
        self.offset_constraints.append(OffsetConstraint(fid, offset))

    def increment_use_count(self):
        self.uses += 1

    # Only one virtual instance always with ID 0
    def is_virtual(self):
        return self.handle == 0

    def is_composite(self):
        return False

    def reset_verification_users(self, depth):
        if depth in self.verification_users:
            del self.verification_users[depth]

    def get_verification_users(self, depth, field, point):
        if depth not in self.verification_users:
            self.verification_users[depth] = dict()
        key = (field,point)
        if key not in self.verification_users[depth]:
            result = list()
            self.verification_users[depth][key] = result
            return result
        return self.verification_users[depth][key]

    def find_verification_use_dependences(self, depth, field, point, op, req):
        assert not self.is_virtual()
        users = self.get_verification_users(depth, field, point)
        result = set()
        logical_op = op.get_logical_op()
        for user in reversed(users):
            # If this is another user generated by the same operation
            # but from a different region requirement then we can 
            # skip the dependence because we'll catch it implicitly
            # as part of the dependences through other region requirements
            if logical_op is user.logical_op and req.index != user.index:
                continue
            dep = compute_dependence_type(user, req)
            if dep == TRUE_DEPENDENCE or dep == ANTI_DEPENDENCE:
                result.add(user.op)
                # If the previous was an exclusive user there is no
                # need to keep going back
                if user.is_write() and user.is_exclusive():
                    break
        return result

    def find_verification_copy_dependences(self, depth, field, point, 
                                           op, index, reading, redop, version):
        assert not self.is_virtual()
        users = self.get_verification_users(depth, field, point)
        result = set()
        if reading:
            assert redop == 0
            inst = PointUser(None, index, READ_ONLY, EXCLUSIVE, 0, version)
        elif redop != 0:
            inst = PointUser(None, index, REDUCE, EXCLUSIVE, redop, version)
        else:
            inst = PointUser(None, index, READ_WRITE, EXCLUSIVE, 0, version)
        logical_op = op.get_logical_op()
        for user in reversed(users):
            # If this user was generated by the same operation check to 
            # see if is another user or a copy operation, users from a
            # different region requirement can be skipped, otherwise
            # we can avoid WAR and WAW dependences, but not true RAW dependences
            if logical_op is user.logical_op and index != user.index:
                if not user.is_realm_op() or not reading or user.is_read_only():
                    continue
            dep = compute_dependence_type(user, inst)
            if dep == TRUE_DEPENDENCE or dep == ANTI_DEPENDENCE:
                # We can safely write the same version number on top of a reader
                if dep == ANTI_DEPENDENCE and redop == 0 and version == user.version:
                    continue
                result.add(user.op)
                # If the previous was an exclusive writer than we 
                # have transitive dependences on everything before
                if user.is_write() and user.is_exclusive():
                    break
        return result

    def add_verification_user(self, depth, field, point, op, req, version):
        assert not self.is_virtual()
        assert not op.is_realm_operation()
        users = self.get_verification_users(depth, field, point)
        users.append(PointUser(op, req.index, req.priv, req.coher, req.redop, version))

    def add_verification_copy_user(self, depth, field, point, op, index, 
                                   reading, redop, version):
        assert not self.is_virtual()
        assert op.is_realm_operation()
        users = self.get_verification_users(depth, field, point)
        if reading:
            assert redop == 0
            users.append(PointUser(op, index, READ_ONLY, EXCLUSIVE, 0, version))
        elif redop != 0:
            users.append(PointUser(op, index, REDUCE, EXCLUSIVE, redop, version))
        else:
            users.append(PointUser(op, index, READ_WRITE, EXCLUSIVE, 0, version))

    def pack_inst_replay_info(self, replay_file):
        replay_file.write(struct.pack('Q', self.handle))
        replay_file.write(struct.pack('I', self.uses))
        replay_file.write(struct.pack('Q', self.processor.uid))
        replay_file.write(struct.pack('Q', self.memory.uid))
        # Now pack the constraints
        assert self.specialized_constraint
        self.specialized_constraint.pack_constraint(replay_file)
        assert self.memory_constraint
        self.memory_constraint.pack_constraint(replay_file)
        assert self.field_constraint
        self.field_constraint.pack_constraint(replay_file) 
        assert self.ordering_constraint
        self.ordering_constraint.pack_constraint(replay_file)
        if self.splitting_constraints:
            replay_file.write(struct.pack('I',len(self.splitting_constraints)))
            for constraint in self.splitting_constraints:
                constraint.pack_constraint(replay_file)
        else:
            replay_file.write(struct.pack('I',0))
        if self.dimension_constraints:
            replay_file.write(struct.pack('I',len(self.dimension_constraints)))
            for constraint in self.dimension_constraints:
                constraint.pack_constraint(replay_file)
        else:
            replay_file.write(struct.pack('I',0))
        if self.alignment_constraints:
            replay_file.write(struct.pack('I',len(self.alignment_constraints)))
            for constraint in self.alignment_constraints:
                constraint.pack_constraint(replay_file)
        else:
            replay_file.write(struct.pack('I',0))
        if self.offset_constraints:
            replay_file.write(struct.pack('I',len(self.offset_constraints)))
            for constraint in self.offset_constraints:
                constraint.pack_constraint(replay_file)
        else:
            replay_file.write(struct.pack('I',0))
        # Now pack the paths for each of the different creator regions
        assert self.creator_regions
        replay_file.write(struct.pack('I',len(self.creator_regions)))
        for region in self.creator_regions:
            path = collections.deque()
            temp = region
            while temp.parent:
                path.appendleft(temp.index_space.color)
                path.appendleft(temp.parent.index_partition.color)
                temp = temp.parent.parent
            replay_file.write(struct.pack('I',len(path)))
            for color in path:
                replay_file.write(struct.pack('i', color.dim))
                for idx in xrange(color.dim):
                    replay_file.write(struct.pack('Q', color.vals[idx]))

class EventHandle(object):
    __slots__ = ['uid']
    def __init__(self, uid):
        self.uid = uid

    def __hash__(self):
        return hash(self.uid)

    def __eq__(self, other):
        return (self.uid) == (other.uid)

    def __str__(self):
        return "ev(" + hex(self.uid) + ")"

    __repr__ = __str__

    def exists(self):
        return (self.uid != 0)

class Event(object):
    __slots__ = ['state', 'handle', 'phase_barrier', 'incoming', 'outgoing',
                 'incoming_ops', 'outgoing_ops', 'incoming_fills', 'outgoing_fills',
                 'incoming_copies', 'outgoing_copies', 'incoming_depparts',
                 'outgoing_depparts', 'generation', 'ap_user_event',
                 'rt_user_event', 'pred_event', 'user_event_triggered', 
                 'barrier_contributors', 'barrier_waiters']
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
        self.incoming_depparts = None
        self.outgoing_depparts = None
        self.ap_user_event = False
        self.rt_user_event = False
        self.pred_event = False
        self.user_event_triggered = False
        # For traversals
        self.generation = 0
        # For phase barriers
        self.barrier_contributors = None
        self.barrier_waiters = None

    def exists(self):
        return self.handle.uid > 0

    def __str__(self):
        return str(self.handle)

    __repr__ = __str__

    def set_ap_user_event(self):
        assert not self.ap_user_event
        assert not self.rt_user_event
        assert not self.pred_event
        self.ap_user_event = True

    def set_rt_user_event(self):
        assert not self.ap_user_event
        assert not self.rt_user_event
        assert not self.pred_event
        self.rt_user_event = True

    def set_pred_event(self):
        assert not self.ap_user_event
        assert not self.rt_user_event
        assert not self.pred_event
        self.pred_event = True

    def set_triggered(self):
        assert not self.user_event_triggered
        self.user_event_triggered = True

    def check_for_user_event_leak(self):
        if self.user_event_triggered:
            return
        if not self.ap_user_event and not self.rt_user_event and not self.pred_event:
            return
        # This is an untriggered user event, report it
        if self.ap_user_event:
            owner = None
            for op in itervalues(self.state.ops):
                if op.finish_event is self:
                    owner = op
                    break
            if owner:
                print("WARNING: "+str(self)+" is an untriggered application "+
                      "user event for "+str(owner))
            else:
                print("WARNING: "+str(self)+" is an untriggered application user event")
            if self.state.assert_on_warning:
                assert False
        elif self.rt_user_event:
            print("WARNING: "+str(self)+" is an untriggered runtime user event")
            if self.state.assert_on_warning:
                assert False
        else:
            print("WARNING: "+str(self)+" is an untriggered predicate event")
            if self.state.assert_on_warning:
                assert False
        print("  Incoming:")
        if self.incoming:
            for ev in self.incoming:
                print("    "+str(ev))
        if self.incoming_ops:
            for op in self.incoming_ops:
                print("    "+str(op))
        if self.incoming_copies:
            for copy in self.incoming_copies:
                print("    "+str(copy))
        if self.incoming_fills:
            for fill in self.incoming_fills:
                print("    "+str(fill))
        if self.incoming_depparts:
            for deppart in self.incoming_depparts:
                print("    "+str(deppart))
        print("  Outgoing:")
        if self.outgoing:
            for ev in self.outgoing:
                print("    "+str(ev))
        if self.outgoing_ops:
            for op in self.outgoing_ops:
                print("    "+str(op))
        if self.outgoing_copies:
            for copy in self.outgoing_copies:
                print("    "+str(copy))
        if self.outgoing_fills:
            for fill in self.outgoing_fills:
                print("    "+str(fill))
        if self.outgoing_depparts:
            for deppart in self.outgoing_depparts:
                print("    "+str(deppart))

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

    def update_incoming_op(self, old, new):
        assert old in self.incoming_ops
        self.incoming_ops.remove(old)
        self.incoming_ops.add(new)

    def update_outgoing_op(self, old, new):
        assert old in self.outgoing_ops
        self.outgoing_ops.remove(old)
        self.outgoing_ops.add(new)

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

    def add_incoming_deppart(self, deppart):
        if self.incoming_depparts is None:
            self.incoming_depparts = set()
        self.incoming_depparts.add(deppart)

    def add_outgoing_deppart(self, deppart):
        if self.outgoing_depparts is None:
            self.outgoing_depparts = set()
        self.outgoing_depparts.add(deppart)

    def add_phase_barrier_contributor(self, op):
        if not self.barrier_contributors:
            self.barrier_contributors = set()
        self.barrier_contributors.add(op)
        self.phase_barrier = True

    def add_phase_barrier_waiter(self, op):
        if not self.barrier_waiters:
            self.barrier_waiters = set()
        self.barrier_waiters.add(op)
        self.phase_barrier = True

class RealmBase(object):
    __slots__ = ['state', 'realm_num', 'creator', 'index_expr', 'field_space',
                 'start_event', 'finish_event', 'physical_incoming', 'physical_outgoing', 
                 'eq_incoming', 'eq_outgoing', 'eq_privileges', 'generation', 
                 'event_context', 'version_numbers', 'across_version_numbers', 
                 'indirections', 'cluster_name']
    def __init__(self, state, realm_num):
        self.state = state
        self.realm_num = realm_num
        self.creator = None
        self.index_expr = None
        self.field_space = None
        self.physical_incoming = set()
        self.physical_outgoing = set()
        self.eq_incoming = None
        self.eq_outgoing = None
        self.eq_privileges = None
        self.start_event = state.get_no_event()
        self.finish_event = state.get_no_event()
        self.generation = 0
        self.event_context = None
        self.version_numbers = None
        self.across_version_numbers = None
        self.indirections = None
        self.cluster_name = None # always none

    def is_realm_operation(self):
        return True

    def is_physical_operation(self):
        return True 

    def add_equivalence_incoming(self, eq, src):
        assert eq in self.eq_privileges
        if self.eq_incoming is None:
            self.eq_incoming = dict()
        if eq not in self.eq_incoming:
            self.eq_incoming[eq] = set()
        self.eq_incoming[eq].add(src)

    def add_equivalence_outgoing(self, eq, dst):
        assert eq in self.eq_privileges
        if self.eq_outgoing is None:
            self.eq_outgoing = dict()
        if eq not in self.eq_outgoing:
            self.eq_outgoing[eq] = set()
        self.eq_outgoing[eq].add(dst)

    def get_context(self):
        assert self.creator is not None
        return self.creator.get_context()

    def get_logical_op(self):
        assert self.creator is not None
        return self.creator

    def get_point_set(self):
        result = PointSet()
        if self.version_numbers:
            for eq_key in iterkeys(self.version_numbers):
                result.add_point(eq_key[0])
        return result

    def record_version_number(self, eq_set):
        if self.version_numbers is None:
            self.version_numbers = dict()
        eq_key = (eq_set.point, eq_set.field, eq_set.tree.tree_id)
        if eq_key in self.version_numbers:
            assert self.version_numbers[eq_key] == eq_set.version_number
            return True
        else:
            self.version_numbers[eq_key] = eq_set.version_number
            return False 

    def record_across_version_number(self, point, field, tree, version):
        if self.across_version_numbers is None:
            self.across_version_numbers = dict()
        eq_key = (point, field, tree)
        if eq_key in self.across_version_numbers:
            assert self.across_version_numbers[eq_key] == version
        else:
            self.across_version_numbers[eq_key] = version

    def check_for_spurious_updates(self, fields, tree, versions):
        # Indirection copies are never spurious so we do not check this currently
        if self.indirections is not None:
            return True
        point_set = self.index_expr.get_point_set()
        for point in point_set.iterator():
            for field in fields:
                eq_key = (point, field, tree)
                if versions is None or eq_key not in versions:
                    print('ERROR: '+str(self.creator)+' generated spurious '+
                            str(self)+' for point '+str(point)+' of '+str(field)+
                            ' in tree '+str(tree))
                    if self.state.eq_graph_on_error:
                        self.state.dump_eq_graph((point, field, tree))
                    if self.state.assert_on_error:
                        assert False
                    return False
        return True

    def compute_physical_reachable(self):
        # Once we reach something that is not an event
        # then we record it and return
        def traverse_node(node, traverser):
            if not node.is_physical_operation():
                return True
            traverser.reachable.add(node)
            return False
        if self.start_event.exists():
            traverser = EventGraphTraverser(forwards=False, use_gen=True,
                generation=self.state.get_next_traversal_generation(),
                op_fn=traverse_node, copy_fn=traverse_node, 
                fill_fn=traverse_node, deppart_fn=traverse_node)
            traverser.reachable = self.physical_incoming
            traverser.run(self.start_event)
            # Keep everything symmetric
            for other in self.physical_incoming:
                other.physical_outgoing.add(self)
        if self.finish_event.exists():
            traverser = EventGraphTraverser(forwards=True, use_gen=True,
                generation=self.state.get_next_traversal_generation(),
                op_fn=traverse_node, copy_fn=traverse_node, 
                fill_fn=traverse_node, deppart_fn=traverse_node)
            traverser.reachable = self.physical_outgoing
            traverser.run(self.finish_event)
            # Keep everything symmetric
            for other in self.physical_outgoing:
                other.physical_incoming.add(self)

    def perform_cycle_check(self, cycle_detector):
        return cycle_detector.check_for_cycles(self, self.physical_outgoing)

    def print_incoming_event_edges(self, printer):
        for src in self.physical_incoming:
            if src.cluster_name is not None:
                printer.println(src.node_name+' -> '+self.node_name+
                            ' [ltail='+src.cluster_name+',style=solid,'+
                            'color=black,penwidth=2];')
            else:
                printer.println(src.node_name+' -> '+self.node_name+
                        ' [style=solid,color=black,penwidth=2];')

    def print_eq_node(self, printer, eq_key):
        pass

    def print_incoming_eq_edges(self, printer, eq_key):
        if self.eq_incoming is None or eq_key not in self.eq_incoming:
            return
        for src in self.eq_incoming[eq_key]:
            if src.cluster_name is not None:
                printer.println(src.node_name+' -> '+self.node_name+
                            ' [ltail='+src.cluster_name+',style=solid,'+
                            'color=black,penwidth=2];')
            else:
                printer.println(src.node_name+' -> '+self.node_name+
                        ' [style=solid,color=black,penwidth=2];')

class Indirections(object):
    __slots__ = ['uid', 'state', 'instances', 'groups']
    def __init__(self, uid, state):
        self.uid = uid
        self.state = state
        self.groups = list()
        self.instances = list()

    def add_indirect_instance(self, index, inst, fid):
        while len(self.instances) <= index:
            self.instances.append(None)
        field = inst.field_space.get_field(fid)
        self.instances[index] = (inst,field) 

    def add_group_instance(self, index, inst, index_space):
        while len(self.groups) <= index:
            self.groups.append(list())
        self.groups[index].append((inst,index_space))

    def get_indirect_instance(self, index):
        return self.instances[index][0]

    def get_indirect_field(self, index):
        return self.instances[index][1]

    def get_group_field(self, index, fid):
        return self.groups[index][0][0].field_space.get_field(fid)

    def get_group_size(self, index):
        return len(self.groups[index])

    def get_group_instance(self, index, offset):
        group = self.groups[index]
        if offset >= len(group):
            return None
        return group[offset][0]

    def has_indirect_instance(self, index, inst, field):
        if index >= len(self.instances):
            return False
        pair = self.instances[index]
        if inst is not pair[0]:
            return False
        if field is not pair[1]:
            return False
        return True

    def has_group_instance(self, index, inst, index_space):
        if index >= len(self.groups):
            return False
        for instance,space in self.groups[index]:
            if inst is not instance:
                continue
            if index_space is not space:
                continue
            return True
        return False

    def exchange(self, other):
        for index in xrange(len(self.groups)):
            group = self.groups[index]
            pair = group[0]
            other.add_group_instance(index, pair[0], pair[1])

class RealmCopy(RealmBase):
    __slots__ = ['start_event', 'finish_event', 'src_fields', 'dst_fields', 
                 'srcs', 'dsts', 'src_tree_id', 'dst_tree_id', 'src_indirections',
                 'dst_indirections', 'redops', 'across', 'node_name']
    def __init__(self, state, finish, realm_num):
        RealmBase.__init__(self, state, realm_num)
        self.finish_event = finish
        if finish.exists():
            finish.add_incoming_copy(self)
        self.src_fields = list()
        self.dst_fields = list()
        self.srcs = list()
        self.dsts = list()
        self.src_tree_id = None
        self.dst_tree_id = None
        self.src_indirections = None
        self.dst_indirections = None
        self.redops = list()
        self.across = None
        self.node_name = 'realm_copy_'+str(realm_num)

    def __str__(self):
        if self.indirections:
            has_src = False
            for index in self.src_indirections:
                if index is not None:
                    has_src = True
                    break
            has_dst = False
            for index in self.dst_indirections:
                if index is not None:
                    has_dst = True
                    break
            assert has_src or has_dst
            if has_src:
                if has_dst:
                    return "Indirect Copy ("+str(self.realm_num)+")"
                else:
                    return "Gather Copy ("+str(self.realm_num)+")"
            else:
                return "Scatter Copy ("+str(self.realm_num)+")"
        else:
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

    def set_tree_properties(self, index_expr, src_tid, dst_tid):
        self.index_expr = index_expr
        self.src_tree_id = src_tid
        self.dst_tree_id = dst_tid

    def set_indirection_properties(self, index_expr, indirections):
        self.index_expr = index_expr
        self.src_indirections = list()
        self.dst_indirections = list()
        self.indirections = indirections

    def is_across(self):
        if self.across is not None:
            return self.across
        if self.src_tree_id != self.dst_tree_id:
            self.across = True
        else:
            self.across = False
            assert len(self.src_fields) == len(self.dst_fields)
            for idx in xrange(len(self.src_fields)):
                if self.src_fields[idx] != self.dst_fields[idx]:
                    self.across = True
                    break
        return self.across

    def update_creator(self, new_creator):
        assert self.creator
        assert new_creator is not self.creator
        self.creator = new_creator

    def add_field(self, src_fid, src, dst_fid, dst, redop):
        # Always get the fields from the source and destination regions
        # which is especially important for handling cross-region copies
        src_field = src.field_space.get_field(src_fid)
        dst_field = dst.field_space.get_field(dst_fid)
        self.src_fields.append(src_field)
        self.dst_fields.append(dst_field)
        self.srcs.append(src)
        self.dsts.append(dst)
        self.redops.append(redop)

    def add_indirect_field(self, src_fid, src, src_index, dst_fid, dst, dst_index, redop):
        assert self.indirections is not None
        if src_index >= 0:
            assert src is None
            src_field = self.indirections.get_group_field(src_index, src_fid)
            self.src_fields.append(src_field)
            self.srcs.append(None)
            self.src_indirections.append(src_index)
        else:
            assert src is not None
            src_field = src.field_space.get_field(src_fid)
            self.src_fields.append(src_field)
            self.srcs.append(src)
            self.src_indirections.append(None)
        if dst_index >= 0:
            assert dst is None
            dst_field = self.indirections.get_group_field(dst_index, dst_fid)
            self.dst_fields.append(dst_field)
            self.dsts.append(None)
            self.dst_indirections.append(dst_index)
        else:
            assert dst is not None
            dst_field = dst.field_space.get_field(dst_fid)
            self.dst_fields.append(dst_field)
            self.dsts.append(dst)
            self.dst_indirections.append(None)
        self.redops.append(redop)

    def find_src_inst(self, src_field):
        assert len(self.src_fields) == len(self.srcs)
        for idx in xrange(len(self.src_fields)):
            if src_field == self.src_fields[idx]:
                return self.srcs[idx]
        assert False

    def print_event_node(self, printer):
        if self.state.detailed_graphs:
            if self.index_expr:
                # This is the case where the runtime told us what
                # the name of the index space was for the copy
                point_set = self.index_expr.get_point_set()
                label = str(self)+" of "+ point_set.point_space_graphviz_string()
            else:
                # This is the case where we had to generate the 
                # copy from our own information
                point_set = self.get_point_set()
                assert point_set
                label = str(self)+" of "+point_set.point_space_graphviz_string()
        else:
            label = str(self)
        if self.creator is not None:
            label += " generated by "+self.creator.html_safe_name
            if self.creator.kind == SINGLE_TASK_KIND:
                label += " (UID: " + str(self.creator.uid) + ")"
        if self.indirections is not None:
            num_columns = 3
            num_fields = len(self.src_fields)
            has_src_indirect = False
            for fidx in xrange(num_fields):
                if self.src_indirections[fidx] is not None:
                    has_src_indirect = True
                    num_columns += 1
                    break 
            has_dst_indirect = False
            for fidx in xrange(num_fields):
                if self.dst_indirections[fidx] is not None:
                    has_dst_indirect = True
                    num_columns += 1
                    break
            lines = [[{ "label" : label, "colspan" : num_columns}]]
            if self.state.detailed_graphs:
                first_field = True
                for fidx in xrange(num_fields):
                    src_field = self.src_fields[fidx]
                    dst_field = self.dst_fields[fidx]
                    src_inst = self.srcs[fidx]
                    dst_inst = self.dsts[fidx]
                    src_index = self.src_indirections[fidx]
                    dst_index = self.dst_indirections[fidx]
                    redop = self.redops[fidx]
                    line = []
                    # Do the field labels first
                    line.append(src_field.html_safe_name)
                    local_rows = 1
                    if has_src_indirect:
                        assert src_index is not None
                        line.append('Src Indirect: '+
                                self.indirections.get_indirect_field(src_index).html_safe_name)
                        local_rows = max(local_rows, 
                                self.indirections.get_group_size(src_index))
                    if has_dst_indirect:
                        assert dst_index is not None
                        line.append('Dst Indirect: '+
                                self.indirections.get_indirect_field(dst_index).html_safe_name)
                        local_rows = max(local_rows, 
                                self.indirections.get_group_size(dst_index))
                    line.append(dst_field.html_safe_name)
                    if first_field:
                        # Count how many rows there are for each field
                        num_rows = num_fields
                        for idx in xrange(num_fields):
                            src_index = self.src_indirections[fidx]
                            count = 1
                            if src_index is not None:
                                size = self.indirections.get_group_size(src_index) 
                                if size > count:
                                    count = size
                            dst_index = self.dst_indirections[fidx]
                            if dst_index is not None:
                                size = self.indirections.get_group_size(dst_index)
                                if size > count:
                                    count = size
                            num_rows += count
                        line.insert(0, {"label" : "Fields",
                                        "rowspan" : num_rows})
                        first_field = False
                    lines.append(line)
                    # Now we do the rows for all the instances
                    for row in xrange(local_rows):
                        line = []
                        if has_src_indirect:
                            idx_inst = self.indirections.get_group_instance(src_index, row) 
                            if idx_inst is not None:
                                line.append(str(idx_inst))
                            else:
                                line.append('^')
                            if row == 0:
                                line.append(str(
                                    self.indirections.get_indirect_instance(src_index)))
                            else:
                                line.append('^')
                        else:
                            if row == 0:
                                line.append(str(src_inst))
                            else:
                                line.append('^')
                        if has_dst_indirect:
                            if row == 0:
                                line.append(str(
                                    self.indirections.get_indirect_instance(dst_index)))
                            else:
                                line.append('^')
                            idx_inst = self.indirections.get_group_instance(dst_index, row)
                            if idx_inst is not None:
                                line.append(str(idx_inst))
                            else:
                                line.append('^')
                        else:
                            if row == 0:
                                line.append(str(dst_inst))
                            else:
                                line.append('^')
                        lines.append(line)
        else:
            lines = [[{ "label" : label, "colspan" : 3 }]]
            if self.state.detailed_graphs:
                num_fields = len(self.src_fields)
                first_field = True
                for fidx in xrange(num_fields):
                    src_field = self.src_fields[fidx]
                    dst_field = self.dst_fields[fidx]
                    src_inst = self.srcs[fidx]
                    dst_inst = self.dsts[fidx]
                    redop = self.redops[fidx]
                    line = []
                    if src_field == dst_field:
                        if redop != 0:
                            line.append(src_field.html_safe_name+' Redop='+str(redop))
                        else:
                            line.append(src_field.html_safe_name)
                    else:
                        if redop != 0:
                            line.append(src_field.html_safe_name+':'+dst_field.html_safe_name+' Redop='+str(redop))
                        else:
                            line.append(src_field.html_safe_name+':'+dst_field.html_safe_name)
                    line.append(str(src_inst)+':'+str(dst_inst))
                    if first_field:
                        line.insert(0, {"label" : "Fields",
                                        "rowspan" : num_fields})
                        first_field = False
                    lines.append(line)
        if self.indirections is not None:
            color = 'darkorange'
        else:
            color = 'darkgoldenrod1'
        for redop in self.redops:
            if redop != 0:
                color = 'tomato'
                break
        size = 14
        label = '<table border="0" cellborder="1" cellspacing="0" cellpadding="3" bgcolor="%s">' % color + \
                "".join([printer.wrap_with_trtd(line) for line in lines]) + '</table>'
        printer.println(self.node_name+' [label=<'+label+'>,fontsize='+str(size)+\
                ',fontcolor=black,shape=box,penwidth=0];')

    def compute_copy_size(self):
        field_size = 0
        for field in self.src_fields:
            field_size += field.size
        point_set = self.index_expr.get_point_set()
        return (field_size * point_set.volume())

    def get_equivalence_privileges(self):
        if self.eq_privileges is None:
            self.eq_privileges = dict()
            point_set = self.index_expr.get_point_set()
            if self.is_across():
                # Copy-across case
                for fidx in range(len(self.src_fields)):
                    field = self.src_fields[fidx]
                    # Check for any indirections
                    if self.src_indirections is None or self.src_indirections[fidx] is None:
                        for point in point_set.iterator():
                            key = (point,field,self.src_tree_id)
                            assert key not in self.eq_privileges
                            self.eq_privileges[key] = READ_ONLY
                    else:
                        # Do the source instances first
                        index = self.src_indirections[fidx]
                        for off in range(self.indirections.get_group_size(index)):
                            inst,space = self.indirections.groups[index][off]
                            for point in space.get_point_set().iterator():
                                key = (point,field,inst.tree_id)
                                self.eq_privileges[key] = READ_ONLY
                        # Then do the indirection field
                        inst = self.indirections.get_indirect_instance(index)
                        field = self.indirections.get_indirect_field(index)
                        for point in point_set.iterator():
                            key = (point,field,inst.tree_id)
                            self.eq_privileges[key] = READ_ONLY
                    field = self.dst_fields[fidx]
                    redop = self.redops[fidx]
                    if self.dst_indirections is None or self.dst_indirections[fidx] is None:
                        for point in point_set.iterator():
                            key = (point,field,self.dst_tree_id)
                            assert key not in self.eq_privileges
                            self.eq_privileges[key] = WRITE_ONLY if redop == 0 else READ_WRITE
                    else:
                        # Do the destination instances first
                        index = self.dst_indirections[fidx]
                        for off in range(self.indirections.get_group_size(index)):
                            inst,space = self.indirections.groups[index][off]
                            for point in space.get_point_set().iterator():
                                key = (point,field,inst.tree_id)
                                self.eq_privileges[key] = WRITE_ONLY if redop == 0 else READ_WRITE
                        # Then do the indirection field
                        inst = self.indirections.get_indirect_instance(index)
                        field = self.indirections.get_indirect_field(index)
                        for point in point_set.iterator():
                            key = (point,field,inst.tree_id)
                            self.eq_privileges[key] = READ_ONLY
            else:
                # Normal copy case
                for point in point_set.iterator():
                    for field in self.src_fields:
                        key = (point,field,self.src_tree_id)
                        assert key not in self.eq_privileges
                        self.eq_privileges[key] = READ_ONLY
        return self.eq_privileges 

class RealmFill(RealmBase):
    __slots__ = ['fields', 'dsts', 'dst_tree_id', 'fill_op', 'across', 'node_name']
    def __init__(self, state, finish, realm_num):
        RealmBase.__init__(self, state, realm_num)
        self.finish_event = finish
        if finish.exists():
            finish.add_incoming_fill(self)
        self.fields = list()
        self.dsts = list()
        self.dst_tree_id = None
        self.fill_op = None  
        self.across = None
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

    def set_fill_op(self, fill_op):
        assert fill_op is not None
        if self.fill_op is None:
            self.fill_op = fill_op
        else:
            assert fill_op is self.fill_op

    def set_tree_properties(self, index_expr, field_space, dst_tid):
        self.index_expr = index_expr
        self.field_space = field_space
        self.dst_tree_id = dst_tid

    def is_across(self):
        if self.across is not None:
            return self.across
        # This happens when we do a fill for initializing a reduction instance
        if self.fill_op is None:
            self.across = False
            return self.across
        req = self.fill_op.reqs[0]
        if req.tid != self.dst_tree_id:
            self.across = True
        else:
            self.across = False
            for field in self.fields:
                if field not in req.fields:
                    self.across = True
                    break
        return self.across

    def update_creator(self, new_creator):
        assert self.creator
        assert new_creator is not self.creator
        self.creator = new_creator

    def add_field(self, fid, dst):
        assert self.field_space is not None
        field = self.field_space.get_field(fid)
        self.fields.append(field)
        self.dsts.append(dst)

    def print_event_node(self, printer):
        if self.state.detailed_graphs:
            if self.index_expr:
                # This is the case where the runtime told us what
                # the name of the index space was for the copy
                label = "Realm Fill ("+str(self.realm_num)+") of "+\
                            self.index_expr.point_space_graphviz_string()
            else:
                # This is the case where we had to generate the 
                # copy from our own information
                point_set = self.get_point_set()
                assert point_set
                label = "Realm Fill ("+str(self.realm_num)+") of "+\
                            point_set.point_space_graphviz_string()
        else:
            label = "Realm Fill ("+str(self.realm_num)+")"
        if self.creator is not None:
            label += " generated by "+self.creator.html_safe_name
        lines = [[{ "label" : label, "colspan" : 3 }]]
        if self.state.detailed_graphs:
            num_fields = len(self.fields)
            first_field = True
            for fidx in xrange(num_fields):
                dst_field = self.fields[fidx]
                dst_inst = self.dsts[fidx]
                line = []
                line.append(dst_field.html_safe_name)
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
                ',fontcolor=black,shape=box,penwidth=0];')

    def compute_fill_size(self):
        field_size = 0
        for field in self.fields:
            field_size += field.size
        shape = self.region.index_space.shape
        if self.intersect:
            shape = shape & self.intersect.index_space.shape
        return (field_size * shape.volume())

    def get_equivalence_privileges(self):
        if self.eq_privileges is None:
            self.eq_privileges = dict()
            point_set = self.index_expr.get_point_set()
            for point in point_set.iterator():
                for field in self.fields:
                    key = (point,field,self.dst_tree_id)
                    assert key not in self.eq_privileges
                    self.eq_privileges[key] = WRITE_ONLY
        return self.eq_privileges

class RealmDeppart(RealmBase):
    __slots__ = ['node_name']
    def __init__(self, state, finish, realm_num):
        RealmBase.__init__(self, state, realm_num)
        self.finish_event = finish
        if finish.exists():
            finish.add_incoming_deppart(self)
        self.node_name = 'realm_deppart_'+str(realm_num)

    def __str__(self):
        return "Realm Deppart ("+str(self.realm_num)+")"

    __repr__ = __str__

    def set_start(self, start):
        self.start_event = start
        if start.exists():
            start.add_outgoing_deppart(self)

    def set_creator(self, creator):
        assert self.creator is None
        self.creator = creator
        self.creator.add_realm_deppart(self)

    def update_creator(self, new_creator):
        assert self.creator
        assert new_creator is not self.creator
        self.creator = new_creator

    def set_index_expr(self, index_expr):
        self.index_expr = index_expr

    def print_event_node(self, printer):
        if self.state.detailed_graphs:
            label = "Realm Deppart ("+str(self.realm_num)+") of "+\
                    self.index_expr.point_space_graphviz_string()
        else:
            label = "Realm Deppart ("+str(self.realm_num)+")"
        if self.creator is not None:
            label += " generated by "+self.creator.html_safe_name
        lines = [[{ "label" : label, "colspan" : 3 }]]
        color = 'magenta'
        size = 14
        label = '<table border="0" cellborder="1" cellspacing="0" cellpadding="3" bgcolor="%s">' % color + \
                "".join([printer.wrap_with_trtd(line) for line in lines]) + '</table>'
        printer.println(self.node_name+' [label=<'+label+'>,fontsize='+str(size)+\
                ',fontcolor=black,shape=box,penwidth=0];')

    def get_equivalence_privileges(self):
        assert self.creator is not None
        if self.eq_privileges is None:
            self.eq_privileges = self.creator.get_equivalence_privileges()
        return self.eq_privileges

class EventGraphTraverser(object):
    def __init__(self, forwards, use_gen, generation,
                 event_fn = None, op_fn = None,
                 copy_fn = None, fill_fn = None,
                 deppart_fn = None,
                 post_event_fn = None, post_op_fn = None,
                 post_copy_fn = None, post_fill_fn = None,
                 post_deppart_fn = None):
        self.forwards = forwards
        self.use_gen = use_gen
        self.generation = generation
        self.functions = list()
        self.functions.append(event_fn)
        self.functions.append(op_fn)
        self.functions.append(copy_fn)
        self.functions.append(fill_fn)
        self.functions.append(deppart_fn)
        self.post_functions = list()
        self.post_functions.append(post_event_fn)
        self.post_functions.append(post_op_fn)
        self.post_functions.append(post_copy_fn)
        self.post_functions.append(post_fill_fn)
        self.post_functions.append(post_deppart_fn)

    def run(self, event):
        nodes = list()
        nodes.append((event,0,True))
        while nodes:
            node,kind,first_pass = nodes[-1]
            if first_pass:
                if self.use_gen:
                    if node.generation == self.generation:
                        nodes.pop()
                        continue
                    else:
                        node.generation = self.generation
                do_next = True
                if self.functions[kind] is not None:
                    do_next = self.functions[kind](node, self)
                if not do_next:
                    nodes.pop()
                    continue
                if kind == 0:
                    # Event 
                    # We'll just assume all events have some kind of children
                    # which will be true in most cases
                    nodes[-1] = (node,0,False)
                    if self.forwards:
                        if node.outgoing is not None:
                            for event in node.outgoing:
                                nodes.append((event,0,True))
                        if node.outgoing_ops is not None:
                            for op in node.outgoing_ops:
                                nodes.append((op,1,True))
                        if node.outgoing_copies is not None:
                            for copy in node.outgoing_copies:
                                nodes.append((copy,2,True))
                        if node.outgoing_fills is not None:
                            for fill in node.outgoing_fills:
                                nodes.append((fill,3,True))
                        if node.outgoing_depparts is not None:
                            for deppart in node.outgoing_depparts:
                                nodes.append((deppart,4,True))
                    else:
                        if node.incoming is not None:
                            for event in node.incoming:
                                nodes.append((event,0,True))
                        if node.incoming_ops is not None:
                            for op in node.incoming_ops:
                                nodes.append((op,1,True))
                        if node.incoming_copies is not None:
                            for copy in node.incoming_copies:
                                nodes.append((copy,2,True))
                        if node.incoming_fills is not None:
                            for fill in node.incoming_fills:
                                nodes.append((fill,3,True))
                        if node.incoming_depparts is not None:
                            for deppart in node.incoming_depparts:
                                nodes.append((deppart,4,True))
                    # We assumed we have children so keep going 
                    continue
                else:
                    # Something with just a start and finish event
                    if self.forwards:
                        if node.finish_event.exists():
                            nodes[-1] = (node,kind,False)
                            nodes.append((node.finish_event,0,True))
                            continue
                    else:
                        if node.start_event.exists():
                            nodes[-1] = (node,kind,False)
                            nodes.append((node.start_event,0,True))
                            continue
            if self.post_functions[kind] is not None:
                self.post_functions[kind](node, self)
            nodes.pop()


class PhysicalTraverser(object):
    def __init__(self, forwards, use_gen, generation,
                 node_fn = None, post_node_fn = None):
        self.forwards = forwards
        self.use_gen = use_gen
        self.generation = generation
        self.node_fn = node_fn
        self.post_node_fn = post_node_fn

    def run(self, node):
        # Do this with DFS for reachability
        nodes = list()
        nodes.append((node,True))
        while nodes:
            node,first_pass = nodes[-1]
            if first_pass:
                if self.use_gen:
                    if node.generation == self.generation:
                        nodes.pop()
                        continue
                    else:
                        assert node.generation < self.generation
                        node.generation = self.generation
                do_next = True
                if self.node_fn is not None:
                    do_next = self.node_fn(node, self)
                if not do_next:
                    nodes.pop()
                    continue
                children = node.physical_outgoing if self.forwards else node.physical_incoming
                if children is not None:
                    nodes[-1] = (node,False)
                    for next_node in children:
                        nodes.append((next_node,True))
                    # We have children to traverse so we're not done with this node yet
                    continue
            if self.post_node_fn is not None:
                self.post_node_fn(node, self)
            nodes.pop()

class GraphPrinter(object):
    # Static member so we only issue this warning once
    zoom_warning = True
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

    def print_pdf_after_close(self, simplify, zoom_graph=False):
        dot_file = self.close()
        pdf_file = self.name+".pdf"
        #svg_file = self.name+".svg"
        #png_file = self.name+".png"
        try:
            if simplify:
                print("Simplifying dot file "+dot_file+" with tred...")
                tred = subprocess.Popen(['tred', dot_file], stdout=subprocess.PIPE)
                print("Done")
                print("Invoking dot to generate file "+pdf_file+"...")
                dot = subprocess.Popen(['dot', '-Tpdf', '-o', pdf_file], stdin=tred.stdout)
                print("Done")
                if dot.wait() != 0:
                    raise Exception('DOT failed')
            else:
                print("Invoking dot to generate file "+pdf_file+"...")
                subprocess.check_call(['dot', '-Tpdf', '-o', pdf_file, dot_file])
                #subprocess.check_call(['dot', '-Tsvg', '-o', svg_file, dot_file])
                #subprocess.check_call(['dot', '-Tpng', '-o', png_file, dot_file])
        except OSError:
            print("WARNING: graphviz is not installed, skipping generation of "+str(self.name))
        except:
            print("WARNING: 'dot' failure, image for graph "+str(self.name)+" not generated")
            subprocess.call(['rm', '-f', 'core', pdf_file])
        # If we are making a zoom graph, then make a directory with the appropriate name
        if zoom_graph:
            try:
                import pydot
                # Make a directory to put this in 
                zoom_dir = 'zoom_'+self.name
                os.mkdir(zoom_dir)
                # Rest of this is courtesy of @manopapad
                nodes = {} # map(string,Node)
                in_edges = {} # map(string,set(Edge))
                out_edges = {} # map(string,set(Edge))
                def collect_nodes_edges(g):
                    for sub in g.get_subgraphs():
                        collect_nodes_edges(sub)
                    for n in g.get_nodes():
                        if n.get_style() == 'invis':
                            # HACK: Assuming invisible nodes aren't connected to anything
                            continue
                        assert(n.get_name() not in nodes)
                        nodes[n.get_name()] = n
                        n.set_URL(n.get_name() + '.svg')
                        in_edges[n.get_name()] = set()
                        out_edges[n.get_name()] = set()
                    for e in g.get_edges():
                        out_edges[e.get_source()].add(e)
                        in_edges[e.get_destination()].add(e)

                # Support both older and newer versions of pydot library
                # See pydot issue 159 on github:
                # https://github.com/erocarrera/pydot/issues/159
                graphs = pydot.graph_from_dot_file(self.filename)
                if type(graphs) == list:
                    # This is the common path
                    assert len(graphs) == 1
                    g = graphs[0]
                else:
                    # This is the deprecated path
                    g = graphs
                collect_nodes_edges(g)

                g.write_svg(zoom_dir + '/zoom.svg')
                g.write_cmap(zoom_dir + '/zoom.map')
                g.write_png(zoom_dir + '/zoom.png')
                with open(zoom_dir+'/index.html', 'w') as f:
                    f.write('<!DOCTYPE html>\n')
                    f.write('<html>\n')
                    f.write('<head></head>\n')
                    f.write('<body>\n')
                    f.write('<img src="zoom.png" usemap="#mainmap"/>\n')
                    f.write('<map id="mainmap" name="mainmap">\n')
                    with open(zoom_dir+'/zoom.map', 'r') as f_map:
                        for line in f_map:
                            f.write(line)
                    f.write('</map>\n')
                    f.write('</body>\n')
                    f.write('</html>\n')

                for n in nodes.values():
                    sub = pydot.Dot()
                    sub.obj_dict['attributes'] = g.get_attributes()
                    sub.add_node(n)
                    for e in out_edges[n.get_name()]:
                        dst = nodes[e.get_destination()]
                        sub.add_node(dst)
                        sub.add_edge(e)
                    for e in in_edges[n.get_name()]:
                        src = nodes[e.get_source()]
                        sub.add_node(src)
                        sub.add_edge(e)
                    sub.write_svg(zoom_dir + '/' + n.get_name() + '.svg')
            except ImportError:
                if self.zoom_warning:
                    print("WARNING: Unable to make zoom plots because the package pydot is not installed")
                    GraphPrinter.zoom_warning = False

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
        for i in xrange(self.depth):
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
            for i in xrange(len(requirements)):
                req = requirements[i]
                region_name = req.logical_node.html_safe_name
                line = [str(i), region_name, req.get_privilege_and_coherence()]
                lines.append(line)
                if detailed:
                    first_field = True
                    for f in req.fields:
                        line = []
                        if first_field:
                            line.append({"label" : "Fields", "rowspan" : len(req.fields)})
                            first_field = False
                        line.append(f.html_safe_name)
                        if mappings is not None and i in mappings:
                            line.append(str(mappings[i][f.fid]))
                        else:
                            line.append('(Unknown instances)')
                        lines.append(line)
        return '<table border="0" cellborder="1" cellpadding="3" cellspacing="0" bgcolor="%s">' % color + \
              "".join([self.wrap_with_trtd(line) for line in lines]) + '</table>'


prefix    = "\[(?P<node>[0-9]+) - (?P<thread>[0-9a-f]+)\](?:\s+[0-9]+\.[0-9]+)? \{\w+\}\{legion_spy\}: "
prefix_pat               = re.compile(prefix)
# Configuration patterns
config_pat               = re.compile(
    prefix+"Legion Spy Logging")
detailed_config_pat      = re.compile(
    prefix+"Legion Spy Detailed Logging")
# Patterns for the shape of the machine
proc_kind_pat            = re.compile(
    prefix+"Processor Kind (?P<kind>[0-9]+) (?P<name>.+)")
mem_kind_pat             = re.compile(
    prefix+"Memory Kind (?P<kind>[0-9]+) (?P<name>.+)")
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
    prefix+"Index Space (?P<uid>[0-9a-f]+) (?P<owner>[0-9]+) (?P<provenance>.*)")
index_name_pat           = re.compile(
    prefix+"Index Space Name (?P<uid>[0-9a-f]+) (?P<name>.+)")
index_part_pat           = re.compile(
    prefix+"Index Partition (?P<pid>[0-9a-f]+) (?P<uid>[0-9a-f]+) (?P<disjoint>[0-2]) "+
           "(?P<complete>[0-2]) (?P<color>[0-9]+) (?P<owner>[0-9]+) (?P<provenance>.*)")
index_part_name_pat      = re.compile(
    prefix+"Index Partition Name (?P<uid>[0-9a-f]+) (?P<name>.+)")
index_subspace_pat       = re.compile(
    prefix+"Index Subspace (?P<pid>[0-9a-f]+) (?P<uid>[0-9a-f]+) (?P<owner>[0-9]+) "+
           "(?P<dim>[0-9]+) (?P<rem>.*)")
field_space_pat          = re.compile(
    prefix+"Field Space (?P<uid>[0-9]+) (?P<owner>[0-9]+) (?P<provenance>.*)")
field_space_name_pat     = re.compile(
    prefix+"Field Space Name (?P<uid>[0-9]+) (?P<name>.+)")
field_create_pat         = re.compile(
    prefix+"Field Creation (?P<uid>[0-9]+) (?P<fid>[0-9]+) (?P<size>[0-9]+) (?P<provenance>.*)")
field_name_pat           = re.compile(
    prefix+"Field Name (?P<uid>[0-9]+) (?P<fid>[0-9]+) (?P<name>.+)")
region_pat               = re.compile(
    prefix+"Region (?P<iid>[0-9a-f]+) (?P<fid>[0-9]+) (?P<tid>[0-9]+) (?P<owner>[0-9]+) (?P<provenance>.*)")
region_name_pat          = re.compile(
    prefix+"Logical Region Name (?P<iid>[0-9a-f]+) (?P<fid>[0-9]+) (?P<tid>[0-9]+) "+
            "(?P<name>.+)")
partition_name_pat       = re.compile(
    prefix+"Logical Partition Name (?P<iid>[0-9a-f]+) (?P<fid>[0-9]+) (?P<tid>[0-9]+) "+
            "(?P<name>.+)")
index_space_point_pat    = re.compile(
    prefix+"Index Space Point (?P<uid>[0-9a-f]+) (?P<dim>[0-9]+) (?P<rem>.*)")
index_space_rect_pat     = re.compile(
    prefix+"Index Space Rect (?P<uid>[0-9a-f]+) (?P<dim>[0-9]+) (?P<rem>.*)")
decimal_pat              = re.compile("\-?[0-9]+")
empty_index_space_pat    = re.compile(
    prefix+"Empty Index Space (?P<uid>[0-9a-f]+)")
index_expr_pat           = re.compile(
    prefix+"Index Space Expression (?P<uid>[0-9a-f]+) (?P<expr>[0-9]+)")
union_expr_pat           = re.compile(
    prefix+"Index Space Union (?P<expr>[0-9]+) (?P<count>[0-9]+)")
intersect_expr_pat       = re.compile(
    prefix+"Index Space Intersection (?P<expr>[0-9]+) (?P<count>[0-9]+)")
diff_expr_pat            = re.compile(
    prefix+"Index Space Difference (?P<expr>[0-9]+) (?P<left>[0-9]+) (?P<right>[0-9]+)")
# Patterns for operations
task_name_pat            = re.compile(
    prefix+"Task ID Name (?P<tid>[0-9]+) (?P<name>.+)")
task_variant_pat         = re.compile(
    prefix+"Task Variant (?P<tid>[0-9]+) (?P<vid>[0-9]+) (?P<inner>[0-1]) "+
    "(?P<leaf>[0-1]) (?P<idem>[0-1]+) (?P<name>.+)")
top_task_pat             = re.compile(
    prefix+"Top Task (?P<tid>[0-9]+) (?P<ctxuid>[0-9]+) (?P<uid>[0-9]+) (?P<name>.+)")
single_task_pat          = re.compile(
    prefix+"Individual Task (?P<ctx>[0-9]+) (?P<tid>[0-9]+) (?P<uid>[0-9]+) "+
            "(?P<index>[0-9]+) (?P<name>.+)")
index_task_pat           = re.compile(
    prefix+"Index Task (?P<ctx>[0-9]+) (?P<tid>[0-9]+) (?P<uid>[0-9]+) "+
            "(?P<index>[0-9]+) (?P<name>.+)")
inline_task_pat          = re.compile(
    prefix+"Inline Task (?P<uid>[0-9]+)")
mapping_pat              = re.compile(
    prefix+"Mapping Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index>[0-9]+)")
close_pat                = re.compile(
    prefix+"Close Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index>[0-9]+) "+
           "(?P<is_inter>[0-1])")
internal_creator_pat     = re.compile(
    prefix+"Internal Operation Creator (?P<uid>[0-9]+) (?P<cuid>[0-9]+) (?P<idx>[0-9]+)")
fence_pat                = re.compile(
    prefix+"Fence Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index>[0-9]+)")
trace_pat                = re.compile(
    prefix+"Trace Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
copy_op_pat              = re.compile(
    prefix+"Copy Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<kind>[0-9]+) "+ 
           "(?P<index>[0-9]+) (?P<src>[0-1]) (?P<dst>[0-1])")
fill_op_pat              = re.compile(
    prefix+"Fill Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index>[0-9]+)")
acquire_op_pat           = re.compile(
    prefix+"Acquire Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index>[0-9]+)")
release_op_pat           = re.compile(
    prefix+"Release Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index>[0-9]+)")
creation_pat             = re.compile(
    prefix+"Creation Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index>[0-9]+)")
deletion_pat             = re.compile(
    prefix+"Deletion Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index>[0-9]+) "+
           "(?P<unordered>[0-1])")
attach_pat               = re.compile(
    prefix+"Attach Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index>[0-9]+) "+
           "(?P<restriction>[0-1])")
detach_pat               = re.compile(
    prefix+"Detach Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index>[0-9]+) "+
           "(?P<unordered>[0-9]+)")
unordered_pat            = re.compile(
    prefix+"Unordered Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index>[0-9]+)")
dynamic_collective_pat   = re.compile(
    prefix+"Dynamic Collective (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index>[0-9]+)")
timing_op_pat            = re.compile(
    prefix+"Timing Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index>[0-9]+)")
tunable_op_pat           = re.compile(
    prefix+"Tunable Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index>[0-9]+)")
all_reduce_op_pat        = re.compile(
    prefix+"All Reduce Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index>[0-9]+)")
predicate_op_pat         = re.compile(
    prefix+"Predicate Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
must_epoch_op_pat        = re.compile(
    prefix+"Must Epoch Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
summary_op_pat        = re.compile(
    prefix+"Summary Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+)")
summary_op_creator_pat        = re.compile(
    prefix+"Summary Operation Creator (?P<uid>[0-9]+) (?P<cuid>[0-9]+)")
dep_partition_op_pat     = re.compile(
    prefix+"Dependent Partition Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) "+
           "(?P<pid>[0-9a-f]+) (?P<kind>[0-9]+) (?P<index>[0-9]+)")
pending_partition_op_pat = re.compile(
    prefix+"Pending Partition Operation (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index>[0-9]+)")
target_partition_pat     = re.compile(
    prefix+"Pending Partition Target (?P<uid>[0-9]+) (?P<pid>[0-9a-f]+) (?P<kind>[0-9]+)")
index_slice_pat          = re.compile(
    prefix+"Index Slice (?P<index>[0-9]+) (?P<slice>[0-9]+)")
slice_slice_pat          = re.compile(
    prefix+"Slice Slice (?P<slice1>[0-9]+) (?P<slice2>[0-9]+)")
slice_point_pat          = re.compile(
    prefix+"Slice Point (?P<slice>[0-9]+) (?P<point>[0-9]+) (?P<dim>[0-9]+) (?P<rem>.*)")
point_point_pat          = re.compile(
    prefix+"Point Point (?P<point1>[0-9]+) (?P<point2>[0-9]+)")
index_point_pat          = re.compile(
    prefix+"Index Point (?P<index>[0-9]+) (?P<point>[0-9]+) (?P<dim>[0-9]+) (?P<rem>.*)")
intra_space_pat          = re.compile(
    prefix+"Intra Space Dependence (?P<point>[0-9]+) (?P<dim>[0-9]+) (?P<rem>.*)")
op_index_pat             = re.compile(
    prefix+"Operation Index (?P<parent>[0-9]+) (?P<index>[0-9]+) (?P<child>[0-9]+)")
op_provenance_pat        = re.compile(
    prefix+"Operation Provenance (?P<uid>[0-9]+) (?P<provenance>.*)")
close_index_pat          = re.compile(
    prefix+"Close Index (?P<parent>[0-9]+) (?P<index>[0-9]+) (?P<child>[0-9]+)")
predicate_false_pat      = re.compile(
    prefix+"Predicate False (?P<uid>[0-9]+)")
# Patterns for logical analysis and region requirements
requirement_pat         = re.compile(
    prefix+"Logical Requirement (?P<uid>[0-9]+) (?P<index>[0-9]+) (?P<is_reg>[0-1]) "+
           "(?P<ispace>[0-9a-f]+) (?P<fspace>[0-9]+) (?P<tid>[0-9]+) (?P<priv>[0-9]+) "+
           "(?P<coher>[0-9]+) (?P<redop>[0-9]+) (?P<pis>[0-9a-f]+)")
req_field_pat           = re.compile(
    prefix+"Logical Requirement Field (?P<uid>[0-9]+) (?P<index>[0-9]+) (?P<fid>[0-9]+)")
projection_func_pat     = re.compile(
    prefix+"Projection Function (?P<pid>[0-9]+) (?P<depth>[0-9]+) (?P<invertible>[0-1])")
req_proj_pat            = re.compile(
    prefix+"Logical Requirement Projection (?P<uid>[0-9]+) (?P<index>[0-9]+) "+
           "(?P<pid>[0-9]+)")
index_launch_domain_pat = re.compile(
    prefix+"Index Launch Rect (?P<uid>[0-9]+) (?P<dim>[0-9]+) (?P<rem>.*)")
mapping_dep_pat         = re.compile(
    prefix+"Mapping Dependence (?P<ctx>[0-9]+) (?P<prev_id>[0-9]+) (?P<pidx>[0-9]+) "+
           "(?P<next_id>[0-9]+) (?P<nidx>[0-9]+) (?P<dtype>[0-9]+)")
future_create_pat       = re.compile(
    prefix+"Future Creation (?P<uid>[0-9]+) (?P<iid>[0-9a-f]+) (?P<dim>[0-9]+) (?P<rem>.*)")
future_use_pat          = re.compile(
    prefix+"Future Usage (?P<uid>[0-9]+) (?P<iid>[0-9a-f]+)")
predicate_use_pat       = re.compile(
    prefix+"Predicate Use (?P<uid>[0-9]+) (?P<pred>[0-9]+)")
# Physical instance and mapping decision patterns
instance_pat            = re.compile(
    prefix+"Physical Instance (?P<eid>[0-9a-f]+) (?P<iid>[0-9a-f]+) (?P<mid>[0-9a-f]+) (?P<redop>[0-9]+) (?P<expr>[0-9]+) (?P<space>[0-9]+) (?P<tid>[0-9]+)")
instance_field_pat      = re.compile(
    prefix+"Physical Instance Field (?P<eid>[0-9a-f]+) (?P<fid>[0-9]+)")
instance_creator_pat    = re.compile(
    prefix+"Physical Instance Creator (?P<eid>[0-9a-f]+) (?P<uid>[0-9]+) "
           "(?P<proc>[0-9a-f]+)")
instance_creator_region_pat = re.compile(
    prefix+"Physical Instance Creation Region (?P<eid>[0-9a-f]+) (?P<ispace>[0-9]+) "
           "(?P<fspace>[0-9]+) (?P<tid>[0-9]+)")
specialized_constraint_pat = re.compile(
    prefix+"Instance Specialized Constraint (?P<eid>[0-9a-f]+) (?P<kind>[0-9]+) "
           "(?P<redop>[0-9]+)")
memory_constraint_pat   = re.compile(
    prefix+"Instance Memory Constraint (?P<eid>[0-9a-f]+) (?P<kind>[0-9]+)")
field_constraint_pat    = re.compile(
    prefix+"Instance Field Constraint (?P<eid>[0-9a-f]+) (?P<contig>[0-1]) "
           "(?P<inorder>[0-1]) (?P<fields>[0-9]+)")
field_constraint_field_pat = re.compile(
    prefix+"Instance Field Constraint Field (?P<eid>[0-9a-f]+) (?P<fid>[0-9]+)")
ordering_constraint_pat = re.compile(
    prefix+"Instance Ordering Constraint (?P<eid>[0-9a-f]+) (?P<contig>[0-1]) "
           "(?P<dims>[0-9]+)")
ordering_constraint_dim_pat = re.compile(
    prefix+"Instance Ordering Constraint Dimension (?P<eid>[0-9a-f]+) (?P<dim>[0-9]+)")
splitting_constraint_pat = re.compile(
    prefix+"Instance Splitting Constraint (?P<eid>[0-9a-f]+) (?P<dim>[0-9]+) "
           "(?P<value>[0-9]+) (?P<chunks>[0-1])")
dimension_constraint_pat = re.compile(
    prefix+"Instance Dimension Constraint (?P<eid>[0-9a-f]+) (?P<dim>[0-9]+) "
           "(?P<eqk>[0-9]+) (?P<value>[0-9]+)")
alignment_constraint_pat = re.compile(
    prefix+"Instance Alignment Constraint (?P<eid>[0-9a-f]+) (?P<fid>[0-9]+) "
           "(?P<eqk>[0-9]+) (?P<align>[0-9]+)")
offset_constraint_pat = re.compile(
    prefix+"Instance Offset Constraint (?P<eid>[0-9a-f]+) (?P<fid>[0-9]+) "
           "(?P<offset>[0-9]+)")
variant_decision_pat    = re.compile(
    prefix+"Variant Decision (?P<uid>[0-9]+) (?P<vid>[0-9]+)")
mapping_decision_pat    = re.compile(
    prefix+"Mapping Decision (?P<uid>[0-9]+) (?P<idx>[0-9]+) (?P<fid>[0-9]+) "
           "(?P<eid>[0-9a-f]+)")
post_decision_pat       = re.compile(
    prefix+"Post Mapping Decision (?P<uid>[0-9]+) (?P<idx>[0-9]+) (?P<fid>[0-9]+) "
           "(?P<eid>[0-9a-f]+)")
task_priority_pat       = re.compile(
    prefix+"Task Priority (?P<uid>[0-9]+) (?P<priority>-?[0-9]+)") # Handle negatives
task_processor_pat      = re.compile(
    prefix+"Task Processor (?P<uid>[0-9]+) (?P<proc>[0-9a-f]+)")
task_premapping_pat     = re.compile(
    prefix+"Task Premapping (?P<uid>[0-9]+) (?P<index>[0-9]+)")
tunable_pat             = re.compile(
    prefix+"Task Tunable (?P<uid>[0-9]+) (?P<idx>[0-9]+) (?P<bytes>[0-9]+) "
           "(?P<value>[0-9a-f]+)")
# Physical event and operation patterns
event_dependence_pat     = re.compile(
    prefix+"Event Event (?P<id1>[0-9a-f]+) (?P<id2>[0-9a-f]+)")
ap_user_event_pat       = re.compile(
    prefix+"Ap User Event (?P<id>[0-9a-f]+)")
rt_user_event_pat       = re.compile(
    prefix+"Rt User Event (?P<id>[0-9a-f]+)")
pred_event_pat          = re.compile(
    prefix+"Pred Event (?P<id>[0-9a-f]+)")
ap_user_event_trig_pat  = re.compile(
    prefix+"Ap User Event Trigger (?P<id>[0-9a-f]+)")
rt_user_event_trig_pat  = re.compile(
    prefix+"Rt User Event Trigger (?P<id>[0-9a-f]+)")
pred_event_trig_pat     = re.compile(
    prefix+"Pred Event Trigger (?P<id>[0-9a-f]+)")
operation_event_pat     = re.compile(
    prefix+"Operation Events (?P<uid>[0-9]+) (?P<id1>[0-9a-f]+) (?P<id2>[0-9a-f]+)")
realm_copy_pat          = re.compile(
    prefix+"Copy Events (?P<uid>[0-9]+) (?P<ispace>[0-9]+) "+
           "(?P<src_tid>[0-9]+) (?P<dst_tid>[0-9]+) "+
           "(?P<preid>[0-9a-f]+) (?P<postid>[0-9a-f]+)")
realm_copy_field_pat    = re.compile(
    prefix+"Copy Field (?P<id>[0-9a-f]+) (?P<srcfid>[0-9]+) "+
           "(?P<srcid>[0-9a-f]+) (?P<dstfid>[0-9]+) (?P<dstid>[0-9a-f]+) (?P<redop>[0-9]+)")
indirect_copy_pat       = re.compile(
    prefix+"Indirect Events (?P<uid>[0-9]+) (?P<ispace>[0-9]+) (?P<indirect>[0-9]+) "+
           "(?P<preid>[0-9a-f]+) (?P<postid>[0-9a-f]+)")
indirect_field_pat      = re.compile(
    prefix+"Indirect Field (?P<id>[0-9a-f]+) (?P<srcfid>[0-9]+) (?P<srcid>[0-9a-f]+) "+
           "(?P<srcidx>-?[0-9]+) (?P<dstfid>[0-9]+) (?P<dstid>[0-9a-f]+) "+
           "(?P<dstidx>-?[0-9]+) (?P<redop>[0-9]+)")
indirect_inst_pat       = re.compile(
    prefix+"Indirect Instance (?P<indirect>[0-9]+) (?P<index>[0-9]+) "+
           "(?P<inst>[0-9a-f]+) (?P<fid>[0-9]+)")
indirect_group_pat      = re.compile(
    prefix+"Indirect Group (?P<indirect>[0-9]+) (?P<index>[0-9]+) "+
           "(?P<inst>[0-9a-f]+) (?P<ispace>[0-9]+)")
realm_fill_pat          = re.compile(
    prefix+"Fill Events (?P<uid>[0-9]+) (?P<ispace>[0-9]+) (?P<fspace>[0-9]+) "+
           "(?P<tid>[0-9]+) (?P<preid>[0-9a-f]+) (?P<postid>[0-9a-f]+) (?P<fill_uid>[0-9]+)")
realm_fill_field_pat    = re.compile(
    prefix+"Fill Field (?P<id>[0-9a-f]+) (?P<fid>[0-9]+) "+
           "(?P<dstid>[0-9a-f]+)")
realm_deppart_pat       = re.compile(
    prefix+"Deppart Events (?P<uid>[0-9]+) (?P<ispace>[0-9]+) "+
           "(?P<preid>[0-9a-f]+) (?P<postid>[0-9a-f]+)")
barrier_arrive_pat      = re.compile(
    prefix+"Phase Barrier Arrive (?P<uid>[0-9]+) (?P<iid>[0-9a-f]+)")
barrier_wait_pat        = re.compile(
    prefix+"Phase Barrier Wait (?P<uid>[0-9]+) (?P<iid>[0-9a-f]+)")
replay_op_pat           = re.compile(
    prefix+"Replay Operation (?P<uid>[0-9]+)")

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
        e1 = state.get_event(int(m.group('id1'),16))
        e2 = state.get_event(int(m.group('id2'),16))
        assert e2.exists()
        if e1.exists():
            e2.add_incoming(e1)
            e1.add_outgoing(e2)
        return True
    m = ap_user_event_pat.match(line)
    if m is not None:
        e = state.get_event(int(m.group('id'),16))
        e.set_ap_user_event()
        return True
    m = rt_user_event_pat.match(line)
    if m is not None:
        e = state.get_event(int(m.group('id'),16))
        e.set_rt_user_event()
        return True
    m = pred_event_pat.match(line)
    if m is not None:
        e = state.get_event(int(m.group('id'),16))
        e.set_pred_event()
        return True
    m = ap_user_event_trig_pat.match(line)
    if m is not None:
        e = state.get_event(int(m.group('id'),16))
        e.set_triggered()
        return True
    m = rt_user_event_trig_pat.match(line)
    if m is not None:
        e = state.get_event(int(m.group('id'),16))
        e.set_triggered()
        return True
    m = pred_event_trig_pat.match(line)
    if m is not None:
        e = state.get_event(int(m.group('id'),16))
        e.set_triggered()
        return True
    m = operation_event_pat.match(line)
    if m is not None:
        e1 = state.get_event(int(m.group('id1'),16))
        e2 = state.get_event(int(m.group('id2'),16))
        op = state.get_operation(int(m.group('uid')))
        op.set_events(e1, e2)
        return True
    m = realm_copy_pat.match(line)
    if m is not None:
        e1 = state.get_event(int(m.group('preid'),16))
        e2 = state.get_event(int(m.group('postid'),16))
        copy = state.get_realm_copy(e2)
        copy.set_start(e1)
        op = state.get_operation(int(m.group('uid')))
        copy.set_creator(op)
        index_expr = state.get_index_expr(int(m.group('ispace')))
        src_tree_id = int(m.group('src_tid'))
        dst_tree_id = int(m.group('dst_tid'))
        copy.set_tree_properties(index_expr, src_tree_id, dst_tree_id)
        return True
    m = realm_copy_field_pat.match(line)
    if m is not None:
        e = state.get_event(int(m.group('id'),16))
        copy = state.get_realm_copy(e)
        src = state.get_instance(int(m.group('srcid'),16))
        dst = state.get_instance(int(m.group('dstid'),16))
        copy.add_field(int(m.group('srcfid')), src, 
                       int(m.group('dstfid')), dst, int(m.group('redop')))
        return True
    m = indirect_copy_pat.match(line)
    if m is not None:
        e1 = state.get_event(int(m.group('preid'),16))
        e2 = state.get_event(int(m.group('postid'),16))
        copy = state.get_realm_copy(e2)
        copy.set_start(e1)
        op = state.get_operation(int(m.group('uid')))
        copy.set_creator(op)
        index_expr = state.get_index_expr(int(m.group('ispace')))
        indirections = state.get_indirections(int(m.group('indirect')))
        copy.set_indirection_properties(index_expr, indirections)
        return True
    m = indirect_field_pat.match(line)
    if m is not None:
        e = state.get_event(int(m.group('id'),16))
        copy = state.get_realm_copy(e)
        src_index = int(m.group('srcidx'))
        dst_index = int(m.group('dstidx'))
        src = None if src_index >= 0 else state.get_instance(int(m.group('srcid'),16))
        dst = None if dst_index >= 0 else state.get_instance(int(m.group('dstid'),16))
        copy.add_indirect_field(int(m.group('srcfid')), src, src_index, 
            int(m.group('dstfid')), dst, dst_index, int(m.group('redop')))
        return True
    m = indirect_inst_pat.match(line)
    if m is not None:
        indirections = state.get_indirections(int(m.group('indirect')))
        inst = state.get_instance(int(m.group('inst'),16))
        indirections.add_indirect_instance(int(m.group('index')), inst, int(m.group('fid')))
        return True
    m = indirect_group_pat.match(line)
    if m is not None:
        indirections = state.get_indirections(int(m.group('indirect')))
        inst = state.get_instance(int(m.group('inst'),16))
        index_space = state.get_index_space(int(m.group('ispace')))
        indirections.add_group_instance(int(m.group('index')), inst, index_space)
        return True
    m = realm_fill_pat.match(line)
    if m is not None:
        e1 = state.get_event(int(m.group('preid'),16))
        e2 = state.get_event(int(m.group('postid'),16))
        fill = state.get_realm_fill(e2)
        fill.set_start(e1)
        op = state.get_operation(int(m.group('uid')))
        fill.set_creator(op)
        index_expr = state.get_index_expr(int(m.group('ispace')))
        field_space = state.get_field_space(int(m.group('fspace')))
        tree_id = int(m.group('tid'))
        fill.set_tree_properties(index_expr, field_space, tree_id)
        fill_uid = int(m.group('fill_uid'))
        if fill_uid > 0:
            fill_op = state.get_operation(fill_uid)
            fill.set_fill_op(fill_op)
        return True
    m = realm_fill_field_pat.match(line)
    if m is not None:
        e = state.get_event(int(m.group('id'),16))
        fill = state.get_realm_fill(e)
        dst = state.get_instance(int(m.group('dstid'),16))
        fill.add_field(int(m.group('fid')), dst)
        return True
    m = realm_deppart_pat.match(line)
    if m is not None:
        e1 = state.get_event(int(m.group('preid'),16))
        e2 = state.get_event(int(m.group('postid'),16))
        deppart = state.get_realm_deppart(e2)
        deppart.set_start(e1)
        op = state.get_operation(int(m.group('uid')))
        deppart.set_creator(op)
        index_expr = state.get_index_expr(int(m.group('ispace')))
        deppart.set_index_expr(index_expr) 
        return True
    m = barrier_arrive_pat.match(line)
    if m is not None:
        e = state.get_event(int(m.group('iid'),16))
        op = state.get_operation(int(m.group('uid')))
        e.add_phase_barrier_contributor(op)
        op.add_arrival_barrier(e)
        return True
    m = barrier_wait_pat.match(line)
    if m is not None:
        e = state.get_event(int(m.group('iid'),16))
        op = state.get_operation(int(m.group('uid')))
        e.add_phase_barrier_waiter(op)
        op.add_wait_barrier(e)
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
    m = projection_func_pat.match(line)
    if m is not None:
        func = state.get_projection_function(int(m.group('pid')))
        func.set_depth(int(m.group('depth')))
        func.set_invertible(True if int(m.group('invertible')) != 0 else False)
        return True
    m = req_proj_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        index = int(m.group('index'))
        func = state.get_projection_function(int(m.group('pid')))
        op.set_projection_function(index, func)
        return True
    m = index_launch_domain_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        dim = int(m.group('dim'))
        lo = Point(dim)
        hi = Point(dim)
        # Get the remainder of the points
        values = decimal_pat.findall(m.group('rem'))
        for index in xrange(dim):
            lo.vals[index] = int(values[2*index])
            hi.vals[index] = int(values[2*index+1])
        op.add_launch_rect(Rect(lo, hi)) 
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
    m = future_create_pat.match(line)
    if m is not None:
        future = state.get_future(int(m.group('iid'),16))
        future.set_creator(int(m.group('uid')))
        dim = int(m.group('dim'))
        point = Point(dim)
        values = decimal_pat.findall(m.group('rem'))
        for index in xrange(dim):
            point.vals[index] = int(values[index])
        future.set_point(point)
        return True 
    m = future_use_pat.match(line)
    if m is not None:
        future = state.get_future(int(m.group('iid'),16))
        future.add_uid(int(m.group('uid')))
        return True
    m = predicate_use_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        pred = state.get_operation(int(m.group('pred')))
        op.set_predicate(pred)
        return True
    # Physical Instances and Mapping decisions happen frequently too
    m = instance_pat.match(line)
    if m is not None:
        mem = state.get_memory(int(m.group('mid'),16))
        inst = state.get_instance(int(m.group('eid'),16))
        inst.set_handle(int(m.group('iid'),16))
        inst.set_memory(mem)
        inst.set_redop(int(m.group('redop')))
        inst_expr = state.get_index_expr(int(m.group('expr')))
        space = state.get_field_space(int(m.group('space')))
        inst.set_properties(inst_expr, space, int(m.group('tid')))
        return True
    m = instance_field_pat.match(line)
    if m is not None:
        inst = state.get_instance(int(m.group('eid'),16))
        inst.add_field(int(m.group('fid')))
        return True
    m = instance_creator_pat.match(line)
    if m is not None:
        inst = state.get_instance(int(m.group('eid'),16))
        proc = state.get_processor(int(m.group('proc'),16))
        inst.set_creator(int(m.group('uid')), proc)
        return True
    m = instance_creator_region_pat.match(line)
    if m is not None:
        inst = state.get_instance(int(m.group('eid'),16))
        region = state.get_region(int(m.group('ispace')), 
            int(m.group('fspace')), int(m.group('tid')))
        inst.add_creator_region(region)
        return True
    m = specialized_constraint_pat.match(line)
    if m is not None:
        inst = state.get_instance(int(m.group('eid'),16))
        inst.set_specialized_constraint(int(m.group('kind')),
                                        int(m.group('redop')))
        return True
    m = memory_constraint_pat.match(line)
    if m is not None:
        inst = state.get_instance(int(m.group('eid'),16))
        inst.set_memory_constraint(int(m.group('kind')))
        return True
    m = field_constraint_pat.match(line)
    if m is not None:
        inst = state.get_instance(int(m.group('eid'),16))
        inst.set_field_constraint(int(m.group('contig')), 
            int(m.group('inorder')), int(m.group('fields')))
        return True
    m = field_constraint_field_pat.match(line)
    if m is not None:
        inst = state.get_instance(int(m.group('eid'),16))
        inst.add_field_constraint_field(int(m.group('fid')))
        return True
    m = ordering_constraint_pat.match(line)
    if m is not None:
        inst = state.get_instance(int(m.group('eid'),16))
        inst.set_ordering_constraint(int(m.group('contig')), 
                                     int(m.group('dims')))
        return True
    m = ordering_constraint_dim_pat.match(line)
    if m is not None:
        inst = state.get_instance(int(m.group('eid'),16))
        inst.add_ordering_constraint_dim(int(m.group('dim')))
        return True
    m = splitting_constraint_pat.match(line)
    if m is not None:
        inst = state.get_instance(int(m.group('eid'),16))
        inst.add_splitting_constraint(int(m.group('dim')),
            int(m.group('value')), int(m.group('chunks')))
        return True
    m = dimension_constraint_pat.match(line)
    if m is not None:
        inst = state.get_instance(int(m.group('eid'),16))
        inst.add_dimesion_constraint(int(m.group('dim')),
            int(m.group('eqk')), int(m.group('value')))
        return True
    m = alignment_constraint_pat.match(line)
    if m is not None:
        inst = state.get_instance(int(m.group('eid'),16))
        inst.add_alignment_constraint(int(m.group('fid')),
            int(m.group('eqk')), int(m.group('align')))
        return True
    m = offset_constraint_pat.match(line)
    if m is not None:
        inst = state.get_instance(int(m.group('eid'),16))
        inst.add_offset_constraint(int(m.group('fid')),
            int(m.group('offset')))
        return True
    m = variant_decision_pat.match(line)
    if m is not None:
        task = state.get_task(int(m.group('uid')))
        variant = state.get_variant(int(m.group('vid')))
        task.set_variant(variant)
        return True
    m = mapping_decision_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        inst = state.get_instance(int(m.group('eid'),16))
        op.add_mapping_decision(int(m.group('idx')),
            int(m.group('fid')), inst)
        return True
    m = post_decision_pat.match(line)
    if m is not None:
        task = state.get_task(int(m.group('uid')))
        inst = state.get_instance(int(m.group('eid'),16))
        task.add_postmapping(int(m.group('idx')),
            int(m.group('fid')), inst)
        return True
    m = task_priority_pat.match(line)
    if m is not None:
        task = state.get_task(int(m.group('uid')))
        task.set_priority(int(m.group('priority')))
        return True
    m = task_processor_pat.match(line)
    if m is not None:
        task = state.get_task(int(m.group('uid')))
        proc = state.get_processor(int(m.group('proc'),16))
        task.set_processor(proc)
        return True
    m = task_premapping_pat.match(line)
    if m is not None:
        task = state.get_task(int(m.group('uid')))
        task.add_premapping(int(m.group('index')))
        return True
    m = tunable_pat.match(line)
    if m is not None:
        task = state.get_task(int(m.group('uid')))
        task.add_tunable(int(m.group('idx')), int(m.group('bytes')), m.group('value'))
        return True
    # Operations near the top since they happen frequently
    m = task_name_pat.match(line)
    if m is not None:
        state.task_names[int(m.group('tid'))] = m.group('name')
        return True
    m = task_variant_pat.match(line)
    if m is not None:
        variant = state.get_variant(int(m.group('vid')))
        variant.initialize(int(m.group('inner')), int(m.group('leaf')),
                           int(m.group('idem')), m.group('name'))
        return True
    m = top_task_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(SINGLE_TASK_KIND)
        op.set_name(m.group('name'))
        op.set_task_id(int(m.group('tid')))
        # Save the top-level uid
        state.top_level_uid = int(m.group('uid'))
        state.top_level_ctx_uid = int(m.group('ctxuid'))
        return True
    m = single_task_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(SINGLE_TASK_KIND)
        op.set_name(m.group('name'))
        op.set_task_id(int(m.group('tid')))
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context, int(m.group('index')))
        return True
    m = index_task_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(INDEX_TASK_KIND)
        op.set_name(m.group('name'))
        op.set_task_id(int(m.group('tid')))
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context, int(m.group('index')))
        return True
    m = inline_task_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.inlined = True
        return True
    m = mapping_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(MAP_OP_KIND)
        op.set_name("Mapping Op")
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context, int(m.group('index')))
        return True
    m = close_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        inter = True if int(m.group('is_inter')) == 1 else False
        if inter:
            op.set_op_kind(INTER_CLOSE_OP_KIND)
            op.set_name("Inter Close Op")
        else:
            op.set_op_kind(POST_CLOSE_OP_KIND)
            op.set_name("Post Close Op")
        
        context = state.get_task(int(m.group('ctx')))
        # Only add this to the context if it not an intermediate
        # close operation, otherwise add it to the context like normal
        # because it as an actual operation
        op.set_context(context, None if inter else int(m.group('index')))
        return True
    m = internal_creator_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        creator = state.get_operation(int(m.group('cuid')))
        op.set_creator(creator, int(m.group('idx')))
        return True
    m = fence_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(FENCE_OP_KIND)
        op.set_name("Fence Op")
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context, int(m.group('index')))
        return True
    m = trace_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_name("Trace Op")
        return True
    m = copy_op_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(COPY_OP_KIND)
        op.set_name("Copy Op")
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context, int(m.group('index')))
        op.copy_kind = int(m.group('kind'))
        collective_src = int(m.group('src'))
        op.collective_src = True if collective_src == 1 and \
                (op.copy_kind == 1 or op.copy_kind == 3) else False
        collective_dst = int(m.group('dst'))
        op.collective_dst = True if collective_dst == 1 and \
                (op.copy_kind == 2 or op.copy_kind == 3) else False
        return True
    m = fill_op_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(FILL_OP_KIND)
        op.set_name("Fill Op")
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context, int(m.group('index')))
        return True
    m = acquire_op_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(ACQUIRE_OP_KIND)
        op.set_name("Acquire Op")
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context, int(m.group('index')))
        return True
    m = release_op_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(RELEASE_OP_KIND)
        op.set_name("Release Op")
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context, int(m.group('index')))
        return True
    m = creation_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(CREATION_OP_KIND)
        op.set_name("Creation Op")
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context, int(m.group('index')))
        return True
    m = deletion_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(DELETION_OP_KIND)
        op.set_name("Deletion Op")
        if int(m.group('unordered')) == 0:
            context = state.get_task(int(m.group('ctx')))
            op.set_context(context, int(m.group('index')))
        return True
    m = attach_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(ATTACH_OP_KIND)
        op.set_name("Attach Op")
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context, int(m.group('index')))
        op.restricted = True if int(m.group('restriction')) == 1 else False
        return True
    m = detach_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(DETACH_OP_KIND)
        op.set_name("Detach Op")
        if int(m.group('unordered')) == 0:
            context = state.get_task(int(m.group('ctx')))
            op.set_context(context, int(m.group('index')))
        return True
    m = unordered_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context, int(m.group('index')))
        return True
    m = dynamic_collective_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(DYNAMIC_COLLECTIVE_OP_KIND)
        op.set_name("Dynamic Collective Op")
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context, int(m.group('index')))
        return True
    m = timing_op_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(TIMING_OP_KIND)
        op.set_name("Timing Op")
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context, int(m.group('index')))
        return True
    m = tunable_op_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(TUNABLE_OP_KIND)
        op.set_name("Tunable Op")
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context, int(m.group('index')))
        return True
    m = all_reduce_op_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(ALL_REDUCE_OP_KIND)
        op.set_name("Reduce Op")
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context, int(m.group('index')))
        return True
    m = predicate_op_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(PREDICATE_OP_KIND)
        op.set_name("Predicate Op")
        # Predicate ops are not recorded in the context for now
        # because they have to outlive when they complete
        return True
    m = must_epoch_op_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(MUST_EPOCH_OP_KIND)
        # Don't add it to the context for now
        return True
    m = summary_op_creator_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        creator = state.get_operation(int(m.group('cuid')))
        creator.set_summary_operation(op)
        op.set_summary_operation(op)
        return True
    m = dep_partition_op_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(DEP_PART_OP_KIND)
        op.set_name("Dependent Partition Op")
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context, int(m.group('index')))
        return True
    m = pending_partition_op_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_op_kind(PENDING_PART_OP_KIND)
        op.set_name("Pending Partition Op")
        context = state.get_task(int(m.group('ctx')))
        op.set_context(context, int(m.group('index')))
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
        values = decimal_pat.findall(m.group('rem'))
        for index in xrange(dim):
            index_point.vals[index] = int(values[index])
        point.set_point(index_point)
        state.point_slice[point] = int(m.group('slice'))
        return True
    m = point_point_pat.match(line)
    if m is not None:
        p1 = state.get_task(int(m.group('point1')))
        p2 = state.get_task(int(m.group('point2')))
        assert p1 not in state.point_point
        assert p2 not in state.point_point
        # Holdoff on doing the merge until after parsing
        state.point_point[p1] = p2
        return True
    m = index_point_pat.match(line)
    if m is not None:
        point = state.get_operation(int(m.group('point')))
        dim = int(m.group('dim'))
        index_point = Point(dim)
        values = decimal_pat.findall(m.group('rem'))
        for index in xrange(dim):
            index_point.vals[index] = int(values[index])
        index = state.get_operation(int(m.group('index')))
        index.add_point_op(point, index_point) 
        return True
    m = intra_space_pat.match(line)
    if m is not None:
        point = state.get_operation(int(m.group('point')))
        dim = int(m.group('dim'))
        index_point = Point(dim)
        values = decimal_pat.findall(m.group('rem'))
        for index in xrange(dim):
            index_point.vals[index] = int(values[index])
        point.add_intra_space_dependence(index_point)
        return True
    m = op_index_pat.match(line)
    if m is not None:
        task = state.get_task(int(m.group('parent')))
        task.add_operation_index(int(m.group('index')),
                                 int(m.group('child')))
        return True
    m = op_provenance_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.provenance = m.group('provenance')
        return True
    m = close_index_pat.match(line)
    if m is not None:
        task = state.get_task(int(m.group('parent')))
        task.add_close_index(int(m.group('index')),
                             int(m.group('child')))
        return True
    m = predicate_false_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_predicate_result(False)
        return True
    # Region tree shape patterns (near the bottom since they are infrequent)
    m = top_index_pat.match(line)
    if m is not None:
        space = state.get_index_space(int(m.group('uid'),16)) 
        space.owner = int(m.group('owner'))
        space.provenance = m.group('provenance')
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
        color= Point(1)
        color.vals[0] = int(m.group('color'))
        part.set_parent(parent, color)
        disjoint = int(m.group('disjoint'))
        if disjoint > 0:
            part.set_disjoint(True if disjoint == 2 else False)
        complete = int(m.group('complete'))
        if complete > 0:
            part.set_complete(True if complete == 2 else False)
        part.owner = int(m.group('owner'))
        part.provenance = m.group('provenance')
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
        values = decimal_pat.findall(m.group('rem'))
        for index in xrange(dim):
            color.vals[index] = int(values[index])
        ispace.set_parent(parent, color)
        ispace.owner = int(m.group('owner'))
        return True
    m = field_space_pat.match(line)
    if m is not None:
        space = state.get_field_space(int(m.group('uid')))
        space.owner = int(m.group('owner'))
        space.provenance = m.group('provenance')
        return True
    m = field_space_name_pat.match(line)
    if m is not None:
        space = state.get_field_space(int(m.group('uid')))
        space.set_name(m.group('name'))
        return True
    m = field_create_pat.match(line)
    if m is not None:
        space = state.get_field_space(int(m.group('uid')))
        field = space.get_field(int(m.group('fid')))
        field.size = int(m.group('size'))
        field.provenance = m.group('provenance')
        return True
    m = field_name_pat.match(line)
    if m is not None:
        space = state.get_field_space(int(m.group('uid')))
        field = space.get_field(int(m.group('fid')))
        field.set_name(m.group('name'))
        return True
    m = region_pat.match(line)
    if m is not None:
        region = state.get_region(int(m.group('iid'),16),
            int(m.group('fid')),int(m.group('tid')))
        region.owner = int(m.group('owner'))
        region.provenance = m.group('provenance')
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
        values = decimal_pat.findall(m.group('rem'))
        for index in xrange(dim):
            point.vals[index] = int(values[index])
        index_space.add_point(point)
        return True
    m = index_space_rect_pat.match(line)
    if m is not None:
        index_space = state.get_index_space(int(m.group('uid'),16))
        dim = int(m.group('dim'))
        lo = Point(dim)
        hi = Point(dim)
        values = decimal_pat.findall(m.group('rem'))
        for index in xrange(dim):
            lo.vals[index] = int(values[2*index])
            hi.vals[index] = int(values[2*index+1])
        if lo == hi:
            index_space.add_point(lo)
        else:
            index_space.add_rect(Rect(lo, hi))
        return True
    m = empty_index_space_pat.match(line)
    if m is not None:
        index_space = state.get_index_space(int(m.group('uid'),16))
        index_space.set_empty()
        return True
    m = index_expr_pat.match(line)
    if m is not None:
        index_space = state.get_index_space(int(m.group('uid'),16))
        expr_id = int(m.group('expr'))
        index_space.set_expr(expr_id)
        index_expr = state.get_index_expr(expr_id)
        index_expr.set_index_space(index_space)
        return True
    m = union_expr_pat.match(line)
    if m is not None:
        index_expr = state.get_index_expr(int(m.group('expr')))
        remainder = line[m.end()+1:]
        expr_ids = list(filter(None, re.split('\W+', remainder)))
        count = int(m.group('count'))
        # Don't handle end-of-line characters well
        assert len(expr_ids) == count
        for expr_id in expr_ids[:count]:
            sub_expr = state.get_index_expr(int(expr_id))
            index_expr.add_union_expr(sub_expr)
        return True
    m = intersect_expr_pat.match(line)
    if m is not None:
        index_expr = state.get_index_expr(int(m.group('expr')))
        remainder = line[m.end()+1:]
        expr_ids = list(filter(None, re.split('\W+', remainder)))
        count = int(m.group('count'))
        # Don't handle end-of-line characters well
        assert len(expr_ids) == count
        for expr_id in expr_ids[:count]:
            sub_expr = state.get_index_expr(int(expr_id))
            index_expr.add_intersect_expr(sub_expr)
        return True
    m = diff_expr_pat.match(line)
    if m is not None:
        index_expr = state.get_index_expr(int(m.group('expr')))
        left_expr = state.get_index_expr(int(m.group('left')))
        right_expr = state.get_index_expr(int(m.group('right')))
        index_expr.set_diff_expr(left_expr, right_expr)
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
        kind_num = int(m.group('kind'))
        proc.set_kind(kind_num, state.get_processor_kind(kind_num))
        return True
    m = memory_pat.match(line)
    if m is not None:
        mem = state.get_memory(int(m.group('mid'),16))
        kind_num = int(m.group('kind'))
        mem.set_kind(kind_num, state.get_memory_kind(kind_num))
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
    m = config_pat.match(line)
    if m is not None:
        state.set_config(False)
        return True
    m = detailed_config_pat.match(line)
    if m is not None:
        state.set_config(True)
        return True
    m = replay_op_pat.match(line)
    if m is not None:
        op = state.get_operation(int(m.group('uid')))
        op.set_replayed()
        return True
    return False

class State(object):
    __slots__ = ['temp_dir', 'verbose', 'top_level_uid', 'top_level_ctx_uid', 
                 'traverser_gen', 'processors', 'memories',
                 'processor_kinds', 'memory_kinds', 'index_exprs', 'index_spaces', 
                 'index_partitions', 'field_spaces', 'regions', 'partitions', 'top_spaces',
                 'trees', 'ops', 'unique_ops', 'tasks', 'task_names', 'variants', 
                 'projection_functions', 'has_mapping_deps', 'instances', 'events', 
                 'copies', 'fills', 'depparts', 'indirections', 'no_event', 'slice_index', 
                 'slice_slice', 'point_slice', 'point_point', 'futures', 'next_generation', 
                 'next_realm_num', 'next_indirections_num', 'detailed_graphs', 
                 'assert_on_error', 'assert_on_warning', 'bad_graph_on_error', 
                 'eq_graph_on_error', 'config', 'detailed_logging']
    def __init__(self, temp_dir, verbose, details, assert_on_error, 
                 assert_on_warning, bad_graph_on_error, eq_graph_on_error):
        self.temp_dir = temp_dir
        self.config = False
        self.detailed_logging = True
        self.verbose = verbose
        self.detailed_graphs = details
        self.assert_on_error = assert_on_error
        self.assert_on_warning = assert_on_warning
        self.bad_graph_on_error = bad_graph_on_error
        self.eq_graph_on_error = eq_graph_on_error
        self.top_level_uid = None
        self.top_level_ctx_uid = None
        self.traverser_gen = 1
        # Machine things
        self.processors = dict()
        self.memories = dict()
        self.processor_kinds = dict()
        self.memory_kinds = dict()
        # Region tree things
        self.index_exprs = dict()
        self.index_spaces = dict()
        self.index_partitions = dict()
        self.field_spaces = dict()
        self.regions = dict()
        self.partitions = dict()
        self.top_spaces = dict()
        self.trees = dict()
        # Logical things 
        self.ops = dict() # Operations can be duplicated from different nodes
        self.unique_ops = None # Filled in by post_parse to deduplicate
        self.tasks = dict()
        self.task_names = dict()
        self.variants = dict()
        self.projection_functions = dict()
        self.has_mapping_deps = False
        # Physical things 
        self.instances = dict()
        self.events = dict()
        self.copies = dict()
        self.fills = dict()
        self.depparts = dict()
        self.no_event = Event(self, EventHandle(0))
        self.indirections = dict()
        # For parsing only
        self.slice_index = dict()
        self.slice_slice = dict()
        self.point_slice = dict()
        self.point_point = dict()
        self.futures = dict()
        # For physical traversals
        self.next_generation = 1
        self.next_realm_num = 1
        self.next_indirections_num = 1

    def set_config(self, detailed):
        self.config = True
        self.detailed_logging = detailed

    def get_next_traversal_generation(self):
        result = self.next_generation
        self.next_generation += 1
        return result

    def parse_log_file(self, file_name):
        print('Reading log file %s...' % file_name)
        try:
            log = open(file_name, 'r')
        except:
            print('ERROR: Unable to find file '+file_name)
            print('Legion Spy will now exit')
            sys.exit(1)
        else:
            with log:
                matches = 0
                skipped = 0
                for line in log:
                    if parse_legion_spy_line(line, self):
                        matches += 1
                    else:
                        skipped += 1
                        # always print a skipped line if it looks like it should have been matched
                        if self.verbose or (prefix_pat.match(line) is not None):
                            print('Skipping line: ' + line.strip())
        if matches == 0:
            print('WARNING: file %s contained no valid lines!' % file_name)
            #if self.assert_on_warning:
            #    assert False
        if self.verbose:
            print('Matched %d lines in %s' % (matches,file_name))
        if skipped > 0:
            print('WARNING: Skipped %d lines when reading %s' % (skipped,file_name))
        return matches

    def post_parse(self, simplify_graphs, need_physical):
        for space in itervalues(self.index_spaces):
            if space.parent is None:
                space.update_depth(-1)
        print('Reducing top-level index space shapes...')
        # Have to do the same sets across all index spaces
        # with the same dimensions in case of copy across
        dim_sets = dict()
        done = 0
        total_spaces = len(self.index_spaces)
        for space in itervalues(self.index_spaces):
            if space.parent is None:
                done = space.compute_reduced_shapes(dim_sets, done, total_spaces)
        # Finish the progress bar if it wasn't already, happens when we have
        # duplicate sub-trees get recorded
        if done < total_spaces and not self.verbose:
            print_progress_bar(total_spaces, total_spaces, length=50)
        print('Done')
        print('Computing refinement points...')
        for dim,index_sets in iteritems(dim_sets):
            point_value = 0
            total_sets = len(index_sets)
            for shape,index_set in iteritems(index_sets):
                point = Point(1)
                point.vals[0] = point_value
                point.shape = shape
                point.index_set = index_set
                point_value += 1
                print_progress_bar(point_value, total_sets, prefix='Dim '+str(dim)+':',length=50)
                for index in index_set:
                    index.add_refined_point(point)
        print('Done')
        # Find the top-level index spaces
        num_index_trees = 0
        for space in itervalues(self.index_spaces):
            if space.parent is None:
                self.top_spaces[num_index_trees] = space
                num_index_trees += 1
                # Check for the dominance property
                space.check_partition_properties()
        # Fill in the parents for all the regions and partitions
        # Make a copy in case things change size
        for region in itervalues(self.regions.copy()):
            region.update_parent()
        for partition in itervalues(self.partitions.copy()):
            partition.update_parent()
        # Find the top-level regions
        for region in itervalues(self.regions):
            if region.parent is None:
                self.trees[region.tree_id] = region
        # Merge the individual tasks
        for p1,p2 in iteritems(self.point_point):
            self.alias_individual_points(p1, p2)
        # Merge the points and slices
        for point,slice_ in iteritems(self.point_slice):
            while slice_ in self.slice_slice:
                slice_ = self.slice_slice[slice_]
            assert slice_ in self.slice_index
            self.slice_index[slice_].add_point_task(point)
        # Flatten summary operations in each context
        for task in itervalues(self.tasks):
            task.flatten_summary_operations()
        # Create the unique set of operations
        self.unique_ops = set(itervalues(self.ops))
        # Add implicit dependencies between point and index operations
        if self.detailed_logging:
            index_owners = set()
            for op in self.unique_ops:
                if op.index_owner:
                    # Skip close operations for transitive dependences
                    if op.kind == INTER_CLOSE_OP_KIND:
                        continue
                    index_owners.add(op.index_owner)
                    point_termination = op.finish_event
                    index_termination = op.index_owner.finish_event
                    if point_termination.exists() and index_termination.exists():
                        index_termination.add_incoming(point_termination)
                        point_termination.add_outgoing(index_termination)
            # Remove index operations from the event graph
            for op in index_owners:
                if op.finish_event.incoming_ops:
                    op.finish_event.incoming_ops.remove(op)
        # Check for any interfering index space launches
        for op in self.unique_ops:
            if op.is_interfering_index_space_launch():
                print("ERROR: Found interfering index space launch: %s!" % str(op))
                if self.assert_on_error:
                    assert False
        # Fill in any task names
        for task in itervalues(self.tasks):
            if task.op is not None and task.op.task_id in self.task_names:
                task.op.set_name(self.task_names[task.op.task_id])
        # Assign the depth of the top context
        op = self.get_operation(self.top_level_uid)
        assert op.context is not None
        op.context.depth = 0
        # Check to see if we have any unknown operations
        for op in self.unique_ops:
            if op.kind is NO_OP_KIND:
                # Ignore provenances for slice tasks we might have recorded
                if op.uid in self.slice_index or op.uid in self.slice_slice:
                    assert op.provenance is not None
                    continue
                print('WARNING: operation %d has unknown operation kind!' % op.uid)
                if self.assert_on_warning:
                    assert False
        # Update the instance 
        for inst in itervalues(self.instances):
            inst.update_creator()
        # Update the instance users
        for op in self.unique_ops:
            op.update_instance_uses() 
        for task in itervalues(self.tasks):
            task.update_instance_uses()
        # Update the futures
        for future in itervalues(self.futures):
            future.update_creator_and_users()     
        # We can delete some of these data structures now that we
        # no longer need them, go go garbage collection
        self.slice_index = None
        self.slice_slice = None
        self.point_slice = None
        if self.detailed_logging:
            print('Computing physical reachable...')
            total_nodes = len(self.unique_ops) + len(self.copies) + \
                            len(self.fills) + len(self.depparts)
            count = 0
            # Compute the physical reachable
            for op in self.unique_ops:
                op.compute_physical_reachable()
                if not self.verbose:
                    count += 1
                    print_progress_bar(count, total_nodes, length=50)
            for copy in itervalues(self.copies):
                copy.compute_physical_reachable()
                if not self.verbose:
                    count += 1
                    print_progress_bar(count, total_nodes, length=50)
            for fill in itervalues(self.fills):
                fill.compute_physical_reachable()
                if not self.verbose:
                    count += 1
                    print_progress_bar(count, total_nodes, length=50)
            for deppart in itervalues(self.depparts):
                deppart.compute_physical_reachable()
                if not self.verbose:
                    count += 1
                    print_progress_bar(count, total_nodes, length=50)
            print('Done')
        if self.verbose:
            print("Found %d processors" % len(self.processors))
            print("Found %d memories" % len(self.memories))
            print("")
            print("Found %d index space trees" % num_index_trees)
            print("Found %d field spaces" % len(self.field_spaces))
            print("Found %d region trees" % len(self.trees))
            print("")
            print("Found %d tasks" % len(self.tasks))
            print("Found %d operations (including tasks)" % len(self.ops))
            print("")
            print("Found %d instances" % len(self.instances))
            print("Found %d events" % len(self.events))
            print("Found %d copies" % len(self.copies))
            print("Found %d fills" % len(self.fills))

    def simplify_physical_graph(self, need_cycle_check=True, 
                                make_equivalence_graphs=False):
        # Check for cycles first, if there are any, then we disable
        # the transitive reduction and print a warning
        if need_cycle_check:
            print("Checking for cycles...")
            if self.perform_cycle_checks(print_result=True):
                print("WARNING: CYCLE DETECTED IN PHYSICAL EVENT GRAPH!!!")
                print("  This usually indicates a runtime bug and should be reported.")
                print("WARNING: DISABLING TRANSITIVE REDUCTION!!!")
                return
        print("Simplifying event graph...")
        def post_traverse_node(node, traverser):
            traverser.postorder.append(node)
        # Build a topological order of everything 
        topological_sorter = PhysicalTraverser(True, True,
            self.get_next_traversal_generation(), None, post_traverse_node)
        topological_sorter.postorder = list()
        # Traverse all the sources 
        for op in self.unique_ops:
            if not op.physical_incoming:
                topological_sorter.run(op)
        for copy in itervalues(self.copies):
            if not copy.physical_incoming:
                topological_sorter.run(copy)
        for fill in itervalues(self.fills):
            if not fill.physical_incoming:
                topological_sorter.run(fill)
        for deppart in itervalues(self.depparts):
            if not deppart.physical_incoming:
                topological_sorter.run(deppart)
        # Now that we have everything sorted based on topology
        # Do the simplification in postorder so we simplify
        # the smallest subgraphs first and only do the largest
        # subgraphs later after the smallest ones are already simplified
        count = 0
        # Mapping from node to index
        index_map = dict()
        # Maintain a map of reachable sets for each of the nodes that
        # we have traversed so far, we represent these as bit arrays
        # so that they are easy to combine without using a full python
        # set data structure
        reachable = dict()
        total_nodes = len(topological_sorter.postorder)
        for src in topological_sorter.postorder:
            if self.verbose:
                print('Simplifying node %s %d of %d' % (str(src), count, total_nodes)) 
            index_map[src] = count
            count += 1
            if not self.verbose:
                print_progress_bar(count, total_nodes, length=50)
            # Create our reachability set and store it
            our_reachable = NodeSet(total_nodes)
            reachable[src] = our_reachable
            if src.physical_outgoing is None or len(src.physical_outgoing) == 0:
                continue
            # Otherwise iterate through our outgoing edges and get the set of 
            # nodes reachable from all of them
            for dst in src.physical_outgoing:
                assert dst in reachable
                our_reachable.union(reachable[dst])
            # Now see which of our nodes can be reached indirectly
            to_remove = None
            for dst in src.physical_outgoing:
                assert dst in index_map
                dst_index = index_map[dst]
                if our_reachable.contains(dst_index):
                    if to_remove is None:
                        to_remove = list()
                    to_remove.append(dst)
                else:
                    # We need to add it to our reachable set
                    our_reachable.add(dst_index)
            if to_remove:
                for dst in to_remove:
                    src.physical_outgoing.remove(dst)
                    dst.physical_incoming.remove(src)
            # We should never remove everything
            assert len(src.physical_outgoing) > 0
        print("Done")
        if make_equivalence_graphs:
            self.compute_equivalence_graphs(topological_sorter.postorder, False)

    def compute_equivalence_graphs(self, postorder=None, need_cycle_check=True):
        if need_cycle_check:
            print("Checking for cycles...")
            if self.perform_cycle_checks(print_result=True):
                print("WARNING: CYCLE DETECTED IN PHYSICAL EVENT GRAPH!!!")
                print("  This usually indicates a runtime bug and should be reported.")
                print("WARNING: DISABLING TRANSITIVE REDUCTION!!!")
                return
        print('Computing equivalence set graphs for verification...')
        # The first thing we're going to do is BFS-like traversal of the
        # graph to propagate reachability values for each equivalence set
        # The main difference to BFS is that we can only traverse a node
        # after we have traversed all the edges leading to that node so 
        # we know we have a complete summary of the propagation of reachability
        # information flowing into the node
        incoming_sets = dict()
        ready_nodes = collections.deque()
        pending_nodes = dict()
        # This is the function for processing a node during the traversal
        def process_node(node):
            if self.verbose:
                print('Processing node '+str(node))
            if node in incoming_sets:
                local_sets = incoming_sets[node]
                del incoming_sets[node]
                for eq in iterkeys(node.get_equivalence_privileges()):
                    if eq in local_sets:
                        # Add the incoming equivalence edges and update
                        eq_incoming = local_sets[eq]
                        assert len(eq_incoming) > 0
                        for src in eq_incoming:
                            node.add_equivalence_incoming(eq, src)
                            src.add_equivalence_outgoing(eq, node)
                        # Who knows why list doesn't have a clear method
                        del eq_incoming[:]
                        eq_incoming.append(node)
                    else:
                        # Update with ourself
                        eq_incoming = list()
                        eq_incoming.append(node)
                        local_sets[eq] = eq_incoming
            elif node.physical_outgoing:
                local_sets = dict()
                for eq in node.get_equivalence_privileges():
                    # Update with ourself
                    eq_incoming = list()
                    eq_incoming.append(node)
                    local_sets[eq] = eq_incoming
            if not node.physical_outgoing:
                return
            for dst in node.physical_outgoing:
                if not dst in incoming_sets:
                    dst_sets = dict()
                    incoming_sets[dst] = dst_sets
                else:
                    dst_sets = incoming_sets[dst]
                for eq,eq_outgoing in iteritems(local_sets):
                    assert len(eq_outgoing) > 0
                    if eq in dst_sets:
                        eq_target = dst_sets[eq]
                        # Merge the two lists together
                        for src in eq_outgoing:
                            if src not in eq_target:
                                eq_target.append(src)
                    else:
                        # Just assign, make sure that we make a copy of it
                        # to prevent aliasing between destination nodes
                        # if there is more than one of them
                        if len(local_sets) > 1 or len(node.physical_outgoing) > 1:
                            dst_sets[eq] = list(eq_outgoing)
                        else: # Only one outgoing so can just move it over
                            dst_sets[eq] = eq_outgoing
                # See if this destination node is ready or not
                if len(dst.physical_incoming) > 1:
                    if dst in pending_nodes:
                        # Increment the number of received incoming edges
                        pending_nodes[dst] += 1
                        # Check to see if it is ready or not yet
                        if pending_nodes[dst] == len(dst.physical_incoming):
                            del pending_nodes[dst]
                            ready_nodes.append(dst)
                    else:
                        # We've seen one edge so far
                        pending_nodes[dst] = 1
                else:
                    # Only one incoming edge so we know it's ready
                    assert len(dst.physical_incoming) == 1
                    ready_nodes.append(dst)
        # Seed the incoming sets with the roots
        count = 0
        total_nodes = len(self.unique_ops) + len(self.copies) + \
                        len(self.fills) + len(self.depparts)
        for op in self.unique_ops:
            if not op.physical_incoming:
                process_node(op)
                if not self.verbose:
                    count += 1
                    print_progress_bar(count, total_nodes, length=50)
        for copy in itervalues(self.copies):
            if not copy.physical_incoming:
                process_node(copy)
                if not self.verbose:
                    count += 1
                    print_progress_bar(count, total_nodes, length=50)
        for fill in itervalues(self.fills):
            if not fill.physical_incoming:
                process_node(fill)
                if not self.verbose:
                    count += 1
                    print_progress_bar(count, total_nodes, length=50)
        for deppart in itervalues(self.depparts):
            if not deppart.physical_incoming:
                process_node(deppart)
                if not self.verbose:
                    count += 1
                    print_progress_bar(count, total_nodes, length=50)
        # Iterate until we've walked the whole graph O(V + E)
        while ready_nodes:
            node = ready_nodes.popleft()
            process_node(node)
            if not self.verbose:
                count += 1
                print_progress_bar(count, total_nodes, length=50)
        # The pending nodes should be empty by the time we are done with this
        print('Done')
        print('Simplifying equivalence event graphs...')
        # Now we need to transitively reduce each of the equivalence set graphs
        # Do this in postorder for the same reason we transitively reduce
        # the physical graph in post order
        if postorder is None:
            def post_traverse_node(node, traverser):
                traverser.postorder.append(node)
            # Build a topological order of everything 
            topological_sorter = PhysicalTraverser(True, True,
                self.get_next_traversal_generation(), None, post_traverse_node)
            topological_sorter.postorder = list()
            # Traverse all the sources 
            for op in self.unique_ops:
                if not op.physical_incoming:
                    topological_sorter.run(op)
            for copy in itervalues(self.copies):
                if not copy.physical_incoming:
                    topological_sorter.run(copy)
            for fill in itervalues(self.fills):
                if not fill.physical_incoming:
                    topological_sorter.run(fill)
            for deppart in itervalues(self.depparts):
                if not deppart.physical_incoming:
                    topological_sorter.run(deppart)
            postorder = topological_sorter.postorder
        # Transitively reduce the equivalence set graphs 
        # We do this in one pass to avoid try and maximize efficiency
        # and locality at the cost of a little bit of memory pressure
        count = 0
        index_map = dict()
        # This will actually be a dict of dicts with one reachable set
        # for each of the different equivalence sets
        reachable = dict()
        total_nodes = len(postorder)
        for src in postorder:
            if self.verbose:
                print('Simplifying equivalence node %s %d of %d' % 
                        (str(src), count, total_nodes)) 
            index_map[src] = count
            count += 1
            if not self.verbose:
                print_progress_bar(count, total_nodes, length=50)
            # Create our reachability dict
            our_reachable = dict()
            reachable[src] = our_reachable
            # if we have no outgoing equivalence edges then there is nothing to do
            if src.eq_outgoing is None or len(src.eq_outgoing) == 0:
                # Still need to populate it
                for eq in iterkeys(src.get_equivalence_privileges()):
                    our_reachable[eq] = NodeSet(total_nodes)
                continue
            # Iterate over all the equivalence classes for this node
            for eq in iterkeys(src.get_equivalence_privileges()):
                # No outgoing equivalence classes for this node so nothing to do
                eq_reachable = NodeSet(total_nodes) 
                our_reachable[eq] = eq_reachable
                if eq not in src.eq_outgoing:
                    continue
                # Iterate through our outgoing edges and get the set of nodes
                # reachable from all of them
                outgoing_edges = src.eq_outgoing[eq]
                for dst in outgoing_edges:
                    assert dst in reachable
                    assert eq in reachable[dst]
                    eq_reachable.union(reachable[dst][eq])
                # Now see which of our nodes can be reached directly
                to_remove = None
                for dst in outgoing_edges:
                    assert dst in index_map
                    dst_index = index_map[dst]
                    if eq_reachable.contains(dst_index):
                        if to_remove is None:
                            to_remove = list()
                        to_remove.append(dst)
                    else:
                        # We need to add it to our reachable set
                        eq_reachable.add(dst_index)
                if to_remove:
                    for dst in to_remove:
                        src.eq_outgoing[eq].remove(dst)
                        dst.eq_incoming[eq].remove(src)
                # We should never remove everything
                assert len(src.eq_outgoing[eq]) > 0
        print('Done')
            
    def alias_individual_points(self, p1, p2):
        # These are two copies of the same individual
        # task from different nodes, merge them into
        # the one from the original node (e.g. the
        # one with the context)
        if p1.op.context:
            assert not p2.op.context
            p1.op.merge(p2.op)
            self.ops[p2.op.uid] = p1.op
            p1.merge(p2)
            del self.tasks[p2.op]
            return p1
        else:
            assert p2.op.context
            p2.op.merge(p1.op)
            self.ops[p1.op.uid] = p2.op
            p2.merge(p1)
            del self.tasks[p1.op]
            return p2

    def alias_index_points(self, p1, p2):
        # These two copies of the same point task from an
        # index space launch so merge them together 
        # Always merge into the one that actually ran as that
        # is the one that appears in the Realm event graph
        # We know it is the one that ran because it will 
        # have a processor
        if p1.processor:
            assert not p2.processor
            # Merge the operations first 
            p1.op.merge(p2.op)
            self.ops[p2.op.uid] = p1.op
            # Now merge the tasks and delete the other task
            p1.merge(p2)
            del self.tasks[p2.op]
            return p1
        else:
            # TODO: what happens when points bounce around
            # multiple times under stealing
            assert p2.processor
            # Merge the operations first
            p2.op.merge(p1.op)
            self.ops[p1.op.uid] = p2.op
            # Now merge the tasks and delete the other task
            p2.merge(p1)
            del self.tasks[p1.op]
            return p2

    def has_aliased_ancestor_tree_only(self, one, two):
        if one is two:
            return (True,one)
        # Make them the same depth
        while one.depth < two.depth:
            two = two.parent
        while two.depth < one.depth:
            one = one.parent
        # Test again
        if one is two:
            return (True,one)
        parent_one = one.parent
        parent_two = two.parent
        while parent_one is not parent_two:
            one = parent_one
            parent_one = one.parent
            two = parent_two
            parent_two = two.parent
        assert parent_one is parent_two
        assert one is not two
        if parent_one.are_children_disjoint(one, two):
            return (False,None)
        return (True,parent_one)

    def perform_logical_analysis(self, perform_checks, sanity_checks):
        # Run the full analysis first, this will confirm that
        # the runtime did what we thought it should do
        for task in itervalues(self.tasks):
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
        if perform_checks:
            print(("Performing physical verification analysis "+
                   "for %s (UID %d)...") % (str(top_task),top_task.op.uid))
        else:
            print(("Performing physical emulation analysis "+
                   "for %s (UID %d)...") % (str(top_task),top_task.op.uid))
        # Perform the physical analysis on all the operations in program order
        if not top_task.perform_task_physical_verification(perform_checks):
            print("FAIL")
            return
        print("Pass")

    def perform_cycle_checks(self, print_result=True):
        # To perform our cycle checks we run a modified version of
        # Tarjan's algorithm for strongly connected components since
        # it runs in O(V + E) time. If we find a SCC that is larger
        # than size 1 then that is a cycle.
        class CycleDetector(object):
            def __init__(self):
                self.indexes = dict()
                self.lowlink = dict()
                self.index = 0
                self.stack = list()

            def check_for_cycles(self, node, edges):
                # First see if we've already traversed this node before
                if node in self.indexes:
                    return False
                # Check for self cycles
                if node in edges:
                    print('SELF CYCLE DETECTED! '+str(node))
                    return True
                # Now run Tarjan's algorithm
                self.indexes[node] = self.index
                self.lowlink[node] = self.index
                self.index += 1
                self.stack.append(node)
                for next_node in edges:
                    if next_node not in self.indexes:
                        if next_node.perform_cycle_check(self):
                            # Short circuit
                            return True
                        self.lowlink[node] = min(self.lowlink[node], 
                                                 self.lowlink[next_node])
                    elif next_node in self.stack:
                        self.lowlink[node] = min(self.lowlink[node],
                                                 self.indexes[next_node])
                # If this node is the root of an SCC then we need to
                # pop it off the stack. For an acyclic graph every node
                # should be its own strongly connected component of size 1
                if self.indexes[node] == self.lowlink[node]:
                    last = self.stack.pop()
                    if last is not node:
                        print('CYCLE DETECTED!')
                        print(str(last))
                        while last is not node:
                            last = self.stack.pop()
                            print(str(last))
                        return True
                return False

        cycle_detector = CycleDetector()
        for op in self.unique_ops: 
            if op.perform_cycle_check(cycle_detector):
                return True
        for copy in itervalues(self.copies):
            if copy.perform_cycle_check(cycle_detector):
                return True
        for fill in itervalues(self.fills):
            if fill.perform_cycle_check(cycle_detector):
                return True
        if print_result:
            print("No cycles detected")
        return False 

    def perform_user_event_leak_checks(self):
        for event in itervalues(self.events):
            event.check_for_user_event_leak()

    def make_region_tree_graphs(self, path, simplify_graphs):
        index_space_printer = GraphPrinter(path, 'index_space_graph', 'TB')
        for node in itervalues(self.index_spaces):
            if node.parent is None:
                node.print_graph(index_space_printer)
        index_space_printer.print_pdf_after_close(simplify_graphs)

        field_space_printer = GraphPrinter(path, 'field_space_graph', 'TB')
        for node in itervalues(self.field_spaces):
            node.print_graph(field_space_printer)
        field_space_printer.print_pdf_after_close(simplify_graphs)

        region_graph_printer = GraphPrinter(path, 'region_graph', 'TB')
        for node in itervalues(self.trees):
            if simplify_graphs:
                node.mark_named_children()
            node.print_graph(region_graph_printer, simplify_graphs)
        region_graph_printer.print_pdf_after_close(simplify_graphs)

    def make_machine_graphs(self, path):
        machine_printer = GraphPrinter(path, 'machine_graph', 'TB')
        for proc in itervalues(self.processors):
            proc.print_node(machine_printer)
        for mem in itervalues(self.memories):
            mem.print_node(machine_printer)
        for proc in itervalues(self.processors):
            proc.print_mem_edges(machine_printer)
        for mem in itervalues(self.memories):
            mem.print_mem_edges(machine_printer)
        machine_printer.print_pdf_after_close(False)

    def make_dataflow_graphs(self, path, simplify_graphs, zoom_graphs):
        total_dataflow_graphs = 0
        for task in itervalues(self.tasks):
            total_dataflow_graphs += task.print_dataflow_graph(path, simplify_graphs, zoom_graphs)
        if self.verbose:
            print("Made "+str(total_dataflow_graphs)+" dataflow graphs")

    def make_event_graph(self, path, zoom_graphs):
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
        printer.print_pdf_after_close(False, zoom_graphs)

    def dump_bad_graph(self, context, tree_id, field):
        print('Dumping bad graph for field '+str(field)+' of region tree '+
                str(tree_id)+' in context '+str(context))
        nodes = list()
        file_name = 'bad_dataflow_graph'
        printer = GraphPrinter(self.temp_dir, file_name)
        for op in context.operations:
            if not op.reqs:
                continue
            found = False
            for req in itervalues(op.reqs):
                if req.logical_node.tree_id != tree_id:
                    continue
                if field not in req.fields:
                    continue
                found = True
                break
            if found:
                if op.inter_close_ops:
                    for inter in op.inter_close_ops:
                        if not inter.reqs:
                            continue
                        assert len(inter.reqs) == 1
                        req = inter.reqs[0]
                        if req.logical_node.tree_id != tree_id:
                            continue
                        if field not in req.fields:
                            continue
                        nodes.append(inter)
                        inter.print_base_node(printer, True)
                nodes.append(op)
                op.print_base_node(printer, True)
        # Now we need to compute the edges for this graph
        incoming = dict()
        outgoing = dict()
        for idx in xrange(len(nodes)):
            op = nodes[idx]
            incoming[op] = set()
            outgoing[op] = set()
            warning = op.transitive_warning_issued
            op.transitive_warning_issued = True
            for pidx in xrange(idx):
                prev_op = nodes[pidx]
                if op.has_transitive_mapping_dependence(prev_op):
                    incoming[op].add(prev_op)
                    outgoing[prev_op].add(op)
            op.transitive_warning_issued = warning
        # Now do the transitive reduction to reduce these edges down
        count = 0
        index_map = dict()
        reachable = dict()
        total_nodes = len(nodes)
        # Now traverse the list in reverse order
        for src_index in xrange(total_nodes-1,-1,-1):
            src = nodes[src_index]
            count += 1 
            index_map[src] = src_index
            our_reachable = NodeSet(total_nodes)
            reachable[src] = our_reachable
            if len(outgoing[src]) == 0:
                continue
            # Otherwise iterate through our outgoing edges and get the set of 
            # nodes reachable from all of them
            for dst in outgoing[src]:
                # Some nodes won't appear in the list of all operations
                # such as must epoch operations which we can safely skip
                if dst not in reachable:
                    assert dst not in nodes 
                    continue
                our_reachable.union(reachable[dst])
            # Now see which of our nodes can be reached indirectly
            to_remove = None
            for dst in outgoing[src]:
                # See comment above for why we can skip some edges
                if dst not in index_map:
                    assert dst not in nodes 
                    continue
                dst_index = index_map[dst]
                if our_reachable.contains(dst_index):
                    if to_remove is None:
                        to_remove = list()
                    to_remove.append(dst)
                else:
                    # We need to add it to our reachable set
                    our_reachable.add(dst_index)
            if to_remove:
                for dst in to_remove:
                    outgoing[src].remove(dst)
                    incoming[dst].remove(src)
            # We should never remove everything
            assert len(outgoing[src]) > 0
            for dst in outgoing[src]:
                # Skip any edges to nodes not in the reachable list
                # (e.g. must epoch operations)
                if dst not in reachable:
                    continue
                printer.println(src.node_name+' -> '+dst.node_name+
                                ' [style=solid,color=black,penwidth=2];')
        printer.print_pdf_after_close(False)

    def dump_eq_graph(self, eq_key):
        print('Dumping equivalence set graph for eq set (point='+str(eq_key[0])+
                ', field='+str(eq_key[1])+', tree='+str(eq_key[2])+')')
        nodes = set()
        # Find all the nodes with this eq_key
        def has_eq_key(node):
            if node.eq_incoming and eq_key in node.eq_incoming:
                return True
            if node.eq_outgoing and eq_key in node.eq_outgoing:
                return True
            return False
        for op in self.unique_ops:
            if has_eq_key(op):
                nodes.add(op)
        for copy in itervalues(self.copies):
            if has_eq_key(copy):
                nodes.add(copy)
        for fill in itervalues(self.fills):
            if has_eq_key(fill):
                nodes.add(fill)
        for deppart in itervalues(self.depparts):
            if has_eq_key(deppart):
                nodes.add(deppart)
        # Now that we've got all the nodes we can print them with a graph printer
        file_name = 'bad_eq_graph'
        printer = GraphPrinter(self.temp_dir, file_name)
        for node in nodes:
            node.print_event_node(printer)
        for node in nodes:
            node.print_incoming_eq_edges(printer, eq_key)
        printer.print_pdf_after_close(False)

    def print_realm_statistics(self):
        print('Total events: '+str(len(self.events)))
        print('Total copies: '+str(len(self.copies)))
        total_copy_bytes = 0
        for copy in itervalues(self.copies):
            total_copy_bytes += copy.compute_copy_size()
        print('  Total bytes moved: '+str(total_copy_bytes))
        print('Total fills:  '+str(len(self.fills)))
        total_fill_bytes = 0
        for fill in itervalues(self.fills):
            total_fill_bytes += fill.compute_fill_size()
        print('  Total bytes filled: '+str(total_fill_bytes))

    def make_replay_file(self):
        file_name = 'legion.rp'
        print('Emitting replay file '+file_name)
        with open(file_name,'wb') as replay_file:
            # Write out processors
            replay_file.write(struct.pack('I',len(self.processors)))
            for proc in itervalues(self.processors):
                replay_file.write(struct.pack('Q', proc.uid))    
                replay_file.write(struct.pack('I', proc.kind_num))
            # Write out memories
            replay_file.write(struct.pack('I',len(self.memories)))
            for mem in itervalues(self.memories):
                replay_file.write(struct.pack('Q', mem.uid))
                replay_file.write(struct.pack('I', mem.kind_num))
            # Write out the instances
            assert len(self.instances) > 0
            # Skip the virtual instance
            replay_file.write(struct.pack('I',len(self.instances)-1))
            for inst in itervalues(self.instances):
                if inst.is_virtual():
                    continue
                inst.pack_inst_replay_info(replay_file)
            # Find all the sets of operations
            total_index = 0
            single_tasks = set()
            index_tasks = set()
            inlines = set()
            copies = set()
            closes = set()
            releases = set()
            for op in self.unique_ops:
                if op.kind == SINGLE_TASK_KIND:
                    # If it doesn't have a task and a processor, then it's not real
                    if not op.task or op.task.processor is None:
                        continue
                    # Dont' count points in index space tasks
                    if op.index_owner:
                        continue
                    # If it was merged we don't count it
                    if op.merged:
                        continue
                    single_tasks.add(op)
                if op.kind == INDEX_TASK_KIND:
                    index_tasks.add(op) 
                    assert op.points is not None
                    total_index += len(op.points)
                elif op.kind == MAP_OP_KIND:
                    inlines.add(op)
                elif op.kind == COPY_OP_KIND:
                    copies.add(op)
                elif op.kind == INTER_CLOSE_OP_KIND:
                    closes.add(op)
                elif op.kind == RELEASE_OP_KIND:
                    releases.add(op)
            # Write out the ID of the top-level task
            replay_file.write(struct.pack('Q',self.top_level_uid))
            # Write out the tasks first 
            replay_file.write(struct.pack('I',len(single_tasks)+total_index))
            for op in single_tasks:
                op.task.pack_task_replay_info(replay_file, op.uid)
            actual_index_tasks = 0
            for task in index_tasks:
                for point in itervalues(task.points):
                    point.pack_task_replay_info(replay_file, task.uid)
                    actual_index_tasks += 1
            assert actual_index_tasks == total_index
            # Write out the inlines
            replay_file.write(struct.pack('I',len(inlines)))
            for op in inlines:
                op.pack_inline_replay_info(replay_file)
            # Write out the copies
            replay_file.write(struct.pack('I',len(copies)))
            for op in copies:
                op.pack_copy_replay_info(replay_file)
            # Write out the closes
            replay_file.write(struct.pack('I',len(closes)))
            for op in closes:
                op.pack_close_replay_info(replay_file)
            # Write out the releases
            replay_file.write(struct.pack('I',len(releases)))
            for op in releases:
                op.pack_release_replay_info(replay_file)

    def print_instance_descriptions(self):
        for inst in itervalues(self.instances):
            # Skip the virtual instance
            if inst.is_virtual():
                continue
            if inst.redop > 0:
                print(str(inst)+' (Reduction Op '+str(inst.redop)+')')
            else:
                print(str(inst))
            print('  Memory '+str(inst.memory))
            print('  '+str(inst.index_expr))
            print('  Fields:')
            for field in inst.fields:
                print('    '+str(field))

    def print_mapping_decisions(self):
        assert self.top_level_uid is not None
        top_task = self.get_task(self.top_level_uid) 
        top_task.print_task_mapping_decisions()

    def print_trees(self):
        for node in itervalues(self.index_spaces):
            if node.parent is None:
                node.print_tree()
        for node in itervalues(self.trees):
            node.print_tree()

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

    def get_index_expr(self, expr_id):
        if expr_id in self.index_exprs:
            return self.index_exprs[expr_id]
        result = IndexExpr(self, expr_id)
        self.index_exprs[expr_id] = result
        return result

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
        if ipart.parent is not None:
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
        if self.top_level_ctx_uid is None or uid != self.top_level_ctx_uid:
            op = self.get_operation(uid)
            if op in self.tasks:
                return self.tasks[op]
            op.set_op_kind(SINGLE_TASK_KIND)
        else:
            op = None
        result = Task(self, op)
        self.tasks[op] = result
        return result

    def get_future(self, iid):
        if iid in self.futures:
            return self.futures[iid]
        result = Future(self, iid)
        self.futures[iid] = result
        return result

    def get_variant(self, vid):
        if vid in self.variants:
            return self.variants[vid]
        result = Variant(self, vid)
        self.variants[vid] = result
        return result

    def get_projection_function(self, pid):
        if pid in self.projection_functions:
            return self.projection_functions[pid]
        result = ProjectionFunction(self, pid)
        self.projection_functions[pid] = result
        return result

    def get_instance(self, eid):
        if eid in self.instances:
            return self.instances[eid]
        result = Instance(self, eid)
        self.instances[eid] = result
        return result

    def get_event(self, iid):
        handle = EventHandle(iid)
        if handle in self.events:
            return self.events[handle]
        result = Event(self, handle)
        self.events[handle] = result 
        return result 

    def get_no_event(self):
        return self.no_event

    def get_indirections(self, uid):
        if uid in self.indirections:
            return self.indirections[uid]
        result = Indirections(uid, self)
        self.indirections[uid] = result
        return result

    def get_realm_copy(self, event):
        assert event.exists()
        if event in self.copies:
            return self.copies[event]
        result = RealmCopy(self, event, self.next_realm_num)
        self.next_realm_num += 1
        self.copies[event] = result
        return result

    def get_realm_fill(self, event):
        assert event.exists()
        if event in self.fills:
            return self.fills[event]
        result = RealmFill(self, event, self.next_realm_num)
        self.next_realm_num += 1
        self.fills[event] = result
        return result

    def get_realm_deppart(self, event):
        assert event.exists()
        if event in self.depparts:
            return self.depparts[event]
        result = RealmDeppart(self, event, self.next_realm_num)
        self.next_realm_num += 1
        self.depparts[event] = result
        return result

    def create_copy(self, creator):
        result = RealmCopy(self, self.no_event, self.next_realm_num)
        self.copies[self.next_realm_num] = result
        self.next_realm_num += 1
        result.set_creator(creator)
        return result

    def create_indirections(self):
        result = Indirections(self, self.next_indirections_num)
        self.indirections[self.next_indirections_num] = result
        self.next_indirections_num += 1
        return result

    def create_fill(self, creator):
        result = RealmFill(self, self.no_event, self.next_realm_num)
        self.fills[self.next_realm_num] = result
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
        for region in itervalues(self.regions):
            region.reset_logical_state()
        for partition in itervalues(self.partitions):
            partition.reset_logical_state()
        # Definitely run the garbage collector here
        gc.collect()

    def reset_physical_state(self, depth):
        for region in itervalues(self.regions):
            region.reset_physical_state(depth)
        for partition in itervalues(self.partitions):
            partition.reset_physical_state(depth)
        # Definitely run the garbage collector here
        gc.collect()

    def reset_verification_state(self, depth):
        for region in itervalues(self.trees):
            region.reset_verification_state(depth)
        # Definitely run the garbage collector here
        gc.collect()

def generate_random_intersecting_rects(dim, max_size):
    lo1 = Point(dim)
    hi1 = Point(dim)
    lo2 = Point(dim)
    hi2 = Point(dim)
    for d in xrange(dim):
        lo1.vals[d] = random.randint(0,max_size)
        hi1.vals[d] = random.randint(lo1.vals[d],max_size)
        lo2.vals[d] = random.randint(0,hi1.vals[d])
        if lo2.vals[d] < lo1.vals[d]:
            # Constrained to get in range
            hi2.vals[d] = random.randint(lo1.vals[d],hi1.vals[d])
        else:
            #Unconstrained since we're going to be in range
            hi2.vals[d] = random.randint(lo2.vals[d],max_size)
    return Rect(lo1,hi1),Rect(lo2,hi2)

def print_bad_geometry_result(dim, first, second, result):
    print('  First rectangle: '+str(first))
    print('  Second rectangle: '+str(second))
    print('  Result rectangles:')
    for rect in result.rects:
        print('    '+str(rect))

def perform_geometry_test(dim, max_size=100):
    assert dim >= 1
    first = Shape()
    second = Shape()
    first_rect,second_rect = generate_random_intersecting_rects(dim,max_size)
    first.add_rect(first_rect)
    second.add_rect(second_rect)
    result = first - second
    total_points = 0
    for point in first_rect.iterator():
        if second_rect.contains_point(point):
            for rect in result.rects:
                if rect.contains_point(point):
                    print('Point '+str(point)+' should have been removed')
                    print_bad_geometry_result(dim, first_rect, second_rect, result)
                    return False
        else:
            total_points += 1
            contain_rect = None
            for rect in result.rects:
                if rect.contains_point(point):
                    if contain_rect is None:
                        contain_rect = rect
                    else:
                        print('Point '+str(point)+' in duplicate result rectangles')
                        print_bad_geometry_result(dim, first_rect, second_rect, result)
                        return False
            if contain_rect is None:
                print('Point '+str(point)+' not in result rectangles')
                print_bad_geometry_result(dim, first_rect, second_rect, result)
                return False
    actual_points = 0
    for rect in result.rects:
        actual_points += rect.volume()
    if actual_points != total_points:
        print("Point total mismatch! exp="+str(total_points)+" != act="+str(actual_points))
        print_bad_geometry_result(dim, first_rect, second_rect, result)
        return False
    return True

def run_geometry_tests(num_tests=10000):
    success = True
    for dim in xrange(1,4):
        print("Testing dimension... "+str(dim))
        for i in xrange(num_tests):
            if i % 100 == 0:
                print('  Running test '+str(i))
            success = perform_geometry_test(dim)
            if not success:
                break
        if not success:
            break
    return success

def main(temp_dir):
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            self.print_usage(sys.stderr)
            print('error: %s' % message, file=sys.stderr)
            print('hint: invoke %s -h for a detailed description of all arguments' % self.prog, file=sys.stderr)
            sys.exit(2)
    parser = MyParser(
        description='Legion Spy runtime analysis and verification')
    parser.add_argument(
        '-l', '--logical', dest='logical_checks', action='store_true',
        help='check logical analysis')
    parser.add_argument(
        '-p', '--physical', dest='physical_checks', action='store_true',
        help='check physical analysis')
    parser.add_argument(
        '-c', '--cycle', dest='cycle_checks', action='store_true',
        help='check for cycles')
    parser.add_argument(
        '-s', '--sanity', dest='sanity_checks', action='store_true',
        help='check basic properties of the dataflow and event graphs')
    parser.add_argument(
        '-w', '--leaks', dest='user_event_leaks', action='store_true',
        help='check for user event leaks')
    parser.add_argument(
        '-g', '--geometry', dest='test_geometry', action='store_true',
        help='check computational geometry')
    parser.add_argument(
        '-r', '--region', dest='region_tree_graphs', action='store_true',
        help='draw region tree graphs')
    parser.add_argument(
        '-m', '--machine', dest='machine_graphs', action='store_true',
        help='draw machine graphs')
    parser.add_argument(
        '-d', '--dataflow', dest='dataflow_graphs', action='store_true',
        help='draw logical dataflow graphs')
    parser.add_argument(
        '-e', '--event', dest='event_graphs', action='store_true',
        help='draw physical event graphs')
    parser.add_argument(
        '-z', '--detail', dest='detailed_graphs', action='store_true',
        help='include extra detail (such as field names) in graphs')
    parser.add_argument(
        '-u', '--unsimplified', dest='simplify_graphs',
        action='store_false', # Note that this defaults to true
        help='keep redundant edges in graphs')
    parser.add_argument(
        '-i', '--instance', dest='instance_descriptions', action='store_true',
        help='print instance descriptions')
    parser.add_argument(
        '-x', '--mapping', dest='mapping_decisions', action='store_true',
        help='print mapping decisions')
    parser.add_argument(
        '-t', '--trees', dest='print_trees', action='store_true',
        help='print index and region trees')
    parser.add_argument(
        '--realm-stats', dest='realm_stats', action='store_true',
        help='print Realm statistics')
    parser.add_argument(
        '-y', '--replay', dest='replay_file', action='store_true',
        help='generate mapper replay file')
    parser.add_argument(
        '-k', '--temporaries', dest='keep_temp_files', action='store_true',
        help='keep temporary generated files')
    parser.add_argument(
        '-v', '--verbose', dest='verbose', action='store_true',
        help='verbose output')
    parser.add_argument(
        '-a', '--assert-error', dest='assert_on_error', action='store_true',
        help='assert on errors')
    parser.add_argument(
        '--assert-warning', dest='assert_on_warning', action='store_true',
        help='assert on warnings (implies -a)')
    parser.add_argument(
        '--zoom', dest='zoom_graphs', action='store_true',
        help='enable generation of "zoom" graphs for all emitted graphs')
    parser.add_argument(
        '-b', '--bad_graph', dest='bad_graph_on_error', action='store_true',
        help='dump bad dataflow graph on failure')
    parser.add_argument(
        '-q', '--eq_graph', dest='eq_graph_on_error', action='store_true',
        help='dump equivalence set graph on failure')
    parser.add_argument(
        dest='filenames', nargs='+',
        help='input legion spy log filenames')
    args = parser.parse_args()

    file_names = args.filenames

    logical_checks = args.logical_checks
    physical_checks = args.physical_checks
    cycle_checks = args.cycle_checks
    region_tree_graphs = args.region_tree_graphs
    machine_graphs = args.machine_graphs
    dataflow_graphs = args.dataflow_graphs
    event_graphs = args.event_graphs
    instance_descriptions = args.instance_descriptions
    mapping_decisions = args.mapping_decisions
    print_trees = args.print_trees
    realm_stats = args.realm_stats
    replay_file = args.replay_file
    detailed_graphs = args.detailed_graphs
    sanity_checks = args.sanity_checks
    user_event_leaks = args.user_event_leaks
    keep_temp_files = args.keep_temp_files
    simplify_graphs = args.simplify_graphs
    verbose = args.verbose
    assert_on_error = args.assert_on_error or args.assert_on_warning
    assert_on_warning = args.assert_on_warning
    test_geometry = args.test_geometry
    zoom_graphs = args.zoom_graphs
    bad_graph_on_error = args.bad_graph_on_error
    eq_graph_on_error = args.eq_graph_on_error

    if test_geometry:
        run_geometry_tests()
    if dataflow_graphs or event_graphs or bad_graph_on_error:
        try:
            subprocess.Popen(['dot', '-V'], stderr=open(os.devnull, 'wb'))
        except OSError:
            print("WARNING: graphviz is not installed, graphs will not be generated")

    state = State(temp_dir, verbose, detailed_graphs, assert_on_error, 
                  assert_on_warning, bad_graph_on_error, eq_graph_on_error)
    total_matches = 0 
    for file_name in file_names:
        total_matches += state.parse_log_file(file_name)
    if verbose:
        print('Matched %d lines across all files.' % total_matches)
    if total_matches == 0:
        print('No matches found! Exiting...')
        return
    state.post_parse(simplify_graphs, physical_checks or event_graphs)
    if logical_checks and not state.detailed_logging:
        print("WARNING: Requested logical analysis but logging information is "+
              "missing. Please compile the runtime with USE_SPY=1 to enable "+
              "validation of the runtime. Disabling logical checks.")
        if state.assert_on_warning:
            assert False
        logical_checks = False
    if physical_checks and not state.detailed_logging:
        print("WARNING: Requested physical analysis but logging information is "+
              "missing. Please compile the runtime with USE_SPY=1 to enable "+
              "validation of the runtime. Disabling physical checks.")
        if state.assert_on_warning:
            assert False
        physical_checks = False
    if logical_checks and sanity_checks and not state.detailed_logging:
        print("WARNING: Requested sanity checks for logical analysis but "+
              "logging information of logical analysis is missing. Please "+
              "compile the runtime with USE_SPY=1 to enable validation "+
              "of the runtime. Disabling sanity checks.")
        if state.assert_on_warning:
            assert False
        sanity_checks = False
    if physical_checks and sanity_checks and not state.detailed_logging:
        print("WARNING: Requested sanity checks for physical analysis but "+
              "logging information of logical analysis is missing. Please "+
              "compile the runtime with USE_SPY=1 to enable validation "+
              "of the runtime. Disabling sanity checks.")
        if state.assert_on_warning:
            assert False
        sanity_checks = False
    if cycle_checks and not state.detailed_logging:
        print("WARNING: Requested cycle checks but logging information is "+
              "missing. Please compile the runtime with USE_SPY=1 to enable "+
              "validation of the runtime. Disabling cycle checks.")
        if state.assert_on_warning:
            assert False
        cycle_checks = False
    if user_event_leaks and not state.detailed_logging:
        print("WARNING: Requested user event leak checks but logging information "+
              "is missing. Please compile the runtime with USE_SPY=1 to enable "+
              "validation of the runtime. Disabling user event leak checks.")
        if state.assert_on_warning:
            assert False
        user_event_leaks = False
    if cycle_checks:
        print("Performing cycle checks...")
        state.perform_cycle_checks()
    # If we are doing logical checks or the user asked for the dataflow
    # graph but we don't have any logical data then perform the logical analysis
    need_logical = dataflow_graphs and not state.detailed_logging 
    if logical_checks or need_logical:
        if need_logical:
            print("INFO: No logical dependence data was found so we are running "+
                  "logical analysis to show the dataflow graphs that the runtime "+
                  "should compute. These are not the actual dataflow graphs computed.")
        print("Performing logical analysis...")
        state.perform_logical_analysis(logical_checks, sanity_checks)
    # If we are doing physical checks or the user asked for the event
    # graph but we don't have any logical data then perform the physical analysis
    need_physical = event_graphs and not state.detailed_logging 
    if physical_checks or need_physical:
        if need_physical:
            print("INFO: No physical dependence data was found so we are running "+
                  "physical analysis to show the event graph that the runtime "+
                  "should compute. This is not the actual event graph computed.")
        elif (simplify_graphs):
            # Simplify the graph before checking it, we need cycle checks
            # if we didn't do them before
            state.simplify_physical_graph(need_cycle_check=not cycle_checks,
                                          make_equivalence_graphs=True)
        else:
            # Doing verification so we still need the equivalence class graphs
            state.compute_equivalence_graphs()
        state.perform_physical_analysis(physical_checks, sanity_checks) 
        # If we generated the graph for printing, then simplify it 
        if need_physical and simplify_graphs:
            state.simplify_physical_graph(need_cycle_check=False)
    elif event_graphs and simplify_graphs:
        state.simplify_physical_graph(need_cycle_check=not cycle_checks)
    if user_event_leaks:
        print("Performing user event leak checks...")
        state.perform_user_event_leak_checks()
    if region_tree_graphs:
        print("Making region tree graphs...")
        state.make_region_tree_graphs(temp_dir, False)
    if machine_graphs:
        print("Making machine graphs...")
        state.make_machine_graphs(temp_dir)
    if dataflow_graphs:
        print("Making dataflow graphs...")
        state.make_dataflow_graphs(temp_dir, simplify_graphs, zoom_graphs)
    if event_graphs:
        print("Making event graphs...")
        state.make_event_graph(temp_dir, zoom_graphs)
    if realm_stats:
        print("Printing Realm statistics...")
        state.print_realm_statistics()
    if replay_file:
        print("Generating mapper replay file...")
        state.make_replay_file()
    if instance_descriptions:
        state.print_instance_descriptions()
    if mapping_decisions:
        state.print_mapping_decisions()
    if print_trees:
        state.print_trees()

    print('Legion Spy analysis complete.  Exiting...')
    if keep_temp_files:
        try:
            subprocess.check_call('cp '+temp_dir+'* .',shell=True)
        except:
            print('WARNING: Unable to copy temporary files into current directory')

if __name__ == "__main__":
    temp_dir = tempfile.mkdtemp()+'/'
    try:
        sys.setrecursionlimit(10000)
        main(temp_dir)
    finally:
        shutil.rmtree(temp_dir)

# EOF

