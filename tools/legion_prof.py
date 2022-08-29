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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import tempfile
import legion_spy
import argparse
import sys
import os
import shutil
import math
import collections
import string
import re
import json
import heapq
import time
import itertools
from functools import reduce
from legion_serializer import LegionProfASCIIDeserializer, LegionProfBinaryDeserializer, GetFileTypeInfo

root_dir = os.path.dirname(os.path.realpath(__file__))

# Make sure this is up to date with realm_c.h
processor_kinds = {
    1 : 'GPU',
    2 : 'CPU',
    3 : 'Utility',
    4 : 'IO',
    5 : 'Proc Group',
    6 : 'Proc Set',
    7 : 'OpenMP',
    8 : 'Python',
}

# Make sure this is up to date with realm_c.h
memory_kinds = {
    0 : 'No MemKind',
    1 : 'GASNet Global',
    2 : 'System',
    3 : 'Registered',
    4 : 'Socket',
    5 : 'Zero-Copy',
    6 : 'Framebuffer',
    7 : 'Disk',
    8 : 'HDF5',
    9 : 'File',
    10 : 'L3 Cache',
    11 : 'L2 Cache',
    12 : 'L1 Cache',
    13 : 'GPU Managed',
    14 : 'GPU Dynamic',
}
# Make sure this is up to date with memory_kinds
memory_node_proc = {
    'No MemKind': 'None',
    'GASNet Global': 'None',
    'System': 'Node_id',
    'Registered': 'Node_id',
    'Socket': 'Node_id',
    'Zero-Copy': 'Node_id',
    'Framebuffer': 'GPU_proc_id',
    'Disk': 'Node_id',
    'HDF5': 'Node_id',
    'File': 'Node_id',
    'L3 Cache': 'Node_id',
    'L2 Cache': 'Proc_id',
    'L1 Cache': 'Proc_id',
    'GPU Managed': 'Node_id',
    'GPU Dynamic': 'GPU_proc_id',
}

memory_kinds_abbr = {
    'No MemKind': ' none',
    'GASNet Global': ' glob',
    'System': ' sys',
    'Registered': ' reg',
    'Socket': ' sock',
    'Zero-Copy': ' zcpy',
    'Framebuffer': ' fb',
    'Disk': ' disk',
    'HDF5': ' hdf5',
    'File': ' file',
    'L3 Cache': ' l3',
    'L2 Cache': ' l2',
    'L1 Cache': ' l1',
    'GPU Managed': ' uvm',
    'GPU Dynamic': ' gpu-dyn'
}

# Make sure this is up to date with legion_types.h
dep_part_kinds = {
    0 : 'Union',
    1 : 'Unions',
    2 : 'Union Reduction',
    3 : 'Intersection',
    4 : 'Intersections',
    5 : 'Intersection Reduction',
    6 : 'Difference',
    7 : 'Differences',
    8 : 'Equal Partition',
    9 : 'Partition by Field',
    10 : 'Partition by Image',
    11 : 'Partition by Image Range',
    12 : 'Partition by Preimage',
    13 : 'Partition by Preimage Range',
    14 : 'Create Association',
    15 : 'Partition by Weights',
}
# Make sure this is up to date with legion_config.h
legion_equality_kind_t = {
    0 : '<',
    1 : '<=',
    2 : '>',
    3 : ' >=',
    4 : '==',
    5 : '!=',
}

legion_dimension_kind_t = {
    0 : 'DIM_X',
    1 : 'DIM_Y',
    2 : 'DIM_Z',
    3 : 'DIM_W',
    4 : 'DIM_V',
    5 : 'DIM_U',
    6 : 'DIM_T',
    7 : 'DIM_S',
    8 : 'DIM_R',
    9 : 'DIM_F',
    10: 'INNER_DIM_X',
    11 : 'OUTER_DIM_X',
    12 : 'INNER_DIM_Y',
    13 : 'OUTER_DIM_Y',
    14 : 'INNER_DIM_Z',
    15 : 'OUTER_DIM_Z',
    16 : 'INNER_DIM_W',
    17 : 'OUTER_DIM_W',
    18 : 'INNER_DIM_V',
    19 : 'OUTER_DIM_V',
    20 : 'INNER_DIM_U',
    21 : 'OUTER_DIM_U',
    22 : 'INNER_DIM_T',
    23 : 'OUTER_DIM_T',
    24 : 'INNER_DIM_S',
    25 : 'OUTER_DIM_S',
    26 : 'INNER_DIM_R',
    27 : 'OUTER_DIM_R',
}

request = {
    0 : 'fill',
    1 : 'reduc',
    2 : 'copy',
}

# Micro-seconds per pixel
US_PER_PIXEL = 100
# Pixels per level of the picture
PIXELS_PER_LEVEL = 40
# Pixels per tick mark
PIXELS_PER_TICK = 200

# prof_uid counter
prof_uid_ctr = 0

def get_prof_uid():
    global prof_uid_ctr
    prof_uid_ctr += 1
    return prof_uid_ctr

def data_tsv_str(level, level_ready, ready, start, end, color, opacity, title,
                  initiation, _in, out, children, parents, prof_uid, op_id=None):
    # replace None with ''
    def xstr(s):
        return str(s or '')
    if (op_id == None):
        str_op_id = ""
    else:
        str_op_id = str(op_id)
    # if initiation == None:
    #     print(op_id, initiation)
    return xstr(level) + "\t" + xstr(level_ready) + "\t" + \
           xstr('%.3f' % ready if ready else ready) + "\t" + \
           xstr('%.3f' % start if start else start) + "\t" + \
           xstr('%.3f' % end if end else end) + "\t" + \
           xstr(color) + "\t" + xstr(opacity) + "\t" + xstr(title) + "\t" + \
           xstr(initiation) + "\t" + xstr(_in) + "\t" + xstr(out) + "\t" + \
           xstr(children) + "\t" + xstr(parents) + "\t" + xstr(prof_uid) + "\t" + \
           str_op_id + "\n"

def slugify(filename):
    # convert spaces to underscores
    slugified = filename.replace(" ", "_")
    # remove 'L" for hex
    if (slugified[-1] == "L"):
        slugified = slugified[:-1]
    # remove special characters
    slugified = slugified.translate("!@#$%^&*(),/?<>\"':;{}[]|/+=`~") if \
            sys.version_info > (3,) else \
            slugified.translate(None, "!@#$%^&*(),/?<>\"':;{}[]|/+=`~")
    return slugified

# Helper function for computing nice colors
def color_helper(step, num_steps):
    assert step <= num_steps
    h = float(step)/float(num_steps)
    i = ~~int(h * 6)
    f = h * 6 - i
    q = 1 - f
    rem = i % 6
    r = 0
    g = 0
    b = 0
    if rem == 0:
      r = 1
      g = f
      b = 0
    elif rem == 1:
      r = q
      g = 1
      b = 0
    elif rem == 2:
      r = 0
      g = 1
      b = f
    elif rem == 3:
      r = 0
      g = q
      b = 1
    elif rem == 4:
      r = f
      g = 0
      b = 1
    elif rem == 5:
      r = 1
      g = 0
      b = q
    else:
      assert False
    r = (~~int(r*255))
    g = (~~int(g*255))
    b = (~~int(b*255))
    r = "%02x" % r
    g = "%02x" % g
    b = "%02x" % b
    return ("#"+r+g+b)

# Helper methods for python 2/3 foolishness
def iteritems(obj):
    return obj.items() if sys.version_info > (3,) else obj.viewitems()

def iterkeys(obj):
    return obj.keys() if sys.version_info > (3,) else obj.viewkeys()

def itervalues(obj):
    return obj.values() if sys.version_info > (3,) else obj.viewvalues()

# Helper function for size
def size_pretty(size):
    if size is not None:
        if size >= (1024*1024*1024):
            # GBs
            size_pretty = '%.3f GiB' % (size / (1024.0*1024.0*1024.0))
        elif size >= (1024*1024):
            # MBs
            size_pretty = '%.3f MiB' % (size / (1024.0*1024.0))
        elif size >= 1024:
            # KBs
            size_pretty = '%.3f KiB' % (size / 1024.0)
        else:
            # Bytes
            size_pretty = str(size) + ' B'
    else:
        size_pretty = 'Unknown'
    return size_pretty

class PathRange(object):
    __slots__ = ['start', 'stop', 'path']
    def __init__(self, start, stop, path):
        assert start <= stop
        self.start = start
        self.stop = stop
        self.path = list(path)
    def __cmp__(self, other):
        self_elapsed = self.elapsed()
        other_elapsed = other.elapsed()
        if self_elapsed == other_elapsed:
            return len(self.path) - len(other.path)
        else:
            if self_elapsed < other_elapsed:
                return -1
            elif self_elapsed == other_elapsed:
                return 0
            else:
                return 1
    def __lt__(self, other):
        return self.__cmp__(other) < 0
    def __gt__(self, other):
        return self.__cmp__(other) > 0
    def clone(self):
        return PathRange(self.start, self.stop, self.path)
    def elapsed(self):
        return self.stop - self.start
    def __repr__(self):
        return "(" + str(self.start) + "," + str(self.stop) + ")"

class HasDependencies(object):
    __slots__ = []
    _abstract_slots = [
        'deps', 'initiation_op', 'initiation', 'path', 'visited'
    ]
    def __init__(self):
        self.deps = {"in": set(), "out": set(), "parents": set(), "children" : set()}
        self.initiation_op = None
        self.initiation = ''

        # for critical path analysis
        self.path = PathRange(0, 0, [])
        self.visited = False
    
    def add_initiation_dependencies(self, state, op_dependencies, transitive_map):
        pass

    def attach_dependencies(self, state, op_dependencies, transitive_map):
        if self.op_id in op_dependencies:
            self.deps["out"] |= op_dependencies[self.op_id]["out"]
            self.deps["in"] |=  op_dependencies[self.op_id]["in"]
            self.deps["parents"] |=  op_dependencies[self.op_id]["parents"]
            self.deps["children"] |=  op_dependencies[self.op_id]["children"]

    def get_unique_tuple(self):
        assert self.proc is not None #TODO: move to owner
        cur_level = self.proc.max_levels - self.level
        return (self.proc.node_id, self.proc.proc_in_node, self.prof_uid)

class HasInitiationDependencies(HasDependencies):
    __slots__ = []
    _abstract_slots = HasDependencies._abstract_slots + ['initialization_op', 'initialization']
    def __init__(self, initiation_op):
        HasDependencies.__init__(self)
        self.initiation_op = initiation_op
        self.initiation = initiation_op.op_id

    def add_initiation_dependencies(self, state, op_dependencies, transitive_map):
        """
        Add the dependencies from the initiation to us
        """
        unique_tuple = self.get_unique_tuple()
        if self.initiation in state.operations:
            op = state.find_op(self.initiation)
            # this op exists
            if op.proc is not None:
                if self.initiation not in op_dependencies:
                    op_dependencies[self.initiation] = {
                        "in" : set(), 
                        "out" : set(),
                        "parents" : set(),
                        "children" : set()
                    }
                if op.stop < self.stop:
                    op_dependencies[self.initiation]["in"].add(unique_tuple)
                else:
                    op_dependencies[self.initiation]["out"].add(unique_tuple)

    def attach_dependencies(self, state, op_dependencies, transitive_map):
        """
        Add the dependencies from the us to the initiation
        """
        # add the out direction
        if self.initiation in state.operations:
            op = state.find_op(self.initiation)
            if op.proc is not None: # this op exists
                op_tuple = op.get_unique_tuple()
                if op.stop < self.stop:
                    self.deps["out"].add(op_tuple)
                else:
                    self.deps["in"].add(op_tuple)

    def get_color(self):
        return self.initiation_op.get_color()

class HasNoDependencies(HasDependencies):
    __slots__ = []
    _abstract_slots = HasDependencies._abstract_slots
    def __init__(self):
        HasDependencies.__init__(self)
    
    def add_initiation_dependencies(self, state, op_dependencies, transitive_map):
        pass

    def attach_dependencies(self, state, op_dependencies, transitive_map):
        pass

class TimeRange(object):
    __slots__ = []
    _abstract_slots = ['create', 'ready', 'start', 'stop', 'trimmed', 'was_removed']
    def __init__(self, create, ready, start, stop):
        assert create is None or create <= ready
        assert ready is None or ready <= start
        assert start is None or start <= stop
        self.create = create
        self.ready = ready
        self.start = start
        self.stop = stop
        self.trimmed = False
        self.was_removed = False

    def __cmp__(self, other):
        # The order chosen here is critical for sort_range. Ranges are
        # sorted by start_event first, and then by *reversed*
        # end_event, so that each range will precede any ranges they
        # contain in the order.
        if self.start < other.start:
            return -1
        if self.start > other.start:
            return 1

        if self.stop > other.stop:
            return -1
        if self.stop < other.stop:
            return 1
        return 0

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def __gt__(self, other):
        return self.__cmp__(other) > 0

    def __repr__(self):
        return "Delay: %d us  Start: %d us  Stop: %d us  Total: %d us" % (
                self.queue_time(), self.start, self.end, self.total_time())

    def total_time(self):
        return self.stop - self.start

    def queue_time(self):
        print(str(self.start - self.ready))
        return self.start - self.ready

    # The following must be overridden by subclasses that need them
    def mapper_time(self):
        pass

    def active_time(self):
        pass

    def application_time(self):
        pass

    def meta_time(self):
        pass

    def is_trimmed(self):
        return self.was_removed

    def trim_time_range(self, start, stop):
        if self.trimmed:
            return not self.was_removed
        self.trimmed = True
        if start is not None and stop is not None:
            if self.stop < start:
                self.was_removed = True
                return False
            if self.start > stop:
                self.was_removed = True
                return False
            # Either we span or we're contained inside
            # Clip our boundaries down to the border
            if self.start < start:
                self.start = 0
            else:
                self.start -= start
            if self.stop > stop:
                self.stop = stop - start
            else:
                self.stop -= start
            if self.create is not None:
                if self.create < start:
                    self.create = 0
                elif self.create > stop:
                    self.create = stop - start
                else:
                    self.create -= start
            if self.ready is not None:
                if self.ready < start:
                    self.ready = 0
                elif self.ready > stop:
                    self.ready = stop - start
                else:
                    self.ready -= start
            # See if we need to filter the waits
            if isinstance(self, HasWaiters):
                trimmed_intervals = list()
                for wait_interval in self.wait_intervals:
                    if wait_interval.end < start:
                        continue
                    if wait_interval.start > stop:
                        continue
                    # Adjust the start location
                    if wait_interval.start < start:
                        wait_interval.start = 0
                    else:
                        wait_interval.start -= start
                    # Adjust the ready location
                    if wait_interval.ready < start:
                        wait_interval.ready = 0
                    elif wait_interval.ready > stop:
                        wait_interval.ready = stop - start
                    else:
                        wait_interval.ready -= start
                    # Adjust the end location
                    if wait_interval.end > stop:
                        wait_interval.end = stop - start
                    else:
                        wait_interval.end -= start
                    trimmed_intervals.append(wait_interval)
                self.wait_intervals = trimmed_intervals
        elif start is not None:
            if self.stop < start:
                self.was_removed = True
                return False
            if self.start < start:
                self.start = 0
            else:
                self.start -= start
            self.stop -= start
            if self.create is not None:
                if self.create < start:
                    self.create = 0
                else:
                    self.create -= start
            if self.ready is not None:
                if self.ready < start:
                    self.ready = 0
                else:
                    self.ready -= start
            if isinstance(self, HasWaiters):
                trimmed_intervals = list()
                for wait_interval in self.wait_intervals:
                    if wait_interval.end < start:
                        continue
                    if wait_interval.start < start:
                        wait_interval.start = 0
                    else:
                        wait_interval.start -= start
                    if wait_interval.ready < start:
                        wait_interval.ready = 0
                    else:
                        wait_interval.ready -= start
                    wait_interval.end -= start
                    trimmed_intervals.append(wait_interval)
                self.wait_intervals = trimmed_intervals
        else:
            assert stop is not None
            if self.start > stop:
                self.was_removed = True
                return False
            if self.stop > stop:
                self.stop = stop
            if self.create is not None and self.create > stop:
                self.create = stop
            if self.ready is not None and self.ready > stop:
                self.ready = stop
            if isinstance(self, HasWaiters):
                trimmed_intervals = list()
                for wait_interval in self.wait_intervals:
                    if wait_interval.start > stop:
                        continue
                    if wait_interval.ready > stop:
                        wait_interval.ready = stop
                    if wait_interval.end > stop:
                        wait_interval.end = stop
                    trimmed_intervals.append(wait_interval)
                self.wait_intervals = trimmed_intervals
        return True

class Processor(object):
    __slots__ = [
        'proc_id', 'node_id', 'proc_in_node', 'kind', 'app_ranges',
        'last_time', 'tasks', 'max_levels', 'max_levels_ready', 'time_points',
        'util_time_points'
    ]
    def __init__(self, proc_id, kind):
        self.proc_id = proc_id
        # PROCESSOR:   tag:8 = 0x1d, owner_node:16,   (unused):28, proc_idx: 12
        # owner_node = proc_id[55:40]
        # proc_idx = proc_id[11:0]
        self.node_id = (proc_id >> 40) & ((1 << 16) - 1)
        self.proc_in_node = (proc_id) & ((1 << 12) - 1)
        self.kind = kind
        self.app_ranges = list()
        self.last_time = None
        self.tasks = list()
        self.max_levels = 0
        self.max_levels_ready = 0
        self.time_points = list()
        self.util_time_points = list()
    def get_short_text(self):
        return self.kind + " Proc " + str(self.proc_in_node)

    def add_task(self, task):
        task.proc = self
        self.tasks.append(task)

    def add_mapper_call(self, call):
        # treating mapper calls like any other task
        call.proc = self
        self.tasks.append(call)

    def add_runtime_call(self, call):
        # treating runtime calls like any other task
        call.proc = self
        self.tasks.append(call)

    def trim_time_range(self, start, stop):
        trimmed_tasks = list()
        for task in self.tasks:
            if task.trim_time_range(start, stop):
                trimmed_tasks.append(task)
        self.tasks = trimmed_tasks 

    def sort_time_range(self):
        time_points_all = list()
        for task in self.tasks:
            if (task.stop-task.start > 10 and task.ready != None):
                time_points_all.append(TimePoint(task.ready, task, True, task.start))
                time_points_all.append(TimePoint(task.stop, task, False, 0))
            else:
                time_points_all.append(TimePoint(task.start, task, True, 0))
                time_points_all.append(TimePoint(task.stop, task, False, 0))

            self.time_points.append(TimePoint(task.start, task, True, 0))
            self.time_points.append(TimePoint(task.stop, task, False, 0))

            self.util_time_points.append(TimePoint(task.start, task, True, 0))
            self.util_time_points.append(TimePoint(task.stop, task, False, 0))
            if isinstance(task, HasWaiters):
                # wait intervals don't count for the util graph
                for wait_interval in task.wait_intervals:
                    self.util_time_points.append(TimePoint(wait_interval.start, 
                                                           task, False, 0))
                    self.util_time_points.append(TimePoint(wait_interval.end, 
                                                           task, True, 0))

        self.util_time_points.sort(key=lambda p: p.time_key)
        self.time_points.sort(key=lambda p: p.time_key)
        free_levels = set()
        # level without ready state
        for point in self.time_points:
            if point.first:
                if free_levels:
                    point.thing.set_level(min(free_levels))
                    free_levels.remove(point.thing.level)
                else:
                    self.max_levels += 1
                    point.thing.set_level(self.max_levels)
            else:
                free_levels.add(point.thing.level)

        # level with ready state
        # add the ready times
        free_levels_ready = set()
        time_points_all.sort(key=lambda p: p.time_key)

        free_levels_ready = set()
        self.max_levels_ready=0;

        for point in time_points_all:
            if point.first:
                if free_levels_ready:
                    point.thing.set_level_ready(min(free_levels_ready))
                    free_levels_ready.remove(point.thing.level_ready)
                else:
                    self.max_levels_ready += 1
                    point.thing.set_level_ready(self.max_levels_ready)
            else:
                free_levels_ready.add(point.thing.level_ready)

    def add_initiation_dependencies(self, state, op_dependencies, transitive_map):
        for point in self.time_points:
            if point.first:
                point.thing.add_initiation_dependencies(state, op_dependencies, transitive_map)

    def attach_dependencies(self, state, op_dependencies, transitive_map):
        for point in self.time_points:
            if point.first:
                point.thing.attach_dependencies(state, op_dependencies, transitive_map)

    def emit_tsv(self, tsv_file, base_level):
        # iterate over tasks in start/ready time order
        for point in self.time_points:
            if point.first:
                point.thing.emit_tsv(tsv_file, base_level,
                                     self.max_levels + 1,
                                     self.max_levels_ready + 1,
                                     point.thing.level,
                                     point.thing.level_ready)
        return base_level + max(self.max_levels, 1) + 1

    def total_time(self):
        total = 0
        for task in self.tasks:
            total += task.total_time()
        return total

    def active_time(self):
        total = 0
        for task in self.tasks:
            total += task.active_time()
        return total

    def application_time(self):
        total = 0
        for task in self.tasks:
            total += task.application_time()
        return total

    def meta_time(self):
        total = 0
        for task in self.tasks:
            total += task.meta_time()
        return total

    def mapper_time(self):
        total = 0
        for task in self.tasks:
            total += task.mapper_time()
        return total

    def print_stats(self, verbose):
        total_time = self.total_time()
        active_time = 0
        application_time = 0
        meta_time = 0
        mapper_time = 0
        active_ratio = 0.0
        application_ratio = 0.0
        meta_ratio = 0.0
        mapper_ratio = 0.0
        if total_time != 0:
            active_time = self.active_time()
            application_time = self.application_time()
            meta_time = self.meta_time()
            mapper_time = self.mapper_time()
            active_ratio = 100.0*float(active_time)/float(total_time)
            application_ratio = 100.0*float(application_time)/float(total_time)
            meta_ratio = 100.0*float(meta_time)/float(total_time)
            mapper_ratio = 100.0*float(mapper_time)/float(total_time)
        if total_time != 0 or verbose:
            print(self)
            print("    Total time: %d us" % total_time)
            print("    Active time: %d us (%.3f%%)" % (active_time, active_ratio))
            print("    Application time: %d us (%.3f%%)" % (application_time, application_ratio))
            print("    Meta time: %d us (%.3f%%)" % (meta_time, meta_ratio))
            print("    Mapper time: %d us (%.3f%%)" % (mapper_time, mapper_ratio))
            print()

    def __repr__(self):
        return '%s Processor %s' % (self.kind, hex(self.proc_id))

    def __cmp__(a, b):
        return a.proc_id - b.proc_id;

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def __gt__(self, other):
        return self.__cmp__(other) > 0

class TimePoint(object):
    __slots__ = ['time', 'thing', 'first', 'time_key']
    def __init__(self, time, thing, first, secondary_sort_key):
        assert time != None
        self.time = time
        self.thing = thing
        self.first = first
        # secondary_sort_key is a parameter used for breaking ties in sorting.
        # In practice, we plan for this to be a nanosecond timestamp,
        # like the time field above.
        self.time_key = (time, 0 if first is True else 1, secondary_sort_key)
    def __cmp__(a, b):
        if a.time_key < b.time_key:
            return -1
        elif a.time_key > b.time_key:
            return 1
        else:
            return 0
    def __lt__(self, other):
        return self.__cmp__(other) < 0
    def __gt__(self, other):
        return self.__cmp__(other) > 0

class Memory(object):
    __slots__ = [
        'mem_id', 'node_id', 'kind', 'capacity', 'instances',
        'time_points', 'max_live_instances', 'last_time', 'affinity'
    ]
    def __init__(self, mem_id, kind, capacity):
        self.mem_id = mem_id
        # MEMORY:      tag:8 = 0x1e, owner_node:16,   (unused):32, mem_idx: 8
        # owner_node = mem_id[55:40]
        self.node_id = (mem_id >> 40) & ((1 << 16) - 1)
        self.kind = kind
        self.capacity = capacity
        self.instances = set()
        self.time_points = list()
        self.max_live_instances = None
        self.last_time = None
        self.affinity = None

    def get_short_text(self):
        if self.affinity is not None:
            return self.affinity.get_short_text()
        else:
            return "[n" + str(self.node_id) + "]" + memory_kinds_abbr[self.kind]

    def add_instance(self, inst):
        self.instances.add(inst)
        inst.mem = self

    def add_affinity(self, affinity):
        self.affinity = affinity

    def init_time_range(self, last_time):
        # Fill in any of our instances that are not complete with the last time
        for inst in self.instances:
            if inst.stop is None:
                inst.stop = last_time
        self.last_time = last_time 

    def trim_time_range(self, start, stop):
        trimmed_instances = set()
        for inst in self.instances:
            if inst.trim_time_range(start, stop):
                trimmed_instances.add(inst)
        self.instances = trimmed_instances

    def sort_time_range(self):
        self.max_live_instances = 0
        for inst in self.instances:
            self.time_points.append(TimePoint(inst.start, inst, True, 0))
            self.time_points.append(TimePoint(inst.stop, inst, False, 0))
        # Keep track of which levels are free
        self.time_points.sort(key=lambda p: p.time_key)
        free_levels = set()
        # Iterate over all the points in sorted order
        for point in self.time_points:
            if point.first:
                # Find a level to assign this to
                if len(free_levels) > 0:
                    point.thing.set_level(min(free_levels))
                    free_levels.remove(point.thing.level)
                else:
                    point.thing.set_level(self.max_live_instances + 1)
                    self.max_live_instances += 1
            else:
                # Finishing this instance so restore its point
                free_levels.add(point.thing.level)

    def emit_tsv(self, tsv_file, base_level):
        max_levels = self.max_live_instances + 1
        if max_levels > 1:
            # iterate over tasks in start time order
            max_levels = max(4, max_levels)
            for point in self.time_points:
                if point.first:
                    point.thing.emit_tsv(tsv_file, base_level,\
                                         max_levels, None,
                                         point.thing.level, None)

        return base_level + max_levels

        # if max_levels > 1:
        #     for instance in self.instances:
        #         assert instance.level is not None
        #         assert instance.create is not None
        #         assert instance.destroy is not None
        #         inst_name = repr(instance)
        #         tsv_file.write("%d\t%ld\t%ld\t%s\t1.0\t%s\n" % \
        #                 (base_level + (max_levels - instance.level),
        #                  instance.create, instance.destroy,
        #                  instance.get_color(), inst_name))
        # return base_level + max_levels


    def print_stats(self, verbose):
        # Compute total and average utilization of memory
        assert self.last_time is not None
        average_usage = 0.0
        max_usage = 0.0
        current_size = 0
        previous_time = 0
        for point in sorted(self.time_points,key=lambda p: p.time_key):
            # First do the math for the previous interval
            usage = float(current_size)/float(self.capacity) if self.capacity != 0 else 0
            if usage > max_usage:
                max_usage = usage
            duration = point.time - previous_time
            # Update the average usage
            average_usage += usage * float(duration)
            # Update the size
            if point.first:
                current_size += point.thing.size  
            else:
                current_size -= point.thing.size
            # Save the time for the next round through
            previous_time = point.time
        # Last interval is empty so don't worry about it
        average_usage /= float(self.last_time) 
        if average_usage > 0.0 or verbose:
            print(self)
            print("    Total Instances: %d" % len(self.instances))
            print("    Maximum Utilization: %.3f%%" % (100.0 * max_usage))
            print("    Average Utilization: %.3f%%" % (100.0 * average_usage))
            print()
  
    def __repr__(self):
        return '%s Memory %s' % (self.kind, hex(self.mem_id))

    def __cmp__(a, b):
        return a.mem_id - b.mem_id

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def __gt__(self, other):
        return self.__cmp__(other) > 0

class MemProcAffinity(object):
    __slots__ = ['mem', 'bandwidth', 'latency', 'best_proc_aff']
    def __init__(self, mem, proc, bandwidth, latency):
        self.mem = mem
        self.best_proc_aff = proc
        self.bandwidth = bandwidth
        self.latency = latency

    def update_best_affinity(self, bandwidth, latency, proc):
        if (bandwidth > self.bandwidth):
            self.best_proc_aff = proc
            self.bandwidth = bandwidth
            self.latency = latency

    def get_short_text(self):
        if memory_node_proc[self.mem.kind] == "None":
            return "[all n]"
        elif memory_node_proc[self.mem.kind] == "Node_id":
            return "[n" + str(self.mem.node_id) + "]" + memory_kinds_abbr[self.mem.kind]
        elif memory_node_proc[self.mem.kind] == "GPU_proc_id":
            return "[n" + str(self.best_proc_aff.node_id) + "][gpu" + str(self.best_proc_aff.proc_in_node) + "]" + memory_kinds_abbr[self.mem.kind]
        elif memory_node_proc[self.mem.kind] == "Proc_id":
            return "[n" + str(self.best_proc_aff.node_id) + "][cpu" + str(self.best_proc_aff.proc_in_node) + "]" + memory_kinds_abbr[self.mem.kind]
        else:
            return ""

class Channel(object):
    __slots__ = [
        'src', 'dst', 'copies', 'time_points', 'max_live_copies', 'last_time'
    ]
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.copies = set()
        self.time_points = list()
        self.max_live_copies = None 
        self.last_time = None

    def node_id(self):
        if self.src is not None and self.src.mem_id != 0:
            # MEMORY:      tag:8 = 0x1e, owner_node:16,   (unused):32, mem_idx: 8
            # owner_node = mem_id[55:40]
            # (mem_id >> 40) & ((1 << 16) - 1)
            return (self.src.mem_id >> 40) & ((1 << 16) - 1)
        elif self.dst is not None and self.dst.mem_id != 0:
            return (self.dst.mem_id >> 40) & ((1 << 16) - 1)
        else:
            return None

    def node_id_src(self):
        if self.src is not None and self.src.mem_id != 0:
            # MEMORY:      tag:8 = 0x1e, owner_node:16,   (unused):32, mem_idx: 8
            # owner_node = mem_id[55:40]
            # (mem_id >> 40) & ((1 << 16) - 1)
            return (self.src.mem_id >> 40) & ((1 << 16) - 1)
        else:
            return None
    # mem_idx: 8
    def mem_idx_str(self, mem):
        if mem is not None:
            if mem.mem_id == 0:
                return "[all n]"
            return str(mem.mem_id & 0xff)
        return "none"

    def node_idx_str(self, mem_id):
        if mem_id == 0:
            return "[all n]"
        return str((mem_id >> 40) & ((1 << 16) - 1))

    def mem_str(self, mem):
        if mem and mem.mem_id == 0:
            return "[all n]"
        elif mem and mem.affinity is not None:
            return mem.affinity.get_short_text()
        elif  mem and mem.affinity is None:
            return "[n" +self.node_idx_str(mem.mem_id) + "] unknown " + self.mem_idx_str(mem)
        assert False

    def node_id_dst(self):
        if self.dst is not None and self.dst.mem_id != 0:
            # MEMORY:      tag:8 = 0x1e, owner_node:16,   (unused):32, mem_idx: 8
            # owner_node = mem_id[55:40]
            # (mem_id >> 40) & ((1 << 16) - 1)
            return (self.dst.mem_id >> 40) & ((1 << 16) - 1)
        else:
            return None

    def get_short_text(self):
        if self.dst is None and self.src is None:
            return "Dependent Partition Channel"
        # fill channel
        elif self.src is None:
            if self.dst.affinity is not None:
                return self.dst.affinity.get_short_text()
            else:
                return "Fill Channel"
        # normal channels
        elif self.src is not None and self.dst is not None:
            return self.mem_str(self.src) + " to " + self.mem_str(self.dst)
        else:
            assert False

    def add_copy(self, copy):
        copy.chan = self
        self.copies.add(copy)

    def init_time_range(self, last_time):
        self.last_time = last_time

    def trim_time_range(self, start, stop):
        trimmed_copies = set()
        for copy in self.copies:
            if copy.trim_time_range(start, stop):
                trimmed_copies.add(copy)
        self.copies = trimmed_copies 

    def sort_time_range(self):
        self.max_live_copies = 0 
        for copy in self.copies:
            self.time_points.append(TimePoint(copy.start, copy, True, 0))
            self.time_points.append(TimePoint(copy.stop, copy, False, 0))
        # Keep track of which levels are free
        self.time_points.sort(key=lambda p: p.time_key)
        free_levels = set()
        # Iterate over all the points in sorted order
        for point in self.time_points:
            if point.first:
                if len(free_levels) > 0:
                    point.thing.level = min(free_levels)
                    free_levels.remove(point.thing.level)
                else:
                    point.thing.level = self.max_live_copies + 1
                    self.max_live_copies += 1
            else:
                # Finishing this instance so restore its point
                free_levels.add(point.thing.level)

    def emit_tsv(self, tsv_file, base_level):
        max_levels = self.max_live_copies + 1
        if max_levels > 1:
            # iterate over tasks in start time order
            max_levels = max(4, max_levels)
            for point in self.time_points:
                if point.first:
                    point.thing.emit_tsv(tsv_file, base_level,\
                                         max_levels, None,
                                         point.thing.level, None)

        return base_level + max_levels

        # max_levels = self.max_live_copies + 1
        # if max_levels > 1:
        #     max_levels = max(4, max_levels)
        #     for copy in self.copies:
        #         assert copy.level is not None
        #         assert copy.start is not None
        #         assert copy.stop is not None
        #         copy_name = repr(copy)
        #         tsv_file.write("%d\t%ld\t%ld\t%s\t1.0\t%s\n" % \
        #                 (base_level + (max_levels - copy.level),
        #                  copy.start, copy.stop,
        #                  copy.get_color(), copy_name))
        # return base_level + max_levels

    def print_stats(self, verbose):
        assert self.last_time is not None 
        total_usage_time = 0
        max_transfers = 0
        current_transfers = 0
        previous_time = 0
        for point in sorted(self.time_points,key=lambda p: p.time_key):
            if point.first:
                if current_transfers == 0:
                    previous_time = point.time
                current_transfers += 1
                if current_transfers > max_transfers:
                    max_transfers = current_transfers
            else:
                current_transfers -= 1
                if current_transfers == 0:
                    total_usage_time += (point.time - previous_time)
        average_usage = float(total_usage_time)/float(self.last_time)
        if average_usage > 0.0 or verbose:
            print(self)
            print("    Total Transfers: %d" % len(self.copies))
            print("    Maximum Executing Transfers: %d" % (max_transfers))
            print("    Average Utilization: %.3f%%" % (100.0 * average_usage))
            print()
        
    def __repr__(self):
        if self.src is None and self.dst is None:
            return 'Dependent Partition Channel'
        if self.src is None:
            return 'Fill ' + self.dst.__repr__() + ' Channel'
        else:
            return self.src.__repr__() + ' to ' + self.dst.__repr__() + ' Channel'

    def __cmp__(a, b):
        if a.src:
            if b.src:
                x = a.src.__cmp__(b.src)
                if x != 0:
                    return x
                if a.dst:
                    if b.dst:
                        return a.dst.__cmp__(b.dst)
                    else:
                        return 1
                else:
                    if b.dst:
                        return -1
                    else:
                        return 0
            else:
                return 1
        else:
            if b.src:
                return -1
            else:
                if a.dst:
                    if b.dst:
                        return a.dst.__cmp__(b.dst)
                    else:
                        return 1
                else:
                    if b.dst:
                        return -1
                    else:
                        return 0

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def __gt__(self, other):
        return self.__cmp__(other) > 0

class WaitInterval(object):
    __slots__ = ['start', 'ready', 'end']
    def __init__(self, start, ready, end):
        self.start = start
        self.ready = ready
        self.end = end

class TaskKind(object):
    __slots__ = ['task_id', 'name']
    def __init__(self, task_id, name):
        self.task_id = task_id
        self.name = name

    def __repr__(self):
        return self.name

class StatObject(object):
    __slots__ = [
        'total_calls', 'total_execution_time', 'all_calls', 'max_call',
        'min_call'
    ]
    def __init__(self):
        self.total_calls = collections.defaultdict(int)
        self.total_execution_time = collections.defaultdict(int)
        self.all_calls = collections.defaultdict(list)
        self.max_call = collections.defaultdict(int)
        self.min_call = collections.defaultdict(lambda: sys.maxsize)

    def get_total_execution_time(self):
        total_execution_time = 0
        for proc_exec_time in itervalues(self.total_execution_time):
            total_execution_time += proc_exec_time
        return total_execution_time

    def get_total_calls(self):
        total_calls = 0
        for proc_calls in itervalues(self.total_calls):
            total_calls += proc_calls
        return total_calls

    def increment_calls(self, exec_time, proc):
        self.total_calls[proc] += 1
        self.total_execution_time[proc] += exec_time
        self.all_calls[proc].append(exec_time)
        if exec_time > self.max_call[proc]:
            self.max_call[proc] = exec_time
        if exec_time < self.min_call[proc]:
            self.min_call[proc] = exec_time

    def print_task_stat(self, total_calls, total_execution_time,
            max_call, max_dev, min_call, min_dev):
        avg = float(total_execution_time) / float(total_calls) \
                if total_calls > 0 else 0
        print('       Total Invocations: %d' % total_calls)
        print('       Total Time: %d us' % total_execution_time)
        print('       Average Time: %.2f us' % avg)
        print('       Maximum Time: %d us (%.3f sig)' % (max_call,max_dev))
        print('       Minimum Time: %d us (%.3f sig)' % (min_call,min_dev))

    def print_stats(self, verbose):
        procs = sorted(iterkeys(self.total_calls))
        total_execution_time = self.get_total_execution_time()
        total_calls = self.get_total_calls()

        avg = float(total_execution_time) / float(total_calls)
        max_call = max(self.max_call.values())
        min_call = min(self.min_call.values())
        stddev = 0
        for proc_calls in self.all_calls.values():
            for call in proc_calls:
                diff = float(call) - avg
                stddev += math.sqrt(diff * diff)
        stddev /= float(total_calls)
        stddev = math.sqrt(stddev)
        max_dev = (float(max_call) - avg) / stddev if stddev != 0.0 else 0.0
        min_dev = (float(min_call) - avg) / stddev if stddev != 0.0 else 0.0

        print('  '+repr(self))
        self.print_task_stat(total_calls, total_execution_time,
                max_call, max_dev, min_call, min_dev)
        print()

        if verbose and len(procs) > 1:
            for proc in sorted(iterkeys(self.total_calls)):
                avg = float(self.total_execution_time[proc]) / float(self.total_calls[proc]) \
                        if self.total_calls[proc] > 0 else 0
                stddev = 0
                for call in self.all_calls[proc]:
                    diff = float(call) - avg
                    stddev += math.sqrt(diff * diff)
                stddev /= float(self.total_calls[proc])
                stddev = math.sqrt(stddev)
                max_dev = (float(self.max_call[proc]) - avg) / stddev if stddev != 0.0 else 0.0
                min_dev = (float(self.min_call[proc]) - avg) / stddev if stddev != 0.0 else 0.0

                print('    On ' + repr(proc))
                self.print_task_stat(self.total_calls[proc],
                        self.total_execution_time[proc],
                        self.max_call[proc], max_dev,
                        self.min_call[proc], min_dev)
                print()

class Variant(StatObject):
    __slots__ = ['variant_id', 'name', 'ops', 'task_kind', 'color', 'message', 'ordered_vc']
    def __init__(self, variant_id, name, message = False, ordered_vc= False):
        StatObject.__init__(self)
        self.variant_id = variant_id
        self.name = name
        # For task variants this dictionary is from op_id -> Task
        # For meta-task variants, this diction is from op_id -> list[MetaTask]
        self.ops = dict()
        self.task_kind = None
        self.color = None
        self.message = message
        self.ordered_vc = ordered_vc

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.variant_id == other.variant_id

    def set_task_kind(self, task_kind):
        assert self.task_kind == None or self.task_kind == task_kind
        self.task_kind = task_kind

    def compute_color(self, step, num_steps):
        assert self.color is None
        self.color = color_helper(step, num_steps)

    def assign_color(self, color):
        assert self.color is None
        self.color = color

    def __repr__(self):
        if self.task_kind:
            title = self.task_kind.name
            if self.name != None and self.name != self.task_kind.name:
                title += ' ['+self.name+']'
        else:
            if self.name != None:
                title = self.name
            else:
                title = ""
        return title

class Base(object):
    __slots__ = ['prof_uid', 'level', 'level_ready']
    def __init__(self):
        self.prof_uid = get_prof_uid()
        self.level = None
        self.level_ready = None

    def set_level(self, level):
        self.level = level

    def set_level_ready(self, level):
        self.level_ready = level

    def get_unique_tuple(self):
        assert self.proc is not None
        cur_level = self.proc.max_levels - self.level
        return (self.proc.node_id, self.proc.proc_in_node, self.prof_uid)

    def get_owner(self):
        return self.proc

class Operation(Base):
    __slots__ = [
        'op_id', 'kind_num', 'kind', 'is_task', 'is_meta', 'is_multi',
        'is_proftask', 'name', 'variant', 'task_kind', 'color', 'owner', 'proc',
        'parent_id', 'provenance'
    ]
    def __init__(self, op_id):
        Base.__init__(self)
        self.op_id = op_id
        self.kind_num = None
        self.kind = None
        self.is_task = False
        self.is_meta = False
        self.is_multi = False
        self.is_proftask = False
        self.name = 'Operation '+str(op_id)
        self.variant = None
        self.task_kind = None
        self.color = None
        self.owner = None
        self.proc = None
        self.parent_id = -1
        self.provenance = None

    def assign_color(self, color_map):
        assert self.color is None
        if self.kind is None:
            self.color = '#000000' # Black
        else:
            assert self.kind_num in color_map
            self.color = color_map[self.kind_num]

    def get_color(self):
        assert self.color is not None
        return self.color

    def get_op_id(self):
        if self.is_proftask:
            return ""
        else:
            return self.op_id

    def get_info(self):
        info = '<'+str(self.op_id)+">"
        return info

    def is_trimmed(self):
        if isinstance(self, TimeRange):
            return TimeRange.is_trimmed(self)
        return False

    def __repr__(self):
        if self.is_task:
            assert self.variant is not None
            title = self.variant.task_kind.name if self.variant.task_kind is not None else 'unnamed'
            if self.variant.name != None and self.variant.name.find("unnamed") > 0:
                title += ' ['+self.variant.name+']'
            return title+' '+self.get_info()
        elif self.is_multi:
            assert self.task_kind is not None
            if self.task_kind.name is not None:
                return self.task_kind.name+' '+self.get_info()
            else:
                return 'Task '+str(self.task_kind.task_id)+' '+self.get_info()
        elif self.is_proftask:
            return 'ProfTask' + (' <{:d}>'.format(self.op_id) if self.op_id >= 0 else '')
        else:
            if self.kind is None:
                return 'Operation '+self.get_info()
            else:
                return self.kind+' Operation '+self.get_info()

class HasWaiters(object):
    __slots__ = []
    _abstract_slots = ['wait_intervals']
    def __init__(self):
        self.wait_intervals = list()

    def add_wait_interval(self, start, ready, end):
        self.wait_intervals.append(WaitInterval(start, ready, end))

    def active_time(self):
        active_time = 0
        start = self.start
        for wait_interval in self.wait_intervals:
            active_time += (wait_interval.start - start)
            start = max(start, wait_interval.end)
        if start < self.stop:
            active_time += (self.stop - start)
        return active_time

    def emit_tsv(self, tsv_file, base_level, max_levels, max_levels_ready,
                 level, level_ready, op_id=None):
        title = repr(self)
        initiation = str(self.initiation)
        color = self.get_color()

        _in = json.dumps(list(self.deps["in"])) if len(self.deps["in"]) > 0 else ""
        out = json.dumps(list(self.deps["out"])) if len(self.deps["out"]) > 0 else ""
        children = json.dumps(list(self.deps["children"])) if len(self.deps["children"]) > 0 else ""
        parents = json.dumps(list(self.deps["parents"])) if len(self.deps["parents"]) > 0 else ""
        if (level_ready != None):
            l_ready = base_level + (max_levels_ready - level_ready);
        else:
            l_ready = None;
        if len(self.wait_intervals) > 0:
            start = self.start
            cur_level = base_level + (max_levels - level)
            for wait_interval in self.wait_intervals:
                init = data_tsv_str(level = cur_level,
                                    level_ready = l_ready,
                                    ready = start,
                                    start = start,
                                    end = wait_interval.start,
                                    color = color,
                                    opacity = "1.0",
                                    title = title,
                                    initiation = initiation,
                                    _in = _in,
                                    out = out,
                                    children = children,
                                    parents = parents,
                                    prof_uid = self.prof_uid,
                                    op_id = op_id)
                # only write once
                _in = ""
                out = ""
                children = ""
                parents = ""
                wait = data_tsv_str(level = cur_level,
                                    level_ready = l_ready,
                                    ready = wait_interval.start,
                                    start = wait_interval.start,
                                    end = wait_interval.ready,
                                    color = color,
                                    opacity = "0.15",
                                    title = title + " (waiting)",
                                    initiation = initiation,
                                    _in = "",
                                    out = "",
                                    children = "",
                                    parents = "",
                                    prof_uid = self.prof_uid,
                                    op_id = op_id)
                ready = data_tsv_str(level = cur_level,
                                     level_ready = l_ready,
                                     ready = wait_interval.ready,
                                     start = wait_interval.ready,
                                     end = wait_interval.end,
                                     color = color,
                                     opacity = "0.45",
                                     title = title + " (ready)",
                                     initiation = initiation,
                                     _in = "",
                                     out = "",
                                     children = "",
                                     parents = "",
                                     prof_uid = self.prof_uid,
                                     op_id = op_id)

                tsv_file.write(init)
                tsv_file.write(wait)
                tsv_file.write(ready)
                start = max(start, wait_interval.end)
            if start < self.stop:
                end = data_tsv_str(level = cur_level,
                                   level_ready = l_ready,
                                   ready = start,
                                   start = start,
                                   end = self.stop,
                                   color = color,
                                   opacity = "1.0",
                                   title = title,
                                   initiation = initiation,
                                   _in = "",
                                   out = "",
                                   children = "",
                                   parents = "",
                                   prof_uid = self.prof_uid,
                                   op_id = op_id)

                tsv_file.write(end)
        else:
            if (level_ready != None):
                l_ready = base_level + (max_levels_ready - level_ready);
            else:
                l_ready = None;
            line = data_tsv_str(level = base_level + (max_levels - level),
                                level_ready = l_ready,
                                ready = self.ready,
                                start = self.start,
                                end = self.stop,
                                color = color,
                                opacity = "1.0",
                                title = title,
                                initiation = initiation,
                                _in = _in,
                                out = out,
                                children = children,
                                parents = parents,
                                prof_uid = self.prof_uid,
                                op_id = op_id)

            tsv_file.write(line)

class Task(Operation, TimeRange, HasDependencies, HasWaiters):
    __slots__ = TimeRange._abstract_slots + HasDependencies._abstract_slots + HasWaiters._abstract_slots + ['base_op', 'initiation', 'proc']
    def __init__(self, variant, op, create, ready, start, stop):
        Operation.__init__(self, op.op_id)
        HasDependencies.__init__(self)
        HasWaiters.__init__(self)
        TimeRange.__init__(self, create, ready, start, stop)

        self.base_op = op
        self.variant = variant
        self.initiation = ""
        self.is_task = True
        # make sure set the parent_id and provenance as we create a new task instance to replace the original operation
        self.parent_id = op.parent_id
        if op.provenance is not None:
            self.provenance = op.provenance

    def assign_color(self, color):
        assert self.color is None
        assert self.base_op.color is None
        assert self.variant is not None
        assert self.variant.color is not None
        self.color = self.variant.color
        self.base_op.color = self.color

    def emit_tsv(self, tsv_file, base_level, max_levels, max_levels_ready,
                 level, level_ready):
        # update the initiation
        self.initiation = self.parent_id
        return HasWaiters.emit_tsv(self, tsv_file, base_level, max_levels,
                                   max_levels_ready,
                                   level,
                                   level_ready,
                                   self.op_id)

    def get_color(self):
        assert self.color is not None
        return self.color

    def get_op_id(self):
        return self.op_id
        
    def get_info(self):
        info = '<'+str(self.op_id)+">"
        return info

    def active_time(self):
        return HasWaiters.active_time(self)

    def application_time(self):
        return self.total_time()

    def meta_time(self):
        return 0

    def mapper_time(self):
        return 0

    def __repr__(self):
        assert self.variant is not None
        # if self.provenance is not None:
        #   print(self.variant, self.op_id, self.provenance)
        return str(self.variant)+' '+self.get_info()

class MetaTask(Base, TimeRange, HasInitiationDependencies, HasWaiters):
    __slots__ = TimeRange._abstract_slots + HasInitiationDependencies._abstract_slots + HasWaiters._abstract_slots + ['variant', 'is_task', 'is_meta', 'proc']
    def __init__(self, variant, initiation_op, create, ready, start, stop):
        Base.__init__(self)
        HasInitiationDependencies.__init__(self, initiation_op)
        HasWaiters.__init__(self)
        TimeRange.__init__(self, create, ready, start, stop)

        self.variant = variant
        self.is_task = True
        self.is_meta = True

    def get_color(self):
        assert self.variant is not None
        assert self.variant.color is not None
        return self.variant.color

    def active_time(self):
        return HasWaiters.active_time(self)

    def application_time(self):
        return self.total_time()

    def meta_time(self):
        return 0

    def mapper_time(self):
        return 0

    def emit_tsv(self, tsv_file, base_level, max_levels, max_levels_ready,
                 level, level_ready):
        return HasWaiters.emit_tsv(self, tsv_file, base_level, max_levels,
                                   max_levels_ready,
                                   level,
                                   level_ready)

    def __repr__(self):
        assert self.variant is not None
        return self.variant.name

class ProfTask(Base, TimeRange, HasNoDependencies):
    __slots__ = TimeRange._abstract_slots + HasNoDependencies._abstract_slots + ['proftask_id', 'color', 'is_task', 'proc']
    def __init__(self, op_id, create, ready, start, stop):
        Base.__init__(self)
        HasNoDependencies.__init__(self)
        TimeRange.__init__(self, None, ready, start, stop)
        self.proftask_id = op_id
        self.color = '#ffc0cb'  # Pink
        self.is_task = True

    def get_color(self):
        return self.color

    def active_time(self):
        return self.total_time()

    def application_time(self):
        return 0

    def meta_time(self):
        return self.total_time()

    def mapper_time(self):
        return 0

    def emit_tsv(self, tsv_file, base_level, max_levels, max_levels_ready,
                 level, level_ready):
        if (level_ready != None):
            l_ready = base_level + (max_levels_ready - level_ready);
        else:
            l_ready = None;
        tsv_line = data_tsv_str(level = base_level + (max_levels - level),
                                level_ready = l_ready,
                                ready = self.start,
                                start = self.start,
                                end = self.stop,
                                color = self.get_color(),
                                opacity = "1.0",
                                title = repr(self),
                                initiation = None,
                                _in = None,
                                out = None,
                                children = None,
                                parents = None,
                                prof_uid = self.prof_uid,
                                op_id = self.proftask_id)
        tsv_file.write(tsv_line)

    def __repr__(self):
        return 'ProfTask' + (' <{:d}>'.format(self.proftask_id) if self.proftask_id >= 0 else '')

class UserMarker(Base, TimeRange, HasNoDependencies):
    __slots__ = TimeRange._abstract_slots + HasNoDependencies._abstract_slots + ['name', 'color', 'is_task']
    def __init__(self, name, start, stop):
        Base.__init__(self)
        HasNoDependencies.__init__(self)
        TimeRange.__init__(self, None, None, start, stop)
        self.name = name
        self.color = '#000000' # Black
        self.is_task = True

    def get_color(self):
        return self.color

    def active_time(self):
        return self.total_time()

    def application_time(self):
        return 0

    def meta_time(self):
        return self.total_time()

    def mapper_time(self):
        return 0

    def emit_tsv(self, tsv_file, base_level, max_levels, max_levels_ready,
                 level, level_ready):
        if (level_ready != None):
            l_ready = base_level + (max_levels_ready - level_ready);
        else:
            l_ready = None;
        tsv_line = data_tsv_str(level = base_level + (max_levels - level),
                                level_ready = l_ready,
                                ready = None,
                                start = self.start,
                                end = self.stop,
                                color = self.get_color(),
                                opacity = "1.0",
                                title = repr(self),
                                initiation = None,
                                _in = None,
                                out = None,
                                children = None,
                                parents = None,
                                prof_uid = self.prof_uid)
        tsv_file.write(tsv_line)

    def __repr__(self):
        return 'User Marker "'+self.name+'"'

class CopyInfo(object):
    __slots__ = [
        'src_inst', 'dst_inst', 'fevent', 'num_fields', 'request_type', 'num_hops'
        ]
    def __init__(self, src_inst, dst_inst, fevent, num_fields, request_type, num_hops):
        self.src_inst = src_inst
        self.dst_inst = dst_inst
        self.fevent = fevent
        self.num_fields = num_fields
        self.request_type = request_type
        self.num_hops = num_hops

    def get_short_text(self):
        return 'src_inst=%s, dst_inst=%s, fields=%s, type=%s, hops=%s' % (hex(self.src_inst), hex(self.dst_inst), self.num_fields, request[self.request_type], self.num_hops)

    def __repr__(self):
        return self.get_short_text()

class Copy(Base, TimeRange, HasInitiationDependencies):
    __slots__ = TimeRange._abstract_slots + HasInitiationDependencies._abstract_slots + ['src', 'dst', 'size', 'chan', 'fevent', 'src_inst', 'dst_inst', 'num_requests', 'copy_info']
    def __init__(self, src, dst, initiation_op, size, create, ready, start, stop, fevent, num_requests):
        Base.__init__(self)
        HasInitiationDependencies.__init__(self, initiation_op)
        TimeRange.__init__(self, create, ready, start, stop)
        self.src = src
        self.dst = dst
        self.size = size
        self.chan = None
        self.fevent = fevent
        self.num_requests = num_requests
        self.copy_info = list()

    def add_copy_info(self, entry):
        self.copy_info.append(entry)

    def get_owner(self):
        return self.chan

    def get_color(self):
        # Get the color from the initiator
        return self.initiation_op.get_color()

    def __repr__(self):
        val =  'size='+ size_pretty(self.size) + ', num reqs=' + str(len(self.copy_info))
        cnt = 0
        for node in self.copy_info:
            val = val + '$req[' + str(cnt) + ']: ' +  node.get_short_text()
            cnt = cnt+1
        return val

    def get_unique_tuple(self):
        assert self.chan is not None
        cur_level = self.chan.max_live_copies+1 - self.level
        return (str(self.chan), self.prof_uid)

    def emit_tsv(self, tsv_file, base_level, max_levels, max_levels_ready,
                 level, level_ready):
        assert self.level is not None
        assert self.start is not None
        assert self.stop is not None
        copy_name = repr(self)
        _in = json.dumps(self.deps["in"]) if len(self.deps["in"]) > 0 else ""
        out = json.dumps(self.deps["out"]) if len(self.deps["out"]) > 0 else ""
        children = json.dumps(list(self.deps["children"])) if len(self.deps["children"]) > 0 else ""
        parents = json.dumps(list(self.deps["parents"])) if len(self.deps["parents"]) > 0 else ""

        tsv_line = data_tsv_str(level = base_level + (max_levels - level),
                                level_ready = None,
                                ready = None,
                                start = self.start,
                                end = self.stop,
                                color = self.get_color(),
                                opacity = "1.0",
                                title = copy_name,
                                initiation = self.initiation,
                                _in = _in,
                                out = out,
                                children = children,
                                parents = parents,
                                prof_uid = self.prof_uid)
        tsv_file.write(tsv_line)

class Fill(Base, TimeRange, HasInitiationDependencies):
    __slots__ = TimeRange._abstract_slots + HasInitiationDependencies._abstract_slots + ['dst', 'chan']
    def __init__(self, dst, initiation_op, create, ready, start, stop):
        Base.__init__(self)
        HasInitiationDependencies.__init__(self, initiation_op)
        TimeRange.__init__(self, create, ready, start, stop)
        self.dst = dst
        self.chan = None

    def get_owner(self):
        return self.chan

    def __repr__(self):
        return 'Fill'

    def get_unique_tuple(self):
        assert self.chan is not None
        cur_level = self.chan.max_live_copies+1 - self.level
        return (str(self.chan), self.prof_uid)

    def emit_tsv(self, tsv_file, base_level, max_levels, max_levels_ready,
                 level, level_ready):
        fill_name = repr(self)
        _in = json.dumps(self.deps["in"]) if len(self.deps["in"]) > 0 else ""
        out = json.dumps(self.deps["out"]) if len(self.deps["out"]) > 0 else ""
        children = json.dumps(list(self.deps["children"])) if len(self.deps["children"]) > 0 else ""
        parents = json.dumps(list(self.deps["parents"])) if len(self.deps["parents"]) > 0 else ""

        tsv_line = data_tsv_str(level = base_level + (max_levels - level),
                                level_ready = None,
                                ready = None,
                                start = self.start,
                                end = self.stop,
                                color = self.get_color(),
                                opacity = "1.0",
                                title = fill_name,
                                initiation = self.initiation,
                                _in = _in,
                                out = out,
                                children = children,
                                parents = parents,
                                prof_uid = self.prof_uid)
        tsv_file.write(tsv_line)

class DepPart(Base, TimeRange, HasInitiationDependencies):
    __slots__ = TimeRange._abstract_slots + HasInitiationDependencies._abstract_slots + ['part_op', 'chan']
    def __init__(self, part_op, initiation_op, create, ready, start, stop):
        Base.__init__(self)
        HasInitiationDependencies.__init__(self, initiation_op)
        TimeRange.__init__(self, create, ready, start, stop)
        self.part_op = part_op
        self.chan = None

    def __repr__(self):
        assert self.part_op in dep_part_kinds
        return dep_part_kinds[self.part_op]

    def get_owner(self):
        return self.chan

    def get_unique_tuple(self):
        assert self.chan is not None
        cur_level = self.chan.max_live_copies+1 - self.level
        return (str(self.chan), self.prof_uid)
        
    def emit_tsv(self, tsv_file, base_level, max_levels, max_levels_ready,
                 level, level_ready):
        deppart_name = repr(self)
        _in = json.dumps(self.deps["in"]) if len(self.deps["in"]) > 0 else ""
        out = json.dumps(self.deps["out"]) if len(self.deps["out"]) > 0 else ""
        children = json.dumps(list(self.deps["children"])) if len(self.deps["children"]) > 0 else ""
        parents = json.dumps(list(self.deps["parents"])) if len(self.deps["parents"]) > 0 else ""

        tsv_line = data_tsv_str(level = base_level + (max_levels - level),
                                level_ready = None,
                                ready = None,
                                start = self.start,
                                end = self.stop,
                                color = self.get_color(),
                                opacity = "1.0",
                                title = deppart_name,
                                initiation = self.initiation,
                                _in = _in,
                                out = out,
                                children = children,
                                parents = parents,
                                prof_uid = self.prof_uid)
        tsv_file.write(tsv_line)

class Instance(Base, TimeRange, HasInitiationDependencies):
    __slots__ = TimeRange._abstract_slots + HasInitiationDependencies._abstract_slots + [
        'inst_id', 'mem', 'size', 'ispace', 'fspace', 'tree_id', 'fields',
        'align_desc', 'dim_order_desc'
    ]
    def __init__(self, inst_id, initiation_op):
        Base.__init__(self)
        HasInitiationDependencies.__init__(self, initiation_op)
        TimeRange.__init__(self, None, None, None, None)

        self.inst_id = inst_id
        self.mem = None
        self.size = None
        self.ispace = []
        self.fspace = []
        self.tree_id = None
        self.fields = {}
        self.align_desc = {}
        self.dim_order_desc = []
    def get_owner(self):
        return self.mem

    def get_unique_tuple(self):
        assert self.mem is not None
        cur_level = self.mem.max_live_instances+1 - self.level
        return (str(self.mem), self.prof_uid)

    def emit_tsv(self, tsv_file, base_level, max_levels, max_levels_ready,
                 level, level_ready):
        assert self.level is not None
        assert self.start is not None
        assert self.stop is not None
        inst_name = repr(self)

        _in = json.dumps(self.deps["in"]) if len(self.deps["in"]) > 0 else ""
        out = json.dumps(self.deps["out"]) if len(self.deps["out"]) > 0 else ""
        children = json.dumps(list(self.deps["children"])) if len(self.deps["children"]) > 0 else ""
        parents = json.dumps(list(self.deps["parents"])) if len(self.deps["parents"]) > 0 else ""

        tsv_line = data_tsv_str(level = base_level + (max_levels - level),
                                level_ready = None,
                                ready = None,
                                start = self.start,
                                end = self.stop,
                                color = self.get_color(),
                                opacity = "1.0",
                                title = inst_name,
                                initiation = self.initiation,
                                _in = _in,
                                out = out,
                                children = children,
                                parents = parents,
                                prof_uid = self.prof_uid)

        tsv_file.write(tsv_line)

    def total_time(self):
        return self.stop - self.start

    def get_color(self):
        # Get the color from the operation
        return self.initiation_op.get_color()

    def __repr__(self):
        # Check to see if we got a profiling callback
        size_pr = size_pretty(self.size)
        output_str = ""
        for pos in range(0, len(self.ispace)):
            output_str = output_str + "Region: " + self.ispace[pos].get_short_text()
            output_str = output_str + " x " + str(self.fspace[pos])
            key = self.fspace[pos]
            fieldlist = []
            count = 0
            if key in self.fields:
                fieldlist = self.fields[key]
                alignlist = self.align_desc[key]
                for pos1 in range(0, len(fieldlist)):
                    f = fieldlist[pos1]
                    al = alignlist[pos1]
                    if (al.has_align):
                        align_str = ":align=" + str(al.align_desc)
                    else:
                        align_str = ""
                    if (pos1 - 1) % 5 == 0:
                        break_str = "$"
                    else:
                        break_str = ""
                    if count == 0:
                        output_str = output_str + '$Fields: [' + str(f) + align_str
                    else:
                        output_str = output_str + ',' + break_str + str(f) + align_str
                    count = count + 1
                if (count > 0):
                    output_str = output_str + ']'
            pend =len(self.ispace)-1
            if (pos != pend):
                output_str = output_str + '$'
        if (len(self.dim_order_desc) > 0):
            cmpx_order = False
            aos = False
            soa = False
            dim_f = 9
            column_major = 0
            row_major = 0
            dim_last = len(self.dim_order_desc)-1
            output_str = output_str + "$Layout Order: "
            for pos in range(0, len(self.dim_order_desc)):
                if pos == 0:
                    if self.dim_order_desc[pos] == dim_f:
                        aos = True
                else:
                    if pos == len(self.dim_order_desc) - 1:
                        if self.dim_order_desc[pos] == dim_f:
                            soa = True
                    else:
                        if self.dim_order_desc[pos] == dim_f:
                            cmpx_order = True
                #SOA + order -> DIM_X, DIM_Y,.. DIM_F-> column_major
                #or .. DIM_Y, DIM_X, DIM_F? -> row_major
                if self.dim_order_desc[dim_last] == dim_f:
                     if self.dim_order_desc[pos] != dim_f:
                         if self.dim_order_desc[pos] == pos:
                             column_major = column_major+1
                         if self.dim_order_desc[pos] == dim_last-pos-1:
                            row_major = row_major+1
                 #AOS + order -> DIM_F, DIM_X, DIM_Y -> column_major
                 # or DIM_F, DIM_Y, DIM_X -> row_major?
                if self.dim_order_desc[0] == dim_f:
                    if self.dim_order_desc[pos] != dim_f:
                        if self.dim_order_desc[pos] == pos-1:
                            column_major = column_major+1
                        if self.dim_order_desc[pos] == dim_last-pos:
                            row_major = row_major+1
            if dim_last == 1:
                output_str = output_str
            else:
                if column_major == dim_last and cmpx_order==False:
                    output_str = output_str  + "[Column Major]"
                else:
                    if row_major == dim_last and cmpx_order==False:
                        output_str = output_str  + "[Row Major]"
            if cmpx_order:
                for pos in range(0, len(self.dim_order_desc)):
                    output_str = output_str + "["
                    output_str = output_str + legion_dimension_kind_t[self.dim_order_desc[pos]]
                    output_str = output_str +  "]"
                    if (pos+1)%4 == 0 and pos != len(self.dim_order_desc)-1:
                        output_str = output_str + "$"
            else:
                if aos == True:
                    output_str = output_str + "[Array-of-structs (AOS)]"
                else:
                    if soa == True:
                        output_str = output_str + "[Struct-of-arrays (SOA)]"

        output_str = output_str + " $Inst: {} $Size: {}"
        return output_str.format(str(hex(self.inst_id)),size_pr)


class MapperCallKind(StatObject):
    __slots__ = ['mapper_call_kind', 'name', 'color']
    def __init__(self, mapper_call_kind, name):
        StatObject.__init__(self)
        self.mapper_call_kind = mapper_call_kind
        self.name = name
        self.color = None

    def __hash__(self):
        return hash(self.mapper_call_kind)

    def __eq__(self, other):
        return self.mapper_call_kind == other.mapper_call_kind

    def __repr__(self):
        return self.name

    def assign_color(self, color):
        assert self.color is None
        self.color = color

class MapperCall(Base, TimeRange, HasInitiationDependencies):
    __slots__ = TimeRange._abstract_slots + HasInitiationDependencies._abstract_slots + ['kind', 'proc']
    def __init__(self, kind, initiation_op, start, stop):
        Base.__init__(self)
        TimeRange.__init__(self, None, None, start, stop)
        HasInitiationDependencies.__init__(self, initiation_op)
        self.kind = kind

    def get_color(self):
        assert self.kind is not None and self.kind.color is not None
        return self.kind.color

    def emit_tsv(self, tsv_file, base_level, max_levels, max_levels_ready,
                 level, level_ready):
        title = repr(self)
        _in = json.dumps(list(self.deps["in"])) if len(self.deps["in"]) > 0 else ""
        out = json.dumps(list(self.deps["out"])) if len(self.deps["out"]) > 0 else ""
        children = json.dumps(list(self.deps["children"])) if len(self.deps["children"]) > 0 else ""
        parents = json.dumps(list(self.deps["parents"])) if len(self.deps["parents"]) > 0 else ""

        if (level_ready != None):
            l_ready = base_level + (max_levels_ready - level_ready);
        else:
            l_ready = None;
        tsv_line = data_tsv_str(level = base_level + (max_levels - level),
                                level_ready = l_ready,
                                ready = self.start,
                                start = self.start,
                                end = self.stop,
                                color = self.get_color(),
                                opacity = "1.0",
                                title = repr(self),
                                initiation = self.initiation,
                                _in = _in,
                                out = out,
                                children = children,
                                parents = parents,
                                prof_uid = self.prof_uid)

        tsv_file.write(tsv_line)

    def active_time(self):
        return self.total_time()

    def application_time(self):
        return 0

    def meta_time(self):
        return 0

    def mapper_time(self):
        return self.total_time()

    def __repr__(self):
        if self.initiation == 0:
            return 'Mapper Call '+str(self.kind)
        else:
            return 'Mapper Call '+str(self.kind)+' for '+str(self.initiation)

class RuntimeCallKind(StatObject):
    __slots__ = ['runtime_call_kind', 'name', 'color']
    def __init__(self, runtime_call_kind, name):
        StatObject.__init__(self)
        self.runtime_call_kind = runtime_call_kind
        self.name = name
        self.color = None

    def __eq__(self, other):
        return self.runtime_call_kind == other.runtime_call_kind

    def assign_color(self, color):
        assert self.color is None
        self.color = color

    def __repr__(self):
        return self.name

class Field(StatObject):
    __slots__ = ['unique_id', 'field_id', 'size', 'name']
    def __init__(self, unique_id, field_id, size, name):
        StatObject.__init__(self)
        self.unique_id = unique_id
        self.field_id = field_id
        self.size = size
        self.name = name

    def __repr__(self):
        if self.name != None:
            return self.name
        return 'fid:' + str(self.field_id)

class Align(StatObject):
    __slots__ = ['field_id', 'eqk', 'align_desc', 'has_align']
    def __init__(self, field_id, eqk, align_desc, has_align):
        StatObject.__init__(self)
        self.field_id = field_id
        self.eqk = eqk
        self.align_desc = align_desc
        self.has_align = has_align

    def __repr__(self):
        if self.has_align != False:
            return 'align:' + (self.align_desc)
        return ''

class FieldSpace(StatObject):
    __slots__ = ['fspace_id', 'name']
    def __init__(self, fspace_id, name):
        StatObject.__init__(self)
        self.fspace_id = fspace_id
        self.name = name

    def __repr__(self):
        if self.name != None:
            return self.name
        return 'fspace:' + str(self.fspace_id)

class LogicalRegion(StatObject):
    __slots__ = ['ispace_id', 'fspace_id', 'tree_id', 'name']
    def __init__(self, ispace_id, fspace_id, tree_id, name):
        StatObject.__init__(self)
        self.ispace_id = ispace_id
        self.fspace_id = fspace_id
        self.tree_id = tree_id
        self.name = name

    def __repr__(self):
        return self.name

class Partition(StatObject):
    __slots__ = ['unique_id', 'parent', 'disjoint', 'point', 'name']
    def __init__(self, unique_id, name):
        StatObject.__init__(self)
        self.unique_id = unique_id
        self.parent = None
        self.disjoint = None
        self.point = None
        self.name = name

    def __repr__(self):
        return self.name

    def get_short_text(self):
        if self.name != None:
            return self.name
        elif self.parent != None and self.parent.name != None:
            return self.parent.name
        else:
            return str(self.point)

    def set_parent(self, parent):
        self.parent = parent

    def set_disjoint(self, disjoint):
        self.disjoint = disjoint

    def set_point(self, point):
        self.point = point

class IndexSpace(StatObject):
    __slots__ = [
        'is_type', 'unique_id', 'dim', 'point', 'rect_lo', 'rect_hi',
        'name', 'parent', 'is_sparse', 'dense_size', 'sparse_size'
    ]
    def __init__(self, is_type, unique_id, dim, values, max_dim):
        StatObject.__init__(self)
        self.is_type = is_type
        self.unique_id = unique_id
        self.dim = dim
        self.point = []
        self.rect_lo = []
        self.rect_hi = []
        self.is_sparse = False
        self.dense_size = 0
        self.sparse_size = 0

        if (self.is_type == 0):
            for index in range(self.dim):
                self.point.append(int(values[index]))
        if (self.is_type == 1):
            for index in range(self.dim):
                self.rect_lo.append(int(values[index]))
                self.rect_hi.append(int(values[max_dim + index]))

        self.name = None
        self.parent = None


    def set_vals(self, is_type, unique_id, dim, values, max_dim):
        self.is_type = is_type
        self.unique_id = unique_id
        self.dim = dim
        self.point = []
        self.rect_lo = []
        self.rect_hi = []
        if (self.is_type == 0):
            for index in range(self.dim):
                self.point.append(int(values[index]))
        if (self.is_type == 1):
            for index in range(self.dim):
                self.rect_lo.append(int(values[index]))
                self.rect_hi.append(int(values[max_dim + index]))


    def setPoint(self, unique_id, dim, values):
        is_type = 0
        self.set_vals(is_type, unique_id, dim, values,0)

    def setRect(self, unique_id, dim, values, max_dim):
        is_type = 1
        self.set_vals(is_type, unique_id, dim, values, max_dim)

    def setEmpty(self, unique_id):
        is_type = 2
        self.set_vals(is_type, unique_id, None, None, 0)

    def set_size(self, dense_size, sparse_size, is_sparse):
        self.dense_size = dense_size
        self.sparse_size = sparse_size
        self.is_sparse = is_sparse

    @classmethod
    def forPoint(cls, unique_id, dim, values):
        is_type = 0
        return cls(is_type, unique_id, dim, values,0)

    @classmethod
    def forRect(cls, unique_id, dim, values, max_dim):
        is_type = 1
        return cls(is_type, unique_id, dim, values, max_dim)

    @classmethod
    def forEmpty(cls, unique_id):
        is_type = 2
        return cls(is_type, unique_id, None, None, 0)

    @classmethod
    def forUnknown(cls, unique_id):
        return cls(None, unique_id, None, None, 0)

    def set_name(self, name):
        self.name = name

    def set_parent(self, parent):
        self.parent = parent

    def __repr__(self):
        return 'Index Space '+ str(self.name)

    def get_short_text(self):
        stext = ""
        if self.name != None:
            stext = self.name
        elif self.parent != None and self.parent.parent != None and self.parent.parent.name != None:
            stext = self.parent.parent.name
        elif self.parent != None and self.parent.parent != None:
            stext = 'ispace:' + str(self.parent.parent.unique_id)
        else:
            stext = 'ispace:' + str(self.unique_id)
        if (self.is_type == None):
            return stext
        if self.is_sparse == True:
            stext = stext + "[sparse:("  + str(self.sparse_size) + " of " + str(self.dense_size) + " points)]"
            return stext
        if (self.is_type == 0):
            for index in range(self.dim):
                stext = stext + '[' + str(self.point[index]) + ']'
        if (self.is_type == 1):
            for index in range(self.dim):
                stext = stext + '[' + str(self.rect_lo[index]) + ':' + str(self.rect_hi[index]) + ']'
        if (self.is_type == 2):
            stext = 'empty index space'
        return stext

class RuntimeCall(Base, TimeRange, HasNoDependencies):
    __slots__ = TimeRange._abstract_slots + HasNoDependencies._abstract_slots + ['kind']
    def __init__(self, kind, start, stop):
        Base.__init__(self)
        TimeRange.__init__(self, None, None, start, stop)
        HasNoDependencies.__init__(self)
        self.kind = kind

    def get_color(self):
        assert self.kind.color is not None
        return self.kind.color

    def emit_tsv(self, tsv_file, base_level, max_levels, max_levels_ready,
                 level, level_ready):
        if (level_ready != None):
            l_ready = base_level + (max_levels_ready - level_ready);
        else:
            l_ready = None;
        tsv_line = data_tsv_str(level = base_level + (max_levels - level),
                                level_ready = l_ready,
                                ready = None,
                                start = self.start,
                                end = self.stop,
                                color = self.get_color(),
                                opacity = "1.0",
                                title = repr(self),
                                initiation = None,
                                _in = None,
                                out = None,
                                children = None,
                                parents = None,
                                prof_uid = self.prof_uid)

        tsv_file.write(tsv_line)

    def active_time(self):
        return self.total_time()

    def application_time(self):
        return 0

    def meta_time(self):
        return self.total_time()

    def mapper_time(self):
        return 0

    def __repr__(self):
        return 'Runtime Call '+str(self.kind)

class LFSR(object):
    __slots__ = ['register', 'max_value', 'taps']
    def __init__(self, size):
        self.register = ''
        # Initialize the register with all zeros
        needed_bits = int(math.log(size,2))+1
        self.max_value = pow(2,needed_bits)
        # We'll use a deterministic seed here so that
        # our results are repeatable
        seed_configuration = '1010010011110011'
        for i in range(needed_bits):
            self.register += seed_configuration[i]
        polynomials = {
          2 : (2,1),
          3 : (3,2),
          4 : (4,3),
          5 : (5,3),
          6 : (6,5),
          7 : (7,6),
          8 : (8,6,5,4),
          9 : (9,5),
          10 : (10,7),
          11 : (11,9),
          12 : (12,11,10,4),
          13 : (13,12,11,8),
          14 : (14,13,12,2),
          15 : (15,14),
          16 : (16,14,13,11),
        }
        # If we need more than 16 bits that is a lot tasks
        assert needed_bits in polynomials
        self.taps = polynomials[needed_bits]

    def get_max_value(self):
        return self.max_value
        
    def get_next(self):
        xor = 0
        for t in self.taps:
            xor += int(self.register[t-1])
        if xor % 2 == 0:
            xor = 0
        else:
            xor = 1
        self.register = str(xor) + self.register[:-1]
        return int(self.register,2)

class StatGatherer(object):
    __slots__ = [
        'state', 'application_tasks', 'meta_tasks', 'mapper_tasks',
        'runtime_tasks'
    ]
    def __init__(self, state):
        self.state = state
        self.application_tasks = set()
        self.meta_tasks = set()
        self.mapper_tasks = set()
        self.runtime_tasks = set()
        for proc in itervalues(state.processors):
            for task in proc.tasks:
                if isinstance(task, Task):
                    self.application_tasks.add(task.variant)
                    task.variant.increment_calls(task.total_time(), proc)
                elif isinstance(task, MetaTask):
                    self.meta_tasks.add(task.variant)
                    task.variant.increment_calls(task.total_time(), proc)
                elif isinstance(task, MapperCall):
                    self.mapper_tasks.add(task.kind)
                    task.kind.increment_calls(task.total_time(), proc)
                elif isinstance(task, RuntimeCall):
                    self.runtime_tasks.add(task.kind)
                    task.kind.increment_calls(task.total_time(), proc)

    def print_stats(self, verbose):
        print("  -------------------------")
        print("  Task Statistics")
        print("  -------------------------")
        for variant in sorted(self.application_tasks,
                                key=lambda v: v.get_total_execution_time(),
                                reverse=True):
            variant.print_stats(verbose)
        print("  -------------------------")
        print("  Meta-Task Statistics")
        print("  -------------------------")
        for variant in sorted(self.meta_tasks,
                                key=lambda v: v.get_total_execution_time(),
                                reverse=True):
            variant.print_stats(verbose)
        print("  -------------------------")
        print("  Mapper Statistics")
        print("  -------------------------")
        for kind in sorted(self.mapper_tasks,
                           key=lambda k: k.get_total_execution_time(),
                           reverse=True):
            kind.print_stats(verbose)

        if len(self.runtime_tasks) > 0:
            print("  -------------------------")
            print("  Runtime Statistics")
            print("  -------------------------")
            for kind in sorted(self.runtime_tasks,
                               key=lambda k: k.get_total_execution_time(),
                               reverse=True):
                kind.print_stats(verbose)


class State(object):
    __slots__ = [
        'max_dim', 'processors', 'memories', 'mem_proc_affinity', 'channels',
        'task_kinds', 'variants', 'meta_variants', 'op_kinds', 'operations',
        'prof_uid_map', 'multi_tasks', 'first_times', 'last_times',
        'last_time', 'mapper_call_kinds', 'mapper_calls', 'runtime_call_kinds', 
        'runtime_calls', 'instances', 'index_spaces', 'partitions', 'logical_regions', 
        'field_spaces', 'fields', 'has_spy_data', 'spy_state', 'callbacks', 'copy_map'
    ]
    def __init__(self):
        self.max_dim = 3
        self.processors = {}
        self.memories = {}
        self.mem_proc_affinity = {}
        self.channels = {}
        self.task_kinds = {}
        self.variants = {}
        self.meta_variants = {}
        self.op_kinds = {}
        self.operations = {}
        self.prof_uid_map = {}
        self.multi_tasks = {}
        self.first_times = {}
        self.last_times = {}
        self.last_time = 0
        self.mapper_call_kinds = {}
        self.mapper_calls = {}
        self.runtime_call_kinds = {}
        self.runtime_calls = {}
        self.instances = {}
        self.index_spaces = {}
        self.partitions = {}
        self.logical_regions = {}
        self.field_spaces = {}
        self.fields = {}
        self.copy_map = {}
        self.has_spy_data = False
        self.spy_state = None
        self.callbacks = {
            "MapperCallDesc": self.log_mapper_call_desc,
            "RuntimeCallDesc": self.log_runtime_call_desc,
            "MetaDesc": self.log_meta_desc,
            "OpDesc": self.log_op_desc,
            "ProcDesc": self.log_proc_desc,
            "MemDesc": self.log_mem_desc,
            "TaskKind": self.log_kind,
            "TaskVariant": self.log_variant,
            "OperationInstance": self.log_operation,
            "MultiTask": self.log_multi,
            "SliceOwner": self.log_slice_owner,
            "TaskWaitInfo": self.log_task_wait_info,
            "MetaWaitInfo": self.log_meta_wait_info,
            "TaskInfo": self.log_task_info,
            "GPUTaskInfo": self.log_gpu_task_info,
            "MetaInfo": self.log_meta_info,
            "CopyInfo": self.log_copy_info,
            "FillInfo": self.log_fill_info,
            "InstCreateInfo": self.log_inst_create,
            "InstUsageInfo": self.log_inst_usage,
            "InstTimelineInfo": self.log_inst_timeline,
            "PartitionInfo": self.log_partition_info,
            "MapperCallInfo": self.log_mapper_call_info,
            "RuntimeCallInfo": self.log_runtime_call_info,
            "ProfTaskInfo": self.log_proftask_info,
            "ProcMDesc": self.log_mem_proc_affinity_desc,
            "IndexSpacePointDesc": self.log_index_space_point_desc,
            "IndexSpaceRectDesc": self.log_index_space_rect_desc,
            "PartDesc": self.log_index_part_desc,
            "IndexPartitionDesc": self.log_index_partition_desc,
            "IndexSpaceEmptyDesc": self.log_index_space_empty_desc,
            "FieldDesc": self.log_field_desc,
            "FieldSpaceDesc": self.log_field_space_desc,
            "IndexSpaceDesc": self.log_index_space_desc,
            "IndexSubSpaceDesc": self.log_index_subspace_desc,
            "LogicalRegionDesc": self.log_logical_region_desc,
            "PhysicalInstRegionDesc": self.log_physical_inst_region_desc,
            "PhysicalInstLayoutDesc": self.log_physical_inst_layout_desc,
            "PhysicalInstDimOrderDesc": self.log_physical_inst_layout_dim_desc,
            "IndexSpaceSizeDesc": self.log_index_space_size_desc,
            "MaxDimDesc": self.log_max_dim,
            "CopyInstInfo": self.log_copy_inst_info
            #"UserInfo": self.log_user_info
        }

    def log_max_dim(self, max_dim):
        self.max_dim = max_dim

    def log_index_space_point_desc(self, unique_id, dim, rem):
        index_space = self.create_index_space_point(unique_id, dim, rem)

    def log_index_space_rect_desc(self, unique_id, dim, rem):
        index_space = self.create_index_space_rect(unique_id, dim, rem)

    def log_index_space_empty_desc(self, unique_id):
        index_space = self.create_index_space_empty(unique_id)

    def log_index_space_desc(self, unique_id, name):
        index_space = self.find_index_space(unique_id)
        index_space.set_name(name)

    def log_logical_region_desc(self, ispace_id, fspace_id, tree_id, name):
        logical_region = self.create_logical_region(ispace_id, fspace_id, tree_id, name)

    def log_field_space_desc(self, unique_id, name):
        field_space = self.create_field_space(unique_id, name)

    def log_field_desc(self, unique_id, field_id, size, name):
        field = self.create_field(unique_id, field_id, size, name)

    def log_index_part_desc(self, unique_id, name):
        part = self.create_partition(unique_id, name)

    def log_index_partition_desc(self,parent_id, unique_id, disjoint, point0):
        part = self.find_partition(unique_id)
        part.parent = self.find_index_space(parent_id)
        part.disjoint = disjoint
        part.point = point0

    def log_index_subspace_desc(self, parent_id, unique_id):
        index_space = self.find_index_space(unique_id)
        index_part_parent = self.find_partition(parent_id)
        index_space.set_parent(index_part_parent)

    def log_physical_inst_region_desc(self, op_id, inst_id, ispace_id, fspace_id, tree_id):
        op = self.find_op(op_id)
        inst = self.create_instance(inst_id, op)
        fspace = self.find_field_space(fspace_id)
        inst.ispace.append(self.find_index_space(ispace_id))
        inst.fspace.append(fspace)
        if fspace not in inst.fields:
            inst.fields[fspace] = []
            inst.align_desc[fspace] = []
        inst.tree_id = tree_id

    def log_index_space_size_desc(self, unique_id, dense_size, sparse_size, is_sparse):
        is_sparse = bool(is_sparse)
        index_space = self.find_index_space(unique_id)
        index_space.set_size(dense_size, sparse_size, is_sparse)

    def log_physical_inst_layout_dim_desc(self, op_id, inst_id, dim, dim_kind):
        op = self.find_op(op_id)
        inst = self.create_instance(inst_id, op)
        inst.dim_order_desc.insert(dim, dim_kind)


    def log_physical_inst_layout_desc(self, op_id, inst_id, field_id, fspace_id,
                                      has_align, eqk, align_desc):
        op = self.find_op(op_id)
        inst = self.create_instance(inst_id, op)
        field = self.find_field(fspace_id, field_id)
        fspace = self.find_field_space(fspace_id)
        if fspace not in inst.fields:
            inst.fields[fspace] = []
            inst.align_desc[fspace] = []
        inst.fields[fspace].append(field)
        align_elem = Align(field_id, eqk, align_desc, bool(has_align))
        inst.align_desc[fspace].append(align_elem)

    def log_task_info(self, op_id, task_id, variant_id, proc_id,
                      create, ready, start, stop):
        variant = self.find_variant(task_id, variant_id)
        task = self.find_task(op_id, variant, create, ready, start, stop)
        if stop > self.last_time:
            self.last_time = stop
        proc = self.find_processor(proc_id)
        proc.add_task(task)

    def log_gpu_task_info(self, op_id, task_id, variant_id, proc_id,
                          create, ready, start, stop, gpu_start, gpu_stop):
        variant = self.find_variant(task_id, variant_id)
        task = self.find_task(op_id, variant, create, ready, gpu_start, gpu_stop)

        if gpu_stop > self.last_time:
            self.last_time = gpu_stop
        proc = self.find_processor(proc_id)
        proc.add_task(task)

    def log_meta_info(self, op_id, lg_id, proc_id, 
                      create, ready, start, stop):
        op = self.find_op(op_id)
        variant = self.find_meta_variant(lg_id)
        meta = self.create_meta(variant, op, create, ready, start, stop)
        if stop > self.last_time:
            self.last_time = stop
        proc = self.find_processor(proc_id)
        proc.add_task(meta)

    def add_copy_map(self,fevent,copy):
        key = fevent
        if key not in self.copy_map:
            self.copy_map[key] = copy

    def log_copy_info(self, op_id, src, dst, size,
                      create, ready, start, stop, fevent, num_requests):
        op = self.find_op(op_id)
        src = self.find_memory(src)
        dst = self.find_memory(dst)
        copy = self.create_copy(src, dst, op, size, create, ready, start, stop, fevent, num_requests)
        self.add_copy_map(fevent,copy)
        if stop > self.last_time:
            self.last_time = stop
        channel = self.find_channel(src, dst)
        channel.add_copy(copy)

    def log_copy_inst_info(self, op_id, src_inst, dst_inst, fevent, num_fields, request_type, num_hops):
        cpy = self.find_copy(fevent)
        entry = self.create_copy_inst_info(src_inst, dst_inst, fevent, num_fields, request_type, num_hops)
        cpy.add_copy_info(entry)

    def log_fill_info(self, op_id, dst, create, ready, start, stop):
        op = self.find_op(op_id)
        dst = self.find_memory(dst)
        fill = self.create_fill(dst, op, create, ready, start, stop)
        if stop > self.last_time:
            self.last_time = stop
        channel = self.find_channel(None, dst)
        channel.add_copy(fill)

    def log_inst_create(self, op_id, inst_id, create):
        op = self.find_op(op_id)
        inst = self.create_instance(inst_id, op)
        # don't overwrite if we have already captured the (more precise)
        #  timeline info
        if inst.stop is None:
            inst.start = create

    def log_inst_usage(self, op_id, inst_id, mem_id, size):
        op = self.find_op(op_id)
        mem = self.find_memory(mem_id)
        inst = self.create_instance(inst_id, op)
        inst.mem = mem
        inst.size = size
        mem.add_instance(inst)

    def log_inst_timeline(self, op_id, inst_id, create, destroy):
        op = self.find_op(op_id)
        inst = self.create_instance(inst_id, op)
        inst.start = create
        inst.stop = destroy
        if destroy > self.last_time:
            self.last_time = destroy 

    def log_partition_info(self, op_id, part_op, create, ready, start, stop):
        op = self.find_op(op_id)
        deppart = self.create_deppart(part_op, op, create, ready, start, stop)
        if stop > self.last_time:
            self.last_time = stop
        channel = self.find_channel(None, None)
        channel.add_copy(deppart)

    def log_user_info(self, proc_id, start, stop, name):
        proc = self.find_processor(proc_id)
        user = self.create_user_marker(name)
        user.start = start
        user.stop = stop
        if stop > self.last_time:
            self.last_time = stop 
        proc.add_task(user)

    def log_task_wait_info(self, op_id, task_id, variant_id, wait_start, wait_ready, wait_end):
        variant = self.find_variant(task_id, variant_id)
        task = self.find_task(op_id, variant)
        assert wait_ready >= wait_start
        assert wait_end >= wait_ready
        task.add_wait_interval(wait_start, wait_ready, wait_end)

    def log_meta_wait_info(self, op_id, lg_id, wait_start, wait_ready, wait_end):
        op = self.find_op(op_id)
        variant = self.find_meta_variant(lg_id)
        assert wait_ready >= wait_start
        assert wait_end >= wait_ready
        assert op_id in variant.ops
        # We know that meta wait infos are logged in order so we always add
        # the wait intervals to the last element in the list
        variant.ops[op_id][-1].add_wait_interval(wait_start, wait_ready, wait_end)

    def log_kind(self, task_id, name, overwrite):
        if task_id not in self.task_kinds:
            self.task_kinds[task_id] = TaskKind(task_id, name)
        elif overwrite == 1 or self.task_kinds[task_id].name is None:
            self.task_kinds[task_id].name = name

    def log_variant(self, task_id, variant_id, name):
        self.log_kind(task_id,name,0)
        task_kind = self.task_kinds[task_id]
        key = (task_id, variant_id)
        if key not in self.variants:
            self.variants[key] = Variant(variant_id, name)
        else:
            self.variants[key].name = name
        self.variants[key].set_task_kind(task_kind)

    def log_operation(self, op_id, parent_id, kind, provenance=None):
        op = self.find_op(op_id)
        #if op_id == 1:
        op.parent_id = parent_id
        assert kind in self.op_kinds
        op.kind_num = kind
        op.kind = self.op_kinds[kind]
        # the provenance is passed as "" by binary serializer
        #   when it is not set
        if provenance == "":
            provenance = None
        op.provenance = provenance
        #TODO:WEI
        # print(op.op_id, op.parent_id, op.provenance)

    def log_multi(self, op_id, task_id):
        op = self.find_op(op_id)
        task_kind = TaskKind(task_id, None)
        if task_id in self.task_kinds:
            task_kind = self.task_kinds[task_id]
        else:
            self.task_kinds[task_id] = task_kind
        op.is_multi = True
        op.task_kind = self.task_kinds[task_id]

    def log_slice_owner(self, parent_id, op_id):
        parent = self.find_op(parent_id)
        op = self.find_op(op_id)
        op.owner = parent

    def log_meta_desc(self, kind, message, ordered_vc, name):
        if kind not in self.meta_variants:
            self.meta_variants[kind] = Variant(kind, name, message, ordered_vc)
        else:
            self.meta_variants[kind].name = name

    def log_proc_desc(self, proc_id, kind):
        assert kind in processor_kinds
        kind = processor_kinds[kind]
        if proc_id not in self.processors:
            self.processors[proc_id] = Processor(proc_id, kind)
        else:
            self.processors[proc_id].kind = kind

    def log_mem_desc(self, mem_id, kind, capacity):
        assert kind in memory_kinds
        kind = memory_kinds[kind]
        if mem_id not in self.memories:
            self.memories[mem_id] = Memory(mem_id, kind, capacity)
        else:
            self.memories[mem_id].kind = kind
            self.memories[mem_id].capacity = capacity

    def log_mem_proc_affinity_desc(self, mem_id, proc_id, bandwidth, latency):
        if mem_id not in self.mem_proc_affinity:
            self.mem_proc_affinity[mem_id] = MemProcAffinity(self.memories[mem_id], self.processors[proc_id],
                                                             bandwidth, latency)
            self.memories[mem_id].add_affinity(self.mem_proc_affinity[mem_id])
        self.mem_proc_affinity[mem_id].update_best_affinity(bandwidth,latency,self.processors[proc_id])

    def log_op_desc(self, kind, name):
        if kind not in self.op_kinds:
            self.op_kinds[kind] = name

    def log_mapper_call_desc(self, kind, name):
        if kind not in self.mapper_call_kinds:
            self.mapper_call_kinds[kind] = MapperCallKind(kind, name)

    def log_mapper_call_info(self, kind, proc_id, op_id, start, stop):
        assert start <= stop
        assert kind in self.mapper_call_kinds
        # For now we'll only add very expensive mapper calls (more than 100 us)
        if (stop - start) < 100:
            return 
        if stop > self.last_time:
            self.last_time = stop
        call = MapperCall(self.mapper_call_kinds[kind],
                          self.find_op(op_id), start, stop)
        # update prof_uid map
        self.prof_uid_map[call.prof_uid] = call
        proc = self.find_processor(proc_id)
        proc.add_mapper_call(call)

    def log_runtime_call_desc(self, kind, name):
        if kind not in self.runtime_call_kinds:
            self.runtime_call_kinds[kind] = RuntimeCallKind(kind, name)

    def log_runtime_call_info(self, kind, proc_id, start, stop):
        assert start <= stop 
        assert kind in self.runtime_call_kinds
        if stop > self.last_time:
            self.last_time = stop
        call = RuntimeCall(self.runtime_call_kinds[kind], start, stop)
        proc = self.find_processor(proc_id)
        proc.add_runtime_call(call)

    def log_proftask_info(self, proc_id, op_id, start, stop):
        # we don't have a unique op_id for the profiling task itself, so we don't 
        # add to self.operations
        if stop > self.last_time:
            self.last_time = stop
        proftask = ProfTask(op_id, start, start, start, stop)
        proc = self.find_processor(proc_id)
        proc.add_task(proftask)

    def find_processor(self, proc_id):
        if proc_id not in self.processors:
            self.processors[proc_id] = Processor(proc_id, None)
        return self.processors[proc_id]

    def find_memory(self, mem_id):
        if mem_id not in self.memories:
            # use 'No MemKind' as the default kind
            self.memories[mem_id] = Memory(mem_id, "No MemKind", None)
        return self.memories[mem_id]

    def find_mem_proc_affinity(self, mem_id):
        if mem_id not in self.mem_proc_affinity:
            assert False
        return self.mem_proc_affinity[mem_id]

    def find_channel(self, src, dst):
        if src is not None:
            key = (src,dst)
            if key not in self.channels:
                self.channels[key] = Channel(src,dst)
            return self.channels[key]
        elif dst is not None:
            # This is a fill channel
            if dst not in self.channels:
                self.channels[dst] = Channel(None,dst)
            return self.channels[dst]
        else:
            # This is the dependent partitioning channel
            if None not in self.channels:
                self.channels[None] = Channel(None,None)
            return self.channels[None]

    def find_variant(self, task_id, variant_id):
        key = (task_id, variant_id)
        if key not in self.variants:
            self.variants[key] = Variant(variant_id, None)
        return self.variants[key]

    def find_meta_variant(self, lg_id):
        if lg_id not in self.meta_variants:
            self.meta_variants[lg_id] = Variant(lg_id, None)
        return self.meta_variants[lg_id]

    def find_op(self, op_id):
        if op_id not in self.operations:
            op = Operation(op_id) 
            self.operations[op_id] = op
            # update prof_uid map
            self.prof_uid_map[op.prof_uid] = op
        return self.operations[op_id]

    def find_copy(self,fevent):
        key = fevent
        if key not in self.copy_map:
            assert False
        return self.copy_map[key]

    def find_task(self, op_id, variant, create=None, ready=None, start=None, stop=None):
        task = self.find_op(op_id)
        # Upgrade this operation to a task if necessary
        if not task.is_task:
            assert create is not None
            assert ready is not None
            assert start is not None
            assert stop is not None
            task = Task(variant, task, create, ready, start, stop) 
            # print(task.op_id, variant, task.provenance)
            variant.ops[op_id] = task
            self.operations[op_id] = task
            # update prof_uid map
            self.prof_uid_map[task.prof_uid] = task
        else:
            assert task.variant == variant
        return task

    def create_index_space_point(self, unique_id, dim, values):
        key = unique_id
        if key not in self.index_spaces:
            index_space = IndexSpace.forPoint(unique_id, dim, values)
            self.index_spaces[key] = index_space
        else:
            index_space = self.index_spaces[key]
            if index_space.is_type is None:
                index_space.setPoint(unique_id, dim, values)
        return index_space

    def create_index_space_rect(self, unique_id, dim, values):
        key = unique_id
        if key not in self.index_spaces:
            index_space = IndexSpace.forRect(unique_id, dim, values, self.max_dim)
            self.index_spaces[key] = index_space
        else:
            index_space = self.index_spaces[key]
            if index_space.is_type is None:
                index_space.setRect(unique_id, dim, values, self.max_dim)
        return index_space

    def create_index_space_empty(self, unique_id):
        key = unique_id
        if key not in self.index_spaces:
            index_space = IndexSpace.forEmpty(key)
            self.index_spaces[key] = index_space
        else:
            index_space = self.index_spaces[key]
            if index_space.is_type is None:
                index_space.setEmpty(key)
        return index_space

    def find_index_space(self, unique_id):
        key = unique_id
        if key not in self.index_spaces:
            index_space = IndexSpace.forUnknown(key)
            self.index_spaces[key] = index_space
        else:
            index_space = self.index_spaces[key]
        return index_space

    def create_logical_region(self, ispace_id, fspace_id, tree_id, name):
        key = (ispace_id, fspace_id)
        if key not in self.logical_regions:
            logical_region = LogicalRegion(ispace_id,fspace_id, tree_id, name)
            self.logical_regions[key] = logical_region
        else:
            logical_region = self.logical_regions[key]
        return logical_region

    def create_field_space(self, unique_id, name):
        key = unique_id
        if key not in self.field_spaces:
            field_space = FieldSpace(unique_id, name)
            self.field_spaces[key] = field_space
        else:
            field_space = self.field_spaces[key]
        field_space.name = name
        return field_space

    def create_partition(self, unique_id, name):
        key = unique_id
        if key not in self.partitions:
            part = Partition(unique_id, name)
            self.partitions[key] = part
        else:
            part = self.partitions[key]
        part.name = name
        return part

    def find_partition(self, unique_id):
        key = unique_id
        if key not in self.partitions:
            part = Partition(unique_id, None)
            self.partitions[key] = part
        else:
            part = self.partitions[key]
        return part

    def find_field_space(self, unique_id):
        key = unique_id
        if key not in self.field_spaces:
            field_space = FieldSpace(unique_id, None)
            self.field_spaces[key] = field_space
        else:
            field_space = self.field_spaces[key]
        return field_space

    def create_field(self, unique_id, field_id, size, name):
        key = (unique_id, field_id)
        if key not in self.fields:
            field = Field(unique_id, field_id, size, name)
            self.fields[key] = field
        else:
            field = self.fields[key]
            field.size = size;
            field.name = name;
        return field

    def find_field(self, unique_id, field_id):
        key = (unique_id, field_id)
        if key not in self.fields:
            field = Field(unique_id, field_id, None, None)
            self.fields[key] = field
        else:
            field = self.fields[key]
        return field

    def create_meta(self, variant, op, create, ready, start, stop):
        meta = MetaTask(variant, op, create, ready, start, stop)
        if op.op_id not in variant.ops:
            variant.ops[op.op_id] = list()
        variant.ops[op.op_id].append(meta)
        # update prof_uid map
        self.prof_uid_map[meta.prof_uid] = meta
        return meta

    def create_copy(self, src, dst,  op, size, create, ready, start, stop, fevent, num_requests):
        copy = Copy(src, dst, op, size, create, ready, start, stop, fevent, num_requests)
        # update prof_uid map
        self.prof_uid_map[copy.prof_uid] = copy
        return copy

    def create_copy_inst_info(self, src_inst, dst_inst, fevent, num_fields, request_type, num_hops):
        copyinfo =  CopyInfo(src_inst, dst_inst, fevent, num_fields, request_type, num_hops)
        return copyinfo

    def create_fill(self, dst, op, create, ready, start, stop):
        fill = Fill(dst, op, create, ready, start, stop)
        # update prof_uid map
        self.prof_uid_map[fill.prof_uid] = fill
        return fill

    def create_deppart(self, part_op, op, create, ready, start, stop):
        deppart = DepPart(part_op, op, create, ready, start, stop)
        # update the prof_uid map
        self.prof_uid_map[deppart.prof_uid] = deppart
        return deppart

    def create_instance(self, inst_id, op):
        # neither instance id nor op id are unique on their own
        key = (inst_id, op.op_id)
        if key not in self.instances:
            inst = Instance(inst_id, op)
            self.instances[key] = inst
            # update prof_uid map
            self.prof_uid_map[inst.prof_uid] = inst
        else:
            inst = self.instances[key]
        return inst

    def find_instance(self, inst_id, op_id):
        key = (inst_id, op_id)
        if key not in self.instances:
            return None
        return self.instances[key]

    def create_user_marker(self, name):
        user = UserMarker(name)
        # update prof_uid map
        self.prof_uid_map[user.prof_uid] = user
        return user

    def trim_time_ranges(self, start, stop):
        assert self.last_time is not None
        if start < 0:
            start = None
        if stop > self.last_time:
            stop = None
        if start is None and stop is None:
            return
        for proc in itervalues(self.processors):
            proc.trim_time_range(start, stop)
        for mem in itervalues(self.memories):
            mem.trim_time_range(start, stop)
        for channel in itervalues(self.channels):
            channel.trim_time_range(start, stop)
        if start is not None and stop is not None:
            self.last_time = stop - start
        elif stop is not None:
            self.last_time = stop
        else:
            self.last_time -= start

    def sort_time_ranges(self):
        assert self.last_time is not None 
        # Processors first
        for proc in itervalues(self.processors):
            proc.last_time = self.last_time
            proc.sort_time_range()
        for mem in itervalues(self.memories):
            mem.init_time_range(self.last_time)
            mem.sort_time_range()
        for channel in itervalues(self.channels):
            channel.init_time_range(self.last_time)
            channel.sort_time_range()

    def check_message_latencies(self, threshold, warn_percentage):
        if threshold < 0:
            raise ValueError('Illegal threshold value, must be positive')
        if warn_percentage < 0.0 or 100.0 < warn_percentage:
            raise ValueError('Illegal warn percentage, must be a percentage')
        # Iterate over all the variants looking for message variants
        # on un-ordered virtual channels because we know that they are
        # launched without an event precondition. Given Realm's current
        # implementation, all of the latency between the create and ready
        # time will then actually be time spent in the network because
        # Realm will not check the event precondition until the active
        # message for the task launch arrives on the remote node. This
        # gives us a way to see how long message latencies are. We'll report
        # a warning to users if enough messages are over the threshold.
        total_messages = 0
        bad_messages = 0
        longest_latency = 0
        for variant in itervalues(self.meta_variants):
            if not variant.message:
                continue
            if variant.ordered_vc:
                continue
            # Iterate over the lists of meta-tasks for each op_id
            for ops in itervalues(variant.ops):
                total_messages += len(ops)
                for op in ops:
                    latency = op.ready - op.create
                    if threshold <= latency:
                        bad_messages += 1
                    if longest_latency < latency:
                        longest_latency = latency
        if total_messages == 0:
            return
        percentage = 100.0 * bad_messages / total_messages
        if warn_percentage <= percentage:
            for _ in range(5):
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("WARNING: A significant number of long latency messages "
                    "were detected during this run meaning that the network "
                    "was likely congested and could be causing a significant "
                    "performance degredation. We detected %d messages that took "
                    "longer than %.2fus to run, representing %.2f%% of %d total "
                    "messages. The longest latency message required %.2fus to "
                    "execute. Please report this case to the Legion developers "
                    "along with an accompanying Legion Prof profile so we can "
                    "better understand why the network is so congested." % 
                    (bad_messages,threshold,percentage, total_messages,longest_latency))
            for _ in range(5):
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def print_processor_stats(self, verbose):
        print('****************************************************')
        print('   PROCESSOR STATS')
        print('****************************************************')
        for proc in sorted(itervalues(self.processors)):
            proc.print_stats(verbose)
        print

    def print_memory_stats(self, verbose):
        print('****************************************************')
        print('   MEMORY STATS')
        print('****************************************************')
        for mem in sorted(itervalues(self.memories)):
            mem.print_stats(verbose)
        print

    def print_channel_stats(self, verbose):
        print('****************************************************')
        print('   CHANNEL STATS')
        print('****************************************************')
        for channel in sorted(itervalues(self.channels)):
            channel.print_stats(verbose)
        print

    def print_task_stats(self, verbose):
        print('****************************************************')
        print('   TASK STATS')
        print('****************************************************')
        stat = StatGatherer(self)
        stat.print_stats(verbose)
        print

    def print_stats(self, verbose):
        self.print_processor_stats(verbose)
        self.print_memory_stats(verbose)
        self.print_channel_stats(verbose)
        self.print_task_stats(verbose)

    def assign_colors(self):
        # Subtract out some colors for which we have special colors
        num_colors = len(self.variants) + len(self.meta_variants) + \
                     len(self.op_kinds) + \
                     len(self.mapper_call_kinds) + len(self.runtime_call_kinds)
        # Use a LFSR to randomize these colors
        lsfr = LFSR(num_colors)
        num_colors = lsfr.get_max_value()
        op_colors = {}
        for variant in sorted(itervalues(self.variants), key=lambda v: (v.task_kind.task_id, v.variant_id)):
            variant.compute_color(lsfr.get_next(), num_colors)
        for variant in sorted(itervalues(self.meta_variants), key=lambda v: v.variant_id):
            if variant.variant_id == 1: # Remote message
                variant.assign_color('#006600') # Evergreen
            elif variant.variant_id == 2: # Post-Execution
                variant.assign_color('#333399') # Deep Purple
            elif variant.variant_id == 6: # Garbage Collection
                variant.assign_color('#990000') # Crimson
            elif variant.variant_id == 7: # Logical Dependence Analysis
                variant.assign_color('#0000ff') # Duke Blue
            elif variant.variant_id == 8: # Operation Physical Analysis
                variant.assign_color('#009900') # Green
            elif variant.variant_id == 9: # Task Physical Analysis
                variant.assign_color('#009900') #Green
            else:
                variant.compute_color(lsfr.get_next(), num_colors)
        for kind in sorted(iterkeys(self.op_kinds)):
            op_colors[kind] = color_helper(lsfr.get_next(), num_colors)
        # Now we need to assign all the operations colors
        for op in itervalues(self.operations):
            op.assign_color(op_colors)
        # Assign all the message kinds different colors
        for kind in sorted(itervalues(self.mapper_call_kinds), key=lambda k: k.mapper_call_kind):
            kind.assign_color(color_helper(lsfr.get_next(), num_colors))
        for kind in sorted(itervalues(self.runtime_call_kinds), key=lambda k: k.runtime_call_kind):
            kind.assign_color(color_helper(lsfr.get_next(), num_colors))

    def show_copy_matrix(self, output_prefix):
        template_file_name = os.path.join(root_dir, "legion_prof_copy.html.template")
        tsv_file_name = output_prefix + ".tsv"
        html_file_name = output_prefix + ".html"
        print('Generating copy visualization files %s and %s' % (tsv_file_name,html_file_name))

        def node_id(memory):
            return (memory.mem_id >> 23) & ((1 << 5) - 1)
        memories = sorted(itervalues(self.memories))

        tsv_file = open(tsv_file_name, "w")
        tsv_file.write("source\ttarget\tremote\ttotal\tcount\taverage\tbandwidth\n")
        for i in range(0, len(memories)):
            for j in range(0, len(memories)):
                src = memories[i]
                dst = memories[j]
                is_remote = node_id(src) != node_id(dst) or \
                    src.kind == memory_kinds[0] or \
                    dst.kind == memory_kinds[0]
                sum = 0.0
                cnt = 0
                bandwidth = 0.0
                channel = self.find_channel(src, dst)
                for copy in channel.copies:
                    time = copy.stop - copy.start
                    sum = sum + time * 1e-6
                    bandwidth = bandwidth + copy.size / time
                    cnt = cnt + 1
                tsv_file.write("%d\t%d\t%d\t%f\t%d\t%f\t%f\n" % \
                        (i, j, int(is_remote), sum, cnt,
                         sum / cnt * 1000 if cnt > 0 else 0,
                         bandwidth / cnt if cnt > 0 else 0))
        tsv_file.close()

        template_file = open(template_file_name, "r")
        template = template_file.read()
        template_file.close()
        html_file = open(html_file_name, "w")
        html_file.write(template % (repr([str(mem).replace("Memory ", "") for mem in memories]),
                                    repr(tsv_file_name)))
        html_file.close()

    def find_unique_dirname(self, dirname):
        if (not os.path.exists(dirname)):
            return dirname
        # if the dirname exists, loop through dirname.i until we
        # find one that doesn't exist
        i = 1
        while (True):
            potential_dir = dirname + "." + str(i)
            if (not os.path.exists(potential_dir)):
                return potential_dir
            i += 1

    def calculate_utilization_data(self, timepoints, owners, count):
        # we assume that the timepoints are sorted before this step

        # loop through all the timepoints. Get the earliest. If it's first,
        # add to the count. if it's second, decrement the count. Store the
        # (time, count) pair.

        assert len(owners) > 0

        isMemory = False
        isChannel = False
        max_count = 0
        if isinstance(owners[0], Channel):
            isChannel = True
        if isinstance(owners[0], Memory):
            isMemory = True
            for mem in owners:
                max_count += mem.capacity
        else:
            max_count = count

        max_count = float(max_count)

        utilization = list()
        count = 0
        last_time = None
        increment = 1.0 / max_count
        for point in timepoints:
            if isMemory:
                if point.first:
                    count += point.thing.size
                else:
                    count -= point.thing.size
            else:
                if point.first:
                    count += 1
                else:
                    count -= 1


            if point.time == last_time:
                if isChannel and count > 0:
                    utilization[-1] = (point.time, 1)
                else:
                    utilization[-1] = (point.time, count / max_count) # update the count
            else:
                if isChannel and count > 0:
                    utilization.append((point.time, 1))
                else:
                    utilization.append((point.time, count / max_count))
            last_time = point.time
        return utilization

        # CODE BELOW USES A BASIC FILTER IN CASE THE SIZE OF THE UTILIZATION
        # IS TOO LARGE

        # # we want to limit to num_points points in the utilization, so only get
        # # every nth element
        # n = max(1, int(len(times) / num_points))
        # utilization = []
        # # FIXME: get correct count average
        # for i in range(0, len(times),n):
        #     sub_times = times[i:i+n]
        #     sub_counts = counts[i:i+n]
        #     num_elems = float(len(sub_times))
        #     avg_time = float(sum(sub_times)) / num_elems
        #     avg_count = float(sum(sub_counts)) / num_elems
        #     utilization.append((avg_time, avg_count))
        # return utilization

    # utilization is 1 for a processor when some event is on the
    # processor
    #
    # adds points where there are one or more events occurring.
    # removes a point when the count drops to 0.
    def convert_to_utilization(self, timepoints, owner):
        proc_utilization = []
        count = 0
        if isinstance(owner, Processor) or isinstance(owner, Channel):
            for point in timepoints:
                if point.first:
                    count += 1
                    if count == 1:
                        proc_utilization.append(point)
                else:
                    count -= 1
                    if count == 0:
                        proc_utilization.append(point)
        else: # For memories, we want just the points
            proc_utilization = timepoints
        return proc_utilization

    # group by processor kind per node. Also compute the children relationships
    def group_node_proc_kind_timepoints(self, timepoints_dict, proc_count):
        # procs
        for proc in itervalues(self.processors):
            if len(proc.tasks) >= 0:
                # add this processor kind to both all and the node group
                groups = [str(proc.node_id), "all"]
                for node in groups:
                    group = node + " (" + proc.kind + ")"
                    if group not in proc_count:
                        proc_count[group] = 0
                    proc_count[group] = proc_count[group]+1
                    if len(proc.tasks) > 0:
                        if group not in timepoints_dict:
                            timepoints_dict[group] = []
                        timepoints_dict[group].append(proc.util_time_points)
        # memories
        for mem in itervalues(self.memories):
            if len(mem.time_points) > 0:
                # add this memory kind to both the all and the node group
                groups = [str(mem.node_id), "all"]
                for node in groups:
                    group = node + " (" + mem.kind + " Memory)"
                    if group not in timepoints_dict:
                        timepoints_dict[group] = [mem.time_points]
                    else:
                        timepoints_dict[group].append(mem.time_points)
        # channels
        for channel in itervalues(self.channels):
            if len(channel.time_points) > 0:
                # add this channel to both the all and the node group
                if (channel.node_id() != None):
                    groups = [str(channel.node_id()), "all"]
                    if channel.node_id_dst() != channel.node_id() and channel.node_id_dst() != None:
                        groups.append(str(channel.node_id_dst()))
                    if channel.node_id_src() != channel.node_id() and channel.node_id_src() != None:
                        groups.append(str(channel.node_id_src()))
                    for node in groups:
                        group = node + " (" + "Channel)"
                        if group not in timepoints_dict:
                            timepoints_dict[group] = [channel.time_points]
                        else:
                            timepoints_dict[group].append(channel.time_points)


    def get_nodes(self):
        nodes = {}
        for proc in itervalues(self.processors):
            if len(proc.tasks) > 0:
                nodes[str(proc.node_id)] = 1
        if (len(nodes) > 1):
            return ["all"] + sorted(nodes.keys())
        else:
            return sorted(nodes.keys())


    def emit_utilization_tsv(self, output_dirname):
        print("emitting utilization")

        # this is a map from node ids to a list of timepoints in that node
        timepoints_dict = {}
        proc_count = {};
        self.group_node_proc_kind_timepoints(timepoints_dict, proc_count)

        # now we compute the structure of the stats (the parent-child
        # relationships
        nodes = self.get_nodes()
        stats_structure = {node: [] for node in sorted(nodes)}

        # for each node grouping, add all the subtypes of processors
        for node in nodes:
            for kind in itervalues(processor_kinds):
                group = str(node) + " (" + kind + ")"
                if group in timepoints_dict:
                    stats_structure[node].append(group)
            for kind in itervalues(memory_kinds):
                group = str(node) + " (" + kind + " Memory)"
                if group in timepoints_dict:
                    stats_structure[node].append(group)
            group = node + " (" + "Channel)"
            if group in timepoints_dict:
                    stats_structure[node].append(group)

        json_file_name = os.path.join(output_dirname, "json", "utils.json")

        with open(json_file_name, "w") as json_file:
            # json.dump(timepoints_dict.keys(), json_file)
            json.dump(stats_structure, json_file, separators=(',', ':'))

        # here we write out the actual tsv stats files
        for tp_group in timepoints_dict:
            timepoints = timepoints_dict[tp_group]
            utilizations = [self.convert_to_utilization(tp, tp[0].thing.get_owner())
                            for tp in timepoints if len(tp) > 0]

            owners = set()
            for tp in timepoints:
                if len(tp) > 0:
                    owners.add(tp[0].thing.get_owner())
            owners = list(owners)

            utilization = None
            if len(owners) == 0:
                print("WARNING: node " + str(tp_group) + " has no prof events. Is this what you expected?")
                utilization = list()
            else:
                count = 0
                if tp_group in proc_count:
                    count = proc_count[tp_group];
                else:
                    count = len(owners);
                utilization = self.calculate_utilization_data(sorted(itertools.chain(*utilizations)), owners, count)

            util_tsv_filename = os.path.join(output_dirname, "tsv", str(tp_group) + "_util.tsv")
            with open(util_tsv_filename, "w") as util_tsv_file:
                util_tsv_file.write("time\tcount\n")
                util_tsv_file.write("0.000\t0.00\n") # initial point
                for util_point in utilization:
                    util_tsv_file.write("%.3f\t%.2f\n" % util_point)

    def simplify_op(self, op_dependencies, op_existence_set, transitive_map, op_path, _dir):
        cur_op_id = op_path[-1]

        if cur_op_id in op_existence_set:
            # we're done, we've found an op that exists
            for op_id in op_path:
                # for the children that exist, add it to the transitive map
                if not op_id in transitive_map[_dir]:
                    transitive_map[_dir][op_id] = []
                transitive_map[_dir][op_id].append(cur_op_id)
        else:
            children = op_dependencies[cur_op_id][_dir]
            for child_op_id in children:
                new_op_path = op_path + [child_op_id]
                self.simplify_op(op_dependencies, op_existence_set, transitive_map,
                                 new_op_path, _dir)

    def simplify_op_dependencies(self, op_dependencies, op_existence_set):
        # The dependence relation is transitive. We take advantage of this to
        # remove non-existant ops in the graph by following "out" and "in"
        # pointers until we get to an existing op
        transitive_map = {"in": {}, "out": {}, "parents": {}, "children": {}}

        for _dir in iterkeys(transitive_map):
            for op_id in iterkeys(op_dependencies):
                if not op_id in transitive_map[_dir]:
                    self.simplify_op(op_dependencies, op_existence_set,
                                     transitive_map, [op_id], _dir)

        for op_id in iterkeys(op_dependencies):
            for _dir in iterkeys(transitive_map):
                if len(op_dependencies[op_id][_dir]) > 0:
                    # replace each op with the transitive map
                    transformed_dependencies = [transitive_map[_dir][op]
                                                if op in transitive_map[_dir] else []
                                                for op in op_dependencies[op_id][_dir]]
                    # flatMap
                    if len(transformed_dependencies) > 0:
                        simplified_dependencies = reduce(list.__add__, 
                                                         transformed_dependencies)
                    else:
                        simplified_dependencies = []
                    
                    op_dependencies[op_id][_dir] = set(simplified_dependencies)
                else:
                    op_dependencies[op_id][_dir] = set()
        return op_dependencies, transitive_map

    # Here, we read the legion spy data! We will use this to draw dependency
    # lines in the prof
    def get_op_dependencies(self, file_names):
        self.spy_state = legion_spy.State(None, False, True, True, True, False, False)

        total_matches = 0

        for file_name in file_names:
            file_type, version = GetFileTypeInfo(file_name)
            if file_type == "ascii":
                total_matches += self.spy_state.parse_log_file(file_name)
        print('Matched %d lines across all files.' % total_matches)
        op_dependencies = {}

        # compute the slice_index, slice_slice, and point_slice dependencies
        # (which legion_spy throws away). We just need to copy this data over
        # before legion spy throws it away
        for _slice, index in iteritems(self.spy_state.slice_index):
            while _slice in self.spy_state.slice_slice:
                _slice = self.spy_state.slice_slice[_slice]
            if index.uid not in op_dependencies:
                op_dependencies[index.uid] = {
                    "in" : set(), 
                    "out" : set(),
                    "parents" : set(),
                    "children" : set()
                }
            if _slice not in op_dependencies:
                op_dependencies[_slice] = {
                    "in" : set(), 
                    "out" : set(),
                    "parents" : set(),
                    "children" : set()
                }
            op_dependencies[index.uid]["out"].add(_slice)
            op_dependencies[_slice]["in"].add(index.uid)

        for _slice1, _slice2 in iteritems(self.spy_state.slice_slice):
            if _slice1 not in op_dependencies:
                op_dependencies[_slice1] = {
                    "in" : set(), 
                    "out" : set(),
                    "parents" : set(),
                    "children" : set()
                }
            if _slice2 not in op_dependencies:
                op_dependencies[_slice2] = {
                    "in" : set(), 
                    "out" : set(),
                    "parents" : set(),
                    "children" : set()
                }
            op_dependencies[_slice1]["out"].add(_slice2)
            op_dependencies[_slice2]["in"].add(_slice1)

        for point, _slice in iteritems(self.spy_state.point_slice):
            while _slice in self.spy_state.slice_slice:
                _slice = self.spy_state.slice_slice[_slice]
            if _slice not in op_dependencies:
                op_dependencies[_slice] = {
                    "in" : set(), 
                    "out" : set(),
                    "parents" : set(),
                    "children" : set()
                }
            if point.op.uid not in op_dependencies:
                op_dependencies[point.op.uid] = {
                    "in" : set(), 
                    "out" : set(),
                    "parents" : set(),
                    "children" : set()
                }
            op_dependencies[_slice]["out"].add(point.op.uid)
            op_dependencies[point.op.uid]["in"].add(_slice)

        # don't simplify graphs
        self.spy_state.post_parse(False, True)

        print("Performing physical analysis...")
        self.spy_state.perform_physical_analysis(False, False)
        self.spy_state.simplify_physical_graph(need_cycle_check=False)

        op = self.spy_state.get_operation(self.spy_state.top_level_uid)
        elevate = dict()
        all_nodes = set()
        printer = legion_spy.GraphPrinter("./", "temp")
        try:
            os.remove("temp.dot")
        except:
            print("Error remove temp.dot file")

        op.print_event_graph(printer, elevate, all_nodes, True) 
        # Now print the edges at the very end
        for node in all_nodes:
            if hasattr(node, 'uid'):
                for src in node.physical_incoming:
                    if hasattr(src, 'uid'):
                        if src.uid not in op_dependencies:
                            op_dependencies[src.uid] = {
                                "in" : set(), 
                                "out" : set(),
                                "parents" : set(),
                                "children" : set()
                            }
                        if node.uid not in op_dependencies:
                            op_dependencies[node.uid] = {
                                "in" : set(), 
                                "out" : set(),
                                "parents" : set(),
                                "children" : set()
                            }
                        op_dependencies[src.uid]["in"].add(node.uid)
                        op_dependencies[node.uid]["out"].add(src.uid)


        # compute implicit dependencies
        for op in itervalues(self.spy_state.ops):
            if op.context is not None and op.context.op is not None:
                child_uid = op.uid
                parent_uid = op.context.op.uid
                if child_uid not in op_dependencies:
                    op_dependencies[child_uid] = {
                        "in" : set(), 
                        "out" : set(),
                        "parents" : set(),
                        "children" : set()
                    }
                if parent_uid not in op_dependencies:
                    op_dependencies[parent_uid] = {
                    "in" : set(), 
                    "out" : set(),
                    "parents" : set(),
                    "children" : set()
                }
                op_dependencies[parent_uid]["children"].add(child_uid)
                op_dependencies[child_uid]["parents"].add(parent_uid)


        # have an existence map for the uids (some uids in the event graph are not
        # actually executed
        op_existence_set = set()

        for op_id, operation in iteritems(self.operations):
            if operation.proc is not None:
                op_existence_set.add(op_id)

        # now apply the existence map
        op_dependencies, transitive_map = self.simplify_op_dependencies(op_dependencies, op_existence_set)

        return op_dependencies, transitive_map

    def convert_op_ids_to_tuples(self, op_dependencies):
        # convert to tuples
        for op_id in op_dependencies:
            for _dir in op_dependencies[op_id]:
                def convert_to_tuple(elem):
                    if not isinstance(elem, tuple):
                        # needs to be converted
                        return self.find_op(elem).get_unique_tuple()
                    else:
                        return elem
                op_dependencies[op_id][_dir] = set(map(convert_to_tuple, 
                                                   op_dependencies[op_id][_dir]))

    def add_initiation_dependencies(self, state, op_dependencies, transitive_map):
        for proc in itervalues(self.processors):
            proc.add_initiation_dependencies(state, op_dependencies, transitive_map)

    def attach_dependencies(self, op_dependencies, transitive_map):
        for proc in itervalues(self.processors):
            proc.attach_dependencies(self, op_dependencies, transitive_map)

    # traverse one op to get the max outbound path from this point
    def traverse_op_for_critical_path(self, op):
        cur_path  = PathRange(0, 0, [])
        if isinstance(op, HasDependencies):
            if op.visited:
                cur_path = op.path
            else:
                op.visited = True
                paths = list()
                for op_tuple in op.deps["out"]:
                    out_op = self.prof_uid_map[op_tuple[2]]
                    path = self.traverse_op_for_critical_path(out_op)
                    start = min(path.start, op.start)
                    stop = max(path.stop, op.stop)
                    newPath = PathRange(start, stop, path.path)
                    newPath.path.append(op)
                    paths.append(newPath)
                if len(paths) > 0:
                    # pick max outbound path
                    cur_path = max(paths)
                    op.path = cur_path
                else:
                    op.path = PathRange(op.start, op.stop, [op])

                cur_path = op.path
        return cur_path

    def get_longest_child(self, op):
        children = []
        for child in op.deps["children"]:
            child_prof_uid = child[2]
            child_op = self.prof_uid_map[child_prof_uid]
            if isinstance(child_op, HasDependencies):
                children.append(child_op)
        if len(children) > 0:
            longest_child = max(children, key=lambda op: op.path)
            return longest_child.path.path + self.get_longest_child(longest_child)
        else:
            return []

    def compute_critical_path(self):
        paths = []
        # compute the critical path for each task
        for proc in itervalues(self.processors):
            for task in proc.tasks:
                if (len(task.deps["parents"]) > 0) or (len(task.deps["out"]) > 0):
                    path = self.traverse_op_for_critical_path(task)
                    paths.append(path)
        # pick the longest critical path
        critical_path = max(paths).clone()

        # add the chilren to the critical path
        all_children = []
        for op in critical_path.path:
            # remove initiation depedencies from the inner critical paths
            longest_child_path = list(filter(lambda p: not isinstance(p, HasInitiationDependencies), self.get_longest_child(op)))
            all_children = all_children + longest_child_path
        critical_path.path = critical_path.path + all_children

        if len(critical_path.path) > 0:
            critical_path_set = set(map(lambda p: p.get_unique_tuple(), critical_path.path))
            def get_path_obj(p):
                return {
                    "tuple": p.get_unique_tuple(),
                    "obj"  : list(p.deps["out"].intersection(critical_path_set))
                }

            critical_path = map(lambda p: get_path_obj(p), critical_path.path)
        return critical_path

    def check_operation_parent_id(self):
        self.operations = OrderedDict(sorted(self.operations.items()))
        for op_id, operation in iteritems(self.operations):
            if operation.parent_id not in self.operations.keys():
                print("Found Operation: ", operation, " with parent_id = ", operation.parent_id, ", parent NOT existed")
                operation.parent_id = 0

    def simplify_critical_path(self, critical_path):
        simplified_critical_path = set()
        if len(critical_path) > 0:
            critical_path_set = set(critical_path)
            p_prof_uid = critical_path[-1][2]
            p = self.prof_uid_map[p_prof_uid]
            intersection = p.deps["out"].intersection(critical_path_set)
            while len(intersection) != 0:
                simplified_critical_path.add(p.get_unique_tuple())
                p_prof_uid = next(iter(intersection))[2]
                p = self.prof_uid_map[p_prof_uid]
                intersection = p.deps["out"].intersection(critical_path_set)
            simplified_critical_path.add(p.get_unique_tuple())
        return list(simplified_critical_path)


    def emit_interactive_visualization(self, output_dirname, show_procs,
                               file_names, show_channels, show_instances, force):
        self.assign_colors()
        # the output directory will either be overwritten, or we will find
        # a new unique name to create new logs
        if force and os.path.exists(output_dirname):
            print("forcing removal of " + output_dirname)
            shutil.rmtree(output_dirname)
        else:
            output_dirname = self.find_unique_dirname(output_dirname)

        print('Generating interactive visualization files in directory ' + output_dirname)
        src_directory = os.path.join(root_dir, "legion_prof_files")

        shutil.copytree(src_directory, output_dirname)

        proc_list = []
        chan_list = []
        mem_list = []
        processor_levels = {}
        channel_levels = {}
        memory_levels = {}
        base_level = 0
        last_time = 0

        ops_file_name = os.path.join(output_dirname, 
                                     "legion_prof_ops.tsv")
        data_tsv_file_name = os.path.join(output_dirname, 
                                          "legion_prof_data.tsv")
        processor_tsv_file_name = os.path.join(output_dirname, 
                                               "legion_prof_processor.tsv")

        scale_json_file_name = os.path.join(output_dirname, "json", 
                                            "scale.json")
        dep_json_file_name = os.path.join(output_dirname, "json", 
                                          "op_dependencies.json")

        data_tsv_header = "level\tlevel_ready\tready\tstart\tend\tcolor\topacity\ttitle\tinitiation\tin\tout\tchildren\tparents\tprof_uid\top_id\n"

        tsv_dir = os.path.join(output_dirname, "tsv")
        json_dir = os.path.join(output_dirname, "json")
        os.mkdir(tsv_dir)
        if not os.path.exists(json_dir):
            os.mkdir(json_dir)
        
        op_dependencies, transitive_map = None, None
        if self.has_spy_data:
            op_dependencies, transitive_map = self.get_op_dependencies(file_names)

        # with open(dep_json_file_name, "w") as dep_json_file:
        #     json.dump(op_dependencies, dep_json_file)

        ops_file = open(ops_file_name, "w")
        ops_file.write("op_id\tparent_id\tdesc\tproc\tlevel\tprovenance\n")
        for op_id, operation in sorted(iteritems(self.operations)):
            if operation.is_trimmed():
                continue
            proc = ""
            level = ""
            provenance = ""
            if (operation.proc is not None):
                proc = repr(operation.proc)
                level = str(operation.level+1)
            if (operation.provenance is not None):
                provenance = operation.provenance
            ops_file.write("%d\t%d\t%s\t%s\t%s\t%s\n" % \
                            (op_id, operation.parent_id, str(operation), proc, level, provenance))
        ops_file.close()

        if show_procs:
            for proc in itervalues(self.processors):
                if self.has_spy_data and len(proc.tasks) > 0:
                    proc.add_initiation_dependencies(self, op_dependencies,
                                                     transitive_map)
                    self.convert_op_ids_to_tuples(op_dependencies)

            for p,proc in sorted(iteritems(self.processors), key=lambda x: x[1]):
                if len(proc.tasks) > 0:
                    if self.has_spy_data:
                        proc.attach_dependencies(self, op_dependencies,
                                                 transitive_map)
                    proc_name = slugify("Proc_" + str(hex(p)))
                    proc_tsv_file_name = os.path.join(tsv_dir, proc_name + ".tsv")
                    with open(proc_tsv_file_name, "w") as proc_tsv_file:
                        proc_tsv_file.write(data_tsv_header)
                        proc_level = proc.emit_tsv(proc_tsv_file, 0)
                    base_level += proc_level
                    processor_levels[proc] = {
                        'levels': proc_level-1, 
                        'tsv': "tsv/" + proc_name + ".tsv"
                    }
                    proc_list.append(proc)

                    last_time = max(last_time, proc.last_time)
        if show_channels:
            for c,chan in sorted(iteritems(self.channels), key=lambda x: x[1]):
                if len(chan.copies) > 0:
                    chan_name = slugify(str(c))
                    chan_tsv_file_name = os.path.join(tsv_dir, chan_name + ".tsv")
                    with open(chan_tsv_file_name, "w") as chan_tsv_file:
                        chan_tsv_file.write(data_tsv_header)
                        chan_level = chan.emit_tsv(chan_tsv_file, 0)
                    base_level += chan_level
                    channel_levels[chan] = {
                        'levels': chan_level-1, 
                        'tsv': "tsv/" + chan_name + ".tsv"
                    }
                    chan_list.append(chan)

                    last_time = max(last_time, chan.last_time)
        if show_instances:
            for m,mem in sorted(iteritems(self.memories), key=lambda x: x[1]):
                if len(mem.instances) > 0:
                    mem_name = slugify("Mem_" + str(hex(m)))
                    mem_tsv_file_name = os.path.join(tsv_dir, mem_name + ".tsv")
                    with open(mem_tsv_file_name, "w") as mem_tsv_file: 
                        mem_tsv_file.write(data_tsv_header)
                        mem_level = mem.emit_tsv(mem_tsv_file, 0)
                    base_level += mem_level
                    memory_levels[mem] = {
                        'levels': mem_level-1, 
                        'tsv': "tsv/" + mem_name + ".tsv"
                    }
                    mem_list.append(mem)

                    last_time = max(last_time, mem.last_time)

        critical_path = list()
        if self.has_spy_data:
            critical_path = self.compute_critical_path()
            # critical_path = self.simplify_critical_path(critical_path)
            # print("Critical path is " + str(critical_range.elapsed()) + "us")

        # print(str(op) + ", " + str(op.path) + ", " + str(op.path_range))


        critical_path_json_file_name = os.path.join(json_dir, "critical_path.json")
        with open(critical_path_json_file_name, "w") as critical_path_json_file:
            json.dump(list(critical_path), critical_path_json_file)

        processor_tsv_file = open(processor_tsv_file_name, "w")
        processor_tsv_file.write("full_text\ttext\ttsv\tlevels\n")
        if show_procs:
            for proc in sorted(proc_list):
                tsv = processor_levels[proc]['tsv']
                levels = processor_levels[proc]['levels']
                processor_tsv_file.write("%s\t%s\t%s\t%d\n" % 
                                (repr(proc), proc.get_short_text(), tsv, levels))
        if show_channels:
            for channel in sorted(chan_list):
                tsv = channel_levels[channel]['tsv']
                levels = channel_levels[channel]['levels']
                processor_tsv_file.write("%s\t%s\t%s\t%d\n" % 
                                (repr(channel), channel.get_short_text(), tsv, levels))
        if show_instances:
            for memory in sorted(mem_list):
                tsv = memory_levels[memory]['tsv']
                levels = memory_levels[memory]['levels']
                processor_tsv_file.write("%s\t%s\t%s\t%d\n" % 
                                (repr(memory), memory.get_short_text(), tsv, levels))
        processor_tsv_file.close()

        num_utils = self.emit_utilization_tsv(output_dirname)
        stats_levels = 4

        scale_data = {
            'start': 0.0,
            'end': math.ceil(last_time * 10. * 1.01) / 10.,
            'stats_levels': stats_levels,
            'max_level': base_level + 1
        }

        with open(scale_json_file_name, "w") as scale_json_file:
            json.dump(scale_data, scale_json_file, separators=(',', ':'))

def main():
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            self.print_usage(sys.stderr)
            print('error: %s' % message, file=sys.stderr)
            print('hint: invoke %s -h for a detailed description of all arguments' % self.prog, file=sys.stderr)
            sys.exit(2)
    parser = MyParser(
        description='Legion Prof: application profiler')
    parser.add_argument(
        '-C', '--copy', dest='show_copy_matrix', action='store_true',
        help='include copy matrix in visualization')
    parser.add_argument(
        '-s', '--statistics', dest='print_stats', action='store_true',
        help='print statistics')
    parser.add_argument(
        '-v', '--verbose', dest='verbose', action='store_true',
        help='print verbose profiling information')
    parser.add_argument(
        '-m', '--ppm', dest='us_per_pixel', action='store',
        type=int, default=US_PER_PIXEL,
        help='micro-seconds per pixel (default %d)' % US_PER_PIXEL)
    parser.add_argument(
        '-o', '--output', dest='output', action='store',
        default='legion_prof',
        help='output directory pathname')
    parser.add_argument(
        '-f', '--force', dest='force', action='store_true',
        help='overwrite output directory if it exists')
    parser.add_argument(
        '--start-trim', dest='start_trim', action='store',
        type=int, default=-1,
        help='start time in micro-seconds to trim the profile')
    parser.add_argument(
        '--stop-trim', dest='stop_trim', action='store',
        type=int, default=-1,
        help='stop time in micro-seconds to trim the profile')
    parser.add_argument(
        dest='filenames', nargs='+',
        help='input Legion Prof log filenames')
    parser.add_argument(
        '--message-threshold', dest='message_threshold', action='store',
        type=float, default=1000,
        help='threshold for warning about message latencies in microseconds')
    parser.add_argument(
        '--message-percentage', dest='message_percentage', action='store',
        type=float, default=5.0,
        help='perentage of messages that must be over the threshold to trigger a warning')
    args = parser.parse_args()

    file_names = args.filenames
    show_all = not args.show_copy_matrix
    show_procs = show_all
    show_channels = show_all
    show_instances = show_all
    show_copy_matrix = args.show_copy_matrix
    force = args.force
    output_dirname = args.output
    copy_output_prefix = output_dirname + "_copy"
    print_stats = args.print_stats
    verbose = args.verbose
    start_trim = args.start_trim
    stop_trim = args.stop_trim
    

    state = State()
    has_matches = False
    has_binary_files = False # true if any of the files are a binary file

    asciiDeserializer = LegionProfASCIIDeserializer(state, state.callbacks)
    binaryDeserializer = LegionProfBinaryDeserializer(state, state.callbacks)

    for file_name in file_names:
        file_type, version = GetFileTypeInfo(file_name)
        if file_type == "binary":
            has_binary_files = True
            break

    for file_name in file_names:
        deserializer = None
        file_type, version = GetFileTypeInfo(file_name)
        if file_type == "binary":
            deserializer = binaryDeserializer
        else:
            deserializer = asciiDeserializer
        if has_binary_files == False or file_type == "binary":
            # only parse the log if it's a binary file, or if all the files
            # are ascii files
            print('Reading log file %s...' % file_name)
            total_matches = deserializer.parse(file_name, verbose)
            print('Matched %s objects' % total_matches)
            if total_matches > 0:
                has_matches = True
        else:
            # In this case, we have an ascii file passed in but we also have
            # binary files. All we need to do is check if it has legion spy
            # data
            deserializer.search_for_spy_data(file_name)

    if not has_matches:
        print('No matches found! Exiting...')
        return

    # See if we need to trim out any boxes before we build the profile
    if not print_stats and ((start_trim > 0) or (stop_trim > 0)):
        if start_trim > 0 and stop_trim > 0:
            if stop_trim > start_trim:
                state.trim_time_ranges(start_trim, stop_trim)
            else:
                print('WARNING: Ignoring invalid trim ranges because stop trim time ('+
                    str(stop_trim)+') comes before start trim time ('+str(start_trim)+')')
        else:
            state.trim_time_ranges(start_trim, stop_trim)

    # Once we are done loading everything, do the sorting
    state.sort_time_ranges()

    # Check the message latencies
    state.check_message_latencies(args.message_threshold, args.message_percentage)

    # sort operations and check parent_id
    state.check_operation_parent_id()

    if print_stats:
        state.print_stats(verbose)
    else:
        state.emit_interactive_visualization(output_dirname, show_procs,
                             file_names, show_channels, show_instances, force)
        if show_copy_matrix:
            state.show_copy_matrix(copy_output_prefix)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("elapsed: " + str(end - start) + "s")
