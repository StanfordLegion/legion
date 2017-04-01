#!/usr/bin/env python

# Copyright 2017 Stanford University, NVIDIA Corporation
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

import tempfile
import legion_spy
import argparse
import sys, os, shutil
import string, re, json, heapq, time, itertools
from collections import defaultdict
from math import sqrt, log
from cgi import escape
from operator import itemgetter
from os.path import dirname, exists, basename

prefix = r'\[(?P<node>[0-9]+) - (?P<thread>[0-9a-f]+)\] \{\w+\}\{legion_prof\}: '
task_info_pat = re.compile(prefix + r'Prof Task Info (?P<opid>[0-9]+) (?P<vid>[0-9]+) (?P<pid>[a-f0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)')
meta_info_pat = re.compile(prefix + r'Prof Meta Info (?P<opid>[0-9]+) (?P<hlr>[0-9]+) (?P<pid>[a-f0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)')
copy_info_pat = re.compile(prefix + r'Prof Copy Info (?P<opid>[0-9]+) (?P<src>[a-f0-9]+) (?P<dst>[a-f0-9]+) (?P<size>[0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)')
copy_info_old_pat = re.compile(prefix + r'Prof Copy Info (?P<opid>[0-9]+) (?P<src>[a-f0-9]+) (?P<dst>[a-f0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)')
fill_info_pat = re.compile(prefix + r'Prof Fill Info (?P<opid>[0-9]+) (?P<dst>[a-f0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)')
inst_create_pat = re.compile(prefix + r'Prof Inst Create (?P<opid>[0-9]+) (?P<inst>[a-f0-9]+) (?P<create>[0-9]+)')
inst_usage_pat = re.compile(prefix + r'Prof Inst Usage (?P<opid>[0-9]+) (?P<inst>[a-f0-9]+) (?P<mem>[a-f0-9]+) (?P<bytes>[0-9]+)')
inst_timeline_pat = re.compile(prefix + r'Prof Inst Timeline (?P<opid>[0-9]+) (?P<inst>[a-f0-9]+) (?P<create>[0-9]+) (?P<destroy>[0-9]+)')
user_info_pat = re.compile(prefix + r'Prof User Info (?P<pid>[a-f0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+) (?P<name>[$()a-zA-Z0-9_]+)')
task_wait_info_pat = re.compile(prefix + r'Prof Task Wait Info (?P<opid>[0-9]+) (?P<vid>[0-9]+) (?P<start>[0-9]+) (?P<ready>[0-9]+) (?P<end>[0-9]+)')
meta_wait_info_pat = re.compile(prefix + r'Prof Meta Wait Info (?P<opid>[0-9]+) (?P<hlr>[0-9]+) (?P<start>[0-9]+) (?P<ready>[0-9]+) (?P<end>[0-9]+)')
kind_pat = re.compile(prefix + r'Prof Task Kind (?P<tid>[0-9]+) (?P<name>[$()a-zA-Z0-9_<>.]+)')
kind_pat_over = re.compile(prefix + r'Prof Task Kind (?P<tid>[0-9]+) (?P<name>[$()a-zA-Z0-9_<>.]+) (?P<over>[0-1])')
variant_pat = re.compile(prefix + r'Prof Task Variant (?P<tid>[0-9]+) (?P<vid>[0-9]+) (?P<name>[$()a-zA-Z0-9_<>.]+)')
operation_pat = re.compile(prefix + r'Prof Operation (?P<opid>[0-9]+) (?P<kind>[0-9]+)')
multi_pat = re.compile(prefix + r'Prof Multi (?P<opid>[0-9]+) (?P<tid>[0-9]+)')
owner_pat = re.compile(prefix + r'Prof Slice Owner (?P<pid>[0-9]+) (?P<opid>[0-9]+)')
meta_desc_pat = re.compile(prefix + r'Prof Meta Desc (?P<hlr>[0-9]+) (?P<kind>[a-zA-Z0-9_ ]+)')
op_desc_pat = re.compile(prefix + r'Prof Op Desc (?P<opkind>[0-9]+) (?P<kind>[a-zA-Z0-9_ ]+)')
proc_desc_pat = re.compile(prefix + r'Prof Proc Desc (?P<pid>[a-f0-9]+) (?P<kind>[0-9]+)')
mem_desc_pat = re.compile(prefix + r'Prof Mem Desc (?P<mid>[a-f0-9]+) (?P<kind>[0-9]+) (?P<size>[0-9]+)')
# Extensions for messages
message_desc_pat = re.compile(prefix + r'Prof Message Desc (?P<mid>[0-9]+) (?P<desc>[a-zA-Z0-9_ ]+)')
message_info_pat = re.compile(prefix + r'Prof Message Info (?P<mid>[0-9]+) (?P<pid>[a-f0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)')
# Extensions for mapper calls
mapper_call_desc_pat = re.compile(prefix + r'Prof Mapper Call Desc (?P<mid>[0-9]+) (?P<desc>[a-zA-Z0-9_ ]+)')
mapper_call_info_pat = re.compile(prefix + r'Prof Mapper Call Info (?P<mid>[0-9]+) (?P<pid>[a-f0-9]+) (?P<uid>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)')
# Extensions for runtime calls
runtime_call_desc_pat = re.compile(prefix + r'Prof Runtime Call Desc (?P<rid>[0-9]+) (?P<desc>[a-zA-Z0-9_ ]+)')
runtime_call_info_pat = re.compile(prefix + r'Prof Runtime Call Info (?P<rid>[0-9]+) (?P<pid>[a-f0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)')
# Self-profiling
proftask_info_pat = re.compile(prefix + r'Prof ProfTask Info (?P<pid>[a-f0-9]+) (?P<opid>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)')

# Make sure this is up to date with lowlevel.h
processor_kinds = {
    1 : 'GPU',
    2 : 'CPU',
    3 : 'Utility',
    4 : 'IO',
}

# Make sure this is up to date with lowlevel.h
memory_kinds = {
    0 : 'GASNet Global',
    1 : 'System',
    2 : 'Registered',
    3 : 'Socket',
    4 : 'Zero-Copy',
    5 : 'Framebuffer',
    6 : 'Disk',
    7 : 'HDF5',
    8 : 'File',
    9 : 'L3 Cache',
    10 : 'L2 Cache',
    11 : 'L1 Cache',
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

def data_tsv_str(level, start, end, color, opacity, title, 
                 initiation, _in, out, prof_uid):
    # replace None with ''
    def xstr(s):
        return str(s or '')
    return xstr(level) + "\t" + xstr(start) + "\t" + xstr(end) + "\t" + \
           xstr(color) + "\t" + xstr(opacity) + "\t" + xstr(title) + "\t" + \
           xstr(initiation) + "\t" + xstr(_in) + "\t" + xstr(out) + "\t" + \
           xstr(prof_uid) + "\n"

def slugify(filename):
    # convert spaces to underscores
    slugified = filename.replace(" ", "_")
    # remove special characters
    slugified = slugified.translate(None, "!@#$%^&*(),/?<>\"':;{}[]|/+=`~")
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

def read_time(string):
    return long(string)/1000

class HasDependencies(object):
    def __init__(self):
        self.deps = {"in": set(), "out": set()}
        self.initiation_op = None
        self.initiation = ''
    
    def add_initiation_dependencies(self, state, op_dependencies, transitive_map):
        pass

    def attach_dependencies(self, state, op_dependencies, transitive_map):
        if self.op_id in transitive_map["out"]:
            self.deps["out"] |= op_dependencies[self.op_id]["out"]
        if self.op_id in transitive_map["in"]:
            self.deps["in"] |=  op_dependencies[self.op_id]["in"]

    def get_unique_tuple(self):
        assert self.proc is not None #TODO: move to owner
        cur_level = self.proc.max_levels - self.level
        return (self.proc.node_id, self.proc.proc_in_node, self.prof_uid)

class HasInitiationDependencies(HasDependencies):
    def __init__(self, initiation_op):
        HasDependencies.__init__(self)
        self.initiation_op = initiation_op
        self.initiation = initiation_op.op_id

    def add_initiation_dependencies(self, state, op_dependencies, transitive_map):
        unique_tuple = self.get_unique_tuple()
        if self.initiation in transitive_map["in"]: # this op exists
            for in_op_id in transitive_map["in"][self.initiation]:
                op_dependencies[in_op_id]["out"].add(unique_tuple)

    def attach_dependencies(self, state, op_dependencies, transitive_map):
        # add the in direction
        if self.initiation in transitive_map["in"]:
            op_tuples = set(map(lambda op: state.find_op(op).get_unique_tuple(),
                            transitive_map["in"][self.initiation]))
            self.deps["in"] |= op_tuples

    def get_color(self):
        return self.initiation_op.get_color()

class HasNoDependencies(HasDependencies):
    def __init__(self):
        HasDependencies.__init__(self)
    
    def add_initiation_dependencies(self, state, op_dependencies, transitive_map):
        pass

    def attach_dependencies(self, state, op_dependencies, transitive_map):
        pass

class TimeRange(object):
    def __init__(self, create, ready, start, stop):
        assert create <= ready
        assert ready <= start
        assert start <= stop
        self.create = create
        self.ready = ready
        self.start = start
        self.stop = stop

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

    def contains(self, other):
        if self.start <= other.start and \
            other.stop <= self.stop:
            return True
        return False


    def __repr__(self):
        return "Start: %d us  Stop: %d us  Total: %d us" % (
                self.start, self.end, self.total_time())


    def total_time(self):
        return self.stop - self.start

    # The following must be overridden by subclasses that need them
    def mapper_time(self):
        pass

    def active_time(self):
        pass

    def application_time(self):
        pass

    def meta_time(self):
        pass

class Processor(object):
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
        self.time_points = list()

    def get_short_text(self):
        return self.kind + " Proc " + str(self.proc_in_node)

    def add_task(self, task):
        task.proc = self
        self.tasks.append(task)

    def add_message(self, message):
        # treating messages like any other task
        message.proc = self
        self.tasks.append(message)

    def add_mapper_call(self, call):
        # treating mapper calls like any other task
        call.proc = self
        self.tasks.append(call)

    def add_runtime_call(self, call):
        # treating runtime calls like any other task
        call.proc = self
        self.tasks.append(call)

    def sort_time_range(self):
        time_points = list()
        for task in self.tasks:
            self.time_points.append(TimePoint(task.start, task, True))
            self.time_points.append(TimePoint(task.stop, task, False))
        self.time_points.sort(key=lambda p: p.time_key)
        free_levels = set()
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

    def add_initiation_dependencies(self, state, op_dependencies, transitive_map):
        for point in self.time_points:
            if point.first:
                point.thing.add_initiation_dependencies(state, op_dependencies, transitive_map)

    def attach_dependencies(self, state, op_dependencies, transitive_map):
        for point in self.time_points:
            if point.first:
                point.thing.attach_dependencies(state, op_dependencies, transitive_map)

    def emit_tsv(self, tsv_file, base_level):
        # iterate over tasks in start time order
        for point in self.time_points:
            if point.first:
                point.thing.emit_tsv(tsv_file, base_level,
                                     self.max_levels + 1,
                                     point.thing.level)
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
        return cmp(a.proc_id, b.proc_id)


class TimePoint(object):
    def __init__(self, time, thing, first):
        self.time = time
        self.thing = thing
        self.first = first
        self.time_key = 2*time + (0 if first is True else 1)
    def __cmp__(a, b):
        return cmp(a.time_key, b.time_key)

class Memory(object):
    def __init__(self, mem_id, kind, size):
        self.mem_id = mem_id
        # MEMORY:      tag:8 = 0x1e, owner_node:16,   (unused):28, mem_idx: 12
        # owner_node = mem_id[55:40]
        self.node_id = (mem_id >> 40) & ((1 << 16) - 1)
        self.mem_in_node = (mem_id) & ((1 << 12) - 1)
        self.kind = kind
        self.total_size = size
        self.instances = set()
        self.time_points = list()
        self.max_live_instances = None
        self.last_time = None

    def get_short_text(self):
        return self.kind + " Memory " + str(self.mem_in_node)

    def add_instance(self, inst):
        self.instances.add(inst)
        inst.mem = self

    def init_time_range(self, last_time):
        # Fill in any of our instances that are not complete with the last time
        for inst in self.instances:
            if inst.stop is None:
                inst.stop = last_time
        self.last_time = last_time 

    def sort_time_range(self):
        self.max_live_instances = 0
        for inst in self.instances:
            self.time_points.append(TimePoint(inst.create, inst, True))
            self.time_points.append(TimePoint(inst.stop, inst, False))
        # Keep track of which levels are free
        self.time_points.sort(key=lambda p: p.time_key)
        free_levels = set()
        # Iterate over all the points in sorted order
        for point in self.time_points:
            if point.first:
                # Find a level to assign this to
                if len(free_levels) > 0:
                    point.thing.set_level(free_levels.pop())
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
                                max_levels, point.thing.level)

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
            usage = float(current_size)/float(self.total_size) if self.total_size <> 0 else 0
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
        return cmp(a.mem_id, b.mem_id)

class Channel(object):
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.copies = set()
        self.time_points = list()
        self.max_live_copies = None 
        self.last_time = None

    def get_short_text(self):
        if self.src is None:
            return "Fill Channel"
        else:
            return "Mem to Mem Channel"

    def add_copy(self, copy):
        copy.chan = self
        self.copies.add(copy)

    def init_time_range(self, last_time):
        self.last_time = last_time

    def sort_time_range(self):
        self.max_live_copies = 0 
        for copy in self.copies:
            self.time_points.append(TimePoint(copy.start, copy, True))
            self.time_points.append(TimePoint(copy.stop, copy, False))
        # Keep track of which levels are free
        self.time_points.sort(key=lambda p: p.time_key)
        free_levels = set()
        # Iterate over all the points in sorted order
        for point in self.time_points:
            if point.first:
                if len(free_levels) > 0:
                    point.thing.level = free_levels.pop()
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
                                max_levels, point.thing.level)

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
        if self.src is None:
            return 'Fill ' + self.dst.__repr__() + ' Channel'
        else:
            return self.src.__repr__() + ' to ' + self.dst.__repr__() + ' Channel'

    def __cmp__(a, b):
        return cmp(a.dst, b.dst)

class WaitInterval(object):
    def __init__(self, start, ready, end):
        self.start = start
        self.ready = ready
        self.end = end

class TaskKind(object):
    def __init__(self, task_id, name):
        self.task_id = task_id
        self.name = name

    def __repr__(self):
        return self.name

class StatObject(object):
    def __init__(self):
        self.total_calls = defaultdict(int)
        self.total_execution_time = defaultdict(int)
        self.all_calls = defaultdict(list)
        self.max_call = defaultdict(int)
        self.min_call = defaultdict(int)

    def get_total_execution_time(self):
        total_execution_time = 0
        for proc_exec_time in self.total_execution_time.itervalues():
            total_execution_time += proc_exec_time
        return total_execution_time

    def get_total_calls(self):
        total_calls = 0
        for proc_calls in self.total_calls.itervalues():
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
        procs = sorted(self.total_calls.iterkeys())
        total_execution_time = self.get_total_execution_time()
        total_calls = self.get_total_calls()

        avg = float(total_execution_time) / float(total_calls)
        max_call = max(self.max_call.values())
        min_call = min(self.min_call.values())
        stddev = 0
        for proc_calls in self.all_calls.values():
            for call in proc_calls:
                diff = float(call) - avg
                stddev += sqrt(diff * diff)
        stddev /= float(total_calls)
        stddev = sqrt(stddev)
        max_dev = (float(max_call) - avg) / stddev if stddev != 0.0 else 0.0
        min_dev = (float(min_call) - avg) / stddev if stddev != 0.0 else 0.0

        print('  '+repr(self))
        self.print_task_stat(total_calls, total_execution_time,
                max_call, max_dev, min_call, min_dev)
        print()

        if verbose and len(procs) > 1:
            for proc in sorted(self.total_calls.iterkeys()):
                avg = float(self.total_execution_time[proc]) / float(self.total_calls[proc]) \
                        if self.total_calls[proc] > 0 else 0
                stddev = 0
                for call in self.all_calls[proc]:
                    diff = float(call) - avg
                    stddev += sqrt(diff * diff)
                stddev /= float(self.total_calls[proc])
                stddev = sqrt(stddev)
                max_dev = (float(self.max_call[proc]) - avg) / stddev if stddev != 0.0 else 0.0
                min_dev = (float(self.min_call[proc]) - avg) / stddev if stddev != 0.0 else 0.0

                print('    On ' + repr(proc))
                self.print_task_stat(self.total_calls[proc],
                        self.total_execution_time[proc],
                        self.max_call[proc], max_dev,
                        self.min_call[proc], min_dev)
                print()


class Variant(StatObject):
    def __init__(self, variant_id, name):
        StatObject.__init__(self)
        self.variant_id = variant_id
        self.name = name
        self.op = dict()
        self.task_kind = None
        self.color = None

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
        return self.name

class Base(object):
    def __init__(self):
        self.prof_uid = get_prof_uid()
        self.level = None

    def set_level(self, level):
        self.level = level

    def get_unique_tuple(self):
        assert self.proc is not None
        cur_level = self.proc.max_levels - self.level
        return (self.proc.node_id, self.proc.proc_in_node, self.prof_uid)

class Operation(Base):
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

    def __repr__(self):
        if self.is_task:
            assert self.variant is not None
            title = self.variant.task_kind.name if self.variant.task_kind is not None else 'unnamed'
            if self.variant.name <> None and self.variant.name.find("unnamed") > 0:
                title += ' ['+self.variant.name+']'
            return title+' '+self.get_info()
        elif self.is_multi:
            assert self.task_kind is not None
            if self.task_kind.name is not None:
                return self.task_kind.name+' '+self.get_info()
            else:
                return 'Task '+str(self.task_kind.task_id)+' '+self.get_info()
        elif self.is_proftask:
            return 'ProfTask' + (' <{:d}>'.format(self.op_id) if self.op_id > 0 else '')
        else:
            if self.kind is None:
                return 'Operation '+self.get_info()
            else:
                return self.kind+' Operation '+self.get_info()

class HasWaiters(object):
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

    def emit_tsv(self, tsv_file, base_level, max_levels, level):
        title = repr(self)
        initiation = str(self.initiation)
        color = self.get_color()

        _in = json.dumps(list(self.deps["in"])) if len(self.deps["in"]) > 0 else ""
        out = json.dumps(list(self.deps["out"])) if len(self.deps["out"]) > 0 else ""

        if len(self.wait_intervals) > 0:
            start = self.start
            cur_level = base_level + (max_levels - level)
            for wait_interval in self.wait_intervals:
                init = data_tsv_str(level = cur_level,
                                    start = start,
                                    end = wait_interval.start,
                                    color = color,
                                    opacity = "1.0",
                                    title = title,
                                    initiation = initiation,
                                    _in = _in,
                                    out = out,
                                    prof_uid = self.prof_uid)
                wait = data_tsv_str(level = cur_level,
                                    start = wait_interval.start,
                                    end = wait_interval.ready,
                                    color = color,
                                    opacity = "0.15",
                                    title = title + " (waiting)",
                                    initiation = initiation,
                                    _in = _in,
                                    out = out,
                                    prof_uid = self.prof_uid)
                ready = data_tsv_str(level = cur_level,
                                     start = wait_interval.ready,
                                     end = wait_interval.end,
                                     color = color,
                                     opacity = "0.45",
                                     title = title + " (ready)",
                                     initiation = initiation,
                                     _in = _in,
                                     out = out,
                                     prof_uid = None)

                tsv_file.write(init)
                tsv_file.write(wait)
                tsv_file.write(ready)
                start = max(start, wait_interval.end)
            if start < self.stop:
                end = data_tsv_str(level = cur_level,
                                   start = start,
                                   end = self.stop,
                                   color = color,
                                   opacity = "1.0",
                                   title = title,
                                   initiation = initiation,
                                   _in = _in,
                                   out = out,
                                   prof_uid = None)

                tsv_file.write(end)
        else:
            line = data_tsv_str(level = base_level + (max_levels - level),
                                start = self.start,
                                end = self.stop,
                                color = color,
                                opacity = "1.0",
                                title = title,
                                initiation = initiation,
                                _in = _in,
                                out = out,
                                prof_uid = self.prof_uid)
            tsv_file.write(line)

class Task(Operation, TimeRange, HasDependencies, HasWaiters):
    def __init__(self, variant, op, create, ready, start, stop):
        Operation.__init__(self, op.op_id)
        HasDependencies.__init__(self)
        HasWaiters.__init__(self)
        TimeRange.__init__(self, create, ready, start, stop)

        self.base_op = op
        self.variant = variant
        self.initiation = ''
        self.is_task = True

    def assign_color(self, color):
        assert self.color is None
        assert self.base_op.color is None
        assert self.variant is not None
        assert self.variant.color is not None
        self.color = self.variant.color
        self.base_op.color = self.color

    def emit_tsv(self, tsv_file, base_level, max_levels, level):
        return HasWaiters.emit_tsv(self, tsv_file, base_level, max_levels, level)

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
        title = self.variant.task_kind.name if self.variant.task_kind is not None else 'unnamed'
        if self.variant.name <> None and self.variant.name.find("unnamed") > 0:
            title += ' ['+self.variant.name+']'
        return title+' '+self.get_info()

class MetaTask(Base, TimeRange, HasInitiationDependencies, HasWaiters):
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

    def emit_tsv(self, tsv_file, base_level, max_levels, level):
        return HasWaiters.emit_tsv(self, tsv_file, base_level, max_levels, level)

    def __repr__(self):
        assert self.variant is not None
        return self.variant.name

class ProfTask(Base, TimeRange, HasNoDependencies):
    def __init__(self, op, create, ready, start, stop):
        Base.__init__(self)
        HasNoDependencies.__init__(self)
        TimeRange.__init__(self, None, ready, start, stop)
        self.proftask_id = op.op_id
        self.color = '#FFC0CB'  # Pink
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

    def emit_tsv(self, tsv_file, base_level, max_levels, level):
        tsv_line = data_tsv_str(level = base_level + (max_levels - level),
                                start = self.start,
                                end = self.stop,
                                color = self.get_color(),
                                opacity = "1.0",
                                title = repr(self),
                                initiation = None,
                                _in = None,
                                out = None,
                                prof_uid = self.prof_uid)
        tsv_file.write(tsv_line)

    def __repr__(self):
        return 'ProfTask' + (' <{:d}>'.format(self.proftask_id) if self.proftask_id > 0 else '')

class UserMarker(Base, TimeRange, HasNoDependencies):
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

    def emit_tsv(self, tsv_file, base_level, max_levels, level):
        tsv_line = data_tsv_str(level = base_level + (max_levels - level),
                                start = self.start,
                                end = self.stop,
                                color = self.get_color(),
                                opacity = "1.0",
                                title = repr(self),
                                initiation = None,
                                _in = None,
                                out = None,
                                prof_uid = self.prof_uid)
        tsv_file.write(tsv_line)

    def __repr__(self):
        return 'User Marker "'+self.name+'"'

class Copy(Base, TimeRange, HasInitiationDependencies):
    def __init__(self, src, dst, initiation_op, size, create, ready, start, stop):
        Base.__init__(self)
        HasInitiationDependencies.__init__(self, initiation_op)
        TimeRange.__init__(self, create, ready, start, stop)
        self.src = src
        self.dst = dst
        self.size = size
        self.chan = None

    def get_color(self):
        # Get the color from the initiator
        return self.initiation_op.get_color()

    def __repr__(self):
        return 'Copy size='+str(self.size)

    def get_unique_tuple(self):
        assert self.chan is not None
        cur_level = self.chan.max_live_copies+1 - self.level
        return (str(self.chan), self.prof_uid)

    def emit_tsv(self, tsv_file, base_level, max_levels, level):
        assert self.level is not None
        assert self.start is not None
        assert self.stop is not None
        copy_name = repr(self)
        _in = json.dumps(self.deps["in"]) if len(self.deps["in"]) > 0 else ""
        out = json.dumps(self.deps["out"]) if len(self.deps["out"]) > 0 else ""

        tsv_line = data_tsv_str(level = base_level + (max_levels - level),
                                start = self.start,
                                end = self.stop,
                                color = self.get_color(),
                                opacity = "1.0",
                                title = copy_name,
                                initiation = self.initiation,
                                _in = _in,
                                out = out,
                                prof_uid = self.prof_uid)
        tsv_file.write(tsv_line)

class Fill(Base, TimeRange, HasInitiationDependencies):
    def __init__(self, dst, initiation_op, create, ready, start, stop):
        Base.__init__(self)
        HasInitiationDependencies.__init__(self, initiation_op)
        TimeRange.__init__(self, create, ready, start, stop)
        self.dst = dst
        self.chan = None

    def __repr__(self):
        return 'Fill'

    def get_unique_tuple(self):
        assert self.chan is not None
        cur_level = self.chan.max_live_copies+1 - self.level
        return (str(self.chan), self.prof_uid)

    def emit_tsv(self, tsv_file, base_level, max_levels, level):
        fill_name = repr(self)
        _in = json.dumps(self.deps["in"]) if len(self.deps["in"]) > 0 else ""
        out = json.dumps(self.deps["out"]) if len(self.deps["out"]) > 0 else ""

        tsv_line = data_tsv_str(level = base_level + (max_levels - level),
                                start = self.start,
                                end = self.stop,
                                color = self.get_color(),
                                opacity = "1.0",
                                title = fill_name,
                                initiation = self.initiation,
                                _in = _in,
                                out = out,
                                prof_uid = self.prof_uid)
        tsv_file.write(tsv_line)

class Instance(Base, TimeRange, HasInitiationDependencies):
    def __init__(self, inst_id, initiation_op):
        Base.__init__(self)
        HasInitiationDependencies.__init__(self, initiation_op)
        TimeRange.__init__(self, None, None, None, None)

        self.inst_id = inst_id
        self.mem = None
        self.size = None

    def get_unique_tuple(self):
        assert self.mem is not None
        cur_level = self.mem.max_live_copies+1 - self.level
        return (str(self.mem), self.prof_uid)

    def emit_tsv(self, tsv_file, base_level, max_levels, level):
        assert self.level is not None
        assert self.create is not None
        assert self.stop is not None
        inst_name = repr(self)

        _in = json.dumps(self.deps["in"]) if len(self.deps["in"]) > 0 else ""
        out = json.dumps(self.deps["out"]) if len(self.deps["out"]) > 0 else ""

        tsv_line = data_tsv_str(level = base_level + (max_levels - level),
                                start = self.create,
                                end = self.stop,
                                color = self.get_color(),
                                opacity = "1.0",
                                title = inst_name,
                                initiation = self.initiation,
                                _in = _in,
                                out = out,
                                prof_uid = self.prof_uid)

        tsv_file.write(tsv_line)


    def get_color(self):
        # Get the color from the operation
        return self.initiation_op.get_color()

    def __repr__(self):
        # Check to see if we got a profiling callback
        if self.size is not None:
            unit = 'B'
            unit_size = self.size
            if self.size > (1024*1024*1024):
                unit = 'GB'
                unit_size /= (1024*1024*1024)
            elif self.size > (1024*1024):
                unit = 'MB'
                unit_size /= (1024*1024)
            elif self.size > 1024:
                unit = 'KB'
                unit_size /= 1024
            size_pretty = str(unit_size) + unit
        else:
            size_pretty = 'Unknown'

        return ("Instance {} Size={}"
                .format(str(hex(self.inst_id)),
                        size_pretty))

class MessageKind(StatObject):
    def __init__(self, message_id, desc):
        StatObject.__init__(self)
        self.message_id = message_id
        self.desc = desc
        self.color = None

    def __eq__(self, other):
        return self.message_id == other.message_id

    def assign_color(self, color):
        assert self.color is None
        self.color = color

    def __repr__(self):
        return self.desc

class Message(Base, TimeRange, HasNoDependencies):
    def __init__(self, kind, start, stop):
        Base.__init__(self)
        TimeRange.__init__(self, None, None, start, stop)
        HasNoDependencies.__init__(self)
        self.kind = kind

    def get_color(self):
        return self.kind.color

    def active_time(self):
        return self.total_time()

    def application_time(self):
        return 0

    def meta_time(self):
        return 0

    def mapper_time(self):
        return 0

    def emit_tsv(self, tsv_file, base_level, max_levels, level):
        tsv_line = data_tsv_str(level = base_level + (max_levels - level),
                                start = self.start,
                                end = self.stop,
                                color = self.get_color(),
                                opacity = "1.0",
                                title = repr(self),
                                initiation = None,
                                _in = None,
                                out = None,
                                prof_uid = self.prof_uid)
        tsv_file.write(tsv_line)

    def __repr__(self):
        return 'Message '+str(self.kind)

class MapperCallKind(StatObject):
    def __init__(self, mapper_call_kind, desc):
        StatObject.__init__(self)
        self.mapper_call_kind = mapper_call_kind
        self.desc = desc
        self.color = None

    def __eq__(self, other):
        return self.mapper_call_kind == other.mapper_call_kind

    def __repr__(self):
        return self.desc

    def assign_color(self, color):
        assert self.color is None
        self.color = color

class MapperCall(Base, TimeRange, HasInitiationDependencies):
    def __init__(self, kind, initiation_op, start, stop):
        Base.__init__(self)
        TimeRange.__init__(self, None, None, start, stop)
        HasInitiationDependencies.__init__(self, initiation_op)
        self.kind = kind

    def get_color(self):
        assert self.kind is not None and self.kind.color is not None
        return self.kind.color

    def emit_tsv(self, tsv_file, base_level, max_levels, level):
        title = repr(self)
        _in = json.dumps(list(self.deps["in"])) if len(self.deps["in"]) > 0 else ""
        out = json.dumps(list(self.deps["out"])) if len(self.deps["out"]) > 0 else ""

        tsv_line = data_tsv_str(level = base_level + (max_levels - level),
                                start = self.start,
                                end = self.stop,
                                color = self.get_color(),
                                opacity = "1.0",
                                title = repr(self),
                                initiation = self.initiation,
                                _in = _in,
                                out = out,
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
    def __init__(self, runtime_call_kind, desc):
        StatObject.__init__(self)
        self.runtime_call_kind = runtime_call_kind
        self.desc = desc
        self.color = None

    def __eq__(self, other):
        return self.runtime_call_kind == other.runtime_call_kind

    def assign_color(self, color):
        assert self.color is None
        self.color = color

    def __repr__(self):
        return self.desc

class RuntimeCall(Base, TimeRange, HasNoDependencies):
    def __init__(self, kind, start, stop):
        Base.__init__(self)
        TimeRange.__init__(self, None, None, start, stop)
        HasNoDependencies.__init__(self)
        self.kind = kind

    def emit_tsv(self, tsv_file, base_level, max_levels, level):
        tsv_line = data_tsv_str(level = base_level + (max_levels - level),
                                start = self.start,
                                end = self.stop,
                                color = self.get_color(),
                                opacity = "1.0",
                                title = repr(self),
                                initiation = None,
                                _in = None,
                                out = None,
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
    def __init__(self, size):
        self.register = ''
        # Initialize the register with all zeros
        needed_bits = int(log(size,2))+1
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
    def __init__(self, state):
        self.state = state
        self.application_tasks = set()
        self.meta_tasks = set()
        self.mapper_tasks = set()
        self.runtime_tasks = set()
        self.message_tasks = set()
        for proc in state.processors.itervalues():
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
                elif isinstance(task, Message):
                    self.message_tasks.add(task.kind)
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

        print("  -------------------------")
        print("  Message Statistics")
        print("  -------------------------")
        for kind in sorted(self.message_tasks,
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
    def __init__(self):
        self.processors = {}
        self.memories = {}
        self.channels = {}
        self.task_kinds = {}
        self.variants = {}
        self.meta_variants = {}
        self.op_kinds = {}
        self.operations = {}
        self.multi_tasks = {}
        self.first_times = {}
        self.last_times = {}
        self.last_time = 0L
        self.message_kinds = {}
        self.messages = {}
        self.mapper_call_kinds = {}
        self.mapper_calls = {}
        self.runtime_call_kinds = {}
        self.runtime_calls = {}
        self.instances = {}
        self.has_spy_state = False

    def parse_log_file(self, file_name, verbose):
        skipped = 0
        with open(file_name, 'rb') as log:  
            matches = 0
            # Keep track of the first and last times
            first_time = 0L
            last_time = 0L
            for line in log:
                if not self.has_spy_state and \
                    (legion_spy.config_pat.match(line) or \
                     legion_spy.detailed_config_pat.match(line)):
                    self.has_spy_state = True
                matches += 1  
                m = task_info_pat.match(line)
                if m is not None:
                    self.log_task_info(long(m.group('opid')),
                                       int(m.group('vid')),
                                       int(m.group('pid'),16),
                                       read_time(m.group('create')),
                                       read_time(m.group('ready')),
                                       read_time(m.group('start')),
                                       read_time(m.group('stop')))
                    continue
                m = meta_info_pat.match(line)
                if m is not None:
                    self.log_meta_info(long(m.group('opid')),
                                       int(m.group('hlr')),
                                       int(m.group('pid'),16),
                                       read_time(m.group('create')),
                                       read_time(m.group('ready')),
                                       read_time(m.group('start')),
                                       read_time(m.group('stop')))
                    continue
                m = copy_info_pat.match(line)
                if m is not None:
                    self.log_copy_info(long(m.group('opid')),
                                       int(m.group('src'),16),
                                       int(m.group('dst'),16),
                                       int(m.group('size')),
                                       read_time(m.group('create')),
                                       read_time(m.group('ready')),
                                       read_time(m.group('start')),
                                       read_time(m.group('stop')))
                    continue
                m = copy_info_old_pat.match(line)
                if m is not None:
                    self.log_copy_info(long(m.group('opid')),
                                       int(m.group('src'),16),
                                       int(m.group('dst'),16),
                                       0,
                                       read_time(m.group('create')),
                                       read_time(m.group('ready')),
                                       read_time(m.group('start')),
                                       read_time(m.group('stop')))
                    continue
                m = fill_info_pat.match(line)
                if m is not None:
                    self.log_fill_info(long(m.group('opid')),
                                       int(m.group('dst'),16),
                                       read_time(m.group('create')),
                                       read_time(m.group('ready')),
                                       read_time(m.group('start')),
                                       read_time(m.group('stop')))
                    continue
                m = inst_create_pat.match(line)
                if m is not None:
                    self.log_inst_create(long(m.group('opid')),
                                         int(m.group('inst'),16),
                                         read_time(m.group('create')))
                    continue
                m = inst_usage_pat.match(line)
                if m is not None:
                    self.log_inst_usage(long(m.group('opid')),
                                        int(m.group('inst'),16),
                                        int(m.group('mem'),16),
                                        long(m.group('bytes')))
                    continue
                m = inst_timeline_pat.match(line)
                if m is not None:
                    self.log_inst_timeline(long(m.group('opid')),
                                           int(m.group('inst'),16),
                                           read_time(m.group('create')),
                                           read_time(m.group('destroy')))
                    continue
                m = user_info_pat.match(line)
                if m is not None:
                    self.log_user_info(int(m.group('pid'), 16),
                                       read_time(m.group('start')),
                                       read_time(m.group('stop')),
                                       m.group('name'))
                    continue
                m = task_wait_info_pat.match(line)
                if m is not None:
                    self.log_task_wait_info(long(m.group('opid')),
                                            int(m.group('vid')),
                                            read_time(m.group('start')),
                                            read_time(m.group('ready')),
                                            read_time(m.group('end')))
                    continue
                m = meta_wait_info_pat.match(line)
                if m is not None:
                    self.log_meta_wait_info(long(m.group('opid')),
                                            int(m.group('hlr')),
                                            read_time(m.group('start')),
                                            read_time(m.group('ready')),
                                            read_time(m.group('end')))
                    continue
                # Put this one first for maximal munch
                m = kind_pat_over.match(line)
                if m is not None:
                    self.log_kind(int(m.group('tid')),
                                  m.group('name'), int(m.group('over')))
                    continue
                m = kind_pat.match(line)
                if m is not None:
                    self.log_kind(int(m.group('tid')), m.group('name'), 1)
                    continue
                m = variant_pat.match(line)
                if m is not None:
                    self.log_variant(int(m.group('tid')),
                                     int(m.group('vid')),
                                     m.group('name'))
                    continue
                m = operation_pat.match(line)
                if m is not None:
                    self.log_operation(long(m.group('opid')),
                                       int(m.group('kind')))
                    continue
                m = multi_pat.match(line)
                if m is not None:
                    self.log_multi(long(m.group('opid')),
                                   int(m.group('tid')))
                    continue
                m = owner_pat.match(line)
                if m is not None:
                    self.log_slice_owner(long(m.group('pid')),
                                         long(m.group('opid')))
                    continue
                m = meta_desc_pat.match(line)
                if m is not None:
                    self.log_meta_desc(int(m.group('hlr')),
                                       m.group('kind'))
                    continue
                m = op_desc_pat.match(line)
                if m is not None:
                    self.log_op_desc(int(m.group('opkind')),
                                     m.group('kind'))
                    continue
                m = proc_desc_pat.match(line)
                if m is not None:
                    kind = int(m.group('kind'))
                    assert kind in processor_kinds
                    self.log_proc_desc(int(m.group('pid'),16),
                                       processor_kinds[kind])
                    continue
                m = mem_desc_pat.match(line)
                if m is not None:
                    kind = int(m.group('kind'))
                    assert kind in memory_kinds
                    self.log_mem_desc(int(m.group('mid'),16),
                                      memory_kinds[kind],
                                      long(m.group('size')))
                    continue
                m = message_desc_pat.match(line)
                if m is not None:
                    self.log_message_desc(int(m.group('mid')),
                                          m.group('desc'))
                    continue
                m = message_info_pat.match(line)
                if m is not None:
                    self.log_message_info(int(m.group('mid')),
                                          int(m.group('pid'),16),
                                          read_time(m.group('start')),
                                          read_time(m.group('stop')))
                    continue
                m = mapper_call_desc_pat.match(line)
                if m is not None:
                    self.log_mapper_call_desc(int(m.group('mid')),
                                              m.group('desc'))
                    continue
                m = mapper_call_info_pat.match(line)
                if m is not None:
                    self.log_mapper_call_info(int(m.group('mid')),
                                              int(m.group('pid'),16),
                                              int(m.group('uid')),
                                              read_time(m.group('start')),
                                              read_time(m.group('stop')))
                    continue
                m = runtime_call_desc_pat.match(line)
                if m is not None:
                    self.log_runtime_call_desc(int(m.group('rid')),
                                               m.group('desc'))
                    continue
                m = runtime_call_info_pat.match(line)
                if m is not None:
                    self.log_runtime_call_info(int(m.group('rid')),
                                               int(m.group('pid'),16),
                                               read_time(m.group('start')),
                                               read_time(m.group('stop')))
                    continue
                m = proftask_info_pat.match(line)
                if m is not None:
                    self.log_proftask_info(int(m.group('pid'),16),
                                           long(m.group('opid')),
                                           read_time(m.group('start')),
                                           read_time(m.group('stop')))
                    continue
                # If we made it here then we failed to match
                matches -= 1 
                skipped += 1
                if verbose:
                    print('Skipping line: %s' % line.strip())
        if skipped > 0:
            print('WARNING: Skipped %d lines in %s' % (skipped, file_name))
        return matches

    def log_task_info(self, op_id, variant_id, proc_id,
                      create, ready, start, stop):
        variant = self.find_variant(variant_id)
        task = self.find_task(op_id, variant, create, ready, start, stop)
        if stop > self.last_time:
            self.last_time = stop
        proc = self.find_processor(proc_id)
        proc.add_task(task)

    def log_meta_info(self, op_id, hlr, proc_id, 
                      create, ready, start, stop):
        op = self.find_op(op_id)
        variant = self.find_meta_variant(hlr)
        meta = self.create_meta(variant, op, create, ready, start, stop)
        if stop > self.last_time:
            self.last_time = stop
        proc = self.find_processor(proc_id)
        proc.add_task(meta)

    def log_copy_info(self, op_id, src_mem, dst_mem, size,
                      create, ready, start, stop):
        op = self.find_op(op_id)
        src = self.find_memory(src_mem)
        dst = self.find_memory(dst_mem)
        copy = self.create_copy(src, dst, op, size, create, ready, start, stop)
        if stop > self.last_time:
            self.last_time = stop
        channel = self.find_channel(src, dst)
        channel.add_copy(copy)

    def log_fill_info(self, op_id, dst_mem,
                      create, ready, start, stop):
        op = self.find_op(op_id)
        dst = self.find_memory(dst_mem)
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
            inst.create = create

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
        inst.create = create
        inst.stop = destroy
        if destroy > self.last_time:
            self.last_time = destroy 

    def log_user_info(self, proc_id, start, stop, name):
        proc = self.find_processor(proc_id)
        user = self.create_user_marker(name)
        user.start = start
        user.stop = stop
        if stop > self.last_time:
            self.last_time = stop 
        proc.add_task(user)

    def log_task_wait_info(self, op_id, variant_id, start, ready, end):
        variant = self.find_variant(variant_id)
        task = self.find_task(op_id, variant)
        assert ready >= start
        assert end >= ready
        task.add_wait_interval(start, ready, end)

    def log_meta_wait_info(self, op_id, hlr, start, ready, end):
        op = self.find_op(op_id)
        variant = self.find_meta_variant(hlr)
        assert ready >= start
        assert end >= ready
        assert op_id in variant.op
        variant.op[op_id].add_wait_interval(start, ready, end)

    def log_kind(self, task_id, name, overwrite):
        if task_id not in self.task_kinds:
            self.task_kinds[task_id] = TaskKind(task_id, name)
        elif overwrite == 1:
            self.task_kinds[task_id].name = name

    def log_variant(self, task_id, variant_id, name):
        assert task_id in self.task_kinds
        task_kind = self.task_kinds[task_id]
        if variant_id not in self.variants:
            self.variants[variant_id] = Variant(variant_id, name)
        else:
            self.variants[variant_id].name = name
        self.variants[variant_id].set_task_kind(task_kind)

    def log_operation(self, op_id, kind):
        op = self.find_op(op_id)
        assert kind in self.op_kinds
        op.kind_num = kind
        op.kind = self.op_kinds[kind]

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

    def log_meta_desc(self, hlr, name):
        if hlr not in self.meta_variants:
            self.meta_variants[hlr] = Variant(hlr, name)
        else:
            self.meta_variants[hlr].name = name

    def log_proc_desc(self, proc_id, kind):
        if proc_id not in self.processors:
            self.processors[proc_id] = Processor(proc_id, kind)
        else:
            self.processors[proc_id].kind = kind

    def log_mem_desc(self, mem_id, kind, size):
        if mem_id not in self.memories:
            self.memories[mem_id] = Memory(mem_id, kind, size)
        else:
            self.memories[mem_id].kind = kind

    def log_op_desc(self, kind, name):
        if kind not in self.op_kinds:
            self.op_kinds[kind] = name

    def log_message_desc(self, kind, desc):
        if kind not in self.message_kinds:
            self.message_kinds[kind] = MessageKind(kind, desc) 

    def log_message_info(self, kind, proc_id, start, stop):
        assert start <= stop
        assert kind in self.message_kinds
        if stop > self.last_time:
            self.last_time = stop
        message = Message(self.message_kinds[kind], start, stop)
        proc = self.find_processor(proc_id)
        proc.add_message(message)

    def log_mapper_call_desc(self, kind, desc):
        if kind not in self.mapper_call_kinds:
            self.mapper_call_kinds[kind] = MapperCallKind(kind, desc)

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
        proc = self.find_processor(proc_id)
        proc.add_mapper_call(call)

    def log_runtime_call_desc(self, kind, desc):
        if kind not in self.runtime_call_kinds:
            self.runtime_call_kinds[kind] = RuntimeCallKind(kind, desc)

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
        proftask = ProfTask(Operation(op_id), start, start, start, stop) 
        proc = self.find_processor(proc_id)
        proc.add_task(proftask)

    def find_processor(self, proc_id):
        if proc_id not in self.processors:
            self.processors[proc_id] = Processor(proc_id, None)
        return self.processors[proc_id]

    def find_memory(self, mem_id):
        if mem_id not in self.memories:
            # use 'system memory' as the default kind
            self.memories[mem_id] = Memory(mem_id, 1, None)
        return self.memories[mem_id]

    def find_channel(self, src, dst):
        if src is not None:
            key = (src,dst)
            if key not in self.channels:
                self.channels[key] = Channel(src,dst)
            return self.channels[key]
        else:
            # This is a fill channel
            if dst not in self.channels:
                self.channels[dst] = Channel(None,dst)
            return self.channels[dst]

    def find_variant(self, variant_id):
        if variant_id not in self.variants:
            self.variants[variant_id] = Variant(variant_id, None)
        return self.variants[variant_id]

    def find_meta_variant(self, hlr_id):
        if hlr_id not in self.meta_variants:
            self.meta_variants[hlr_id] = Variant(hlr_id, None)
        return self.meta_variants[hlr_id]

    def find_op(self, op_id):
        if op_id not in self.operations:
            self.operations[op_id] = Operation(op_id) 
        return self.operations[op_id]

    def find_task(self, op_id, variant, create=None, ready=None, start=None, stop=None):
        op = self.find_op(op_id)
        # Upgrade this operation to a task if necessary
        if not op.is_task:
            assert create is not None
            assert ready is not None
            assert start is not None
            assert stop is not None
            op = Task(variant, op, create, ready, start, stop) 
            variant.op[op_id] = op
            self.operations[op_id] = op
        else:
            assert op.variant == variant
        return op

    def create_meta(self, variant, op, create, ready, start, stop):
        result = MetaTask(variant, op, create, ready, start, stop)
        variant.op[op.op_id] = result
        return result

    def create_copy(self, src, dst, op, size, create, ready, start, stop):
        return Copy(src, dst, op, size, create, ready, start, stop)

    def create_fill(self, dst, op, create, ready, start, stop):
        return Fill(dst, op, create, ready, start, stop)

    def create_instance(self, inst_id, op):
        # neither instance id nor op id are unique on their own
        key = (inst_id, op.op_id)
        if key not in self.instances:
            inst = Instance(inst_id, op)
            self.instances[key] = inst
        else:
            inst = self.instances[key]
        return inst

    def create_user_marker(self, name):
        return UserMarker(name)

    def build_time_ranges(self):
        assert self.last_time is not None 
        # Processors first
        for proc in self.processors.itervalues():
            proc.last_time = self.last_time
            proc.sort_time_range()
        for mem in self.memories.itervalues():
            mem.init_time_range(self.last_time)
            mem.sort_time_range()
        for channel in self.channels.itervalues():
            channel.init_time_range(self.last_time)
            channel.sort_time_range()

    def print_processor_stats(self, verbose):
        print('****************************************************')
        print('   PROCESSOR STATS')
        print('****************************************************')
        for proc in sorted(self.processors.itervalues()):
            proc.print_stats(verbose)
        print

    def print_memory_stats(self, verbose):
        print('****************************************************')
        print('   MEMORY STATS')
        print('****************************************************')
        for mem in sorted(self.memories.itervalues()):
            mem.print_stats(verbose)
        print

    def print_channel_stats(self, verbose):
        print('****************************************************')
        print('   CHANNEL STATS')
        print('****************************************************')
        for channel in sorted(self.channels.itervalues()):
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
                     len(self.op_kinds) + len(self.message_kinds) + \
                     len(self.mapper_call_kinds) + len(self.runtime_call_kinds)
        # Use a LFSR to randomize these colors
        lsfr = LFSR(num_colors)
        num_colors = lsfr.get_max_value()
        op_colors = {}
        for variant in self.variants.itervalues():
            variant.compute_color(lsfr.get_next(), num_colors)
        for variant in self.meta_variants.itervalues():
            if variant.variant_id == 1: # Remote message
                variant.assign_color('#006600') # Evergreen
            elif variant.variant_id == 2: # Post-Execution
                variant.assign_color('#333399') # Deep Purple
            elif variant.variant_id == 6: # Garbage Collection
                variant.assign_color('#990000') # Crimson
            elif variant.variant_id == 7: # Logical Dependence Analysis
                variant.assign_color('#0000FF') # Duke Blue
            elif variant.variant_id == 8: # Operation Physical Analysis
                variant.assign_color('#009900') # Green
            elif variant.variant_id == 9: # Task Physical Analysis
                variant.assign_color('#009900') #Green
            else:
                variant.compute_color(lsfr.get_next(), num_colors)
        for kind in self.op_kinds.iterkeys():
            op_colors[kind] = color_helper(lsfr.get_next(), num_colors)
        # Now we need to assign all the operations colors
        for op in self.operations.itervalues():
            op.assign_color(op_colors)
        # Assign all the message kinds different colors
        for kinds in (self.message_kinds,
                      self.mapper_call_kinds,
                      self.runtime_call_kinds):
            for kind in kinds.itervalues():
                kind.assign_color(color_helper(lsfr.get_next(), num_colors))

    def show_copy_matrix(self, output_prefix):
        template_file_name = os.path.join(dirname(sys.argv[0]),
                "legion_prof_copy.html.template")
        tsv_file_name = output_prefix + ".tsv"
        html_file_name = output_prefix + ".html"
        print('Generating copy visualization files %s and %s' % (tsv_file_name,html_file_name))

        def node_id(memory):
            return (memory.mem_id >> 23) & ((1 << 5) - 1)
        memories = sorted(self.memories.itervalues())

        tsv_file = open(tsv_file_name, "w")
        tsv_file.write("source\ttarget\tremote\ttotal\tcount\taverage\tbandwidth\n")
        for i in range(0, len(memories)):
            for j in range(0, len(memories)):
                src = memories[i]
                dst = memories[j]
                is_remote = node_id(src) <> node_id(dst) or \
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
        if (not exists(dirname)):
            return dirname
        # if the dirname exists, loop through dirname.i until we
        # find one that doesn't exist
        i = 1
        while (True):
            potential_dir = dirname + "." + str(i)
            if (not exists(potential_dir)):
                return potential_dir
            i += 1

    def calculate_statistics_data(self, timepoints, max_count, num_points=100000):
        # we assume that the timepoints are sorted before this step

        # loop through all the timepoints. Get the earliest. If it's first,
        # add to the count. if it's second, decrement the count. Store the
        # (time, count) pair.

        # times = list()
        # counts = list()
        statistics = list()
        count = 0
        last_time = 0
        increment = 1.0 / float(max_count)
        for point in timepoints:
            if point.first:
                count += increment
            else:
                count -= increment
            if point.time != last_time:
                statistics.append((point.time, count))
            last_time = point.time
        return statistics

        # CODE BELOW USES A BASIC FILTER IN CASE THE SIZE OF THE STATISTICS
        # IS TOO LARGE

        # # we want to limit to num_points points in the statistics, so only get
        # # every nth element
        # n = max(1, int(len(times) / num_points))
        # statistics = []
        # # FIXME: get correct count average
        # for i in range(0, len(times),n):
        #     sub_times = times[i:i+n]
        #     sub_counts = counts[i:i+n]
        #     num_elems = float(len(sub_times))
        #     avg_time = float(sum(sub_times)) / num_elems
        #     avg_count = float(sum(sub_counts)) / num_elems
        #     statistics.append((avg_time, avg_count))
        # return statistics

    # utilization is 1 for a processor when some event is on the
    # processor
    #
    # adds points where there are one or more events occurring.
    # removes a point when the count drops to 0.
    def convert_to_utilization(self, timepoints):
        proc_utilization = []
        count = 0
        for point in timepoints:
            if point.first:
                count += 1
                if count == 1:
                    proc_utilization.append(point)
            else:
                count -= 1
                if count == 0:
                    proc_utilization.append(point)
        return proc_utilization

    # group by processor kind per node. Also compute the children relationships
    def group_node_proc_kind_timepoints(self, timepoints_dict):
        for proc in self.processors.itervalues():
            if len(proc.tasks) > 0:
                # add this processor kind to both all and the node group
                groups = [str(proc.node_id), "all"]
                for node in groups:
                    group = node + " (" + proc.kind + ")"
                    if group not in timepoints_dict:
                        timepoints_dict[group] = [proc.time_points]
                    else:
                        timepoints_dict[group].append(proc.time_points)

    def group_node_proc_timepoints(self, timepoints_dict):
        for proc in self.processors.itervalues():
            if len(proc.tasks) > 0:
                # add this processor to both all and the node group
                groups = [str(proc.node_id), "all"]
                for group in groups:
                    if group not in timepoints_dict:
                        timepoints_dict[group] = [proc.time_points]
                    else:
                        timepoints_dict[group].append(proc.time_points)

    def get_nodes(self):
        nodes = {}
        for proc in self.processors.itervalues():
            if len(proc.tasks) > 0:
                nodes[str(proc.node_id)] = 1
        if (len(nodes) > 1):
            return ["all"] + sorted(nodes.keys())
        else:
            return sorted(nodes.keys())


    def emit_statistics_tsv(self, output_dirname):
        print("emitting statistics")

        # this is a map from node ids to a list of timepoints in that node
        timepoints_dict = {}

        #self.group_node_proc_timepoints(timepoints_dict)
        self.group_node_proc_kind_timepoints(timepoints_dict)

        # add in the all group
        timepoints_dict["all"] = [proc.time_points 
                                    for proc in self.processors.values()
                                    if len(proc.tasks) > 0]

        # now we compute the structure of the stats (the parent-child
        # relationships
        nodes = self.get_nodes()
        stats_structure = {node: [] for node in nodes}

        # for each node grouping, add all the subtypes of processors
        for node in nodes:
            for kind in processor_kinds.itervalues():
                group = str(node) + " (" + kind + ")"
                if group in timepoints_dict:
                    stats_structure[node].append(group)

        json_file_name = os.path.join(output_dirname, "json", "stats.json")

        with open(json_file_name, "w") as json_file:
            # json.dump(timepoints_dict.keys(), json_file)
            json.dump(stats_structure, json_file)

        # here we write out the actual tsv stats files
        for tp_group in timepoints_dict:
            timepoints = timepoints_dict[tp_group]
            utilizations = [self.convert_to_utilization(tp)
                            for tp in timepoints]

            max_count = len(timepoints)
            statistics = None
            if max_count == 0:
                print("WARNING: node " + str(tp_group) + " has no prof events. Is this what you expected?")
                statistics = list()
            else:
                statistics = self.calculate_statistics_data(sorted(itertools.chain(*utilizations)), max_count)

            stats_tsv_filename = os.path.join(output_dirname, "tsv", str(tp_group) + "_stats.tsv")
            with open(stats_tsv_filename, "w") as stats_tsv_file:
                stats_tsv_file.write("time\tcount\n")
                stats_tsv_file.write("0.00\t0.00\n") # initial point
                for stat_point in statistics:
                    stats_tsv_file.write("%.2f\t%.2f\n" % stat_point)

    def simplify_op(self, op_dependencies, op_existence_map, transitive_map, op_path, _dir):
        cur_op_id = op_path[-1]

        if op_existence_map[cur_op_id]:
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
                self.simplify_op(op_dependencies, op_existence_map, transitive_map,
                                 new_op_path, _dir)

    def simplify_op_dependencies(self, op_dependencies, op_existence_map):
        # The dependence relation is transitive. We take advantage of this to
        # remove non-existant ops in the graph by following "out" and "in"
        # pointers until we get to an existing op
        transitive_map = {"in": {}, "out": {}}

        for _dir in ["in", "out"]:
            for op_id in op_dependencies.iterkeys():
                if not op_id in transitive_map[_dir]:
                    self.simplify_op(op_dependencies, op_existence_map, transitive_map,
                                     [op_id], _dir)

        for op_id in op_dependencies.iterkeys():
            for _dir in ["in", "out"]:
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
        spy_state = legion_spy.State(False, True, True, True)

        total_matches = 0

        for file_name in file_names:
            total_matches += spy_state.parse_log_file(file_name)
        print('Matched %d lines across all files.' % total_matches)
        op_dependencies = {}

        # compute the slice_index, slice_slice, and point_slice dependencies
        # (which legion_spy throws away). We just need to copy this data over
        # before legion spy throws it away
        for _slice, index in spy_state.slice_index.iteritems():
            while _slice in spy_state.slice_slice:
                _slice = spy_state.slice_slice[_slice]
            if index.uid not in op_dependencies:
                op_dependencies[index.uid] = {"in": set(), "out": set()}
            if _slice not in op_dependencies:
                op_dependencies[_slice] = {"in": set(), "out": set()}
            op_dependencies[index.uid]["out"].add(_slice)
            op_dependencies[_slice]["in"].add(index.uid)

        for _slice1, _slice2 in spy_state.slice_slice.iteritems():
            if _slice1 not in op_dependencies:
                op_dependencies[_slice1] = {"in": set(), "out": set()}
            if _slice2 not in op_dependencies:
                op_dependencies[_slice2] = {"in": set(), "out": set()}
            op_dependencies[_slice1]["out"].add(_slice2)
            op_dependencies[_slice2]["in"].add(_slice1)

        for point, _slice in spy_state.point_slice.iteritems():
            while _slice in spy_state.slice_slice:
                _slice = spy_state.slice_slice[_slice]
            if _slice not in op_dependencies:
                op_dependencies[_slice] = {"in": set(), "out": set()}
            if point.op.uid not in op_dependencies:
                op_dependencies[point.op.uid] = {"in": set(), "out": set()}
            op_dependencies[_slice]["out"].add(point.op.uid)
            op_dependencies[point.op.uid]["in"].add(_slice)

        # don't simplify graphs
        spy_state.post_parse(False, True)

        print("Performing physical analysis...")
        # don't perform any checks (too slow!)
        spy_state.perform_physical_analysis(False, False)

        op = spy_state.get_operation(spy_state.top_level_uid)
        elevate = dict()
        all_nodes = set()
        printer = legion_spy.GraphPrinter("./", "temp")
        op.print_event_graph(printer, elevate, all_nodes, True) 
        # Now print the edges at the very end
        for node in all_nodes:
            if hasattr(node, 'uid'):
                for src in node.physical_incoming:
                    if hasattr(src, 'uid'):
                        if src.uid not in op_dependencies:
                            op_dependencies[src.uid] = {"in" : set(), "out" : set()}
                        if node.uid not in op_dependencies:
                            op_dependencies[node.uid] = {"in" : set(), "out" : set()}
                        op_dependencies[src.uid]["in"].add(node.uid)
                        op_dependencies[node.uid]["out"].add(src.uid)

        # have an existence map for the uids (some uids in the event graph are not
        # actually executed
        op_existence_map = {}

        for op_id, operation in self.operations.iteritems():
            op_existence_map[op_id] = operation.proc is not None

        # now apply the existence map
        op_dependencies, transitive_map = self.simplify_op_dependencies(op_dependencies, op_existence_map)

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
        for proc in self.processors.itervalues():
            proc.add_initiation_dependencies(state, op_dependencies, transitive_map)

    def attach_dependencies(self, state, op_dependencies, transitive_map):
        for proc in self.processors.itervalues():
            proc.attach_dependencies(state, op_dependencies, transitive_map)


    def emit_interactive_visualization(self, output_dirname, show_procs,
                               file_names, show_channels, show_instances, force):
        self.assign_colors()
        # the output directory will either be overwritten, or we will find
        # a new unique name to create new logs
        if force and exists(output_dirname):
            print("forcing removal of " + output_dirname)
            shutil.rmtree(output_dirname)
        else:
            output_dirname = self.find_unique_dirname(output_dirname)

        print('Generating interactive visualization files in directory ' + output_dirname)
        src_directory = os.path.join(dirname(sys.argv[0]), "legion_prof_files")

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

        data_tsv_header = "level\tstart\tend\tcolor\topacity\ttitle\tinitiation\tin\tout\tprof_uid\n"

        tsv_dir = os.path.join(output_dirname, "tsv")
        json_dir = os.path.join(output_dirname, "json")
        os.mkdir(tsv_dir)
        if not exists(json_dir):
            os.mkdir(json_dir)
        
        op_dependencies, transitive_map = None, None
        if self.has_spy_state:
            op_dependencies, transitive_map = self.get_op_dependencies(file_names)

        # with open(dep_json_file_name, "w") as dep_json_file:
        #     json.dump(op_dependencies, dep_json_file)

        ops_file = open(ops_file_name, "w")
        ops_file.write("op_id\tdesc\tproc\tlevel\n")
        for op_id, operation in self.operations.iteritems():
            proc = ""
            level = ""
            if (operation.proc is not None):
                proc = repr(operation.proc)
                level = str(operation.level+1)
            ops_file.write("%d\t%s\t%s\t%s\n" % \
                            (op_id, str(operation), proc, level))
        ops_file.close()

        if show_procs:
            for proc in self.processors.itervalues():
                if self.has_spy_state and len(proc.tasks) > 0:
                    proc.add_initiation_dependencies(self, op_dependencies, transitive_map)
                    self.convert_op_ids_to_tuples(op_dependencies)

            for p,proc in sorted(self.processors.iteritems(), key=lambda x: x[1]):
                if len(proc.tasks) > 0:
                    if self.has_spy_state:
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
            for c,chan in sorted(self.channels.iteritems(), key=lambda x: x[1]):
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
            for m,mem in sorted(self.memories.iteritems(), key=lambda x: x[1]):
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

        num_stats = self.emit_statistics_tsv(output_dirname)
        stats_levels = 4

        scale_data = {
            'start': 0,
            'end': last_time * 1.01,
            'stats_levels': stats_levels,
            'max_level': base_level + 1
        }

        with open(scale_json_file_name, "w") as scale_json_file:
            json.dump(scale_data, scale_json_file)

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
        dest='filenames', nargs='+',
        help='input Legion Prof log filenames')
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

    state = State()
    has_matches = False
    for file_name in file_names:
        print('Reading log file %s...' % file_name)
        total_matches = state.parse_log_file(file_name, verbose)
        print('Matched %s lines' % total_matches)
        if total_matches > 0:
            has_matches = True
    if not has_matches:
        print('No matches found! Exiting...')
        return

    # Once we are done loading everything, do the sorting
    state.build_time_ranges()

    if print_stats:
        state.print_stats(verbose) 
    else:
        state.emit_interactive_visualization(output_dirname, show_procs,
                             file_names, show_channels, show_instances, force)
        if show_copy_matrix:
            state.show_copy_matrix(copy_output_prefix)

if __name__ == '__main__':
    main()
