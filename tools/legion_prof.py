#!/usr/bin/env python

# Copyright 2015 Stanford University
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

import sys, os, shutil
import string, re
from math import sqrt
from getopt import getopt

prefix = r'\[(?P<node>[0-9]+) - (?P<thread>[0-9a-f]+)\] \{\w+\}\{legion_prof\}: '
processor_pat = re.compile(prefix + r'Prof Processor (?P<proc>[a-f0-9]+) (?P<utility>[0-1]) (?P<kind>[0-9]+)')
memory_pat = re.compile(prefix + r'Prof Memory (?P<mem>[a-f0-9]+) (?P<kind>[0-9]+)')
task_variant_pat = re.compile(prefix + r'Prof Task Variant (?P<tid>[0-9]+) (?P<name>\w+)')
unique_task_pat = re.compile(prefix + r'Prof Unique Task (?P<proc>[a-f0-9]+) (?P<uid>[0-9]+) (?P<tid>[0-9]+) (?P<dim>[0-9]+) (?P<p0>[0-9\-]+) (?P<p1>[0-9\-]+) (?P<p2>[0-9\-]+)')
unique_map_pat = re.compile(prefix + r'Prof Unique Map (?P<proc>[a-f0-9]+) (?P<uid>[0-9]+) (?P<puid>[0-9]+)')
unique_close_pat = re.compile(prefix + r'Prof Unique Close (?P<proc>[a-f0-9]+) (?P<uid>[0-9]+) (?P<puid>[0-9]+)')
unique_copy_pat = re.compile(prefix + r'Prof Unique Copy (?P<proc>[a-f0-9]+) (?P<uid>[0-9]+) (?P<puid>[0-9]+)')
event_pat = re.compile(prefix + r'Prof Event (?P<proc>[a-f0-9]+) (?P<kind>[0-9]+) (?P<uid>[0-9]+) (?P<time>[0-9]+)')
create_pat = re.compile(prefix + r'Prof Create Instance (?P<iid>[a-f0-9]+) (?P<mem>[0-9]+) (?P<redop>[0-9]+) (?P<bf>[0-9]+) (?P<time>[0-9]+)')
field_pat = re.compile(prefix + r'Prof Instance Field (?P<iid>[a-f0-9]+) (?P<fid>[0-9]+) (?P<size>[0-9]+)')
destroy_pat = re.compile(prefix + r'Prof Destroy Instance (?P<iid>[a-f0-9]+) (?P<time>[0-9]+)')

# List of event kinds from legion_profiling.h
event_kind_ids = {
    'PROF_BEGIN_DEP_ANALYSIS': 0,
    'PROF_END_DEP_ANALYSIS': 1,
    'PROF_BEGIN_PREMAP_ANALYSIS': 2,
    'PROF_END_PREMAP_ANALYSIS': 3,
    'PROF_BEGIN_MAP_ANALYSIS': 4,
    'PROF_END_MAP_ANALYSIS': 5,
    'PROF_BEGIN_EXECUTION': 6,
    'PROF_END_EXECUTION': 7,
    'PROF_BEGIN_WAIT': 8,
    'PROF_END_WAIT': 9,
    'PROF_BEGIN_SCHEDULER': 10,
    'PROF_END_SCHEDULER': 11,
    'PROF_COMPLETE': 12,
    'PROF_LAUNCH': 13,
    'PROF_BEGIN_POST': 14,
    'PROF_END_POST': 15,
    'PROF_BEGIN_TRIGGER': 16,
    'PROF_END_TRIGGER': 17,
    'PROF_BEGIN_GC': 18,
    'PROF_END_GC': 19
}
event_kind_names = {
    0: 'PROF_BEGIN_DEP_ANALYSIS',
    1: 'PROF_END_DEP_ANALYSIS',
    2: 'PROF_BEGIN_PREMAP_ANALYSIS',
    3: 'PROF_END_PREMAP_ANALYSIS',
    4: 'PROF_BEGIN_MAP_ANALYSIS',
    5: 'PROF_END_MAP_ANALYSIS',
    6: 'PROF_BEGIN_EXECUTION',
    7: 'PROF_END_EXECUTION',
    8: 'PROF_BEGIN_WAIT',
    9: 'PROF_END_WAIT',
    10: 'PROF_BEGIN_SCHEDULER',
    11: 'PROF_END_SCHEDULER',
    12: 'PROF_COMPLETE',
    13: 'PROF_LAUNCH',
    14: 'PROF_BEGIN_POST',
    15: 'PROF_END_POST',
    16: 'PROF_BEGIN_TRIGGER',
    17: 'PROF_END_TRIGGER',
    18: 'PROF_BEGIN_GC',
    19: 'PROF_END_GC',
}
# Range Kinds
DEPENDENCE_RANGE = 0
PREMAP_RANGE = 1
MAPPING_RANGE = 2
LAUNCH_RANGE = 3
EXECUTION_RANGE = 4
POST_RANGE = 5
WAIT_RANGE = 6
COMPLETION_RANGE = 7
SCHEDULE_RANGE = 8
TRIGGER_RANGE = 9
GC_RANGE = 10
MESSAGE_RANGE = 11
COPY_RANGE = 12

# Helper functions for manipulating event kinds.
def event_kind_is_begin(kind):
    return kind % 2 == 0

def event_kind_is_end(kind):
    return kind % 2 == 1

def event_kind_category(kind):
    return kind / 2

# Micro-seconds per pixel
US_PER_PIXEL = 100
# Pixels per level of the picture
PIXELS_PER_LEVEL = 40
# Pixels per tick mark
PIXELS_PER_TICK = 200

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

class Point(object):
    def __init__(self, dim, p0, p1, p2):
        self.dim = dim
        self.indexes = (p0, p1, p2)[:max(dim, 1)]

    def __eq__(self, other):
        return self.dim == other.dim and self.indexes == other.indexes

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.dim, self.indexes))

    def __repr__(self):
        return '(%s)' % ','.join(self.indexes)

class TaskVariant(object):
    def __init__(self, task_id, name):
        self.task_id = task_id
        self.name = name
        self.color = None

    def compute_color(self, step, num_steps):
        assert self.color is None
        self.color = color_helper(step, num_steps)

    def __repr__(self):
        return 'Task ID %s %s' % (self.task_id, self.name)

class UniqueOp(object):
    def __init__(self, proc, uid, color_string):
        self.proc = proc
        self.uid = uid
        self.color_string = color_string
        self.dependence_range = None
        self.trigger_ranges = list() 
        self.premap_range = None
        self.mapping_range = None
        self.launch_range = None
        self.execution_range = None
        self.post_range = None
        self.completion_range = None
        self.waits = list()
        self.schedules = list()
        self.garbage_ranges = list()
        self.messages = list()

        # Used construction of event begin/end pairs.
        self.event_match = dict()

    def add_event(self, event, proc):
        if event.kind_id == 12:
            # Handle the completion case
            if self.execution_range is None:
                return True 
            assert self.completion_range is None
            #self.completion_range = \
            #    EventRange(self, self.execution_range.end_event, event, COMPLETION_RANGE)
            # Don't add the completion range
        elif event.kind_id == 13:
            # Handle the launch case
            if self.execution_range is None:
                return True 
            assert self.launch_range is None
            self.launch_range = \
                EventRange(self, event, self.execution_range.start_event, LAUNCH_RANGE)
        elif event.kind_id == 8 or event.kind_id == 9:
            return True
        elif event.is_end():
            # If it is an end event
            key = (event.category,proc)
            #assert key in self.event_match
            if key not in self.event_match:
                return True
            begin_event = self.event_match[key].pop()
            assert begin_event.category == event.category
            if event.kind_id == 1:
                assert self.dependence_range is None
                time_range = EventRange(self, begin_event, event, DEPENDENCE_RANGE)
                self.dependence_range = time_range
                proc.add_time_range(time_range)
            elif event.kind_id == 3:
                assert self.premap_range is None
                time_range = EventRange(self, begin_event, event, PREMAP_RANGE)
                self.premap_range = time_range
                proc.add_time_range(time_range)
            elif event.kind_id == 5:
                assert self.mapping_range is None
                time_range = EventRange(self, begin_event, event, MAPPING_RANGE)
                self.mapping_range = time_range
                proc.add_time_range(time_range)
            elif event.kind_id == 7:
                assert self.execution_range is None
                time_range = EventRange(self, begin_event, event, EXECUTION_RANGE)
                self.execution_range = time_range
                proc.add_time_range(time_range)
            elif event.kind_id == 9:
                time_range = EventRange(self, begin_event, event, WAIT_RANGE)
                self.waits.append(time_range)
                proc.add_time_range(time_range)
            elif event.kind_id == 11:
                time_range = EventRange(self, begin_event, event, SCHEDULE_RANGE)
                self.schedules.append(time_range)
                proc.add_time_range(time_range)
            elif event.kind_id == 15:
                assert self.post_range is None
                time_range = EventRange(self, begin_event, event, POST_RANGE)
                self.post_range = time_range
                proc.add_time_range(time_range)
            elif event.kind_id == 17:
                time_range = EventRange(self, begin_event, event, TRIGGER_RANGE)
                self.trigger_ranges.append(time_range)
                proc.add_time_range(time_range)
            elif event.kind_id == 19:
                time_range = EventRange(self, begin_event, event, GC_RANGE)
                self.garbage_ranges.append(time_range)
                proc.add_time_range(time_range)
            elif event.kind_id == 21:
                time_range = EventRange(self, begin_event, event, MESSAGE_RANGE)
                self.messages.append(time_range)
                proc.add_time_range(time_range)
            elif event.kind_id == 23:
                time_range = EventRange(self, begin_event, event, COPY_RANGE)
                self.messages.append(time_range)
                proc.add_time_range(time_range)
            else:
                assert False
        else:
            assert event.is_begin()
            key = (event.category,proc)
            if key not in self.event_match:
                self.event_match[key] = list()
            self.event_match[key].append(event)
        return True

    def waiting_time(self):
        result = 0
        for w in self.waits:
            result = result + w.cumulative_time()
        return result

class UniqueTask(UniqueOp):
    def __init__(self, proc, uid, variant, point, color_string):
        UniqueOp.__init__(self, proc, uid, color_string)
        self.variant = variant
        self.point = point

    def __repr__(self):
        result = '%s (UID %s) Point (%u' % (self.variant, self.uid, self.point.indexes[0])
        if len(self.point.indexes) > 1:
            result = result+(',%u' % self.point.indexes[1])
        if len(self.point.indexes) > 2:
            result = result+(',%u' % self.point.indexes[2])
        result = result+')'
        #if self.launch_range is not None:
        #    result = result+' Launch '+str(self.launch_range.start_event.abs_time)
        #if self.completion_range is not None:
        #    result = result+' Completion '+str(self.completion_range.end_event.abs_time)
        return result

    def get_variant(self):
        return self.variant

    def is_mapping(self):
        return False

    def is_close(self):
        return False

    def is_copy(self):
        return False

class UniqueMap(UniqueOp):
    def __init__(self, proc, uid, parent):
        UniqueOp.__init__(self, proc, uid, "#009999")
        self.parent = parent

    def __repr__(self):
        return 'Map (UID %s) in %s' % (self.uid,self.get_variant().name)

    def get_variant(self):
        return self.parent.get_variant()

    def is_mapping(self):
        return True

    def is_close(self):
        return False

    def is_copy(self):
        return False

class UniqueClose(UniqueOp):
    def __init__(self, proc, uid, parent):
        UniqueOp.__init__(self, proc, uid, "#FF3300")
        self.parent = parent

    def __repr__(self):
        return 'Close (UID %s) in %s' % (self.uid,self.get_variant().name)

    def get_variant(self):
        return self.parent.get_variant()

    def is_mapping(self):
        return False

    def is_close(self):
        return True

    def is_copy(self):
        return False

class UniqueCopy(UniqueOp):
    def __init__(self, proc, uid, parent):
        UniqueOp.__init__(self, proc, uid, "#FF3300")
        self.parent = parent

    def __repr__(self):
        return 'Copy (UID %s) in %s' % (self.uid, self.get_variant().name)

    def get_variant(self):
        return self.parent.get_variant()

    def is_mapping(self):
        return False

    def is_close(self):
        return False

    def is_copy(self):
        return False

class UniqueScheduler(UniqueOp):
    def __init__(self, proc, uid):
        UniqueOp.__init__(self, proc, uid, "#0099CC")

    def __repr__(self):
        return 'Scheduler'

    def get_variant(self):
        return repr(self)

class Event(object):
    def __init__(self, kind_id, unique_op, time):
        self.kind_id = kind_id
        self.unique_op = unique_op
        self.abs_time = time

    def is_begin(self):
        return event_kind_is_begin(self.kind_id)

    def is_end(self):
        return event_kind_is_end(self.kind_id)

    @property
    def category(self):
        return event_kind_category(self.kind_id)

    def __cmp__(self, other):
        if other is None:
             return -1
        if self.abs_time < other.abs_time:
            return -1
        elif self.abs_time == other.abs_time:
            return 0
        else:
            return 1

    def __repr__(self):
        return '%s for %s' % (event_kind_names[self.kind_id], self.unique_op)

class TimeRange(object):
    def __init__(self, start_event, end_event):
        assert start_event is not None
        assert end_event is not None
        assert start_event <= end_event
        self.start_event = start_event
        self.end_event = end_event
        self.subranges = list()

    def __cmp__(self, other):
        # The order chosen here is critical for sort_range. Ranges are
        # sorted by start_event first, and then by *reversed*
        # end_event, so that each range will precede any ranges they
        # contain in the order.
        if self.start_event < other.start_event:
            return -1
        if self.start_event > other.start_event:
            return 1

        if self.end_event > other.end_event:
            return -1
        if self.end_event < other.end_event:
            return 1
        return 0

    def contains(self, other):
        #if self.start_event > other.end_event:
        #    return False
        #if self.end_event < other.start_event:
        #    return False
        if self.start_event <= other.start_event and \
            other.end_event <= self.end_event:
            return True
        # Otherwise they overlap one way or the other
        # but neither contains the other
        return False

    def add_range(self, other):
        assert self.contains(other)
        self.subranges.append(other)

    def sort_range(self):
        self.subranges.sort()

        removed = set()
        stack = []
        for subrange in self.subranges:
            while len(stack) > 0 and not stack[-1].contains(subrange):
                stack.pop()
            if len(stack) > 0:
                stack[-1].add_range(subrange)
                removed.add(subrange)
            stack.append(subrange)

        self.subranges = [s for s in self.subranges if s not in removed]

    def cumulative_time(self):
        return self.end_event.abs_time - self.start_event.abs_time

    def non_cumulative_time(self):
        total_time = self.cumulative_time()
        start_subrange = 0
        end_subrange = 0
        for r in self.subranges:
            # the following does not work because of overlapping subranges
            #total_time = total_time - r.cumulative_time()
            if end_subrange <= r.start_event.abs_time:
                total_time = total_time - (end_subrange - start_subrange)
                start_subrange = r.start_event.abs_time
            end_subrange = r.end_event.abs_time
        total_time = total_time - (end_subrange - start_subrange)

        assert total_time >= 0
        return total_time


    def max_levels(self):
        max_lev = 0
        for idx in range(len(self.subranges)):
            levels = self.subranges[idx].max_levels()
            if levels > max_lev:
                max_lev = levels
        return max_lev+1

    def emit_svg_range(self, printer):
        self.emit_svg(printer, 0)

    def __repr__(self):
        return "Start: %d us  Stop: %d us  Total: %d us" % (
            self.start_event.abs_time,
            self.end_event.abs_time,
            self.cumulative_time())

class BaseRange(TimeRange):
    def __init__(self, proc, start_event, end_event):
        TimeRange.__init__(self, start_event, end_event)
        self.proc = proc

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return False

    def emit_svg(self, printer, level):
        title = repr(self.proc)
        printer.emit_time_line(level, self.start_event.abs_time, self.end_event.abs_time, title)
        for subrange in self.subranges:
            subrange.emit_svg(printer, level + 1)

    def update_task_stats(self, stat, proc):
        for r in self.subranges:
            r.update_task_stats(stat, proc)

    def active_time(self):
        total = 0
        for subrange in self.subranges:
            total = total + subrange.active_time()
        return total

    def application_time(self):
        total = 0
        for subrange in self.subranges:
            total = total + subrange.application_time()
        return total

    def meta_time(self):
        total = 0
        for subrange in self.subranges:
            total = total + subrange.meta_time()
        return total

class EventRange(TimeRange):
    def __init__(self, op, start_event, end_event, range_kind):
        TimeRange.__init__(self, start_event, end_event)
        self.op = op
        self.range_kind = range_kind

    def emit_svg(self, printer, level):
        if self.range_kind == DEPENDENCE_RANGE:
            color = "#0000FF" # Duke Blue
            title = "Dependence Analysis for "+repr(self.op)+" "+repr(self)
        elif self.range_kind == PREMAP_RANGE:
            color = "#6600FF" # Purple
            title = "Premap Analysis for "+repr(self.op)+" "+repr(self)
        elif self.range_kind == MAPPING_RANGE:
            color = "#009900" # Green
            title = "Mapping Analysis for "+repr(self.op)+" "+repr(self)
        elif self.range_kind == EXECUTION_RANGE:
            color = self.op.color_string
            title = "Execution of "+repr(self.op)+" "+repr(self)
        elif self.range_kind == POST_RANGE:
            color = "#333399" # Deep Purple 
            title = "Post Execution of "+repr(self.op)+" "+repr(self)
        elif self.range_kind == TRIGGER_RANGE:
            color = "#FF6600" # Orange
            title = "Trigger Execution of "+repr(self.op)+" "+repr(self)
        elif self.range_kind == WAIT_RANGE:
            color = "#FFFFFF" # White
            title = "Waiting on "+repr(self.op)+" "+repr(self)
        elif self.range_kind == SCHEDULE_RANGE:
            color = self.op.color_string
            title = "Scheduler "+repr(self)
        elif self.range_kind == GC_RANGE:
            color = "#990000" # Crimson
            title = "Garbage Collection"
        elif self.range_kind == MESSAGE_RANGE:
            color = "#006600" # Evergreen
            title = "Message Handler"
        elif self.range_kind == COPY_RANGE:
            color = "#8B0000" # Dark Red
            title = "Low-Level Copy "+repr(self)
        else:
            assert False
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for subrange in self.subranges:
            subrange.emit_svg(printer, level + 1)

    def update_task_stats(self, stat, proc):
        if self.range_kind == SCHEDULE_RANGE:
            stat.update_scheduler(self.cumulative_time(), self.non_cumulative_time(), proc)
        elif self.range_kind == GC_RANGE:
            stat.update_gc(self.cumulative_time(), self.non_cumulative_time(), proc)
        elif self.range_kind <> WAIT_RANGE:
            variant = self.op.get_variant()
            cum_time = self.cumulative_time()
            non_cum_time = self.non_cumulative_time()
            if self.range_kind == DEPENDENCE_RANGE:
                if self.op.is_mapping():
                    stat.update_inline_dep_analysis(variant, cum_time, non_cum_time, proc)
                elif self.op.is_close():
                    stat.update_close_dep_analysis(variant, cum_time, non_cum_time, proc)
                elif self.op.is_copy():
                    stat.update_copy_dep_analysis(variant, cum_time, non_cum_time, proc)
                else:
                    stat.update_dependence_analysis(variant, cum_time, non_cum_time, proc)
            elif self.range_kind == PREMAP_RANGE:
                stat.update_premappings(variant, cum_time, non_cum_time, proc)
            elif self.range_kind == MAPPING_RANGE:
                if self.op.is_mapping():
                    stat.update_inline_mappings(variant, cum_time, non_cum_time, proc)
                elif self.op.is_close():
                    stat.update_close_operations(variant, cum_time, non_cum_time, proc)
                elif self.op.is_copy():
                    stat.update_copy_operations(variant, cum_time, non_cum_time, proc)
                else:
                    stat.update_mapping_analysis(variant, cum_time, non_cum_time, proc)
            elif self.range_kind == EXECUTION_RANGE:
                stat.update_invocations(variant, cum_time, non_cum_time, proc)
            elif self.range_kind == POST_RANGE:
                stat.update_post(variant, cum_time, non_cum_time, proc)
            elif self.range_kind == TRIGGER_RANGE:
                stat.update_trigger(variant, cum_time, non_cum_time, proc)
        for r in self.subranges:
            r.update_task_stats(stat, proc)

    def active_time(self):
        if self.range_kind == EXECUTION_RANGE:
            result = self.cumulative_time() - self.op.waiting_time()
            for subrange in self.subranges:
                result = result + subrange.active_time()
            return result
        elif self.range_kind == WAIT_RANGE:
            result = 0
            for subrange in self.subranges:
                result = result + subrange.active_time()
            return result
        else:
            return self.cumulative_time()

    def application_time(self):
        if self.range_kind == EXECUTION_RANGE:
            result = self.cumulative_time() - self.op.waiting_time()
            for subrange in self.subranges:
                result = result + subrange.application_time()
            return result
        elif self.range_kind == WAIT_RANGE:
            result = 0
            for subrange in self.subranges:
                result = result + subrange.application_time()
            return result
        else:
            return 0

    def meta_time(self):
        if self.range_kind == EXECUTION_RANGE:
            result = 0
            for subrange in self.subranges:
                result = result + subrange.meta_time()
            return result
        elif self.range_kind == WAIT_RANGE:
            result = 0
            for subrange in self.subranges:
                result = result + subrange.meta_time()
            return result
        else:
            return self.cumulative_time()

class Processor(object):
    def __init__(self, proc_id, utility, kind):
        self.proc_id = proc_id
        self.utility = utility
        if kind == 1 or kind == 2: # Kind 2 is a utility proc
            self.kind = 'CPU'
        elif kind == 3:
            self.kind = 'COPY'
        elif kind == 0:
            self.kind = 'GPU'
        else:
            print 'WARNING: Unrecognized processor kind %s' % kind
            self.kind = 'OTHER PROC KIND'
        self.scheduler_op = UniqueScheduler(self, 0)
        self.timing_ranges = list()
        self.full_range = None

    def init_time_range(self, last_time):
        self.full_range = BaseRange(
            self,
            Event(event_kind_ids['PROF_BEGIN_SCHEDULER'], self.scheduler_op, 0L),
            Event(event_kind_ids['PROF_END_SCHEDULER'], self.scheduler_op, last_time))

    def add_time_range(self, timing_range):
        self.timing_ranges.append(timing_range)

    def sort_time_range(self):
        for r in self.timing_ranges:
            self.full_range.add_range(r)
        self.full_range.sort_range()

    def print_stats(self):
        # Figure out the total time for this processor
        # The amount of time the processor was active
        # The amount of time spent on application tasks
        # The amount of time spent on meta tasks
        total_time = self.full_range.cumulative_time()
        active_time = self.full_range.active_time()
        application_time = self.full_range.application_time()
        meta_time = self.full_range.meta_time()
        active_ratio = 100.0*float(active_time)/float(total_time)
        application_ratio = 100.0*float(application_time)/float(total_time)
        meta_ratio = 100.0*float(meta_time)/float(total_time)
        print self
        print "    Total time: %d us" % total_time
        print "    Active time: %d us (%.3f%%)" % (active_time, active_ratio)
        print "    Application time: %d us (%.3f%%)" % (application_time, application_ratio)
        print "    Meta time: %d us (%.3f%%)" % (meta_time, meta_ratio)
        print

    def emit_svg(self, printer):
        # First figure out the max number of levels + 1 for padding
        max_levels = self.full_range.max_levels() + 1
        # Skip any empty processors
        if max_levels > 1:
            printer.init_processor(max_levels)
            self.full_range.emit_svg_range(printer)

    def update_task_stats(self, stat):
        self.full_range.update_task_stats(stat, self.proc_id)

    def __repr__(self):
        if self.kind != 'COPY':
            return '%s Processor %s%s' % (
                self.kind,
                hex(self.proc_id),
                (' (Utility)' if self.utility else ''))
        else:
            return 'Low-Level Copies'

class Memory(object):
    def __init__(self, mem, kind):
        self.mem = mem
        self.kind = kind
        self.instances = set()
        self.max_live_instances = None
        self.time_points = None

    def add_instance(self, inst):
        if inst not in self.instances:
            self.instances.add(inst)

    def __repr__(self):
        return 'Memory %s' % hex(self.mem)

    def print_stats(self):
        print self
        print "    Total Instances: %d" % len(self.instances)

    def get_title(self):
        title = ""
        if self.kind == 0:
            title = "Global "
        elif self.kind == 1:
            title = "System "
        elif self.kind == 2:
            title = "Pinned "
        elif self.kind == 3:
            title = "Socket "
        elif self.kind == 4:
            title = "Zero-Copy "
        elif self.kind == 5:
            title = "GPU Framebuffer "
        elif self.kind == 6:
            title = "Disk "
        elif self.kind == 7:
            title = "HDF "
        elif self.kind == 8:
            title = "L3 Cache "
        elif self.kind == 9:
            title = "L2 Cache "
        elif self.kind == 10:
            title = "L1 Cache "
        else:
            print "WARNING: Unsupported memory type in LegionProf: "+str(self.kind)
        title = title + "Memory "+str(hex(self.mem))
        return title

    def sort_time_range(self):
        self.max_live_instances = 0
        self.time_points = list()
        live_instances = set()
        dead_instances = set()
        # Go through all of our instances and mark down
        # their creation and destruction times
        while len(dead_instances) < len(self.instances):
            # Find the next event
            next_time = None
            creation = None
            target_inst = None
            for inst in self.instances:
                if inst in dead_instances:
                    # Skip this since we're done with it
                    continue
                elif inst in live_instances:
                    # Look at its destruction time
                    if next_time is None:
                        next_time = inst.destroy_time
                        creation = False
                        target_inst = inst
                    elif inst.destroy_time < next_time:
                        next_time = inst.destroy_time
                        creation = False
                        target_inst = inst
                else:
                    # Look at its creation time
                    if next_time is None:
                        next_time = inst.create_time
                        creation = True
                        target_inst = inst
                    elif inst.create_time < next_time:
                        next_time = inst.create_time
                        creation = True
                        target_inst = inst
            # wonchan: this assertion does not hold when instances are leaked
            #assert next_time is not None
            self.time_points.append((next_time,creation,target_inst))
            if creation:
                assert target_inst not in live_instances
                live_instances.add(target_inst)
                # Check to see if we can update the max live instances
                total_live = len(live_instances) - len(dead_instances)
                if total_live > self.max_live_instances:
                    self.max_live_instances = total_live
            else:
                assert target_inst in live_instances
                assert target_inst not in dead_instances
                dead_instances.add(target_inst)

    def emit_svg(self, printer, end_time):
        printer.init_memory(self.max_live_instances+1)
        levels = dict()
        for time,create,inst in self.time_points:
            if create:
                # Find a level to place the instance at
                level = None
                for lev,lev_inst in levels.iteritems():
                    if lev_inst is None:
                        level = lev
                        break
                # If we didn't find a level, make a new one
                if level is None:
                    level = len(levels)+1
                    assert len(levels) <= self.max_live_instances
                levels[level] = inst
                inst.emit_svg(printer, level)
            else:
                # Remove the instance from the levels
                found = False
                for lev,lev_inst in levels.iteritems():
                    if inst == lev_inst:
                        levels[lev] = None
                        found = True
                        break
                assert found
        printer.emit_time_line(0, 0, end_time, self.get_title())

class Instance(object):
    def __init__(self, iid):
        self.iid = iid
        self.memory = None 
        self.redop = None 
        self.blocking_factor = None 
        self.create_time = None 
        self.destroy_time = None
        self.color = None
        self.fields = dict()

    def set_create(self, memory, redop, blocking_factor, create_time):
        assert self.memory is None or self.memory == memory
        assert self.redop is None or self.redop == redop
        assert self.blocking_factor is None or self.blocking_factor == blocking_factor
        self.memory = memory
        self.redop = redop
        self.blocking_factor = blocking_factor
        if self.create_time is None:
            self.create_time = create_time

    def set_destroy(self, time):
        if not (self.destroy_time is None):
            print(self.destroy_time)
        assert self.destroy_time is None
        self.destroy_time = time

    def add_field(self, fid, size):
        if fid not in self.fields:
            self.fields[fid] = size
        else:
            assert self.fields[fid] == size

    def compute_color(self, step, num_steps):
        assert self.color is None
        self.color = color_helper(step, num_steps)

    def get_title(self):
        title = "Instance "+str(hex(self.iid))+" blocking factor "+str(self.blocking_factor)
        title = title+" fields: "
        for fid,size in self.fields.iteritems():
            title = title + "("+str(fid)+","+str(size)+" bytes)"
        return title

    def emit_svg(self, printer, level):
        assert self.color is not None
        printer.emit_timing_range(self.color, level, self.create_time, self.destroy_time, self.get_title())

class SVGPrinter(object):
    def __init__(self, file_name, html_file):
        self.target = open(file_name,'w')
        self.file_name = file_name
        self.html_file = html_file
        assert self.target is not None
        self.offset = 0
        self.target.write('<svg xmlns="http://www.w3.org/2000/svg">\n')
        self.max_width = 0
        self.max_height = 0

    def close(self):
        self.emit_time_scale()
        self.target.write('</svg>\n')
        self.target.close()
        # Round up the max width and max height to a multiple of 100
        while ((self.max_width % 100) != 0):
            self.max_width = self.max_width + 1
        while ((self.max_height % 100) != 0):
            self.max_height = self.max_height + 1
        # Also emit the html file
        html_target = open(self.html_file,'w')
        html_target.write('<!DOCTYPE HTML PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"\n')
        html_target.write('"DTD?xhtml1-transitional.dtd">\n')
        html_target.write('<html>\n')
        html_target.write('<head>\n')
        html_target.write('<title>Legion Prof</title>\n')
        html_target.write('</head>\n')
        html_target.write('<body leftmargin="0" marginwidth="0" topmargin="0" marginheight="0" bottommargin="0">\n')
        html_target.write('<embed src="'+self.file_name+'" width="'+str(self.max_width)+'px" height="'+str(self.max_height)+'px" type="image/svg+xml" />')
        html_target.write('</body>\n')
        html_target.write('</html>\n')
        html_target.close()

    def init_processor(self, total_levels):
        self.offset = self.offset + total_levels

    def init_memory(self, total_levels):
        self.offset = self.offset + total_levels

    def emit_timing_range(self, color, level, start, finish, title):
        self.target.write('  <g>\n')
        self.target.write('    <title>'+title+'</title>\n')
        assert level <= self.offset
        x_start = start//US_PER_PIXEL
        x_length = (finish-start)//US_PER_PIXEL
        y_start = (self.offset-level)*PIXELS_PER_LEVEL
        y_length = PIXELS_PER_LEVEL
        self.target.write('    <rect x="'+str(x_start)+'" y="'+str(y_start)+'" width="'+str(x_length)+'" height="'+str(y_length)+'"\n')
        self.target.write('     fill="'+str(color)+'" stroke="black" stroke-width="1" />')
        self.target.write('</g>\n')
        if (x_start+x_length) > self.max_width:
            self.max_width = x_start + x_length
        if (y_start+y_length) > self.max_height:
            self.max_height = y_start + y_length

    def emit_time_scale(self):
        x_end = self.max_width
        y_val = int(float(self.offset + 1.5)*PIXELS_PER_LEVEL)
        self.target.write('    <line x1="'+str(0)+'" y1="'+str(y_val)+'" x2="'+str(x_end)+'" y2="'+str(y_val)+'" stroke-width="2" stroke="black" />\n')
        y_tick_max = y_val + int(0.2*PIXELS_PER_LEVEL)
        y_tick_min = y_val - int(0.2*PIXELS_PER_LEVEL)
        us_per_tick  = US_PER_PIXEL * PIXELS_PER_TICK
        # Compute the number of tick marks
        for idx in range(x_end // PIXELS_PER_TICK):
            x_tick = idx*PIXELS_PER_TICK
            self.target.write('    <line x1="'+str(x_tick)+'" y1="'+str(y_tick_min)+'" x2="'+str(x_tick)+'" y2="'+str(y_tick_max)+'" stroke-width="2" stroke="black" />\n')
            title = "%d us" % (idx*us_per_tick)
            self.target.write('    <text x="'+str(x_tick)+'" y="'+str(y_tick_max+int(0.2*PIXELS_PER_LEVEL))+'" fill="black">'+title+'</text>\n')
        if (y_val+PIXELS_PER_LEVEL) > self.max_height:
            self.max_height = y_val + PIXELS_PER_LEVEL


    def emit_time_line(self, level, start, finish, title):
        x_start = start//US_PER_PIXEL
        x_end = finish//US_PER_PIXEL
        y_val = int(float(self.offset-level + 0.7)*PIXELS_PER_LEVEL)
        self.target.write('    <line x1="'+str(x_start)+'" y1="'+str(y_val)+'" x2="'+str(x_end)+'" y2="'+str(y_val)+'" stroke-width="2" stroke="black"/>')
        self.target.write('    <text x="'+str(x_start)+'" y="'+str(y_val-int(0.2*PIXELS_PER_LEVEL))+'" fill="black">'+title+'</text>')
        if ((self.offset-level)+1)*PIXELS_PER_LEVEL > self.max_height:
            self.max_height = ((self.offset-level)+1)*PIXELS_PER_LEVEL

class CallTracker(object):
    def __init__(self):
        self.invocations = 0
        self.cum_time = 0
        self.non_cum_time = 0
        self.min_time = 0
        self.max_time = 0
        self.min_proc = None
        self.max_proc = None
        self.all_vals = list()

    def increment(self, cum, non_cum, proc):
        if self.invocations == 0:
            self.min_time = cum
            self.max_time = cum
            self.min_proc = proc
            self.max_proc = proc
        else:
            if cum < self.min_time:
                self.min_time = cum
                self.min_proc = proc
            if cum > self.max_time:
                self.max_time = cum
                self.max_proc = proc
        self.invocations = self.invocations + 1
        self.cum_time = self.cum_time + cum
        self.non_cum_time = self.non_cum_time + non_cum
        self.all_vals.append(cum)

    def is_empty(self):
        return self.invocations == 0

    def print_stats(self, total_time):
        print "                Total Invocations: "+str(self.invocations)
        if self.invocations > 0:
            stddev = 0
            assert self.invocations == len(self.all_vals)
            avg = float(self.cum_time)/float(self.invocations)
            max_dev = 0.0
            min_dev = 0.0
            if self.invocations > 1:
                for val in self.all_vals:
                    diff = float(val) - avg
                    stddev = stddev + sqrt(diff * diff)
                stddev = stddev / float(self.invocations)
                stddev = sqrt(stddev)
                max_dev = (float(self.max_time) - avg) / stddev if stddev != 0 else 0.0
                min_dev = (float(self.min_time) - avg) / stddev if stddev != 0 else 0.0
            print "                Cumulative Time: %d us (%.3f%%)" % (self.cum_time,100.0*float(self.cum_time)/float(total_time))
            print "                Non-Cumulative Time: %d us (%.3f%%)" % (self.non_cum_time,100.0*float(self.non_cum_time)/float(total_time))
            print "                Average Cum Time: %.3f us" % (float(self.cum_time)/float(self.invocations))
            print "                Average Non-Cum Time: %.3f us" % (float(self.non_cum_time)/float(self.invocations))
            print "                Maximum Cum Time: %.3f us (%.3f sig) on processor %s" % (float(self.max_time),max_dev,hex(self.max_proc))
            print "                Minimum Cum Time: %.3f us (%.3f sig) on processor %s" % (float(self.min_time),min_dev,hex(self.min_proc))

class StatVariant(object):
    def __init__(self, var):
        self.var = var
        self.invocations = CallTracker()
        self.inline_dep_analysis = CallTracker()
        self.inline_mappings = CallTracker()
        self.close_dep_analysis = CallTracker()
        self.close_operations = CallTracker()
        self.copy_dep_analysis = CallTracker()
        self.copy_operations = CallTracker()
        self.dependence_analysis = CallTracker()
        self.premappings = CallTracker()
        self.mapping_analysis = CallTracker()
        self.post_operations = CallTracker()
        self.triggers = CallTracker()

    def update_invocations(self, cum, non_cum, proc):
        self.invocations.increment(cum, non_cum, proc)

    def update_inline_dep_analysis(self, cum, non_cum, proc):
        self.inline_dep_analysis.increment(cum, non_cum, proc)

    def update_inline_mappings(self, cum, non_cum, proc):
        self.inline_mappings.increment(cum, non_cum, proc)

    def update_close_dep_analysis(self, cum, non_cum, proc):
        self.close_dep_analysis.increment(cum, non_cum, proc)

    def update_close_operations(self, cum, non_cum, proc):
        self.close_operations.increment(cum, non_cum, proc)

    def update_copy_dep_analysis(self, cum, non_cum, proc):
        self.copy_dep_analysis.increment(cum, num_cum, proc)

    def update_copy_operations(self, cum, non_cum, proc):
        self.copy_operations.increment(cum, non_cum, proc)

    def update_dependence_analysis(self, cum, non_cum, proc):
        self.dependence_analysis.increment(cum, non_cum, proc)

    def update_premappings(self, cum, non_cum, proc):
        self.premappings.increment(cum, non_cum, proc)

    def update_mapping_analysis(self, cum, non_cum, proc):
        self.mapping_analysis.increment(cum, non_cum, proc)

    def update_post_operations(self, cum, non_cum, proc):
        self.post_operations.increment(cum, non_cum, proc)

    def update_trigger(self, cum, non_cum, proc):
        self.triggers.increment(cum, non_cum, proc)

    def cumulative_time(self):
        time = 0
        time = time + self.invocations.cum_time
        time = time + self.inline_dep_analysis.cum_time
        time = time + self.inline_mappings.cum_time
        time = time + self.close_dep_analysis.cum_time
        time = time + self.close_operations.cum_time
        time = time + self.copy_dep_analysis.cum_time
        time = time + self.copy_operations.cum_time
        time = time + self.dependence_analysis.cum_time
        time = time + self.premappings.cum_time
        time = time + self.mapping_analysis.cum_time
        time = time + self.post_operations.cum_time
        time = time + self.triggers.cum_time
        return time

    def non_cumulative_time(self):
        time = 0
        time = time + self.invocations.non_cum_time
        time = time + self.inline_dep_analysis.non_cum_time
        time = time + self.inline_mappings.non_cum_time
        time = time + self.close_dep_analysis.non_cum_time
        time = time + self.close_operations.non_cum_time
        time = time + self.copy_dep_analysis.non_cum_time
        time = time + self.copy_operations.non_cum_time
        time = time + self.dependence_analysis.non_cum_time
        time = time + self.premappings.non_cum_time
        time = time + self.mapping_analysis.non_cum_time
        time = time + self.post_operations.non_cum_time
        time = time + self.triggers.non_cum_time
        return time

    def print_stats(self, total_time, cumulative, verbose):
        title_str = repr(self.var)
        to_add = 50 - len(title_str)
        if to_add > 0:
            for idx in range(to_add):
                title_str = title_str+' '
        cum_time = self.cumulative_time()
        non_cum_time = self.non_cumulative_time()
        if cumulative:
            cum_per = 100.0*(float(cum_time)/float(total_time))
            title_str = title_str+("%d us (%.3f%%)" % (cum_time,cum_per))
        else:
            non_cum_per = 100.0*(float(non_cum_time)/float(total_time))
            title_str = title_str+("%d us (%.3f%%)" % (non_cum_time,non_cum_per))
        print "    "+title_str
        # Not verbose, print out the application and meta timings
        if not verbose:
            app_cum_time = self.invocations.cum_time
            app_non_cum_time = self.invocations.non_cum_time
            meta_cum_time = cum_time - app_cum_time
            meta_non_cum_time = non_cum_time - app_non_cum_time
            print "          Executions (APP):"
            self.invocations.print_stats(total_time)
            print "          Meta Execution Time (META):"
            print "                Cumulative Time: %d us (%.3f%%)" % \
                (meta_cum_time,100.0*float(meta_cum_time)/float(total_time))
            print "                Non-Cumulative Time: %d us (%.3f%%)" % \
                (meta_non_cum_time,100.0*float(meta_non_cum_time)/float(total_time))
        else:
            self.emit_call_stat(self.invocations,"Executions (APP):",total_time)
            self.emit_call_stat(self.dependence_analysis,"Dependence Analysis (META):",total_time)
            self.emit_call_stat(self.premappings,"Premapping Analysis (META):",total_time)
            self.emit_call_stat(self.mapping_analysis,"Mapping Analysis (META):",total_time)
            self.emit_call_stat(self.inline_dep_analysis,"Inline Mapping Dependence (META):",total_time)
            self.emit_call_stat(self.inline_mappings,"Inline Mapping Analysis (META):",total_time)
            self.emit_call_stat(self.close_dep_analysis,"Close Dependence Analysis (META):",total_time)
            self.emit_call_stat(self.close_operations,"Close Mapping Analysis (META):",total_time)
            self.emit_call_stat(self.copy_dep_analysis,"Copy Dependence Analysis (META):",total_time)
            self.emit_call_stat(self.copy_operations,"Copy Mapping Analysis (META):",total_time)
            self.emit_call_stat(self.triggers,"Trigger Calls (META):",total_time)
            self.emit_call_stat(self.post_operations,"Post Operations (META):",total_time)

    def emit_call_stat(self, calls, string, total_time):
        if not calls.is_empty():
            print "         "+string 
            calls.print_stats(total_time)

class StatGatherer(object):
    def __init__(self):
        self.variants = dict()
        self.scheduler = CallTracker()
        self.gcs = CallTracker()
        self.executing_task = list()
        self.dependence_analysis = CallTracker()
        self.mapping_analysis = CallTracker()

    def initialize_variant(self, var):
        assert var not in self.variants
        self.variants[var] = StatVariant(var)

    def update_invocations(self, var, cum, non_cum, proc):
        assert var in self.variants
        self.variants[var].update_invocations(cum, non_cum, proc)

    def update_inline_dep_analysis(self, var, cum, non_cum, proc):
        assert var in self.variants
        self.variants[var].update_inline_dep_analysis(cum, non_cum, proc)
        self.dependence_analysis.increment(cum, non_cum, proc)

    def update_inline_mappings(self, var, cum, non_cum, proc):
        assert var in self.variants
        self.variants[var].update_inline_mappings(cum, non_cum, proc)
        self.mapping_analysis.increment(cum, non_cum, proc)

    def update_close_dep_analysis(self, var, cum, non_cum, proc):
        assert var in self.variants
        self.variants[var].update_close_dep_analysis(cum, non_cum, proc)
        self.dependence_analysis.increment(cum, non_cum, proc)

    def update_close_operations(self, var, cum, non_cum, proc):
        assert var in self.variants
        self.variants[var].update_close_operations(cum, non_cum, proc)
        self.mapping_analysis.increment(cum, non_cum, proc)

    def update_copy_dep_analysis(self, var, cum_non_cum, proc):
        assert var in self.variants
        self.variants[var].update_copy_dep_analysis(cum, non_cum, proc)
        self.dependence_analysis.increment(cum, non_cum, proc)

    def update_copy_operations(self, var, cum, non_cum, proc):
        assert var in self.variants
        self.variants[var].update_copy_operations(cum, non_cum, proc)
        self.mapping_analysis.increment(cum, non_cum, proc)

    def update_dependence_analysis(self, var, cum, non_cum, proc):
        assert var in self.variants
        self.variants[var].update_dependence_analysis(cum, non_cum, proc)
        self.dependence_analysis.increment(cum, non_cum, proc)

    def update_premappings(self, var, cum, non_cum, proc):
        assert var in self.variants
        self.variants[var].update_premappings(cum, non_cum, proc)
        self.mapping_analysis.increment(cum, non_cum, proc)

    def update_mapping_analysis(self, var, cum, non_cum, proc):
        assert var in self.variants
        self.variants[var].update_mapping_analysis(cum, non_cum, proc)
        self.mapping_analysis.increment(cum, non_cum, proc)

    def update_post(self, var, cum, non_cum, proc):
        assert var in self.variants
        self.variants[var].update_post_operations(cum, non_cum, proc)

    def update_trigger(self, var, cum, non_cum, proc):
        assert var in self.variants
        self.variants[var].update_trigger(cum, non_cum, proc)

    def update_scheduler(self, cum, non_cum, proc):
        self.scheduler.increment(cum, non_cum, proc)

    def update_gc(self, cum, non_cum, proc):
        self.gcs.increment(cum, non_cum, proc)

    def print_stats(self, total_time, cumulative, verbose):
        print "  -------------------------"
        print "  Task Statistics"
        print "  -------------------------"
        # Sort the tasks based on either their cumulative
        # or non-cumulative time
        task_list = list()
        for v,var in self.variants.iteritems():
            task_list.append(var)
        if cumulative:
            task_list.sort(key=lambda t: t.cumulative_time())
        else:
            task_list.sort(key=lambda t: t.non_cumulative_time())
        task_list.reverse()
        for t in task_list:
            t.print_stats(total_time, cumulative, verbose)
        print "  -------------------------"
        print "  Meta-Task Statistics"
        print "  -------------------------"
        if not self.scheduler.is_empty():
            print "  Scheduler (META):"
            self.scheduler.print_stats(total_time)
        if not self.gcs.is_empty():
            print "  Garbage Collection (META):"
            self.gcs.print_stats(total_time)
        if not self.dependence_analysis.is_empty():
            print "  Total Dependence Analyses (META):"
            self.dependence_analysis.print_stats(total_time)
        if not self.mapping_analysis.is_empty():
            print "  Total Mapping Analyses (META):"
            self.mapping_analysis.print_stats(total_time)

class State(object):
    def __init__(self):
        self.processors = {}
        self.memories = {}
        self.task_variants = {}
        self.unique_ops = {}
        self.instances = {}
        self.last_time = None

    def create_processor(self, proc_id, utility, kind):
        if proc_id not in self.processors:
            self.processors[proc_id] = Processor(proc_id, utility, kind)

    def create_memory(self, mem, kind):
        if mem not in self.memories:
            self.memories[mem] = Memory(mem, kind)

    def create_task_variant(self, task_id, name):
        if task_id not in self.task_variants:
            self.task_variants[task_id] = TaskVariant(task_id, name)
        else:
            assert self.task_variants[task_id].name == name

    def create_unique_task(self, proc_id, uid, task_id, dim, p0, p1, p2):
        if uid in self.unique_ops:
            return
        assert proc_id in self.processors
        assert task_id in self.task_variants

        self.unique_ops[uid] = UniqueTask(
            proc = self.processors[proc_id],
            variant = self.task_variants[task_id],
            uid = uid,
            point = Point(dim, p0, p1, p2),
            color_string = color_helper(task_id % len(self.task_variants), len(self.task_variants)))

    def create_unique_map(self, proc_id, uid, parent_uid):
        assert uid not in self.unique_ops
        if parent_uid not in self.unique_ops:
            return False
        assert proc_id in self.processors

        self.unique_ops[uid] = UniqueMap(
            proc = self.processors[proc_id],
            uid = uid,
            parent = self.unique_ops[parent_uid])
        return True

    def create_unique_close(self, proc_id, uid, parent_uid):
        assert uid not in self.unique_ops
        if parent_uid not in self.unique_ops:
            return False
        assert proc_id in self.processors

        self.unique_ops[uid] = UniqueClose(
            proc = self.processors[proc_id],
            uid = uid,
            parent = self.unique_ops[parent_uid])
        return True

    def create_unique_copy(self, proc_id, uid, parent_uid):
        assert uid not in self.unique_ops
        if parent_uid not in self.unique_ops:
            return False
        assert proc_id in self.processors
        self.unique_ops[uid] = UniqueCopy(
            proc = self.processors[proc_id],
            uid = uid,
            parent = self.unique_ops[parent_uid])
        return True

    def create_event(self, proc_id, uid, kind_id, time):
        assert proc_id in self.processors
        if uid <> 0 and uid not in self.unique_ops:
            return False

        if uid == 0:
            unique_op = self.processors[proc_id].scheduler_op
        else:
            unique_op = self.unique_ops[uid]

        event = Event(
            unique_op = unique_op,
            kind_id = kind_id,
            time = time)
        return unique_op.add_event(event, self.processors[proc_id])

    def create_instance(self, iid, mem, redop, bf, time):
        if iid not in self.instances:
            self.instances[iid] = [Instance(iid)]
        else:
            self.instances[iid].append(Instance(iid))
        self.instances[iid][-1].set_create(self.memories[mem],
                                       redop, bf, time)
        assert mem in self.memories
        self.memories[mem].add_instance(self.instances[iid][-1])

    def add_instance_field(self, iid, fid, size):
        assert iid in self.instances
        self.instances[iid][-1].add_field(fid, size)

    def destroy_instance(self, iid, time):
        assert iid in self.instances
        #if iid not in self.instances:
        #    self.instances[iid] = [Instance(iid)]
        self.instances[iid][-1].set_destroy(time)

    def build_time_ranges(self):
        assert self.last_time is not None

        for proc in self.processors.itervalues():
            proc.init_time_range(self.last_time)

        # Now that we have all the time ranges added, sort themselves
        for proc in self.processors.itervalues():
            proc.sort_time_range()
        for mem in self.memories.itervalues():
            mem.sort_time_range()

    def print_processor_stats(self):
        print '****************************************************'
        print '   PROCESSOR STATS'
        print '****************************************************'
        for p,proc in sorted(self.processors.iteritems()):
            proc.print_stats()
        print

    def print_memory_stats(self):
        print '****************************************************'
        print '   MEMORY STATS'
        print '****************************************************'
        for m,mem in sorted(self.memories.iteritems()):
            mem.print_stats()
        print

    def print_task_stats(self, cumulative, verbose):
        print '****************************************************'
        print '   TASK STATS'
        print '****************************************************'
        stat = StatGatherer()
        for variant in self.task_variants.itervalues():
            stat.initialize_variant(variant)
        for proc in self.processors.itervalues():
            proc.update_task_stats(stat)
        # Total time is the overall execution time multiplied by the number of processors
        total_time = self.last_time * len(self.processors)
        stat.print_stats(total_time, cumulative, verbose)
        print

    def generate_svg_picture(self, file_name, html_file):
        # Before doing this, generate all the colors
        num_variants = len(self.task_variants)
        idx = 0
        for variant in self.task_variants.itervalues():
            variant.compute_color(idx, num_variants)
            idx = idx + 1
        printer = SVGPrinter(file_name, html_file)
        for p,proc in sorted(self.processors.iteritems()):
            proc.emit_svg(printer)
        printer.close()

    def generate_mem_picture(self, file_name, html_file):
        # Check for any leaked instances
        # and generate colors
        num_instances = 0
        for inst_list in self.instances.itervalues():
            num_instances += len(inst_list)
        idx = 0
        for i,inst_list in self.instances.iteritems():
            for inst in inst_list:
                if inst.destroy_time is None:
                    inst.set_destroy(self.last_time)
                    print 'INFO: Instance %u leaked' % inst.iid
                inst.compute_color(idx, num_instances)
                idx = idx + 1
        printer = SVGPrinter(file_name, html_file)
        for m,mem in sorted(self.memories.iteritems(),key=lambda x: x[0]):
            mem.emit_svg(printer, self.last_time)
        printer.close()

def parse_log_file(file_name, state):
    with open(file_name, 'rb') as log:
        matches = 0
        # Also find the time for the last event.
        last_time = 0L
        replay_lines = list()
        for line in log:
            matches += 1
            m = processor_pat.match(line)
            if m is not None:
                state.create_processor(
                    proc_id = int(m.group('proc'),16),
                    utility = int(m.group('utility')) == 1,
                    kind = int(m.group('kind')))
                continue
            m = memory_pat.match(line)
            if m is not None:
                state.create_memory(
                    mem = int(m.group('mem'),16),
                    kind = int(m.group('kind')))
                continue
            m = task_variant_pat.match(line)
            if m is not None:
                state.create_task_variant(
                    task_id = int(m.group('tid')),
                    name = m.group('name'))
                continue
            m = unique_task_pat.match(line)
            if m is not None:
                state.create_unique_task(
                    proc_id = int(m.group('proc'),16),
                    uid = int(m.group('uid')),
                    task_id = int(m.group('tid')),
                    dim = int(m.group('dim')),
                    p0 = int(m.group('p0')),
                    p1 = int(m.group('p1')),
                    p2 = int(m.group('p2')))
                continue
            m = unique_map_pat.match(line)
            if m is not None:
                if not state.create_unique_map(
                    proc_id = int(m.group('proc'),16),
                    uid = int(m.group('uid')),
                    parent_uid = int(m.group('puid'))):
                    replay_lines.append(line)
                continue
            m = unique_close_pat.match(line)
            if m is not None:
                if not state.create_unique_close(
                    proc_id = int(m.group('proc'),16),
                    uid = int(m.group('uid')),
                    parent_uid = int(m.group('puid'))):
                    replay_lines.append(line)
                continue
            m = unique_copy_pat.match(line)
            if m is not None:
                if not state.create_unique_copy(
                    proc_id = int(m.group('proc'),16),
                    uid = int(m.group('uid')),
                    parent_uid = int(m.group('puid'))):
                    replay_lines.append(line)
                continue
            m = event_pat.match(line)
            if m is not None:
                time = long(m.group('time'))
                if not state.create_event(
                    proc_id = int(m.group('proc'),16),
                    kind_id = int(m.group('kind')),
                    uid = int(m.group('uid')),
                    time = time):
                    replay_lines.append(line)
                if time > last_time:
                    last_time = time
                continue
            m = create_pat.match(line)
            if m is not None:
                state.create_instance(
                    iid = int(m.group('iid'),16),
                    mem = int(m.group('mem')),
                    redop = int(m.group('redop')),
                    bf = int(m.group('bf')),
                    time = long(m.group('time')))
                continue
            m = field_pat.match(line)
            if m is not None:
                state.add_instance_field(
                    iid = int(m.group('iid'),16),
                    fid = int(m.group('fid')),
                    size = int(m.group('size')))
                continue
            m = destroy_pat.match(line)
            if m is not None:
                state.destroy_instance(
                      iid = int(m.group('iid'),16),
                      time = long(m.group('time')))
                continue
            # If we made it here, then we failed to match.
            matches -= 1
            print 'Skipping line: %s' % line.strip()
    if state.last_time == None:
        state.last_time = last_time
    elif state.last_time < last_time:
        state.last_time = last_time
    while len(replay_lines) > 0:
        to_delete = set()
        for line in replay_lines:
            m = unique_map_pat.match(line)
            if m is not None:
                if state.create_unique_map(
                    proc_id = int(m.group('proc'),16),
                    uid = int(m.group('uid')),
                    parent_uid = int(m.group('puid'))):
                    to_delete.add(line)
                continue
            m = unique_close_pat.match(line)
            if m is not None:
                if state.create_unique_close(
                    proc_id = int(m.group('proc'),16),
                    uid = int(m.group('uid')),
                    parent_uid = int(m.group('puid'))):
                    to_delete.add(line)
                continue
            m = unique_copy_pat.match(line)
            if m is not None:
                if state.create_unique_copy(
                    proc_id = int(m.group('proc'),16),
                    uid = int(m.group('uid')),
                    parent_uid = int(m.group('puid'))):
                    to_delete.add(line)
                continue
            m = event_pat.match(line)
            if m is not None:
                time = long(m.group('time'))
                if state.create_event(
                    proc_id = int(m.group('proc'),16),
                    kind_id = int(m.group('kind')),
                    uid = int(m.group('uid')),
                    time = time):
                    to_delete.add(line)
                continue
        if len(to_delete) == 0:
            print "ERROR: NO FORWARD PROGRESS ON REPLAY LINES!  BAD LEGION PROF ASSUMPTION!"
            break
        for line in to_delete:
            replay_lines.remove(line)
    return matches

def usage():
    print 'Usage: '+sys.argv[0]+' [-c] [-p] [-v] <file_name>'
    print '  -c : perform cumulative analysis'
    print '  -p : generate HTML and SVG files for pictures'
    print '  -v : print verbose profiling information'
    print '  -m <ppm> : set the micro-seconds per pixel for images (default %d)' % (US_PER_PIXEL)
    print '  -i : generate HTML and SVG files for memory pictures'
    sys.exit(1)

def main():
    opts, args = getopt(sys.argv[1:],'cpvim:d:')
    opts = dict(opts)
    if len(args) == 0:
        usage()
    file_names = args
    cumulative = False
    generate_pictures = False
    generate_instance = False
    verbose = False
    if '-c' in opts:
        cumulative = True
    if '-p' in opts:
        generate_pictures = True
    if '-v' in opts:
        verbose = True
    if '-m' in opts:
        global US_PER_PIXEL
        US_PER_PIXEL = int(opts['-m'])
    if '-i' in opts:
        generate_instance = True
    svg_file_name = 'legion_prof.svg'
    html_file_name = 'legion_prof.html'
    mem_file_name = 'legion_prof_mem.svg'
    html_mem_file_name = 'legion_prof_mem.html'

    state = State()
    has_matches = False
    for file_name in file_names:
        print 'Loading log file %s...' % file_name
        total_matches = parse_log_file(file_name, state)
        print 'Matched %s lines' % total_matches
        if total_matches > 0:
            has_matches = True
    if not has_matches:
        print 'No matches. Exiting...'
        return
    # Now have the state build the time ranges for each processor
    state.build_time_ranges()

    # Print the per-processor statistics
    state.print_processor_stats()

    # Print the per-memory statistics
    state.print_memory_stats()

    # Print the per-task statistics
    state.print_task_stats(cumulative, verbose)

    # Generate the svg profiling picture
    if generate_pictures:
        print 'Generating SVG execution profile in %s...' % svg_file_name
        state.generate_svg_picture(svg_file_name, html_file_name)
        print 'Done!'
    if generate_instance:
        print 'Generating SVG memory profile in %s...' % mem_file_name
        state.generate_mem_picture(mem_file_name,html_mem_file_name)
        print 'Done!'

if __name__ == '__main__':
    main()

