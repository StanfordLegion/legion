#!/usr/bin/env python

# Copyright 2013 Stanford University
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
from getopt import getopt

prefix = "\[(?P<node>[0-9]+) - (?P<thread>[0-9a-f]+)\] \{\w+\}\{legion_prof\}: "
variant_pat       = re.compile(prefix+"Prof Task Variant (?P<tid>[0-9]+) (?P<leaf>[0-1]) (?P<name>\w+)")
processor_pat     = re.compile(prefix+"Prof Processor (?P<proc>[0-9a-f]+) (?P<utility>[0-1]) (?P<kind>[0-9]+)")
task_event_pat    = re.compile(prefix+"Prof Task Event (?P<proc>[0-9a-f]+) (?P<kind>[0-9]+) (?P<tid>[0-9]+) (?P<uid>[0-9]+) (?P<time>[0-9]+) (?P<dim>[0-9]+) (?P<p0>[0-9\-]+) (?P<p1>[0-9\-]+) (?P<p2>[0-9\-]+)")
scheduler_pat     = re.compile(prefix+"Prof Scheduler (?P<proc>[0-9]+) (?P<kind>[0-9]+) (?P<time>[0-9]+)")
memory_pat        = re.compile(prefix+"Prof Memory (?P<mem>[0-9a-f]+) (?P<kind>[0-9]+)")
create_pat        = re.compile(prefix+"Prof Create Instance (?P<iid>[0-9a-f]+) (?P<uid>[0-9]+) (?P<mem>[0-9a-f]+) (?P<redop>[0-9]+) (?P<factor>[0-9]+) (?P<time>[0-9]+)")
destroy_pat       = re.compile(prefix+"Prof Destroy Instance (?P<uid>[0-9]+) (?P<time>[0-9]+)")
field_pat         = re.compile(prefix+"Prof Instance Field (?P<uid>[0-9]+) (?P<fid>[0-9]+) (?P<size>[0-9]+)")
sub_task_pat      = re.compile(prefix+"Prof Subtask (?P<suid>[0-9]+) (?P<tid>[0-9]+) (?P<uid>[0-9]+) (?P<dim>[0-9]+) (?P<p0>[0-9]+) (?P<p1>[0-9]+) (?P<p2>[0-9]+)")

BEGIN_INDEX_SPACE_CREATE = 0
END_INDEX_SPACE_CREATE = 1
BEGIN_INDEX_SPACE_DESTROY = 2
END_INDEX_SPACE_DESTROY = 3
BEGIN_INDEX_PARTITION_CREATE = 4
END_INDEX_PARTITION_CREATE = 5
BEGIN_INDEX_PARTITION_DESTROY = 6
END_INDEX_PARTITION_DESTROY = 7
BEGIN_GET_INDEX_PARTITION = 8
END_GET_INDEX_PARTITION = 9
BEGIN_GET_INDEX_SUBSPACE = 10
END_GET_INDEX_SUBSPACE = 11
BEGIN_GET_INDEX_DOMAIN = 12
END_GET_INDEX_DOMAIN = 13
BEGIN_GET_INDEX_PARTITION_COLOR_SPACE = 14
END_GET_INDEX_PARTITION_COLOR_SPACE = 15
BEGIN_SAFE_CAST = 16
END_SAFE_CAST = 17
BEGIN_CREATE_FIELD_SPACE = 18
END_CREATE_FIELD_SPACE = 19
BEGIN_DESTROY_FIELD_SPACE = 20
END_DESTROY_FIELD_SPACE = 21
BEGIN_ALLOCATE_FIELDS = 22
END_ALLOCATE_FIELDS = 23
BEGIN_FREE_FIELDS = 24
END_FREE_FIELDS = 25
BEGIN_CREATE_REGION = 26
END_CREATE_REGION = 27
BEGIN_DESTROY_REGION = 28
END_DESTROY_REGION = 29
BEGIN_DESTROY_PARTITION = 30
END_DESTROY_PARTITION = 31
BEGIN_GET_LOGICAL_PARTITION = 32
END_GET_LOGICAL_PARTITION = 33
BEGIN_GET_LOGICAL_SUBREGION = 34
END_GET_LOGICAL_SUBREGION = 35
BEGIN_MAP_REGION = 36
END_MAP_REGION = 37
BEGIN_UNMAP_REGION = 38
END_UNMAP_REGION = 39
BEGIN_TASK_DEP_ANALYSIS = 40
END_TASK_DEP_ANALYSIS = 41
BEGIN_MAP_DEP_ANALYSIS = 42
END_MAP_DEP_ANALYSIS = 43
BEGIN_DEL_DEP_ANALYSIS = 44
END_DEL_DEP_ANALYSIS = 45
BEGIN_SCHEDULER = 46
END_SCHEDULER = 47
BEGIN_TASK_MAP = 48
END_TASK_MAP = 49
BEGIN_TASK_RUN = 50
END_TASK_RUN = 51
BEGIN_TASK_CHILDREN_MAPPED = 52
END_TASK_CHILDREN_MAPPED = 53
BEGIN_TASK_FINISH = 54
END_TASK_FINISH = 55
TASK_LAUNCH = 56

# Micro-seconds per pixel
US_PER_PIXEL = 1000
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
        self.indexes = list()
        self.indexes.append(p0)
        if dim > 1:
            self.indexes.append(p1)
        if dim > 2:
            self.indexes.append(p2)

    def matches(self, dim, p0, p1, p2):
        if self.dim <> dim:
            return False
        if self.indexes[0] <> p0:
            return False
        if dim > 1:
            if self.indexes[1] <> p1:
                return False
        if dim > 2:
            if self.indexes[2] <> p2:
                return False
        return True

    def to_string(self):
        result = "("+str(self.indexes[0])
        if self.dim > 1:
            result = result + "," + str(self.indexes[1])
        if self.dim > 2:
            result = result + "," + str(self.indexes[2])
        result = result + ")"
        return result

    def __repr__(self):
        return "Point: "+self.to_string()

    def __str__(self):
        return self.to_string()

class TaskInstance(object):
    def __init__(self, unique_task, point):
        self.unique_task = unique_task
        self.point = point
        self.begin_map = None
        self.end_map = None
        self.launch = None
        self.begin_run = None
        self.end_run = None
        self.begin_children_mapped = None
        self.end_children_mapped = None
        self.begin_task_finish = None
        self.end_task_finish = None
        self.map_processor = None
        self.run_processor = None
        self.children_processor = None
        self.finish_processor = None
        # Create index space
        self.create_index_spaces = dict()
        self.index_space_create = None
        # Destroy index space
        self.destroy_index_spaces = dict()
        self.index_space_destroy = None
        # Create index partitions
        self.create_index_partitions = dict()
        self.index_partition_create = None
        # Destroy index partitions
        self.destroy_index_partitions = dict()
        self.index_partition_destroy = None
        # Get index partitions
        self.get_index_partitions = dict()
        self.get_index_part = None
        # Get index subspaces
        self.get_index_subspaces = dict()
        self.get_index_sub = None
        # Get index domains
        self.get_index_domains = dict()
        self.get_index_dom = None
        # Get index partition color space
        self.get_index_partition_color_spaces = dict()
        self.get_index_color = None
        # Safe cast
        self.safe_casts = dict()
        self.cast = None
        # Create field space
        self.create_field_spaces = dict()
        self.field_space_create = None
        # Destroy field space
        self.destroy_field_spaces = dict()
        self.field_space_destroy = None
        # Allocate fields
        self.allocate_fields = dict()
        self.alloc_fields = None
        # Free fields
        self.free_fields = dict()
        self.free_fs = None
        # Create regions
        self.create_regions = dict()
        self.create_reg = None
        # Destroy regions
        self.destroy_regions = dict()
        self.region_destroy = None
        # Destroy partitions
        self.destroy_partitions = dict()
        self.partition_destroy = None
        # Get logical partition
        self.get_logical_partitions = dict()
        self.get_logical_part = None
        # Get logical subregion
        self.get_logical_subregions = dict()
        self.get_logical_sub = None
        # Inline map
        self.map_regions = dict()
        self.inline_map = None
        # Inline unmap
        self.unmap_regions = dict()
        self.inline_unmap = None
        # Task dependence analysis
        self.task_dep_analysis = dict()
        # Mapping dependence analysis
        self.map_dep_analysis = dict()
        self.map_dep = None
        # Deletion dependence anlaysis
        self.del_dep_analysis = dict()
        self.del_dep = None
        # Keep track of our set of subtasks
        self.subtasks = set()

    def add_task_event(self, processor, kind, event):
        if kind == BEGIN_TASK_MAP:
            assert self.begin_map == None
            assert self.map_processor == None
            self.begin_map = event
            self.map_processor = processor
        elif kind == END_TASK_MAP:
            assert self.end_map == None
            assert self.map_processor == processor
            self.end_map = event
        elif kind == TASK_LAUNCH:
            assert self.launch == None
            self.launch = event
        elif kind == BEGIN_TASK_RUN:
            assert self.begin_run == None
            assert self.run_processor == None
            self.begin_run = event
            self.run_processor = processor
        elif kind == END_TASK_RUN:
            assert self.end_run == None
            assert self.run_processor == processor
            self.end_run = event
        elif kind == BEGIN_TASK_CHILDREN_MAPPED:
            assert self.begin_children_mapped == None
            assert self.children_processor == None
            self.begin_children_mapped = event
            self.children_processor = processor
        elif kind == END_TASK_CHILDREN_MAPPED:
            assert self.end_children_mapped == None
            assert self.children_processor == processor
            self.end_children_mapped = event
        elif kind == BEGIN_TASK_FINISH:
            assert self.begin_task_finish == None
            assert self.finish_processor == None
            self.begin_task_finish = event
            self.finish_processor = processor
        elif kind == END_TASK_FINISH:
            assert self.end_task_finish == None
            assert self.finish_processor == processor
            self.end_task_finish = event
        elif kind == BEGIN_INDEX_SPACE_CREATE:
            assert self.index_space_create == None
            self.index_space_create = event
        elif kind == END_INDEX_SPACE_CREATE:
            assert self.index_space_create <> None
            time_range = CreateIndexSpaceRange(self,self.index_space_create,event)
            self.create_index_spaces[time_range] = processor
            self.index_space_create = None
        elif kind == BEGIN_INDEX_SPACE_DESTROY:
            assert self.index_space_destroy == None
            self.index_space_destroy = event
        elif kind == END_INDEX_SPACE_DESTROY:
            assert self.index_space_destroy <> None
            time_range = DestroyIndexSpaceRange(self,self.index_space_destroy,event)
            self.destroy_index_spaces[time_range] = processor
            self.index_space_destroy = None
        elif kind == BEGIN_INDEX_PARTITION_CREATE:
            assert self.index_partition_create == None
            self.index_partition_create = event
        elif kind == END_INDEX_PARTITION_CREATE:
            assert self.index_partition_create <> None
            time_range = CreateIndexPartitionRange(self,self.index_partition_create,event)
            self.create_index_partitions[time_range] = processor
            self.index_partition_create = None
        elif kind == BEGIN_INDEX_PARTITION_DESTROY:
            assert self.index_partition_destroy == None
            self.index_partition_destroy = event
        elif kind == END_INDEX_PARTITION_DESTROY:
            assert self.index_partition_destroy <> None
            time_range = DestroyIndexPartitionRange(self,self.index_partition_destroy,event)
            self.destroy_index_partitions[time_range] = processor
            self.index_partition_destroy = None
        elif kind == BEGIN_GET_INDEX_PARTITION:
            assert self.get_index_part == None
            self.get_index_part = event
        elif kind == END_GET_INDEX_PARTITION:
            assert self.get_index_part <> None
            time_range = GetIndexPartitionRange(self,self.get_index_part,event)
            self.get_index_partitions[time_range] = processor
            self.get_index_part = None
        elif kind == BEGIN_GET_INDEX_SUBSPACE:
            assert self.get_index_sub == None
            self.get_index_sub = event
        elif kind == END_GET_INDEX_SUBSPACE:
            assert self.get_index_sub <> None
            time_range = GetIndexSubspaceRange(self,self.get_index_sub,event)
            self.get_index_subspaces[time_range] = processor
            self.get_index_sub = None
        elif kind == BEGIN_GET_INDEX_DOMAIN:
            assert self.get_index_dom == None
            self.get_index_dom = event
        elif kind == END_GET_INDEX_DOMAIN:
            assert self.get_index_dom <> None
            time_range = GetIndexDomainRange(self,self.get_index_dom,event)
            self.get_index_domains[time_range] = processor
            self.get_index_dom = None
        elif kind == BEGIN_GET_INDEX_PARTITION_COLOR_SPACE:
            assert self.get_index_color == None
            self.get_index_color = event
        elif kind == END_GET_INDEX_PARTITION_COLOR_SPACE:
            assert self.get_index_color <> None
            time_range = GetIndexColorSpaceRange(self,self.get_index_color,event)
            self.get_index_partition_color_spaces[time_range] = processor
            self.get_index_color = None
        elif kind == BEGIN_SAFE_CAST:
            assert self.cast == None
            self.cast = event
        elif kind == END_SAFE_CAST:
            assert self.cast <> None
            time_range = SafeCastRange(self,self.cast,event)
            self.safe_casts[time_range] = processor
            self.cast = None
        elif kind == BEGIN_CREATE_FIELD_SPACE:
            assert self.field_space_create == None
            self.field_space_create = event
        elif kind == END_CREATE_FIELD_SPACE:
            assert self.field_space_create <> None
            time_range = CreateFieldSpaceRange(self,self.field_space_create,event)
            self.create_field_spaces[time_range] = processor
            self.field_space_create = None
        elif kind == BEGIN_DESTROY_FIELD_SPACE:
            assert self.field_space_destroy == None
            self.field_space_destroy = event
        elif kind == END_DESTROY_FIELD_SPACE:
            assert self.field_space_destroy <> None
            time_range = DestroyFieldSpaceRange(self,self.field_space_destroy,event)
            self.destroy_field_spaces[time_range] = processor
            self.field_space_destroy = None
        elif kind == BEGIN_ALLOCATE_FIELDS:
            assert self.alloc_fields == None
            self.alloc_fields = event
        elif kind == END_ALLOCATE_FIELDS:
            assert self.alloc_fields <> None
            time_range = AllocFieldRange(self,self.alloc_fields,event)
            self.allocate_fields[time_range] = processor
            self.alloc_fields = None
        elif kind == BEGIN_FREE_FIELDS:
            assert self.free_fs == None
            self.free_fs = event
        elif kind == END_FREE_FIELDS:
            assert self.free_fs <> None
            time_range = FreeFieldsRange(self,self.free_fs,event)
            self.free_fields[time_range] = processor
            self.free_fs = None
        elif kind == BEGIN_CREATE_REGION:
            assert self.create_reg == None
            self.create_reg = event
        elif kind == END_CREATE_REGION:
            assert self.create_reg <> None
            time_range = CreateRegionRange(self,self.create_reg,event)
            self.create_regions[time_range] = processor
            self.create_reg = None
        elif kind == BEGIN_DESTROY_REGION:
            assert self.region_destroy == None
            self.region_destroy = event
        elif kind == END_DESTROY_REGION:
            assert self.region_destroy <> None
            time_range = DestroyRegionRange(self,self.region_destroy,event)
            self.destroy_regions[time_range] = processor
            self.region_destroy = None
        elif kind == BEGIN_DESTROY_PARTITION:
            assert self.partition_destroy == None
            self.partition_destroy = event
        elif kind == END_DESTROY_PARTITION:
            assert self.partition_destroy <> None
            time_range = DestroyPartitionRange(self,self.partition_destroy,event)
            self.destroy_partitions[time_range] = processor
            self.partition_destroy = None
        elif kind == BEGIN_GET_LOGICAL_PARTITION:
            assert self.get_logical_part == None
            self.get_logical_part = event
        elif kind == END_GET_LOGICAL_PARTITION:
            assert self.get_logical_part <> None
            time_range = GetLogicalPartitionRange(self,self.get_logical_part,event)
            self.get_logical_partitions[time_range] = processor
            self.get_logical_part = None
        elif kind == BEGIN_GET_LOGICAL_SUBREGION:
            assert self.get_logical_sub == None
            self.get_logical_sub = event
        elif kind == END_GET_LOGICAL_SUBREGION:
            assert self.get_logical_sub <> None
            time_range = GetLogicalSubregionRange(self,self.get_logical_sub,event)
            self.get_logical_subregions[time_range] = processor
            self.get_logical_sub = None
        elif kind == BEGIN_MAP_REGION:
            assert self.inline_map == None
            self.inline_map = event
        elif kind == END_MAP_REGION:
            assert self.inline_map <> None
            time_range = InlineMapRange(self,self.inline_map,event)
            self.map_regions[time_range] = processor
            self.inline_map = None
        elif kind == BEGIN_UNMAP_REGION:
            assert self.inline_unmap == None
            self.inline_unmap = event
        elif kind == END_UNMAP_REGION:
            assert self.inline_unmap <> None
            time_range = InlineUnmapRange(self,self.inline_unmap,event)
            self.unmap_regions[time_range] = processor
            self.inline_unmap = None
        elif kind == BEGIN_TASK_DEP_ANALYSIS:
            assert self.task_dep == None
            self.task_dep = event
        elif kind == END_TASK_DEP_ANALYSIS:
            assert self.task_dep <> None
            time_range = TaskDepRange(self,self.task_dep,event)
            self.task_dep_analysis[time_range] = processor
            self.task_dep = None
        elif kind == BEGIN_MAP_DEP_ANALYSIS:
            assert self.map_dep == None
            self.map_dep = event
        elif kind == END_MAP_DEP_ANALYSIS:
            assert self.map_dep <> None
            time_range = MapDepRange(self,self.map_dep,event)
            self.map_dep_analysis[time_range] = processor
            self.map_dep = None
        elif kind == BEGIN_DEL_DEP_ANALYSIS:
            assert self.del_dep == None
            self.del_dep = event
        elif kind == END_DEL_DEP_ANALYSIS:
            assert self.del_dep <> None
            time_range = DelDepRange(self,self.del_dep,event)
            self.del_dep_analysis[time_range] = processor
            self.del_dep = None
        else:
            # These are handled at the granularity of UniqueTasks
            assert kind <> BEGIN_TASK_DEP_ANALYSIS
            assert kind <> END_TASK_DEP_ANALYSIS

    def add_subtask(self, subtask):
        assert subtask not in self.subtasks 
        self.subtasks.add(subtask)
        subtask.add_parent(self)

    def add_time_ranges(self):
        # Might not have a map processor if it is the top level task
        assert self.map_processor <> None
        assert self.begin_map <> None
        assert self.end_map <> None
        self.map_processor.add_range(TaskMapRange(self,self.begin_map,self.end_map))
        assert self.run_processor <> None
        assert self.begin_run <> None
        assert self.end_run <> None
        self.run_processor.add_range(TaskRunRange(self,self.begin_run,self.end_run))
        if self.children_processor <> None:
            assert self.begin_children_mapped <> None
            assert self.end_children_mapped <> None
            self.children_processor.add_range(TaskChildRange(self,self.begin_children_mapped,self.end_children_mapped))
        assert self.finish_processor <> None
        assert self.begin_task_finish <> None
        assert self.end_task_finish <> None
        self.finish_processor.add_range(TaskFinishRange(self,self.begin_task_finish,self.end_task_finish))
        for r,proc in self.create_index_spaces.iteritems():
            proc.add_range(r)
        for r,proc in self.destroy_index_spaces.iteritems():
            proc.add_range(r)
        for r,proc in self.create_index_partitions.iteritems():
            proc.add_range(r)
        for r,proc in self.destroy_index_partitions.iteritems():
            proc.add_range(r)
        for r,proc in self.get_index_partitions.iteritems():
            proc.add_range(r)
        for r,proc in self.get_index_subspaces.iteritems():
            proc.add_range(r)
        for r,proc in self.get_index_domains.iteritems():
            proc.add_range(r)
        for r,proc in self.get_index_partition_color_spaces.iteritems():
            proc.add_range(r)
        for r,proc in self.safe_casts.iteritems():
            proc.add_range(r)
        for r,proc in self.create_field_spaces.iteritems():
            proc.add_range(r)
        for r,proc in self.destroy_field_spaces.iteritems():
            proc.add_range(r)
        for r,proc in self.allocate_fields.iteritems():
            proc.add_range(r)
        for r,proc in self.free_fields.iteritems():
            proc.add_range(r)
        for r,proc in self.create_regions.iteritems():
            proc.add_range(r)
        for r,proc in self.destroy_regions.iteritems():
            proc.add_range(r)
        for r,proc in self.destroy_partitions.iteritems():
            proc.add_range(r)
        for r,proc in self.get_logical_partitions.iteritems():
            proc.add_range(r)
        for r,proc in self.get_logical_subregions.iteritems():
            proc.add_range(r)
        for r,proc in self.map_regions.iteritems():
            proc.add_range(r)
        for r,proc in self.unmap_regions.iteritems():
            proc.add_range(r)
        for r,proc in self.task_dep_analysis.iteritems():
            proc.add_range(r)
        for r,proc in self.map_dep_analysis.iteritems():
            proc.add_range(r)
        for r,proc in self.del_dep_analysis.iteritems():
            proc.add_range(r)

    def get_title(self):
        return self.unique_task.get_title()+" Point "+self.point.to_string()

    def get_color(self):
        return self.unique_task.get_color()

    def get_unique_task(self):
        return self.unique_task

    def get_variant(self):
        return self.unique_task.get_variant()

class UniqueTask(object):
    def __init__(self, variant, unique_id):
        self.variant = variant
        self.unique_id = unique_id
        self.points = dict()
        # Mapping analysis is done at the granularity of unique
        # tasks and not individual points so keep track of it here
        # and ignore the point values.  We'll accumulate this
        # into the enclosing parent task's dependence analysis
        # cost during add_time_ranges
        self.dep_analysis = None
        self.dep_processor = None
        # The parent task is a variant
        self.parent_task = None

    def add_task_event(self, processor, kind, point, event):
        if kind == BEGIN_TASK_DEP_ANALYSIS:
            assert self.dep_analysis == None
            assert self.dep_processor == None
            self.dep_analysis = event
            self.dep_processor = processor
        elif kind == END_TASK_DEP_ANALYSIS:
            assert self.dep_analysis <> None 
            assert self.dep_processor == processor
            dep_range = TaskDepRange(self,self.dep_analysis,event)
            self.dep_analysis = dep_range
        else:
            # Do the normal thing here
            if point not in self.points:
                self.points[point] = TaskInstance(self, point)
            self.points[point].add_task_event(processor, kind, event)

    def add_parent(self, parent):
        assert self.parent_task == None
        self.parent_task = parent

    def add_time_ranges(self):
        for p,t in self.points.iteritems():
            t.add_time_ranges()
        # Also put our dependence analysis range in
        if self.dep_analysis <> None:
            assert self.dep_processor <> None
            self.dep_processor.add_range(self.dep_analysis)

    def get_title(self):
        return self.variant.get_title()+" (UID "+str(self.unique_id)+")" 

    def get_color(self):
        return self.variant.get_color()

    def get_variant(self):
        return self.variant

    def add_subtask(self, point, subtask):
        if point not in self.points:
            self.points[point] = TaskInstance(self, point)
        self.points[point].add_subtask(subtask)

class TaskVariant(object):
    def __init__(self, task_id, leaf, name):
        self.task_id = task_id
        self.leaf = leaf
        self.name = name
        self.color = None 
        self.unique_tasks = dict()

    def add_task_event(self, processor, event, kind, uid, point):
        if uid not in self.unique_tasks:
            self.unique_tasks[uid] = UniqueTask(self, uid)
        self.unique_tasks[uid].add_task_event(processor, kind, point, event)

    def add_time_ranges(self):
        for u,t in self.unique_tasks.iteritems():
            t.add_time_ranges()

    def get_title(self):
        return "Task ID "+str(self.task_id)+" "+str(self.name)

    def compute_color(self, step, num_steps):
        assert self.color == None
        self.color = color_helper(step, num_steps)

    def get_color(self):
        return self.color

    def get_task_id(self):
        return self.task_id

    def has_unique_task(self, uid):
        if uid in self.unique_tasks:
            return self.unique_tasks[uid]
        return None

    def add_subtask(self, uid, point, subtask):
        if uid not in self.unique_tasks:
            self.unique_tasks[uid] = UniqueTask(self, uid)
        self.unique_tasks[uid].add_subtask(point, subtask)

class ScheduleInstance(object):
    def __init__(self):
        self.begin_schedule = None
        self.end_schedule = None
        self.schedule_processor = None

    def add_scheduling_event(self, processor, kind, event):
        assert kind == BEGIN_SCHEDULER or kind == END_SCHEDULER
        if kind == BEGIN_SCHEDULER:
            assert self.schedule_processor == None
            self.begin_schedule = event
            self.schedule_processor = processor
        else:
            assert self.schedule_processor == processor
            self.end_schedule = event

class Event(object):
    def __init__(self, time, proc):
        self.abs_time = time
        self.processor = proc

    def __cmp__(self, other):
        if other is None:
            return -1
        if self.abs_time < other.abs_time:
            return -1
        elif self.abs_time == other.abs_time:
            return 0
        else:
            return 1

# Some fancy stuff to emulate dynamic dispatch on pure virtual
# functions in python since they aren't natively supported
# Borrowed from http://code.activestate.com/recipes/266468/
class AbstractMethod(object):
    def __init__(self, func):
        self._function = func

    def __get__(self, obj, type):
        return self.AbstractMethodHelper(self._function, type)

    class AbstractMethodHelper(object):
        def __init__(self, func, cls):
            self._function = func
            self._class = cls

        def __call__(self, *args, **kwargs):
            raise TypeError('Abstract method `'+ self._class.__name__ \
                  + '.' + self._function + '\' called')

class Metaclass(type):
    def __init__(cls, name, bases, *args, **kwargs):
        super(Metaclass, cls).__init__(cls, name, bases)
        cls.__new__ = staticmethod(cls.new)
        abstractmethods = []
        ancestors = list(cls.__mro__)
        ancestors.reverse()
        for ancestor in ancestors:
            for clsname, clst in ancestor.__dict__.items():
                if isinstance(clst, AbstractMethod):
                    abstractmethods.append(clsname)
                else:
                    if clsname in abstractmethods:
                        abstractmethods.remove(clsname)
        abstractmethods.sort()
        setattr(cls, '__abstractmethods__', abstractmethods)

    def new(self, cls, *args, **kwargs):
        if len(cls.__abstractmethods__):
            raise NotImplementedError('Can\'t instantiate class `' + \
                                      cls.__name__ + '\';\n' + \
                                      'Abstract methods: ' + \
                                      ", ".join(cls.__abstractmethods__))
        return object.__new__(self)

class TimeRange(object):
    __metaclass__ = Metaclass
    is_app_range = AbstractMethod('is_app_range')
    is_meta_range = AbstractMethod('is_meta_range')
    emit_svg = AbstractMethod('emit_svg')
    def __init__(self, start_event, end_event):
        assert start_event <> None
        assert end_event <> None
        assert start_event <= end_event
        self.start_event = start_event
        self.end_event = end_event
        self.subranges = list()

    def contains(self, other_range):
        if self.start_event > other_range.end_event:
            return False
        if self.end_event < other_range.start_event:
            return False
        if self.start_event <= other_range.start_event and \
            other_range.end_event <= self.end_event:
            return True
        # Otherwise they overlap one way or the other
        # but neither contains the other
        return False

    def add_range(self, other_range):
        assert self.contains(other_range)
        self.subranges.append(other_range)

    def sort_range(self):
        cur_idx = 0
        while cur_idx < len(self.subranges):
            cur_range = self.subranges[cur_idx]
            # Try adding it to any of the other ranges
            added = False
            for idx in range(len(self.subranges)):
                if idx == cur_idx:
                    continue
                if self.subranges[idx].contains(cur_range):
                    self.subranges[idx].add_range(cur_range)
                    added = True
                    break
            if added:
                # Remove this entry from the list, keep the same cur_idx
                self.subranges.remove(cur_range)
            else:
                # Didn't add it, so go onto the next range
                cur_idx = cur_idx + 1
        # Now recursively sort all of our subranges
        for idx in range(len(self.subranges)):
            self.subranges[idx].sort_range()

    def get_app_range(self):
        return self.is_app_range()

    def get_meta_range(self):
        return self.is_meta_range()

    def total_time(self):
        return (self.end_event.abs_time - self.start_event.abs_time)

    def non_cummulative_time(self):
        total_time = self.total_time()
        for r in self.subranges:
            total_time = total_time - r.total_time()
        assert total_time >= 0
        return total_time

    def active_time(self):
        total_time = 0
        for idx in range(len(self.subranges)):
            total_time = total_time + self.subranges[idx].total_time()
        return total_time

    def application_time(self):
        if self.get_app_range():
            app_time = self.total_time()
            # Then subtract out any subranges that are not application time
            for idx in range(len(self.subranges)):
                if self.subranges[idx].get_meta_range():
                    app_time = app_time - self.subranges[idx].meta_time()
            assert app_time >= 0
            return app_time
        else:
            # Otherwise this is meta, so count up any components that are app
            app_time = 0
            for idx in range(len(self.subranges)):
                if self.subranges[idx].get_app_range():
                    app_time = app_time + self.subranges[idx].application_time()
            return app_time

    def meta_time(self):
        if self.get_meta_range():
            meta_time = self.total_time()
            # Now subtract out any ranges that are application time
            for idx in range(len(self.subranges)):
                if self.subranges[idx].get_app_range():
                    meta_time = meta_time - self.subranges[idx].application_time()
            assert meta_time >= 0
            return meta_time
        else:
            # Otherwise this is application, so add up any meta components
            meta_time = 0
            for idx in range(len(self.subranges)):
                if self.subranges[idx].get_meta_range():
                    meta_time = meta_time + self.subranges[idx].meta_time()
            return meta_time

    def max_levels(self):
        max_lev = 0
        for idx in range(len(self.subranges)):
            levels = self.subranges[idx].max_levels()
            if levels > max_lev:
                max_lev = levels
        return max_lev+1

    def emit_svg_range(self, printer):
        self.emit_svg(printer, 0) 

    def get_timing_string(self):
        result = "Start: %d us  Stop: %d us  Total: %d us" % (self.start_event.abs_time,self.end_event.abs_time,self.total_time())
        return result


class BaseRange(TimeRange):
    def __init__(self, proc, start_event, end_event):
        TimeRange.__init__(self, start_event, end_event)
        self.proc = proc

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return False

    def emit_svg(self, printer, level):
        title = self.proc.get_title()
        printer.emit_time_line(level, self.start_event.abs_time,self.end_event.abs_time,title) 
        for idx in range(len(self.subranges)):
            self.subranges[idx].emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        for r in self.subranges:
            r.update_task_stats(stat)

class TaskMapRange(TimeRange):
    def __init__(self, inst, begin_map, end_map):
        TimeRange.__init__(self, begin_map, end_map)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#009900" # Green
        title = "Task Mapping for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for idx in range(len(self.subranges)):
            self.subranges[idx].emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time() 
        stat.update_task_map(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)
        

class TaskWaitRange(TimeRange):
    def __init__(self, launch_event, run_event):
        TimeRange.__init__(self, launch_event, run_event)

class TaskRunRange(TimeRange):
    def __init__(self, inst, begin_run, end_run):
        TimeRange.__init__(self, begin_run, end_run)
        self.inst = inst

    def is_app_range(self):
        return True

    def is_meta_range(self):
        return False

    def emit_svg(self, printer, level):
        color = self.inst.get_color()
        title = "Task Execution for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for idx in range(len(self.subranges)):
            self.subranges[idx].emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_task_run(variant, cum_time, non_cum_time)
        stat.start_task(variant)
        for r in self.subranges:
            r.update_task_stats(stat)
        stat.stop_task(variant)

class TaskChildRange(TimeRange):
    def __init__(self, inst, begin_children, end_children):
        TimeRange.__init__(self, begin_children, end_children)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0099999" # Turquoise
        title = "All Children Mapped for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for idx in range(len(self.subranges)):
            self.subranges[idx].emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_task_children_map(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class TaskFinishRange(TimeRange):
    def __init__(self, inst, begin_finish, end_finish):
        TimeRange.__init__(self, begin_finish, end_finish)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF" # Duke Blue
        title = "Task Finish for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for idx in range(len(self.subranges)):
            self.subranges[idx].emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_task_finish(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class CreateIndexSpaceRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self,begin,end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF" # Duke Blue
        title = "Create Index Space for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_create_index_space(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class DestroyIndexSpaceRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self,begin,end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Destroy Index Space for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_destroy_index_space(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class CreateIndexPartitionRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self,begin,end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Create Index Partition for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_create_index_partition(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class DestroyIndexPartitionRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self,begin,end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Destroy Index Partition for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_destroy_index_partition(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class GetIndexPartitionRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Get Index Partition for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_get_index_partition(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class GetIndexSubspaceRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Get Index Subspace for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_get_index_subspace(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class GetIndexDomainRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        self.inst = inst
        
    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Get Index Domain for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_get_index_domain(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class GetIndexColorSpaceRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Get Index Color Space for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_get_index_color_space(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class SafeCastRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Safe Cast for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_safe_cast(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class CreateFieldSpaceRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Create Field Space for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_create_field_space(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class DestroyFieldSpaceRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Destroy Field Space for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_destroy_field_space(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class AllocFieldRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Allocate Fields for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_allocate_fields(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class FreeFieldsRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Free Fields for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_free_fields(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class CreateRegionRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Create Logical Region for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_create_logical_region(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class DestroyRegionRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Destroy Logical Region for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_destroy_logical_region(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class DestroyPartitionRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Destroy Logical Partition for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_destroy_logical_partition(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class GetLogicalPartitionRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Get Logical Partition for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_get_logical_partition(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class GetLogicalSubregionRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Get Logical Subregion for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_get_logical_subregion(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class InlineMapRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Inline Mapping for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_inline_mapping(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class InlineUnmapRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Inline Unmap for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_inline_unmap(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class TaskDepRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        # This is actually a UniqueTask
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Task Dependence Analysis for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_task_dependence_analysis(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class MapDepRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Inline Mapping Dependence Analysis for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_inline_mapping_dependence_analysis(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class DelDepRange(TimeRange):
    def __init__(self, inst, begin, end):
        TimeRange.__init__(self, begin, end)
        self.inst = inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0000FF"
        title = "Deletion Dependence Analysis for "+self.inst.get_title()+" "+self.get_timing_string()
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for r in self.subranges:
            r.emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        variant = self.inst.get_variant()
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_deletion_dependence_analysis(variant, cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

#color = "#006600" # Dark Green 
#color = "#6600CC" # Deep Purple

class ScheduleRange(TimeRange):
    def __init__(self, sched_inst):
        TimeRange.__init__(self, sched_inst.begin_schedule, sched_inst.end_schedule)
        self.sched_inst = sched_inst

    def is_app_range(self):
        return False

    def is_meta_range(self):
        return True

    def emit_svg(self, printer, level):
        color = "#0099CC" # Carolina Blue
        title = "Runtime Scheduler"
        printer.emit_timing_range(color, level, self.start_event.abs_time, self.end_event.abs_time, title)
        for idx in range(len(self.subranges)):
            self.subranges[idx].emit_svg(printer, level+1)

    def update_task_stats(self, stat):
        cum_time = self.total_time()
        non_cum_time = self.non_cummulative_time()
        stat.update_scheduler(cum_time, non_cum_time)
        for r in self.subranges:
            r.update_task_stats(stat)

class Processor(object):
    def __init__(self, proc_id, utility, kind):
        self.proc_id = proc_id
        self.utility = utility
        if kind == 1 or kind == 2: # Kind 2 is a utility proc
            self.proc_kind = "CPU"
        elif kind == 0:
            self.proc_kind = "GPU"
        else:
            print "WARNING: Unrecognized processor kind "+str(kind)
            self.proc_kind = "OTHER PROC KIND"
        self.current_scheduling = None
        self.sched_instances = list()
        self.full_range = None

    def add_scheduling_event(self, kind, event):
        if self.current_scheduling <> None:
            assert kind == END_SCHEDULER
            self.current_scheduling.add_scheduling_event(self, kind, event)
            self.sched_instances.append(self.current_scheduling)
            self.current_scheduling = None
        else:
            assert kind == BEGIN_SCHEDULER
            self.current_scheduling = ScheduleInstance()
            self.current_scheduling.add_scheduling_event(self, kind, event)

    def init_time_range(self, last_time):
        self.full_range = BaseRange(self,Event(long(0),self), Event(last_time,self))

    def add_range(self, timing_range):
        assert self.full_range.contains(timing_range)
        self.full_range.add_range(timing_range)

    def sort_time_range(self):
        self.full_range.sort_range()

    def add_time_ranges(self):
        for s in self.sched_instances:
            self.add_range(ScheduleRange(s))

    def get_title(self):
        result = self.proc_kind+" Processor "+str(hex(self.proc_id))
        if self.utility:
            result = result + " (Utility)"
        return result

    def print_stats(self):
        # Figure out the total time for this processor
        # The amount of time the processor was active
        # The amount of time spent on application tasks
        # The amount of time spent on meta tasks
        total_time = self.full_range.total_time()
        active_time = self.full_range.active_time()
        application_time = self.full_range.application_time()
        meta_time = self.full_range.meta_time()
        active_ratio = 100.0*float(active_time)/float(total_time)
        application_ratio = 100.0*float(application_time)/float(total_time)
        meta_ratio = 100.0*float(meta_time)/float(total_time)
        print self.get_title()
        print "    Total time: %d us" % total_time
        print "    Active time: %d us (%.3f%%)" % (active_time, active_ratio)
        print "    Application time: %d us (%.3f%%)" % (application_time, application_ratio)
        print "    Meta time: %d us (%.3f%%)" % (meta_time, meta_ratio)
        print ""

    def emit_svg(self, printer):
        # First figure out the max number of levels + 1 for padding
        max_levels = self.full_range.max_levels() + 1
        printer.init_processor(max_levels)
        self.full_range.emit_svg_range(printer)

    def update_task_stats(self, stat):
        self.full_range.update_task_stats(stat)
        
class Memory(object):
    def __init__(self, mem, kind):
        self.mem = mem
        self.kind = kind
        self.instances = set()
        self.max_live_instances = None
        self.time_points = None

    def add_instance(self, inst):
        assert inst not in self.instances
        self.instances.add(inst)

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
            title = "L3 Cache "
        elif self.kind == 7:
            title = "L2 Cache "
        elif self.kind == 8:
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
                    if next_time == None:
                        next_time = inst.destroy_time
                        creation = False
                        target_inst = inst
                    elif inst.destroy_time < next_time:
                        next_time = inst.destroy_time
                        creation = False
                        target_inst = inst
                else:
                    # Look at its creation time
                    if next_time == None:
                        next_time = inst.create_time
                        creation = True
                        target_inst = inst
                    elif inst.create_time < next_time:
                        next_time = inst.create_time
                        creation = True
                        target_inst = inst
            # Should have found something
            assert next_time <> None
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
                    if lev_inst == None:
                        level = lev
                        break
                # If we didn't find a level, make a new one
                if level == None:
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
    def __init__(self, iid, uid, memory, redop, factor, create_time):
        self.iid = iid
        self.uid = uid
        self.memory = memory
        self.redop = redop
        self.blocking_factor = factor
        self.create_time = create_time
        self.destroy_time = None
        self.color = None
        self.fields = dict()

    def set_destroy(self, time):
        assert self.destroy_time == None
        self.destroy_time = time

    def add_field(self, fid, size):
        assert fid not in self.fields
        self.fields[fid] = size

    def compute_color(self, step, num_steps):
        assert self.color == None
        self.color = color_helper(step, num_steps)

    def get_title(self):
        title = "Instance "+str(hex(self.iid))+" blocking factor "+str(self.blocking_factor)
        title = title+" fields: "
        for fid,size in self.fields.iteritems():
            title = title + "("+str(fid)+","+str(size)+" bytes)"
        return title

    def emit_svg(self, printer, level):
        assert self.color <> None
        printer.emit_timing_range(self.color, level, self.create_time, self.destroy_time, self.get_title())

class SVGPrinter(object):
    def __init__(self, file_name, html_file):
        self.target = open(file_name,'w')
        self.file_name = file_name
        self.html_file = html_file
        assert self.target <> None
        self.offset = 0
        self.target.write('<svg xmlns="http://www.w3.org/2000/svg">\n')
        self.max_width = 0
        self.max_height = 0

    def close(self):
        self.emit_time_scale()
        self.target.write('</svg>\n')
        self.target.close()
        # Round up the max width and max height to a multiple of 100
        while ((self.max_width % 100) <> 0):
            self.max_width = self.max_width + 1
        while ((self.max_height % 100) <> 0):
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

    def increment(self, cum, non_cum):
        self.invocations = self.invocations + 1
        self.cum_time = self.cum_time + cum
        self.non_cum_time = self.non_cum_time + non_cum

    def is_empty(self):
        return self.invocations == 0

    def print_stats(self, total_time):
        print "                Total Invocations: "+str(self.invocations)
        if self.invocations > 0:
            print "                Cummulative Time: %d us (%.3f%%)" % (self.cum_time,100.0*float(self.cum_time)/float(total_time))
            print "                Non-Cummulative Time: %d us (%.3f%%)" % (self.non_cum_time,100.0*float(self.non_cum_time)/float(total_time))
            print "                Average Cum Time: %.3f us" % (float(self.cum_time)/float(self.invocations))
            print "                Average Non-Cum Time: %.3f us" % (float(self.non_cum_time)/float(self.invocations))

class StatVariant(object):
    def __init__(self, var):
        self.var = var
        self.mapping_calls = CallTracker()
        self.run_calls = CallTracker()
        self.children_map_calls = CallTracker()
        self.finish_calls = CallTracker()
        self.create_index_space_calls = CallTracker()
        self.destroy_index_space_calls = CallTracker()
        self.create_index_partition_calls = CallTracker()
        self.destroy_index_partition_calls = CallTracker()
        self.get_index_partition_calls = CallTracker()
        self.get_index_subspace_calls = CallTracker()
        self.get_index_domain_calls = CallTracker()
        self.get_index_color_space_calls = CallTracker()
        self.safe_cast_calls = CallTracker()
        self.create_field_space_calls = CallTracker()
        self.destroy_field_space_calls = CallTracker()
        self.allocate_fields_calls = CallTracker()
        self.free_fields_calls = CallTracker()
        self.create_logical_region_calls = CallTracker()
        self.destroy_logical_region_calls = CallTracker()
        self.destroy_logical_partition_calls = CallTracker()
        self.get_logical_partition_calls = CallTracker()
        self.get_logical_subregion_calls = CallTracker()
        self.inline_mapping_calls = CallTracker()
        self.inline_unmap_calls = CallTracker()
        self.task_dependence_calls = CallTracker()
        self.inline_map_dep_calls = CallTracker()
        self.deletion_dep_calls = CallTracker()

    def update_task_map(self, cum, non_cum):
        self.mapping_calls.increment(cum, non_cum)

    def update_task_run(self, cum, non_cum):
        self.run_calls.increment(cum, non_cum)

    def update_task_children_map(self, cum, non_cum):
        self.children_map_calls.increment(cum, non_cum)

    def update_task_finish(self, cum, non_cum):
        self.finish_calls.increment(cum, non_cum)

    def update_inline_map(self, cum, non_cum):
        self.inline_mapping.increment(cum, non_cum)

    def update_create_index_space(self, cum, non_cum):
        self.create_index_space_calls.increment(cum, non_cum)

    def update_destroy_index_space(self, cum, non_cum):
        self.destroy_index_space_calls.increment(cum, non_cum)

    def update_create_index_partition(self, cum, non_cum):
        self.create_index_partition_calls.increment(cum, non_cum)

    def update_destroy_index_partition(self, cum, non_cum):
        self.destroy_index_partition_calls.increment(cum, non_cum)

    def update_get_index_partition(self, cum, non_cum):
        self.get_index_partition_calls.increment(cum, non_cum)

    def update_get_index_subspace(self, cum, non_cum):
        self.get_index_subspace_calls.increment(cum, non_cum)

    def update_get_index_domain(self, cum, non_cum):
        self.get_index_domain_calls.increment(cum, non_cum)

    def update_get_index_color_space(self, cum, non_cum):
        self.get_index_color_space_calls.increment(cum, non_cum)

    def update_safe_cast(self, cum, non_cum):
        self.safe_cast_calls.increment(cum, non_cum)

    def update_create_field_space(self, cum, non_cum):
        self.create_field_space_calls.increment(cum, non_cum)

    def update_destroy_field_space(self, cum, non_cum):
        self.destroy_field_space_calls.increment(cum, non_cum)

    def update_allocate_fields(self, cum, non_cum):
        self.allocate_fields_calls.increment(cum, non_cum)

    def update_free_fields(self, cum, non_cum):
        self.free_fields_calls.increment(cum, non_cum)

    def update_create_logical_region(self, cum, non_cum):
        self.create_logical_region_calls.increment(cum, non_cum)

    def update_destroy_logical_region(self, cum, non_cum):
        self.destroy_logical_region_calls.increment(cum, non_cum)

    def update_destroy_logical_partition(self, cum, non_cum):
        self.destroy_logical_partition_calls.increment(cum, non_cum)

    def update_get_logical_partition(self, cum, non_cum):
        self.get_logical_partition_calls.increment(cum, non_cum)

    def update_get_logical_subregion(self, cum, non_cum):
        self.get_logical_subregion_calls.increment(cum, non_cum)

    def update_inline_mapping(self, cum, non_cum):
        self.inline_mapping_calls.increment(cum, non_cum)

    def update_inline_unmap(self, cum, non_cum):
        self.inline_unmap_calls.increment(cum, non_cum)

    def update_task_dependence_analysis(self, cum, non_cum):
        self.task_dependence_calls.increment(cum, non_cum)

    def update_inline_mapping_dependence_analysis(self, cum, non_cum):
        self.inline_map_dep_calls.increment(cum, non_cum)

    def update_deletion_dependence_analysis(self, cum, non_cum):
        self.deletion_dep_calls.increment(cum, non_cum)

    def cummulative_time(self):
        time = 0
        time = time + self.mapping_calls.cum_time
        time = time + self.run_calls.cum_time
        time = time + self.children_map_calls.cum_time
        time = time + self.finish_calls.cum_time
        time = time + self.create_index_space_calls.cum_time
        time = time + self.destroy_index_space_calls.cum_time
        time = time + self.create_index_partition_calls.cum_time
        time = time + self.destroy_index_partition_calls.cum_time
        time = time + self.get_index_partition_calls.cum_time
        time = time + self.get_index_subspace_calls.cum_time
        time = time + self.get_index_domain_calls.cum_time
        time = time + self.get_index_color_space_calls.cum_time
        time = time + self.safe_cast_calls.cum_time
        time = time + self.create_field_space_calls.cum_time
        time = time + self.destroy_field_space_calls.cum_time
        time = time + self.allocate_fields_calls.cum_time
        time = time + self.free_fields_calls.cum_time
        time = time + self.create_logical_region_calls.cum_time
        time = time + self.destroy_logical_region_calls.cum_time
        time = time + self.destroy_logical_partition_calls.cum_time
        time = time + self.get_logical_partition_calls.cum_time
        time = time + self.get_logical_subregion_calls.cum_time
        time = time + self.inline_mapping_calls.cum_time
        time = time + self.inline_unmap_calls.cum_time
        time = time + self.task_dependence_calls.cum_time
        time = time + self.inline_map_dep_calls.cum_time
        time = time + self.deletion_dep_calls.cum_time
        return time

    def non_cummulative_time(self):
        time = 0
        time = time + self.mapping_calls.non_cum_time
        time = time + self.run_calls.non_cum_time
        time = time + self.children_map_calls.non_cum_time
        time = time + self.finish_calls.non_cum_time
        time = time + self.create_index_space_calls.non_cum_time
        time = time + self.destroy_index_space_calls.non_cum_time
        time = time + self.create_index_partition_calls.non_cum_time
        time = time + self.destroy_index_partition_calls.non_cum_time
        time = time + self.get_index_partition_calls.non_cum_time
        time = time + self.get_index_subspace_calls.non_cum_time
        time = time + self.get_index_domain_calls.non_cum_time
        time = time + self.get_index_color_space_calls.non_cum_time
        time = time + self.safe_cast_calls.non_cum_time
        time = time + self.create_field_space_calls.non_cum_time
        time = time + self.destroy_field_space_calls.non_cum_time
        time = time + self.allocate_fields_calls.non_cum_time
        time = time + self.free_fields_calls.non_cum_time
        time = time + self.create_logical_region_calls.non_cum_time
        time = time + self.destroy_logical_region_calls.non_cum_time
        time = time + self.destroy_logical_partition_calls.non_cum_time
        time = time + self.get_logical_partition_calls.non_cum_time
        time = time + self.get_logical_subregion_calls.non_cum_time
        time = time + self.inline_mapping_calls.non_cum_time
        time = time + self.inline_unmap_calls.non_cum_time
        time = time + self.task_dependence_calls.non_cum_time
        time = time + self.inline_map_dep_calls.non_cum_time
        time = time + self.deletion_dep_calls.non_cum_time
        return time

    def print_stats(self, total_time, cummulative, verbose):
        title_str = self.var.get_title() 
        to_add = 50 - len(title_str)
        if to_add > 0:
            for idx in range(to_add):
                title_str = title_str+' '
        cum_time = self.cummulative_time()
        non_cum_time = self.non_cummulative_time()
        if cummulative:
            cum_per = 100.0*(float(cum_time)/float(total_time))
            title_str = title_str+("%d us (%.3f%%)" % (cum_time,cum_per))
        else:
            non_cum_per = 100.0*(float(non_cum_time)/float(total_time))
            title_str = title_str+("%d us (%.3f%%)" % (non_cum_time,non_cum_per))
        print "    "+title_str
        if not verbose:
            # Not verbose, print out the application and meta timings    
            app_cum_time = self.run_calls.cum_time
            app_non_cum_time = self.run_calls.non_cum_time
            meta_cum_time = cum_time - app_cum_time
            meta_non_cum_time = non_cum_time - app_non_cum_time
            print "          Executions (APP):"
            self.run_calls.print_stats(total_time)
            print "          Meta Execution Time (META):"
            print "                Cummulative Time: %d us (%.3f%%)" % (meta_cum_time,100.0*float(meta_cum_time)/float(total_time))
            print "                Non-Cummulative Time: %d us (%.3f%%)" % (meta_non_cum_time,100.0*float(meta_non_cum_time)/float(total_time))
        else:
            self.emit_call_stat(self.run_calls,"Executions (APP):",total_time)
            self.emit_call_stat(self.mapping_calls,"Mapping Calls (META):",total_time)
            self.emit_call_stat(self.children_map_calls,"Children Mapped Calls (META):",total_time)
            self.emit_call_stat(self.finish_calls,"Finish Task Calls (META):",total_time)
            self.emit_call_stat(self.create_index_space_calls,"Create Index Space Calls (META):",total_time)
            self.emit_call_stat(self.destroy_index_space_calls,"Destroy Index Space Calls (META):",total_time)
            self.emit_call_stat(self.create_index_partition_calls,"Create Index Partition Calls (META):",total_time)
            self.emit_call_stat(self.destroy_index_partition_calls,"Destroy Index Partition Calls (META):",total_time)
            self.emit_call_stat(self.get_index_partition_calls,"Get Index Partition Calls (META):",total_time)
            self.emit_call_stat(self.get_index_subspace_calls,"Get Index Subspace Calls (META):",total_time)
            self.emit_call_stat(self.get_index_domain_calls,"Get Index Domain Calls (META):",total_time)
            self.emit_call_stat(self.get_index_color_space_calls,"Get Index Color Space Calls (META):",total_time)
            self.emit_call_stat(self.safe_cast_calls,"Safe Cast Calls (META):",total_time)
            self.emit_call_stat(self.create_field_space_calls,"Create Field Space Calls (META):",total_time)
            self.emit_call_stat(self.destroy_field_space_calls,"Destroy Field Space Calls (META):",total_time)
            self.emit_call_stat(self.allocate_fields_calls,"Allocate Field Calls (META):",total_time)
            self.emit_call_stat(self.free_fields_calls,"Free Field Calls (META):",total_time)
            self.emit_call_stat(self.create_logical_region_calls,"Create Logical Region Calls (META):",total_time)
            self.emit_call_stat(self.destroy_logical_region_calls,"Destroy Logical Region Calls (META):",total_time)
            self.emit_call_stat(self.destroy_logical_partition_calls,"Destroy Logical Partition Calls (META):",total_time)
            self.emit_call_stat(self.get_logical_partition_calls,"Get Logical Partition Calls (META):",total_time)
            self.emit_call_stat(self.get_logical_subregion_calls,"Get Logical Subreigon Calls (META):",total_time)
            self.emit_call_stat(self.inline_mapping_calls,"Inline Mappings (META):",total_time)
            self.emit_call_stat(self.inline_unmap_calls,"Inline Unmaps (META):",total_time)
            self.emit_call_stat(self.task_dependence_calls,"Task Dependence Analyses (META):",total_time)
            self.emit_call_stat(self.inline_map_dep_calls,"Inline Mapping Dependence Analyses (META):",total_time)
            self.emit_call_stat(self.deletion_dep_calls,"Deletion Dependence Analyses (META):",total_time)

    def emit_call_stat(self, calls, string, total_time):
        if not calls.is_empty():
            print "         "+string 
            calls.print_stats(total_time)

class StatGatherer(object):
    def __init__(self):
        self.variants = dict()
        self.scheduler = CallTracker()
        self.executing_task = list()

    def initialize_variant(self, var):
        assert var not in self.variants
        self.variants[var] = StatVariant(var)

    def update_task_map(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_task_map(cum, non_cum)

    def update_task_run(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_task_run(cum, non_cum)

    def start_task(self, var):
        assert var in self.variants
        self.executing_task.append(var)

    def stop_task(self, var):
        assert len(self.executing_task) > 0
        last = self.executing_task.pop()
        assert last == var

    def update_task_children_map(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_task_children_map(cum, non_cum)

    def update_task_finish(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_task_finish(cum, non_cum)

    def update_create_index_space(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_create_index_space(cum, non_cum)

    def update_destroy_index_space(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_destroy_index_space(cum, non_cum)

    def update_create_index_partition(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_create_index_partition(cum, non_cum)

    def update_destroy_index_partition(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_destroy_index_partition(cum, non_cum)

    def update_get_index_partition(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_get_index_partition(cum, non_cum)

    def update_get_index_subspace(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_get_index_subspace(cum, non_cum)

    def update_get_index_domain(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_get_index_domain(cum, non_cum)

    def update_get_index_color_space(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_get_index_color_space(cum, non_cum)

    def update_safe_cast(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_safe_cast(cum, non_cum)

    def update_create_field_space(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_create_field_space(cum, non_cum)

    def update_destroy_field_space(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_destroy_field_space(cum, non_cum)

    def update_allocate_fields(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_allocate_fields(cum, non_cum)

    def update_free_fields(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_free_fields(cum, non_cum)

    def update_create_logical_region(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_create_logical_region(cum, non_cum)

    def update_destroy_logical_region(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_destroy_logical_region(cum, non_cum)

    def update_destroy_logical_partition(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_destroy_logical_partition(cum, non_cum)

    def update_get_logical_partition(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_get_logical_partition(cum, non_cum)

    def update_get_logical_subregion(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_get_logical_subregion(cum, non_cum)

    def update_inline_mapping(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_inline_mapping(cum, non_cum)

    def update_inline_unmap(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_inline_unmap(cum, non_cum)

    def update_task_dependence_analysis(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_task_dependence_analysis(cum, non_cum)

    def update_inline_mapping_dependence_analysis(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_inline_mapping_dependence_analysis(cum, non_cum)

    def update_deletion_dependence_analysis(self, var, cum, non_cum):
        assert var in self.variants
        self.variants[var].update_deletion_dependence_analysis(cum, non_cum)

    def update_scheduler(self, cum, non_cum):
        self.scheduler.increment(cum, non_cum)

    def print_stats(self, total_time, cummulative, verbose):
        print "  -------------------------"
        print "  Task Statistics"
        print "  -------------------------"
        # Sort the tasks based on either their cummulative
        # or non-cummulative time
        task_list = list()
        for v,var in self.variants.iteritems():
            task_list.append(var)
        if cummulative:
            task_list.sort(key=lambda t: t.cummulative_time())
        else:
            task_list.sort(key=lambda t: t.non_cummulative_time())
        task_list.reverse()
        for t in task_list:
            t.print_stats(total_time, cummulative, verbose)
        print "  -------------------------"
        print "  Meta-Task Statistics"
        print "  -------------------------"
        if not self.scheduler.is_empty():
            print "  Scheduler (META):"
            self.scheduler.print_stats(total_time)
        

class State(object):
    def __init__(self):
        self.events = list()
        self.processors = dict()
        self.memories = dict()
        self.task_variants = dict()
        self.instances = dict()
        self.points = list()
        self.last_time = None

    def create_variant(self, tid, leaf, name):
        if tid not in self.task_variants:
            self.task_variants[tid] = TaskVariant(tid, leaf, name)
        else:
            assert self.task_variants[tid].leaf == leaf
            assert self.task_variants[tid].name == name

    def create_processor(self, proc, utility, kind):
        assert proc not in self.processors
        self.processors[proc] = Processor(proc, utility, kind)

    def create_task_event(self, proc, kind, tid, uid, time, dim, p0, p1, p2):
        if proc not in self.processors:
            return False
        if tid not in self.task_variants:
            return False
        event = Event(time, self.processors[proc])
        self.events.append(event)
        point = self.find_point(dim, p0, p1, p2)
        self.task_variants[tid].add_task_event(self.processors[proc], event, kind, uid, point)
        return True

    def create_scheduler_event(self, proc, kind, time):
        if proc not in self.processors:
            return False
        event = Event(time, self.processors[proc])
        self.events.append(event)
        self.processors[proc].add_scheduling_event(kind, event)
        return True

    def add_memory(self, mem, kind):
        assert mem not in self.memories
        self.memories[mem] = Memory(mem, kind)

    def add_instance_creation(self, iid, uid, mem, redop, factor, time):
        if mem not in self.memories:
            return False
        assert uid not in self.instances
        inst = Instance(iid, uid, self.memories[mem], redop, factor, time)
        self.instances[uid] = inst
        self.memories[mem].add_instance(inst)
        return True

    def add_instance_destroy(self, uid, time):
        if uid not in self.instances:
            return False
        self.instances[uid].set_destroy(time)
        return True

    def add_instance_field(self, uid, fid, size):
        if uid not in self.instances:
            return False
        self.instances[uid].add_field(fid, size)
        return True

    def set_subtask(self, suid, tid, uid, dim, p0, p1, p2):
        if tid not in self.task_variants:
            return False
        # See if we can find the UniqueTask
        sub_task = None
        for v,var in self.task_variants.iteritems():
            sub_task = var.has_unique_task(suid)
            if sub_task <> None:
                break
        if sub_task == None:
            return False
        point = self.find_point(dim, p0, p1, p2)
        self.task_variants[tid].add_subtask(uid, point, sub_task)
        return True

    def find_point(self, dim, p0, p1, p2):
        for p in self.points:
            if p.matches(dim, p0, p1, p2):
                return p
        # otherwise the point didn't exist, so make it
        point = Point(dim, p0, p1, p2)
        self.points.append(point)
        return point

    def build_time_ranges(self):
        assert self.last_time <> None # should have been set by now
        for p,proc in self.processors.iteritems():
            proc.init_time_range(self.last_time)
        # Now have each of the objects add their time ranges
        for v,var in self.task_variants.iteritems():
            var.add_time_ranges()
        for p,proc in self.processors.iteritems():
            proc.add_time_ranges()
        # Now that we have all the time ranges added, sort themselves
        for p,proc in self.processors.iteritems():
            proc.sort_time_range()
        for m,mem in self.memories.iteritems():
            mem.sort_time_range()

    def print_processor_stats(self):
        print "****************************************************"
        print "   PROCESSOR STATS"
        print "****************************************************"
        for p,proc in self.processors.iteritems():
            proc.print_stats()
        print ""

    def print_task_stats(self, cummulative, verbose):
        print "****************************************************"
        print "   TASK STATS"
        print "****************************************************"
        stat = StatGatherer()  
        for v,var, in self.task_variants.iteritems():
            stat.initialize_variant(var)
        for p,proc in self.processors.iteritems():
            proc.update_task_stats(stat)
        # Total time is the overall execution time multiplied by the number of processors
        total_time = self.last_time * len(self.processors)
        stat.print_stats(total_time, cummulative, verbose)
        print ""

    def generate_svg_picture(self, file_name, html_file):
        # Before doing this, generate all the colors
        num_variants = len(self.task_variants)
        idx = 0
        for v,var in self.task_variants.iteritems():
            var.compute_color(idx, num_variants)
            idx = idx + 1
        printer = SVGPrinter(file_name, html_file)
        for p,proc in sorted(self.processors.iteritems(),key=lambda x: x[0]):
            proc.emit_svg(printer)
        printer.close()

    def generate_mem_picture(self, file_name, html_file):
        # Before doing this, generate all the colors
        num_instances = len(self.instances)
        idx = 0
        for i,inst in self.instances.iteritems():
            inst.compute_color(idx, num_instances)
            idx = idx + 1
        printer = SVGPrinter(file_name, html_file)
        for m,mem in sorted(self.memories.iteritems(),key=lambda x: x[0]):
            mem.emit_svg(printer, self.last_time)
        printer.close()


def parse_log_file(file_name, state):
    log = open(file_name,'r')
    matches = 0
    # Also find the time for the last event
    last_time = long(0)
    # Handle cases for out of order printing
    replay_lines = list()
    for line in log:
        matches = matches + 1
        m = variant_pat.match(line)
        if m <> None:
            state.create_variant(int(m.group('tid')),True if (int(m.group('leaf'))) == 1 else False, m.group('name'))
            continue
        m = processor_pat.match(line)
        if m <> None:
            state.create_processor(int(m.group('proc'),16),True if (int(m.group('utility'))) == 1 else False,int(m.group('kind')))
            continue
        m = task_event_pat.match(line)
        if m <> None:
            time = long(m.group('time'))
            if not state.create_task_event(int(m.group('proc'),16), int(m.group('kind')), int(m.group('tid')), int(m.group('uid')),
                                              time, int(m.group('dim')), int(m.group('p0')), int(m.group('p1')), int(m.group('p2'))):  
                replay_lines.append(line)
            if time > last_time:
                last_time = time
            continue
        m = scheduler_pat.match(line)
        if m <> None:
            time = long(m.group('time'))
            if not state.create_scheduler_event(int(m.group('proc'),16), int(m.group('kind')), time):
                replay_lines.append(line)
            if time > last_time:
                last_time = time
            continue
        m = memory_pat.match(line)
        if m <> None:
            state.add_memory(int(m.group('mem'),16), int(m.group('kind')))
            continue
        m = create_pat.match(line)
        if m <> None:
            time = long(m.group('time'))
            if not state.add_instance_creation(int(m.group('iid'),16),int(m.group('uid')),
                                        int(m.group('mem'),16),int(m.group('redop')),int(m.group('factor')),time):
                replay_lines.append(line)
            if time > last_time:
                last_time = time
            continue
        m = destroy_pat.match(line)
        if m <> None:
            time = long(m.group('time'))
            if not state.add_instance_destroy(int(m.group('uid')), time):
                replay_lines.append(line)
            if time > last_time:
                last_time = time
            continue
        m = field_pat.match(line)
        if m <> None:
            if not state.add_instance_field(int(m.group('uid')), int(m.group('fid')), int(m.group('size'))):
                replay_lines.append(line)
            continue
        m = sub_task_pat.match(line)
        if m <> None:
            if not state.set_subtask(int(m.group('suid')), int(m.group('tid')), int(m.group('uid')), int(m.group('dim')),
                                      int(m.group('p0')), int(m.group('p1')), int(m.group('p2'))):
                replay_lines.append(line)
            continue
        #If we made it here, then we failed to match
        matches = matches - 1
    log.close()
    while len(replay_lines) > 0:
        to_delete = set()
        for line in replay_lines:
            m = task_event_pat.match(line)
            if m <> None:
                if state.create_task_event(int(m.group('proc'),16), int(m.group('kind')), int(m.group('tid')), int(m.group('uid')),
                                           time, int(m.group('dim')), int(m.group('p0')), int(m.group('p1')), int(m.group('p2'))):  
                    to_delete.add(line)
                continue
            m = scheduler_pat.match(line)
            if m <> None:
                if state.create_scheduler_event(int(m.group('proc'),16), int(m.group('kind')), time):
                    to_delete.add(line)   
                continue   
            m = create_pat.match(line)
            if m <> None:
                time = long(m.group('time'))
                if state.add_instance_creation(int(m.group('iid'),16),int(m.group('uid')),int(m.group('mem'),16),
                                                    int(m.group('redop')),int(m.group('factor')),time):
                    to_delete.add(line)
                continue
            m = destroy_pat.match(line)
            if m <> None:
                time = long(m.group('time'))
                if state.add_instance_destroy(int(m.group('uid')), time):
                    to_delete.add(line)
                continue
            m = field_pat.match(line)
            if m <> None:
                if state.add_instance_field(int(m.group('uid')), int(m.group('fid')), int(m.group('size'))):
                    to_delete.add(line)
                continue
            m = sub_task_pat.match(line)
            if m <> None:
                if state.set_subtask(int(m.group('suid')), int(m.group('tid')), int(m.group('uid')), int(m.group('dim')),
                                      int(m.group('p0')), int(m.group('p1')), int(m.group('p2'))):
                    to_delete.add(line)
                continue;
        if len(to_delete) == 0:
            print "ERROR: NO FORWARD PROGRESS ON REPLAY LINES!  BAD LEGION PROF ASSUMPTION!"
            assert False
        for line in to_delete:
            replay_lines.remove(line)
    state.last_time = last_time
    return matches

def usage():
    print "Usage: "+sys.argv[0]+" [-c] [-p] [-v] <file_name>"
    print "  -c : perform cummulative analysis"
    print "  -p : generate HTML and SVG files for pictures"
    print "  -v : print verbose profiling information"
    print "  -m <ppm> : set the pixels per micro-second for images"
    sys.exit(1)

def main():
    opts, args = getopt(sys.argv[1:],'cpvm:')
    opts = dict(opts)
    if len(args) <> 1:
        usage()
    file_name = args[0]
    cummulative = False
    generate_pictures = False
    verbose = False
    if '-c' in opts:
        cummulative = True
    if '-p' in opts:
        generate_pictures = True
    if '-v' in opts:
        verbose = True
    if '-m' in opts:
        global US_PER_PIXEL
        US_PER_PIXEL = int(opts['-m'])
    svg_file_name = "legion_prof.svg"
    html_file_name = "legion_prof.html"
    mem_file_name = "legion_prof_mem.svg"
    html_mem_file_name = "legion_prof_mem.html"

    print 'Loading log file '+file_name+'...'
    state = State()
    total_matches = parse_log_file(file_name, state)
    print 'Matched '+str(total_matches)+' lines'
    if total_matches == 0:
        print 'No matches. Exiting...'
        return
    # Now have the state build the time ranges for each processor
    state.build_time_ranges()

    # Print the per-processor statistics
    state.print_processor_stats()

    # Print the per-task statistics
    state.print_task_stats(cummulative, verbose)

    # Generate the svg profiling picture
    if generate_pictures:
        print "Generating SVG execution profile in "+svg_file_name+"..."
        state.generate_svg_picture(svg_file_name,html_file_name)
        print "Done!"
        print "Generating SVG memory profile in "+mem_file_name+"..."
        state.generate_mem_picture(mem_file_name,html_mem_file_name)
        print "Done!"

if __name__ == "__main__":
    main()

