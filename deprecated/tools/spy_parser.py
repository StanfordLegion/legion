#!/usr/bin/env python3

# Copyright 2023 Stanford University
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

#from spy_state import *
from spy_analysis import *
import sys, re

# All of these calls are based on the print statements in legion_logging.h

prefix    = "\[(?P<node>[0-9]+) - (?P<thread>[0-9a-f]+)\] \{\w+\}\{legion_spy\}: "
prefix_pat              = re.compile(prefix)
# Logger calls for the shape of the machine
utility_pat             = re.compile(prefix+"Utility (?P<pid>[0-9a-f]+)")
processor_pat           = re.compile(prefix+"Processor (?P<pid>[0-9a-f]+) (?P<kind>[0-9]+)")
memory_pat              = re.compile(prefix+"Memory (?P<mid>[0-9a-f]+) (?P<capacity>[0-9]+) (?P<kind>[0-9]+)")
proc_mem_pat            = re.compile(prefix+"Processor Memory (?P<pid>[0-9a-f]+) (?P<mid>[0-9a-f]+) (?P<band>[0-9]+) (?P<lat>[0-9]+)")
mem_mem_pat             = re.compile(prefix+"Memory Memory (?P<mone>[0-9a-f]+) (?P<mtwo>[0-9a-f]+) (?P<band>[0-9]+) (?P<lat>[0-9]+)")

# Calls for the shape of region trees
top_index_pat           = re.compile(prefix+"Index Space (?P<uid>[0-9a-f]+)")
top_index_name_pat      = re.compile(prefix+"Index Space Name (?P<uid>[0-9a-f]+) (?P<name>[-$()\w. ]+)")
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

# Logger calls for operations
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

# Logger calls for phase barriers
phase_barrier_pat       = re.compile(prefix+"Phase Barrier (?P<uid>[0-9a-f]+)")

# Logger calls for logical mapping dependence analysis
requirement_pat         = re.compile(prefix+"Logical Requirement (?P<uid>[0-9]+) (?P<index>[0-9]+) (?P<is_reg>[0-1]) (?P<ispace>[0-9a-f]+) (?P<fspace>[0-9]+) (?P<tid>[0-9]+) (?P<priv>[0-9]+) (?P<coher>[0-9]+) (?P<redop>[0-9]+)")
req_field_pat           = re.compile(prefix+"Logical Requirement Field (?P<uid>[0-9]+) (?P<index>[0-9]+) (?P<fid>[0-9]+)")
mapping_dep_pat         = re.compile(prefix+"Mapping Dependence (?P<ctx>[0-9]+) (?P<prev_id>[0-9]+) (?P<pidx>[0-9]+) (?P<next_id>[0-9]+) (?P<nidx>[0-9]+) (?P<dtype>[0-9]+)")

# Logger calls for dynamic independence analysis
independent_ispace_pat  = re.compile(prefix+"Index Space Independence (?P<pid>[0-9a-f]+) (?P<uid1>[0-9a-f]+) (?P<uid2>[0-9a-f]+)")
independent_ipart_pat   = re.compile(prefix+"Index Partition Independence (?P<pid>[0-9a-f]+) (?P<uid1>[0-9a-f]+) (?P<uid2>[0-9a-f]+)")

# Logger calls for physical dependence analysis
task_inst_req_pat       = re.compile(prefix+"Task Instance Requirement (?P<uid>[0-9]+) (?P<idx>[0-9]+) (?P<index>[0-9]+)")

# Logger calls for events
event_event_pat         = re.compile(prefix+"Event Event (?P<idone>[0-9a-f]+) (?P<genone>[0-9]+) (?P<idtwo>[0-9a-f]+) (?P<gentwo>[0-9]+)")
implicit_event_pat      = re.compile(prefix+"Implicit Event (?P<idone>[0-9a-f]+) (?P<genone>[0-9]+) (?P<idtwo>[0-9a-f]+) (?P<gentwo>[0-9]+)")
op_event_pat            = re.compile(prefix+"Op Events (?P<uid>[0-9]+) (?P<startid>[0-9a-f]+) (?P<startgen>[0-9]+) (?P<termid>[0-9a-f]+) (?P<termgen>[0-9]+)")
copy_event_pat          = re.compile(prefix+"Copy Events (?P<startid>[0-9a-f]+) (?P<startgen>[0-9]+) (?P<termid>[0-9a-f]+) (?P<termgen>[0-9]+)")
copy_req_pat            = re.compile(prefix+"Copy Requirement (?P<startid>[0-9a-f]+) (?P<startgen>[0-9]+) (?P<termid>[0-9a-f]+) (?P<termgen>[0-9]+) (?P<index>[0-9]+) (?P<is_reg>[0-1]) (?P<ispace>[0-9a-f]+) (?P<fspace>[0-9]+) (?P<tid>[0-9]+) (?P<priv>[0-9]+) (?P<coher>[0-9]+) (?P<redop>[0-9]+)")
copy_field_pat          = re.compile(prefix+"Copy Field (?P<startid>[0-9a-f]+) (?P<startgen>[0-9]+) (?P<termid>[0-9a-f]+) (?P<termgen>[0-9]+) (?P<index>[0-9]+) (?P<fid>[0-9]+)")
copy_inst_pat           = re.compile(prefix+"Copy Instance (?P<startid>[0-9a-f]+) (?P<startgen>[0-9]+) (?P<termid>[0-9a-f]+) (?P<termgen>[0-9]+) (?P<index>[0-9]+) (?P<iid>[0-9a-f]+)")
copy_ctx_pat            = re.compile(prefix+"Copy Context (?P<startid>[0-9a-f]+) (?P<startgen>[0-9]+) (?P<termid>[0-9a-f]+) (?P<termgen>[0-9]+) (?P<ctx>[0-9]+)")

# Logger calls for physical instance usage 
physical_inst_pat       = re.compile(prefix+"Physical Instance (?P<iid>[0-9a-f]+) (?P<mid>[0-9a-f]+) (?P<index>[0-9a-f]+) (?P<field>[0-9]+) (?P<tid>[0-9]+) (?P<blocking>[0-9]+)")
physical_reduc_pat      = re.compile(prefix+"Reduction Instance (?P<iid>[0-9a-f]+) (?P<mid>[0-9a-f]+) (?P<index>[0-9a-f]+) (?P<field>[0-9]+) (?P<tid>[0-9]+) (?P<fold>[0-1]) (?P<indirect>[0-9]+)")
inst_field_pat          = re.compile(prefix+"Instance Field (?P<iid>[0-9a-f]+) (?P<fid>[0-9]+)")
op_user_pat             = re.compile(prefix+"Op Instance User (?P<uid>[0-9]+) (?P<idx>[0-9]+) (?P<iid>[0-9a-f]+)")
op_field_user_pat       = re.compile(prefix+"Op Instance Field User (?P<uid>[0-9]+) (?P<idx>[0-9]+) (?P<iid>[0-9a-f]+) (?P<fid>[0-9]+)")
op_proc_user_pat        = re.compile(prefix+"Op Processor User (?P<uid>[0-9]+) (?P<pid>[0-9a-f]+)")

def parse_log_line(line, state):
    # Machine shapes
    m = utility_pat.match(line)
    if m <> None:
        if state.add_utility(int(m.group('pid'),16)):
            return True
    m = processor_pat.match(line)
    if m <> None:
        if state.add_processor(int(m.group('pid'),16), int(m.group('kind'))):
            return True
    m = memory_pat.match(line)
    if m <> None:
        if state.add_memory(int(m.group('mid'),16), int(m.group('capacity')), int(m.group('kind'))):
            return True
    m = proc_mem_pat.match(line)
    if m <> None:
        if state.set_proc_mem(int(m.group('pid'),16), int(m.group('mid'),16), int(m.group('band')), int(m.group('lat'))):
            return True
    m = mem_mem_pat.match(line)
    if m <> None:
        if state.set_mem_mem(int(m.group('mone'),16), int(m.group('mtwo'),16), int(m.group('band')), int(m.group('lat'))):
            return True
    # Region tree shapes
    m = top_index_pat.match(line)
    if m <> None:
        if state.add_index_space(int(m.group('uid'),16)):
            return True
    m = top_index_name_pat.match(line)
    if m <> None:
        if state.add_index_space_name(int(m.group('uid'),16), m.group('name')):
            return True
    m = index_part_pat.match(line)
    if m <> None:
        if state.add_index_partition(int(m.group('pid'),16), int(m.group('uid'),16), True if (int(m.group('disjoint'))) == 1 else False, int(m.group('dim')), int(m.group('val1')), int(m.group('val2')), int(m.group('val3'))):
            return True
    m = index_part_name_pat.match(line)
    if m <> None:
        if state.add_index_partition_name(int(m.group('uid'),16), m.group('name')):
            return True
    m = index_subspace_pat.match(line)
    if m <> None:
        if state.add_index_subspace(int(m.group('pid'),16), int(m.group('uid'),16), int(m.group('dim')), int(m.group('val1')), int(m.group('val2')), int(m.group('val3'))):
            return True
    m = field_space_pat.match(line)
    if m <> None:
        if state.add_field_space(int(m.group('uid'))):
            return True
    m = field_space_name_pat.match(line)
    if m <> None:
        if state.add_field_space_name(int(m.group('uid')), m.group('name')):
            return True
    m = field_create_pat.match(line)
    if m <> None:
        if state.add_field(int(m.group('uid')), int(m.group('fid'))):
            return True
    m = field_name_pat.match(line)
    if m <> None:
        if state.add_field_name(int(m.group('uid')), int(m.group('fid')), m.group('name')):
            return True
    m = region_pat.match(line)
    if m <> None:
        if state.add_region(int(m.group('iid'),16), int(m.group('fid')), int(m.group('tid'))):
            return True 
    m = region_name_pat.match(line)
    if m <> None:
        if state.add_region_name(int(m.group('iid'),16), int(m.group('fid')), int(m.group('tid')), m.group('name')):
            return True
    m = partition_name_pat.match(line)
    if m <> None:
        if state.add_partition_name(int(m.group('iid'),16), int(m.group('fid')), int(m.group('tid')), m.group('name')):
            return True
    # Operations
    m = top_task_pat.match(line)
    if m <> None:
        if state.add_top_task(int(m.group('tid')), int(m.group('uid')), m.group('name')):
            return True
    m = single_task_pat.match(line)
    if m <> None:
        if state.add_single_task(int(m.group('ctx')), int(m.group('tid')), int(m.group('uid')), m.group('name')):
            return True
    m = index_task_pat.match(line)
    if m <> None:
        if state.add_index_task(int(m.group('ctx')), int(m.group('tid')), int(m.group('uid')), m.group('name')):
            return True
    m = mapping_pat.match(line)
    if m <> None:
        if state.add_mapping(int(m.group('ctx')), int(m.group('uid'))):
            return True
    m = close_pat.match(line)
    if m <> None:
        if state.add_close(int(m.group('ctx')), int(m.group('uid')), True if int(m.group('is_inter')) == 1 else False):
            return True
    m = close_creator_pat.match(line)
    if m <> None:
        if state.set_close_creator(int(m.group('uid')), int(m.group('cuid')), int(m.group('idx'))):
            return True
    m = fence_pat.match(line)
    if m <> None:
        if state.add_fence(int(m.group('ctx')), int(m.group('uid'))):
            return True
    m = copy_op_pat.match(line)
    if m <> None:
        if state.add_copy_op(int(m.group('ctx')), int(m.group('uid'))):
            return True
    m = fill_op_pat.match(line)
    if m <> None:
        if state.add_fill_op(int(m.group('ctx')), int(m.group('uid'))):
            return True
    m = acquire_op_pat.match(line)
    if m <> None:
        if state.add_acquire_op(int(m.group('ctx')), int(m.group('uid'))):
            return True
    m = release_op_pat.match(line)
    if m <> None:
        if state.add_release_op(int(m.group('ctx')), int(m.group('uid'))):
            return True
    m = deletion_pat.match(line)
    if m <> None:
        if state.add_deletion(int(m.group('ctx')), int(m.group('uid'))):
            return True
    m = dep_partition_op_pat.match(line)
    if m <> None:
        if state.add_dependent_partition_op(int(m.group('ctx')), int(m.group('uid')), int(m.group('pid'), 16), int(m.group('kind'))):
            return True
    m = pending_partition_op_pat.match(line)
    if m <> None:
        if state.add_pending_partition_op(int(m.group('ctx')), int(m.group('uid'))):
            return True
    m = target_partition_pat.match(line)
    if m <> None:
        if state.set_pending_partition_target(int(m.group('uid')), int(m.group('pid'), 16), int(m.group('kind'))):
            return True
    m = index_slice_pat.match(line)
    if m <> None:
        if state.add_index_slice(int(m.group('index')),int(m.group('slice'))):
            return True
    m = slice_slice_pat.match(line)
    if m <> None:
        if state.add_slice_slice(int(m.group('slice1')),int(m.group('slice2'))):
            return True
    m = slice_point_pat.match(line)
    if m <> None:
        if state.add_slice_point(int(m.group('slice')),int(m.group('point')), int(m.group('dim')), int(m.group('val1')), int(m.group('val2')), int(m.group('val3'))):
            return True
    m = point_point_pat.match(line)
    if m <> None:
        if state.add_point_point(int(m.group('point1')),int(m.group('point2'))):
            return True
    # Phase Barriers
    m = phase_barrier_pat.match(line)
    if m <> None:
        if state.add_phase_barrier(int(m.group('uid'), 16)):
            return True
    # Mapping dependence analysis
    m = requirement_pat.match(line)
    if m <> None:
        if state.add_requirement(int(m.group('uid')), int(m.group('index')), True if (int(m.group('is_reg')))==1 else False, int(m.group('ispace'),16), int(m.group('fspace')), int(m.group('tid')), int(m.group('priv')), int(m.group('coher')), int(m.group('redop'))):
            return True
    m = req_field_pat.match(line)
    if m <> None:
        if state.add_req_field(int(m.group('uid')), int(m.group('index')), int(m.group('fid'))):
            return True
    m = mapping_dep_pat.match(line)
    if m <> None:
        if state.add_mapping_dependence(int(m.group('ctx')), int(m.group('prev_id')), int(m.group('pidx')), int(m.group('next_id')), int(m.group('nidx')), int(m.group('dtype'))):
            return True
    # Dynamic independence analysis
    m = independent_ispace_pat.match(line)
    if m <> None:
        if state.add_independent_index_spaces(int(m.group('pid'),16), int(m.group('uid1'),16), int(m.group('uid2'),16)):
            return True
    m = independent_ipart_pat.match(line)
    if m <> None:
        if state.add_independent_index_partitions(int(m.group('pid'),16), int(m.group('uid1'),16), int(m.group('uid2'),16)):
            return True
    # Physical dependence analysis
    m = task_inst_req_pat.match(line)
    if m <> None:
        if state.add_instance_requirement(int(m.group('uid')), int(m.group('idx')), int(m.group('index'))):
            return True
    # Physical Analysis
    m = event_event_pat.match(line)
    if m <> None:
        if state.add_event_dependence(int(m.group('idone'),16), int(m.group('genone')), int(m.group('idtwo'),16), int(m.group('gentwo'))):
            return True
    m = implicit_event_pat.match(line)
    if m <> None:
        if state.add_implicit_dependence(int(m.group('idone'),16), int(m.group('genone')), int(m.group('idtwo'),16), int(m.group('gentwo'))):
            return True
    m = op_event_pat.match(line)
    if m <> None:
        if state.add_op_events(int(m.group('uid')), int(m.group('startid'),16), int(m.group('startgen')), int(m.group('termid'),16), int(m.group('termgen'))):
            return True
    m = copy_event_pat.match(line)
    if m <> None:
        if state.add_copy_event(int(m.group('startid'),16), int(m.group('startgen')), int(m.group('termid'),16), int(m.group('termgen'))):
            return True
    m = copy_req_pat.match(line)
    if m <> None:
        if state.add_copy_req(int(m.group('startid'),16), int(m.group('startgen')), int(m.group('termid'),16), int(m.group('termgen')), int(m.group('index')), True if (int(m.group('is_reg')))==1 else False, int(m.group('ispace'),16), int(m.group('fspace')), int(m.group('tid')), int(m.group('priv')), int(m.group('coher')), int(m.group('redop'))):
            return True
    m = copy_field_pat.match(line)
    if m <> None:
        if state.add_copy_field(int(m.group('startid'),16), int(m.group('startgen')), int(m.group('termid'),16), int(m.group('termgen')), int(m.group('index')), int(m.group('fid'))):
            return True
    m = copy_inst_pat.match(line)
    if m <> None:
        if state.add_copy_instance(int(m.group('startid'),16), int(m.group('startgen')), int(m.group('termid'),16), int(m.group('termgen')), int(m.group('index')), int(m.group('iid'),16)):
            return True
    m = copy_ctx_pat.match(line)
    if m <> None:
        if state.add_copy_context(int(m.group('startid'),16), int(m.group('startgen')), int(m.group('termid'),16), int(m.group('termgen')), int(m.group('ctx'))):
            return True
    # Physical instance usage
    m = physical_inst_pat.match(line)
    if m <> None:
        if state.add_physical_instance(int(m.group('iid'),16), int(m.group('mid'),16), int(m.group('index'),16), int(m.group('field')), int(m.group('tid')), int(m.group('blocking'))):
            return True
    m = physical_reduc_pat.match(line)
    if m <> None:
        if state.add_reduction_instance(int(m.group('iid'),16), int(m.group('mid'),16), int(m.group('index'),16), int(m.group('field')), int(m.group('tid')), True if (int(m.group('fold')) == 1) else False, int(m.group('indirect'))):
            return True
    m = inst_field_pat.match(line)
    if m <> None:
        if state.add_instance_field(int(m.group('iid'),16), int(m.group('fid'))):
            return True
    m = op_user_pat.match(line)
    if m <> None:
        if state.add_op_user(int(m.group('uid')), int(m.group('idx')), int(m.group('iid'),16)):
            return True
    m = op_field_user_pat.match(line)
    if m <> None:
        if state.add_op_field_user(int(m.group('uid')), int(m.group('idx')), int(m.group('iid'),16), int(m.group('fid'))):
            return True
    m = op_proc_user_pat.match(line)
    if m <> None:
        if state.add_op_proc_user(int(m.group('uid')), int(m.group('pid'),16)):
            return True
    return False

def parse_log_file(file_name, state):
    log = open(file_name, 'r')
    matches = 0
    # Since some lines might match, but are out of order due to things getting
    # printed to the log file in weird orders, try reparsing lines
    replay_lines = list()
    for line in log:
        # Do a quick check to see if we match the prefix
        # If not then we can skip the line entirely
        m = prefix_pat.match(line)
        if m == None:
            continue
        # If we made it here then we failed to match
        if parse_log_line(line, state):
            matches += 1
        else:
            replay_lines.append(line)
    log.close()
    # Now see if we have lines that need to be replayed
    while len(replay_lines) > 0:
        to_delete = set()
        for line in replay_lines:
            if parse_log_line(line, state):
                to_delete.add(line)

        # Now check to make sure we actually did something
        # If not then we're not making forward progress which is bad
        if len(to_delete) == 0:
            print "ERROR: NO PROGRESS PARSING! BUG IN LEGION SPY LOGGING ASSUMPTIONS!"
            for line in replay_lines:
                print line
            assert False
        # Now delete any lines to delete and go again until we're done
        for line in to_delete:
            replay_lines.remove(line)
    return matches

# EOF

