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

from spy_state import *
import sys, re

# All of these calls are based on the print statements in legion_logging.h

prefix    = "\[(?P<node>[0-9]+) - (?P<thread>[0-9a-f]+)\] \{\w+\}\{legion_spy\}: "
# Logger calls for the shape of the machine
utility_pat             = re.compile(prefix+"Utility (?P<pid>[0-9a-f]+)")
processor_pat           = re.compile(prefix+"Processor (?P<pid>[0-9a-f]+) (?P<util>[0-9a-f]+) (?P<kind>[0-9]+)")
memory_pat              = re.compile(prefix+"Memory (?P<mid>[0-9a-f]+) (?P<capacity>[0-9]+)")
proc_mem_pat            = re.compile(prefix+"Processor Memory (?P<pid>[0-9a-f]+) (?P<mid>[0-9a-f]+) (?P<band>[0-9]+) (?P<lat>[0-9]+)")
mem_mem_pat             = re.compile(prefix+"Memory Memory (?P<mone>[0-9a-f]+) (?P<mtwo>[0-9a-f]+) (?P<band>[0-9]+) (?P<lat>[0-9]+)")

# Calls for the shape of region trees
top_index_pat           = re.compile(prefix+"Index Space (?P<uid>[0-9a-f]+)")
index_part_pat          = re.compile(prefix+"Index Partition (?P<pid>[0-9a-f]+) (?P<uid>[0-9]+) (?P<disjoint>[0-1]) (?P<color>[0-9]+)")
index_subspace_pat      = re.compile(prefix+"Index Subspace (?P<pid>[0-9]+) (?P<uid>[0-9a-f]+) (?P<color>[0-9]+)")
field_space_pat         = re.compile(prefix+"Field Space (?P<uid>[0-9]+)")
field_create_pat        = re.compile(prefix+"Field Creation (?P<uid>[0-9]+) (?P<fid>[0-9]+)")
region_pat              = re.compile(prefix+"Region (?P<iid>[0-9a-f]+) (?P<fid>[0-9]+) (?P<tid>[0-9]+)")

# Logger calls for operations
top_task_pat            = re.compile(prefix+"Top Task (?P<hid>[0-9a-f]+) (?P<gen>[0-9]+) (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<tid>[0-9]+)")
task_pat                = re.compile(prefix+"Task Operation (?P<uid>[0-9]+) (?P<tid>[0-9]+) (?P<pid>[0-9]+) (?P<ctx>[0-9]+) (?P<hid>[0-9a-f]+) (?P<gen>[0-9]+) (?P<index>[0-1])")
mapping_pat             = re.compile(prefix+"Mapping Operation (?P<uid>[0-9]+) (?P<pid>[0-9]+) (?P<ctx>[0-9]+) (?P<hid>[0-9a-f]+) (?P<gen>[0-9]+)")
deletion_pat            = re.compile(prefix+"Deletion Operation (?P<uid>[0-9]+) (?P<pid>[0-9]+) (?P<ctx>[0-9]+) (?P<hid>[0-9a-f]+) (?P<gen>[0-9]+)")
task_name_pat           = re.compile(prefix+"Task Name (?P<uid>[0-9]+) (?P<name>\w+)")

# Logger calls for logical mapping dependence analysis
requirement_pat         = re.compile(prefix+"Logical Requirement (?P<uid>[0-9]+) (?P<index>[0-9]+) (?P<is_reg>[0-1]) (?P<ispace>[0-9a-f]+) (?P<fspace>[0-9]+) (?P<tid>[0-9]+) (?P<priv>[0-9]+) (?P<coher>[0-9]+) (?P<redop>[0-9]+)")
req_field_pat           = re.compile(prefix+"Logical Requirement Field (?P<uid>[0-9]+) (?P<index>[0-9]+) (?P<fid>[0-9]+)")
mapping_dep_pat         = re.compile(prefix+"Mapping Dependence (?P<pid>[0-9]+) (?P<ctx>[0-9]+) (?P<hid>[0-9a-f]+) (?P<gen>[0-9]+) (?P<prev_id>[0-9]+) (?P<pidx>[0-9]+) (?P<next_id>[0-9]+) (?P<nidx>[0-9]+) (?P<dtype>[0-9]+)")

# Logger calls for physical dependence analysis
task_inst_req_pat       = re.compile(prefix+"Task Instance Requirement (?P<uid>[0-9]+) (?P<ctx>[0-9]+) (?P<gen>[0-9]+) (?P<hid>[0-9a-f]+) (?P<idx>[0-9]+) (?P<index>[0-9a-f]+)")

# Logger calls for events
event_event_pat         = re.compile(prefix+"Event Event (?P<idone>[0-9a-f]+) (?P<genone>[0-9]+) (?P<idtwo>[0-9a-f]+) (?P<gentwo>[0-9]+)")
task_event_pat          = re.compile(prefix+"Task Events (?P<hid>[0-9a-f]+) (?P<gen>[0-9]+) (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index_space>[0-1]) (?P<point>[0-9]+) (?P<startid>[0-9a-f]+) (?P<startgen>[0-9]+) (?P<termid>[0-9a-f]+) (?P<termgen>[0-9]+)")
index_term_pat          = re.compile(prefix+"Index Termination (?P<uid>[0-9]+) (?P<termid>[0-9a-f]+) (?P<termgen>[0-9]+)")
copy_event_pat          = re.compile(prefix+"Copy Events (?P<srcman>[0-9]+) (?P<dstman>[0-9]+) (?P<index>[0-9a-f]+) (?P<field>[0-9]+) (?P<tree>[0-9]+) (?P<startid>[0-9a-f]+) (?P<startgen>[0-9]+) (?P<termid>[0-9a-f]+) (?P<termgen>[0-9]+) (?P<mask>[0-9\,]+)")
reduce_event_pat        = re.compile(prefix+"Reduce Events (?P<srcman>[0-9]+) (?P<dstman>[0-9]+) (?P<index>[0-9a-f]+) (?P<field>[0-9]+) (?P<tree>[0-9]+) (?P<startid>[0-9a-f]+) (?P<startgen>[0-9]+) (?P<termid>[0-9a-f]+) (?P<termgen>[0-9]+) (?P<redop>[0-9]+) (?P<mask>[0-9\,]+)")
map_event_pat           = re.compile(prefix+"Map Events (?P<uid>[0-9]+) (?P<startid>[0-9a-f]+) (?P<startgen>[0-9]+) (?P<termid>[0-9a-f]+) (?P<termgen>[0-9]+)")
point_1_pat             = re.compile(prefix+"Point Value 1 (?P<point_id>[0-9]+) (?P<idx1>[0-9]+)")
point_2_pat             = re.compile(prefix+"Point Value 2 (?P<point_id>[0-9]+) (?P<idx1>[0-9]+) (?P<idx2>[0-9]+)")
point_3_pat             = re.compile(prefix+"Point Value 3 (?P<point_id>[0-9]+) (?P<idx1>[0-9]+) (?P<idx2>[0-9]+) (?P<idx3>[0-9]+)")
point_event_pat         = re.compile(prefix+"Point Task Events (?P<hid>[0-9a-f]+) (?P<gen>[0-9]+) (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index_space>[0-1]) (?P<point_id>[0-9]+) (?P<startid>[0-9a-f]+) (?P<startgen>[0-9]+) (?P<termid>[0-9a-f]+) (?P<termgen>[0-9]+)")

# Logger calls for physical instance usage 
physical_inst_pat       = re.compile(prefix+"Physical Instance (?P<iid>[0-9a-f]+) (?P<mid>[0-9a-f]+) (?P<index>[0-9a-f]+) (?P<field>[0-9]+) (?P<tid>[0-9]+)")
inst_manager_pat        = re.compile(prefix+"Instance Manager (?P<iid>[0-9a-f]+) (?P<manager>[0-9]+)")
physical_reduc_pat      = re.compile(prefix+"Reduction Instance (?P<iid>[0-9a-f]+) (?P<mid>[0-9a-f]+) (?P<index>[0-9a-f]+) (?P<field>[0-9]+) (?P<tid>[0-9]+) (?P<fold>[0-1]) (?P<indirect>[0-9]+)")
reduc_manager_pat       = re.compile(prefix+"Reduction Manager (?P<iid>[0-9a-f]+) (?P<manager>[0-9]+)")
task_user_pat           = re.compile(prefix+"Task Instance User (?P<uid>[0-9]+) (?P<ctx>[0-9]+) (?P<gen>[0-9]+) (?P<hid>[0-9a-f]+) (?P<idx>[0-9]+) (?P<manager>[0-9]+)")
mapping_user_pat            = re.compile(prefix+"Mapping Instance User (?P<uid>[0-9]+) (?P<manager>[0-9]+)")

# Logger calls for timing analysis
exec_info_pat           = re.compile(prefix+"Execution Information (?P<uid>[0-9]+) (?P<ctx>[0-9]+) (?P<gen>[0-9]+) (?P<hid>[0-9a-f]+) (?P<proc>[0-9]+)")
begin_info_pat          = re.compile(prefix+"Begin Task Timing (?P<uid>[0-9]+) (?P<ctx>[0-9]+) (?P<gen>[0-9]+) (?P<hid>[0-9a-f]+) (?P<start>[0-9]+)")
end_info_pat            = re.compile(prefix+"End Task Timing (?P<uid>[0-9]+) (?P<ctx>[0-9]+) (?P<gen>[0-9]+) (?P<hid>[0-9a-f]+) (?P<end>[0-9]+)")

def parse_log_file(file_name, state):
    log = open(file_name, 'r')
    matches = 0
    # Since some lines might match, but are out of order due to things getting
    # printed to the log file in weird orders, try reparsing lines
    replay_lines = list()
    for line in log:
        matches = matches + 1
        # Machine shapes
        m = utility_pat.match(line)
        if m <> None:
            state.add_utility(int(m.group('pid'),16))
            continue
        m = processor_pat.match(line)
        if m <> None:
            if not state.add_processor(int(m.group('pid'),16), int(m.group('util'),16), int(m.group('kind'))):
                replay_lines.append(line)
            continue
        m = memory_pat.match(line)
        if m <> None:
            state.add_memory(int(m.group('mid'),16), int(m.group('capacity')))
            continue
        m = proc_mem_pat.match(line)
        if m <> None:
            if not state.set_proc_mem(int(m.group('pid'),16), int(m.group('mid'),16), int(m.group('band')), int(m.group('lat'))):
                replay_lines.append(line)
            continue
        m = mem_mem_pat.match(line)
        if m <> None:
            if not state.set_mem_mem(int(m.group('mone'),16), int(m.group('mtwo'),16), int(m.group('band')), int(m.group('lat'))):
                replay_lines.append(line)
            continue
        # Region tree shapes
        m = top_index_pat.match(line)
        if m <> None:
            state.add_index_space(int(m.group('uid'),16))
            continue
        m = index_part_pat.match(line)
        if m <> None:
            if not state.add_index_partition(int(m.group('pid'),16), int(m.group('uid')), True if (int(m.group('disjoint'))) == 1 else False, int(m.group('color'))):
                replay_lines.append(line)
            continue
        m = index_subspace_pat.match(line)
        if m <> None:
            if not state.add_index_subspace(int(m.group('pid')), int(m.group('uid'),16), int(m.group('color'))):
                replay_lines.append(line)
            continue
        m = field_space_pat.match(line)
        if m <> None:
            state.add_field_space(int(m.group('uid')))
            continue
        m = field_create_pat.match(line)
        if m <> None:
            if not state.add_field(int(m.group('uid')), int(m.group('fid'))):
                replay_lines.append(line)
            continue
        m = region_pat.match(line)
        if m <> None:
            if not state.add_region(int(m.group('iid'),16), int(m.group('fid')), int(m.group('tid'))):
                replay_lines.append(line)
            continue 
        # Operations
        m = top_task_pat.match(line)
        if m <> None:
            state.add_top_task(int(m.group('hid'),16), int(m.group('gen')), int(m.group('uid')), int(m.group('ctx')), int(m.group('tid')))
            continue
        m = task_pat.match(line)
        if m <> None:
            if not state.add_task(int(m.group('uid')), int(m.group('tid')), int(m.group('pid')), int(m.group('ctx')), int(m.group('hid'),16), int(m.group('gen')), True if (int(m.group('index')) == 1) else False):
                replay_lines.append(line)
            continue
        m = mapping_pat.match(line)
        if m <> None:
            if not state.add_mapping(int(m.group('uid')), int(m.group('pid')), int(m.group('ctx')), int(m.group('hid'),16), int(m.group('gen'))):
                replay_lines.append(line)
            continue
        m = deletion_pat.match(line)
        if m <> None:
            if not state.add_deletion(int(m.group('uid')), int(m.group('pid')), int(m.group('ctx')), int(m.group('hid'),16), int(m.group('gen'))):
                replay_lines.append(line)
            continue
        m = task_name_pat.match(line)
        if m <> None:
            if not state.add_name(int(m.group('uid')), m.group('name')):
                replay_lines.append(line)
            continue
        # Mapping dependence analysis
        m = requirement_pat.match(line)
        if m <> None:
            if not state.add_requirement(int(m.group('uid')), int(m.group('index')), True if (int(m.group('is_reg')))==1 else False, int(m.group('ispace'),16), int(m.group('fspace')), int(m.group('tid')), int(m.group('priv')), int(m.group('coher')), int(m.group('redop'))):
                replay_lines.append(line)
            continue
        m = req_field_pat.match(line)
        if m <> None:
            if not state.add_req_field(int(m.group('uid')), int(m.group('index')), int(m.group('fid'))):
                replay_lines.append(line)
            continue
        m = mapping_dep_pat.match(line)
        if m <> None:
            if not state.add_mapping_dependence(int(m.group('pid')), int(m.group('ctx')), int(m.group('hid'),16), int(m.group('gen')),  int(m.group('prev_id')), int(m.group('pidx')), int(m.group('next_id')), int(m.group('nidx')), int(m.group('dtype'))):
                replay_lines.append(line)
            continue
        # Physical dependence analysis
        m = task_inst_req_pat.match(line)
        if m <> None:
            if not state.add_instance_requirement(int(m.group('uid')), int(m.group('ctx')), int(m.group('gen')), int(m.group('hid'),16), int(m.group('idx')), int(m.group('index'),16)):
                replay_lines.append(line)
            continue
        # Physical Analysis
        m = task_event_pat.match(line)
        if m <> None:
            point = Point(-1, 1)
            point.add_value(int(m.group('point')))
            if not state.add_task_instance(int(m.group('uid')), int(m.group('ctx')), int(m.group('hid'),16), int(m.group('gen')), True if (int(m.group('index_space')))==1 else False, point, int(m.group('startid'),16), int(m.group('startgen')), int(m.group('termid'),16), int(m.group('termgen'))):
                replay_lines.append(line)
            continue
        m = event_event_pat.match(line)
        if m <> None:
            state.add_event_dependence(int(m.group('idone'),16), int(m.group('genone')), int(m.group('idtwo'),16), int(m.group('gentwo')))
            continue
        m = index_term_pat.match(line)
        if m <> None:
            if not state.add_index_term(int(m.group('uid')), int(m.group('termid'),16), int(m.group('termgen'))):
                replay_lines.append(line)
            continue
        m = map_event_pat.match(line)
        if m <> None:
            if not state.add_map_instance(int(m.group('uid')), int(m.group('startid'),16), int(m.group('startgen')), int(m.group('termid'),16), int(m.group('termgen'))):
                replay_lines.append(line)
            continue
        m = point_1_pat.match(line)
        if m <> None:
            state.add_point(int(m.group('node')), int(m.group('point_id')), 1, int(m.group('idx1')))
            continue
        m = point_2_pat.match(line)
        if m <> None:
            state.add_point(int(m.group('node')), int(m.group('point_id')), 2, int(m.group('idx1')), int(m.group('idx2')))
            continue
        m = point_3_pat.match(line)
        if m <> None:
            state.add_point(int(m.group('node')), int(m.group('point_id')), 3, int(m.group('idx1')), int(m.group('idx2')), int(m.group('idx3')))
            continue
        m = point_event_pat.match(line)
        if m <> None:
            point = state.get_point(int(m.group('node')), int(m.group('point_id')))
            if not state.add_task_instance(int(m.group('uid')), int(m.group('ctx')), int(m.group('hid'),16), int(m.group('gen')), True if (int(m.group('index_space')))==1 else False, point, int(m.group('startid'),16), int(m.group('startgen')), int(m.group('termid'),16), int(m.group('termgen'))):
                replay_lines.append(line)
            continue
        m = copy_event_pat.match(line)
        if m <> None:
            if not state.add_copy_instance(int(m.group('srcman')), int(m.group('dstman')), int(m.group('index'),16), int(m.group('field')), int(m.group('tree')), int(m.group('startid'),16), int(m.group('startgen')), int(m.group('termid'),16), int(m.group('termgen')), m.group('mask')):
                replay_lines.append(line)
            continue
        m = reduce_event_pat.match(line)
        if m <> None:
            if not state.add_reduce_instance(int(m.group('srcman')), int(m.group('dstman')), int(m.group('index'),16), int(m.group('field')), int(m.group('tree')), int(m.group('startid'),16), int(m.group('startgen')), int(m.group('termid'),16), int(m.group('termgen')), int(m.group('redop')), m.group('mask')):
                replay_lines.append(line)
            continue
        # Physical instance usage
        m = physical_inst_pat.match(line)
        if m <> None:
            if not state.add_physical_instance(int(m.group('iid'),16), int(m.group('mid'),16), int(m.group('index'),16), int(m.group('field')), int(m.group('tid'))):
                replay_lines.append(line)
            continue
        m = inst_manager_pat.match(line)
        if m <> None:
            if not state.add_instance_manager(int(m.group('iid'),16), int(m.group('manager'))):
                replay_lines.append(line)
            continue
        m = physical_reduc_pat.match(line)
        if m <> None:
            if not state.add_reduction_instance(int(m.group('iid'),16), int(m.group('mid'),16), int(m.group('index'),16), int(m.group('field')), int(m.group('tid')), True if (int(m.group('fold')) == 1) else False, int(m.group('indirect'))):
                replay_lines.append(line)
            continue
        m = reduc_manager_pat.match(line)
        if m <> None:
            if not state.add_reduction_manager(int(m.group('iid'),16), int(m.group('manager'))):
                replay_lines.append(line)
            continue
        m = task_user_pat.match(line)
        if m <> None:
            if not state.add_task_user(int(m.group('uid')), int(m.group('ctx')), int(m.group('gen')), int(m.group('hid'),16), int(m.group('idx')), int (m.group('manager'))):
                replay_lines.append(line)
            continue
        m = mapping_user_pat.match(line)
        if m <> None:
            if not state.add_mapping_user(int(m.group('uid')), int(m.group('manager'))):
                replay_lines.append(line)
            continue
        # Timing Analysis
        m = exec_info_pat.match(line)
        if m <> None:
            if not state.set_exec_info(int(m.group('uid')), int(m.group('ctx')), int(m.group('gen')), int(m.group('hid'),16), int(m.group('proc'))):
                replay_lines.append(line)
            continue
        m = begin_info_pat.match(line)
        if m <> None:
            if not state.set_task_start(int(m.group('uid')), int(m.group('ctx')), int(m.group('gen')), int(m.group('hid'),16), long(m.group('start'))):
                replay_lines.append(line)
            continue
        m = end_info_pat.match(line)
        if m <> None:
            if not state.set_task_end(int(m.group('uid')), int(m.group('ctx')), int(m.group('gen')), int(m.group('hid'),16), long(m.group('end'))):
                replay_lines.append(line)
            continue

        # If we made it here then we failed to match
        matches = matches - 1
    log.close()
    # Now see if we have lines that need to be replayed
    while len(replay_lines) > 0:
        to_delete = set()
        for line in replay_lines:
            m = processor_pat.match(line)
            if m <> None:
                if state.add_processor(int(m.group('pid'),16), int(m.group('util'),16), int(m.group('kind'))):
                    to_delete.add(line)
                continue
            m = proc_mem_pat.match(line)
            if m <> None:
                if state.set_proc_mem(int(m.group('pid'),16), int(m.group('mid'),16), int(m.group('band')), int(m.group('lat'))):
                    to_delete.add(line)
                continue
            m = mem_mem_pat.match(line)
            if m <> None:
                if state.set_mem_mem(int(m.group('mone'),16), int(m.group('mtwo'),16), int(m.group('band')), int(m.group('lat'))):
                    to_delete.add(line)
                continue
            m = index_part_pat.match(line)
            if m <> None:
                if state.add_index_partition(int(m.group('pid'),16), int(m.group('uid')), True if (int(m.group('disjoint'))) == 1 else False, int(m.group('color'))):
                    to_delete.add(line)
                continue
            m = index_subspace_pat.match(line)
            if m <> None:
                if state.add_index_subspace(int(m.group('pid')), int(m.group('uid'),16), int(m.group('color'))):
                    to_delete.add(line)
                continue
            m = field_create_pat.match(line)
            if m <> None:
                if state.add_field(int(m.group('uid')), int(m.group('fid'))):
                    to_delete.add(line)
                continue
            m = region_pat.match(line)
            if m <> None:
                if state.add_region(int(m.group('iid'),16), int(m.group('fid')), int(m.group('tid'))):
                    to_delete.add(line)
                continue
            m = task_pat.match(line)
            if m <> None:
                if state.add_task(int(m.group('uid')), int(m.group('tid')), int(m.group('pid')), int(m.group('ctx')), int(m.group('hid'),16), int(m.group('gen')), True if (int(m.group('index')) == 1) else False):
                    to_delete.add(line)
                continue
            m = mapping_pat.match(line)
            if m <> None:
                if state.add_mapping(int(m.group('uid')), int(m.group('pid')), int(m.group('ctx')), int(m.group('hid'),16), int(m.group('gen'))):
                    to_delete.add(line)
                continue
            m = deletion_pat.match(line)
            if m <> None:
                if state.add_deletion(int(m.group('uid')), int(m.group('pid')), int(m.group('ctx')), int(m.group('hid'),16), int(m.group('gen'))):
                    to_delete.add(line)
                continue
            m = task_name_pat.match(line)
            if m <> None:
                if state.add_name(int(m.group('uid')), m.group('name')):
                    to_delete.add(line)
                continue
            m = requirement_pat.match(line)
            if m <> None:
                if state.add_requirement(int(m.group('uid')), int(m.group('index')), True if (int(m.group('is_reg')))==1 else False, int(m.group('ispace'),16), int(m.group('fspace')), int(m.group('tid')), int(m.group('priv')), int(m.group('coher')), int(m.group('redop'))):
                    to_delete.add(line)
                continue
            m = req_field_pat.match(line)
            if m <> None:
                if state.add_req_field(int(m.group('uid')), int(m.group('index')), int(m.group('fid'))):
                    to_delete.add(line)
                continue
            m = mapping_dep_pat.match(line)
            if m <> None:
                if state.add_mapping_dependence(int(m.group('pid')), int(m.group('ctx')), int(m.group('hid'),16), int(m.group('gen')),  int(m.group('prev_id')), int(m.group('pidx')), int(m.group('next_id')), int(m.group('nidx')), int(m.group('dtype'))):
                    to_delete.add(line)
                continue
            m = task_inst_req_pat.match(line)
            if m <> None:
                if state.add_instance_requirement(int(m.group('uid')), int(m.group('ctx')), int(m.group('gen')), int(m.group('hid'),16), int(m.group('idx')), int(m.group('index'),16)):
                    to_delete.add(line)
                continue
            m = task_event_pat.match(line)
            if m <> None:
                point = Point(-1, 1)
                point.add_value(int(m.group('point')))
                if state.add_task_instance(int(m.group('uid')), int(m.group('ctx')), int(m.group('hid'),16), int(m.group('gen')), True if (int(m.group('index_space')))==1 else False, point, int(m.group('startid'),16), int(m.group('startgen')), int(m.group('termid'),16), int(m.group('termgen'))):
                    to_delete.add(line)
                continue
            m = index_term_pat.match(line)
            if m <> None:
                if state.add_index_term(int(m.group('uid')), int(m.group('termid'),16), int(m.group('termgen'))):
                    to_delete.add(line)
                continue
            m = map_event_pat.match(line)
            if m <> None:
                if state.add_map_instance(int(m.group('uid')), int(m.group('startid'),16), int(m.group('startgen')), int(m.group('termid'),16), int(m.group('termgen'))):
                    to_delete.add(line)
                continue
            m = point_event_pat.match(line)
            if m <> None:
                point = state.get_point(int(m.group('node')), int(m.group('point_id')))
                if state.add_task_instance(int(m.group('uid')), int(m.group('ctx')), int(m.group('hid'),16), int(m.group('gen')), True if (int(m.group('index_space')))==1 else False, point, int(m.group('startid'),16), int(m.group('startgen')), int(m.group('termid'),16), int(m.group('termgen'))):
                    to_delete.add(line)
                continue
            m = copy_event_pat.match(line)
            if m <> None:
                if state.add_copy_instance(int(m.group('srcman')), int(m.group('dstman')), int(m.group('index'),16), int(m.group('field')), int(m.group('tree')), int(m.group('startid'),16), int(m.group('startgen')), int(m.group('termid'),16), int(m.group('termgen')), m.group('mask')):
                    to_delete.add(line)
                continue
            m = reduce_event_pat.match(line)
            if m <> None:
                if state.add_reduce_instance(int(m.group('srcman')), int(m.group('dstman')), int(m.group('index'),16), int(m.group('field')), int(m.group('tree')), int(m.group('startid'),16), int(m.group('startgen')), int(m.group('termid'),16), int(m.group('termgen')), int(m.group('redop')), m.group('mask')):
                    to_delete.add(line)
                continue
            m = physical_inst_pat.match(line)
            if m <> None:
                if state.add_physical_instance(int(m.group('iid'),16), int(m.group('mid'),16), int(m.group('index'),16), int(m.group('field')), int(m.group('tid'))):
                    to_delete.add(line)
                continue
            m = inst_manager_pat.match(line)
            if m <> None:
                if state.add_instance_manager(int(m.group('iid'),16), int(m.group('manager'))):
                    to_delete.add(line)
                continue
            m = physical_reduc_pat.match(line)
            if m <> None:
                if state.add_reduction_instance(int(m.group('iid'),16), int(m.group('mid'),16), int(m.group('index'),16), int(m.group('field')), int(m.group('tid')), True if (int(m.group('fold')) == 1) else False, int(m.group('indirect'))):
                    to_delete.add(line)
                continue
            m = reduc_manager_pat.match(line)
            if m <> None:
                if state.add_reduction_manager(int(m.group('iid'),16), int(m.group('manager'))):
                    to_delete.add(line)
                continue
            m = task_user_pat.match(line)
            if m <> None:
                if state.add_task_user(int(m.group('uid')), int(m.group('ctx')), int(m.group('gen')), int(m.group('hid'),16), int(m.group('idx')), int (m.group('manager'))):
                    to_delete.add(line)
                continue
            m = mapping_user_pat.match(line)
            if m <> None:
                if state.add_mapping_user(int(m.group('uid')), int(m.group('manager'))):
                    to_delete.add(line)
                continue
            m = exec_info_pat.match(line)
            if m <> None:
                if state.set_exec_info(int(m.group('uid')), int(m.group('ctx')), int(m.group('gen')), int(m.group('hid'),16), int(m.group('proc'))):
                    to_delete.add(line)
                continue
            m = begin_info_pat.match(line)
            if m <> None:
                if state.set_task_start(int(m.group('uid')), int(m.group('ctx')), int(m.group('gen')), int(m.group('hid'),16), long(m.group('start'))):
                    to_delete.add(line)
                continue
            m = end_info_pat.match(line)
            if m <> None:
                if state.set_task_end(int(m.group('uid')), int(m.group('ctx')), int(m.group('gen')), int(m.group('hid'),16), long(m.group('end'))):
                    to_delete.add(line)
                continue
        # Now check to make sure we actually did something
        # If not then we're not making forward progress which is bad
        if len(to_delete) == 0:
            print "ERROR: NO PROGRESS PARSING! BUG IN LEGION SPY LOGGING ASSUMPTIONS!"
            assert False
        # Now delete any lines to delete and go again until we're done
        for line in to_delete:
            replay_lines.remove(line)
    return matches

# EOF

