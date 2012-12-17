#!/usr/bin/python

# Copyright 2012 Stanford University
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

#!/usr/bin/python

from cent_state import *
import sys, re

# All of these calls are based on the print statements in legion_logging.h

prefix = "\[[0-9]+ - [0-9a-f]+\] \{\w+\}\{legion_spy\}: "
# Logger calls for the shape of the machine
utility_pat             = re.compile(prefix+"Utility (?P<pid>[0-9]+)")
processor_pat           = re.compile(prefix+"Processor (?P<pid>[0-9]+) (?P<util>[0-9]+) (?P<kind>[0-9]+)")
memory_pat              = re.compile(prefix+"Memory (?P<mid>[0-9]+) (?P<capacity>[0-9]+)")
proc_mem_pat            = re.compile(prefix+"Processor Memory (?P<pid>[0-9]+) (?P<mid>[0-9]+) (?P<band>[0-9]+) (?P<lat>[0-9]+)")
mem_mem_pat             = re.compile(prefix+"Memory Memory (?P<mone>[0-9]+) (?P<mtwo>[0-9]+) (?P<band>[0-9]+) (?P<lat>[0-9]+)")

# Calls for the shape of region trees
top_index_pat           = re.compile(prefix+"Index Space (?P<uid>[0-9]+)")
index_part_pat          = re.compile(prefix+"Index Partition (?P<pid>[0-9]+) (?P<uid>[0-9]+) (?P<disjoint>[0-1]) (?P<color>[0-9]+)")
index_subspace_pat      = re.compile(prefix+"Index Subspace (?P<pid>[0-9]+) (?P<uid>[0-9]+) (?P<color>[0-9]+)")
field_space_pat         = re.compile(prefix+"Field Space (?P<uid>[0-9]+)")
field_create_pat        = re.compile(prefix+"Field Creation (?P<uid>[0-9]+) (?P<fid>[0-9]+)")
region_pat              = re.compile(prefix+"Region (?P<iid>[0-9]+) (?P<fid>[0-9]+) (?P<tid>[0-9]+)")

# Logger calls for operations
top_task_pat            = re.compile(prefix+"Top Task (?P<hid>[0-9]+) (?P<gen>[0-9]+) (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<tid>[0-9]+)")
task_pat                = re.compile(prefix+"Task Operation (?P<uid>[0-9]+) (?P<tid>[0-9]+) (?P<pid>[0-9]+) (?P<ctx>[0-9]+) (?P<hid>[0-9]+) (?P<gen>[0-9]+) (?P<index>[0-1])")
mapping_pat             = re.compile(prefix+"Mapping Operation (?P<uid>[0-9]+) (?P<pid>[0-9]+) (?P<ctx>[0-9]+) (?P<hid>[0-9]+) (?P<gen>[0-9]+)")
deletion_pat            = re.compile(prefix+"Deletion Operation (?P<uid>[0-9]+) (?P<pid>[0-9]+) (?P<ctx>[0-9]+) (?P<hid>[0-9]+) (?P<gen>[0-9]+)")
task_name_pat           = re.compile(prefix+"Task Name (?P<uid>[0-9]+) (?P<name>\w+)")

# Logger calls for logical mapping dependence analysis
requirement_pat         = re.compile(prefix+"Logical Requirement (?P<uid>[0-9]+) (?P<index>[0-9]+) (?P<is_reg>[0-1]) (?P<ispace>[0-9]+) (?P<fspace>[0-9]+) (?P<tid>[0-9]+) (?P<priv>[0-9]+) (?P<coher>[0-9]+) (?P<redop>[0-9]+)")
req_field_pat           = re.compile(prefix+"Logical Requirement Field (?P<uid>[0-9]+) (?P<index>[0-9]+) (?P<fid>[0-9]+)")
mapping_dep_pat         = re.compile(prefix+"Mapping Dependence (?P<pid>[0-9]+) (?P<ctx>[0-9]+) (?P<hid>[0-9]+) (?P<gen>[0-9]+) (?P<prev_id>[0-9]+) (?P<pidx>[0-9]+) (?P<next_id>[0-9]+) (?P<nidx>[0-9]+) (?P<dtype>[0-9]+)")

# Logger calls for physical dependence analysis
task_inst_req_pat       = re.compile(prefix+"Task Instance Requirement (?P<uid>[0-9]+) (?P<ctx>[0-9]+) (?P<gen>[0-9]+) (?P<hid>[0-9]+) (?P<idx>[0-9]+) (?P<index>[0-9]+)")

# Logger calls for events
event_event_pat         = re.compile(prefix+"Event Event (?P<idone>[0-9]+) (?P<genone>[0-9]+) (?P<idtwo>[0-9]+) (?P<gentwo>[0-9]+)")
task_event_pat          = re.compile(prefix+"Task Events (?P<hid>[0-9]+) (?P<gen>[0-9]+) (?P<ctx>[0-9]+) (?P<uid>[0-9]+) (?P<index_space>[0-1]) (?P<point>[0-9]+) (?P<startid>[0-9]+) (?P<startgen>[0-9]+) (?P<termid>[0-9]+) (?P<termgen>[0-9]+)")
index_term_pat          = re.compile(prefix+"Index Termination (?P<uid>[0-9]+) (?P<termid>[0-9]+) (?P<termgen>[0-9]+)")
copy_event_pat          = re.compile(prefix+"Copy Events (?P<srcman>[0-9]+) (?P<dstman>[0-9]+) (?P<index>[0-9]+) (?P<field>[0-9]+) (?P<tree>[0-9]+) (?P<startid>[0-9]+) (?P<startgen>[0-9]+) (?P<termid>[0-9]+) (?P<termgen>[0-9]+) (?P<mask>[0-9\,]+)")
reduce_event_pat        = re.compile(prefix+"Reduce Events (?P<srcman>[0-9]+) (?P<dstman>[0-9]+) (?P<index>[0-9]+) (?P<field>[0-9]+) (?P<tree>[0-9]+) (?P<startid>[0-9]+) (?P<startgen>[0-9]+) (?P<termid>[0-9]+) (?P<termgen>[0-9]+) (?P<redop>[0-9]+) (?P<mask>[0-9\,]+)")
map_event_pat           = re.compile(prefix+"Map Events (?P<uid>[0-9]+) (?P<startid>[0-9]+) (?P<startgen>[0-9]+) (?P<termid>[0-9]+) (?P<termgen>[0-9]+)")

# Logger calls for physical instance usage 
physical_inst_pat       = re.compile(prefix+"Physical Instance (?P<iid>[0-9]+) (?P<mid>[0-9]+) (?P<index>[0-9]+) (?P<field>[0-9]+) (?P<tid>[0-9]+)")
inst_manager_pat        = re.compile(prefix+"Instance Manager (?P<iid>[0-9]+) (?P<manager>[0-9]+)")
physical_reduc_pat      = re.compile(prefix+"Reduction Instance (?P<iid>[0-9]+) (?P<mid>[0-9]+) (?P<index>[0-9]+) (?P<field>[0-9]+) (?P<tid>[0-9]+) (?P<fold>[0-1]) (?P<indirect>[0-9]+)")
reduc_manager_pat       = re.compile(prefix+"Reduction Manager (?P<iid>[0-9]+) (?P<manager>[0-9]+)")
task_user_pat           = re.compile(prefix+"Task Instance User (?P<uid>[0-9]+) (?P<ctx>[0-9]+) (?P<gen>[0-9]+) (?P<hid>[0-9]+) (?P<idx>[0-9]+) (?P<manager>[0-9]+)")
mapping_user_pat            = re.compile(prefix+"Mapping Instance User (?P<uid>[0-9]+) (?P<manager>[0-9]+)")

def parse_log_file(file_name, state):
    log = open(file_name, 'r')
    matches = 0
    for line in log:
        matches = matches + 1
        # Machine shapes
        m = utility_pat.match(line)
        if m <> None:
            state.add_utility(int(m.group('pid')))
            continue
        m = processor_pat.match(line)
        if m <> None:
            state.add_processor(int(m.group('pid')), int(m.group('util')), int(m.group('kind')))
            continue
        m = memory_pat.match(line)
        if m <> None:
            state.add_memory(int(m.group('mid')), int(m.group('capacity')))
            continue
        m = proc_mem_pat.match(line)
        if m <> None:
            state.set_proc_mem(int(m.group('pid')), int(m.group('mid')), int(m.group('band')), int(m.group('lat')))
            continue
        m = mem_mem_pat.match(line)
        if m <> None:
            state.set_mem_mem(int(m.group('mone')), int(m.group('mtwo')), int(m.group('band')), int(m.group('lat')))
            continue
        # Region tree shapes
        m = top_index_pat.match(line)
        if m <> None:
            state.add_index_space(int(m.group('uid')))
            continue
        m = index_part_pat.match(line)
        if m <> None:
            state.add_index_partition(int(m.group('pid')), int(m.group('uid')), True if (int(m.group('disjoint'))) == 1 else False, int(m.group('color')))
            continue
        m = index_subspace_pat.match(line)
        if m <> None:
            state.add_index_subspace(int(m.group('pid')), int(m.group('uid')), int(m.group('color')))
            continue
        m = field_space_pat.match(line)
        if m <> None:
            state.add_field_space(int(m.group('uid')))
            continue
        m = field_create_pat.match(line)
        if m <> None:
            state.add_field(int(m.group('uid')), int(m.group('fid')))
            continue
        m = region_pat.match(line)
        if m <> None:
            state.add_region(int(m.group('iid')), int(m.group('fid')), int(m.group('tid')))
            continue 
        # Operations
        m = top_task_pat.match(line)
        if m <> None:
            state.add_top_task(int(m.group('hid')), int(m.group('gen')), int(m.group('uid')), int(m.group('ctx')), int(m.group('tid')))
            continue
        m = task_pat.match(line)
        if m <> None:
            state.add_task(int(m.group('uid')), int(m.group('tid')), int(m.group('pid')), int(m.group('ctx')), int(m.group('hid')), int(m.group('gen')), True if (int(m.group('index')) == 1) else False)
            continue
        m = mapping_pat.match(line)
        if m <> None:
            state.add_mapping(int(m.group('uid')), int(m.group('pid')), int(m.group('ctx')), int(m.group('hid')), int(m.group('gen')))
            continue
        m = deletion_pat.match(line)
        if m <> None:
            state.add_deletion(int(m.group('uid')), int(m.group('pid')), int(m.group('ctx')), int(m.group('hid')), int(m.group('gen')))
            continue
        m = task_name_pat.match(line)
        if m <> None:
            state.add_name(int(m.group('uid')), m.group('name'))
            continue
        # Mapping dependence analysis
        m = requirement_pat.match(line)
        if m <> None:
            state.add_requirement(int(m.group('uid')), int(m.group('index')), True if (int(m.group('is_reg')))==1 else False, int(m.group('ispace')), int(m.group('fspace')), int(m.group('tid')), int(m.group('priv')), int(m.group('coher')), int(m.group('redop')))
            continue
        m = req_field_pat.match(line)
        if m <> None:
            state.add_req_field(int(m.group('uid')), int(m.group('index')), int(m.group('fid')))
            continue
        m = mapping_dep_pat.match(line)
        if m <> None:
            state.add_mapping_dependence(int(m.group('pid')), int(m.group('ctx')), int(m.group('hid')), int(m.group('gen')),  int(m.group('prev_id')), int(m.group('pidx')), int(m.group('next_id')), int(m.group('nidx')), int(m.group('dtype')))
            continue
        # Physical dependence analysis
        m = task_inst_req_pat.match(line)
        if m <> None:
            state.add_instance_requirement(int(m.group('uid')), int(m.group('ctx')), int(m.group('gen')), int(m.group('hid')), int(m.group('idx')), int(m.group('index')))
            continue
        # Physical Analysis
        m = task_event_pat.match(line)
        if m <> None:
            state.add_task_instance(int(m.group('uid')), int(m.group('ctx')), int(m.group('hid')), int(m.group('gen')), True if (int(m.group('index_space')))==1 else False, int(m.group('point')), int(m.group('startid')), int(m.group('startgen')), int(m.group('termid')), int(m.group('termgen')))
            continue
        m = event_event_pat.match(line)
        if m <> None:
            state.add_event_dependence(int(m.group('idone')), int(m.group('genone')), int(m.group('idtwo')), int(m.group('gentwo')))
            continue
        m = index_term_pat.match(line)
        if m <> None:
            state.add_index_term(int(m.group('uid')), int(m.group('termid')), int(m.group('termgen')))
            continue
        m = map_event_pat.match(line)
        if m <> None:
            state.add_map_instance(int(m.group('uid')), int(m.group('startid')), int(m.group('startgen')), int(m.group('termid')), int(m.group('termgen')))
            continue
        m = copy_event_pat.match(line)
        if m <> None:
            state.add_copy_instance(int(m.group('srcman')), int(m.group('dstman')), int(m.group('index')), int(m.group('field')), int(m.group('tree')), int(m.group('startid')), int(m.group('startgen')), int(m.group('termid')), int(m.group('termgen')), m.group('mask'))
            continue
        m = reduce_event_pat.match(line)
        if m <> None:
            state.add_reduce_instance(int(m.group('srcman')), int(m.group('dstman')), int(m.group('index')), int(m.group('field')), int(m.group('tree')), int(m.group('startid')), int(m.group('startgen')), int(m.group('termid')), int(m.group('termgen')), int(m.group('redop')), m.group('mask'))
            continue
        # Physical instance usage
        m = physical_inst_pat.match(line)
        if m <> None:
            state.add_physical_instance(int(m.group('iid')), int(m.group('mid')), int(m.group('index')), int(m.group('field')), int(m.group('tid')))
            continue
        m = inst_manager_pat.match(line)
        if m <> None:
            state.add_instance_manager(int(m.group('iid')), int(m.group('manager')))
            continue
        m = physical_reduc_pat.match(line)
        if m <> None:
            state.add_reduction_instance(int(m.group('iid')), int(m.group('mid')), int(m.group('index')), int(m.group('field')), int(m.group('tid')), True if (int(m.group('fold')) == 1) else False, int(m.group('indirect')))
            continue
        m = reduc_manager_pat.match(line)
        if m <> None:
            state.add_reduction_manager(int(m.group('iid')), int(m.group('manager')))
            continue
        m = task_user_pat.match(line)
        if m <> None:
            state.add_task_user(int(m.group('uid')), int(m.group('ctx')), int(m.group('gen')), int(m.group('hid')), int(m.group('idx')), int (m.group('manager')))
            continue
        m = mapping_user_pat.match(line)
        if m <> None:
            state.add_mapping_user(int(m.group('uid')), int(m.group('manager')))
            continue

        # If we made it here then we failed to match
        matches = matches - 1
    return matches

# EOF

