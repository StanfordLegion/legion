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

import sys, os, re, argparse, itertools

scriptPath = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.append(scriptPath + '../')
from legion_serializer import LegionProfBinaryDeserializer
from legion_serializer import LegionProfASCIIDeserializer

noop = lambda **kwargs: None

# these will be maps of ids to strings
task_names = {}
meta_task_names = {}

task_kinds = {}

# these will be maps of tasks to time spans
task_spans = {}


def log_kind(task_id, name, overwrite):
    if (task_id not in task_kinds) or (overwrite == 1):
        task_kinds[task_id] = name

def log_meta_desc(kind, name):
    meta_task_names[kind] = name

def log_variant(task_id, variant_id, name):
    assert task_id in task_kinds
    task_name = task_kinds[task_id]
    task_names[variant_id] = task_name
    meta_task_names[variant_id] = name

def log_task_info(op_id, task_id, variant_id, proc_id, create, ready, start, stop):
    assert variant_id in task_names
    task_name = task_names[variant_id]
    time_range = (start, stop)
    if task_name not in task_spans:
        task_spans[task_name] = []
    task_spans[task_name].append(time_range)

def log_gpu_task_info(op_id, task_id, variant_id, proc_id, create, ready, start, stop, gpu_start, gpu_stop):
    assert variant_id in task_names
    task_name = task_names[variant_id]
    time_range = (gpu_start, gpu_stop)
    if task_name not in task_spans:
        task_spans[task_name] = []
    task_spans[task_name].append(time_range)

def log_meta_info(op_id, lg_id, proc_id, create, ready, start, stop):
    assert lg_id in meta_task_names
    task_name = meta_task_names[lg_id]
    time_range = (start, stop)
    if task_name not in task_spans:
        task_spans[task_name] = []
    task_spans[task_name].append(time_range)

callbacks = {
    "MapperCallDesc": noop,
    "RuntimeCallDesc": noop,
    "MetaDesc": log_meta_desc,
    "OpDesc": noop,
    "ProcDesc": noop,
    "MemDesc": noop,
    "TaskKind":  log_kind,
    "TaskVariant": log_variant,
    "OperationInstance": noop,
    "MultiTask": noop,
    "SliceOwner": noop,
    "TaskWaitInfo": noop,
    "MetaWaitInfo": noop,
    "TaskInfo":  log_task_info,
    "GPUTaskInfo": log_gpu_task_info,
    "MetaInfo":  log_meta_info,
    "CopyInfo": noop,
    "FillInfo": noop,
    "InstCreateInfo": noop,
    "InstUsageInfo": noop,
    "InstTimelineInfo": noop,
    "PartitionInfo": noop,
    "MessageInfo": noop,
    "MapperCallInfo": noop,
    "RuntimeCallInfo": noop,
    "ProfTaskInfo": noop,
    "ProcMDesc": noop,
    "IndexSpacePointDesc": noop,
    "IndexSpaceRectDesc": noop,
    "PartDesc": noop,
    "IndexPartitionDesc": noop,
    "IndexSpaceEmptyDesc": noop,
    "FieldDesc": noop,
    "FieldSpaceDesc": noop,
    "IndexSpaceDesc": noop,
    "IndexSubSpaceDesc": noop,
    "LogicalRegionDesc": noop,
    "PhysicalInstRegionDesc": noop,
    "PhysicalInstLayoutDesc": noop,
    "PhysicalInstDimOrderDesc": noop,
    "IndexSpaceSizeDesc": noop,
    "MaxDimDesc": noop,
    "CopyInstInfo": noop,
}


class Dummy(object):
    def __init__(self):
        self.has_spy_data = False

def main():
    parser = argparse.ArgumentParser()

    # usage: python3 span_example.py <task0> <task1> <logfile0> <logfile1>
    parser.add_argument("-t", "--tasks", nargs=2, type=str,
                        help="The two task names you want to find the span between. These are interpreted as regexes")
    parser.add_argument("-s", "--skip", nargs=2, type=int,
                        default=(0, 0),
                        help="The number of task occurrences you want to prune in the span.")
    parser.add_argument("-a", "--ascii", action="store_true",
                        dest="ascii_parser",
                        help="Use ASCII parser.")
    parser.add_argument(dest='filenames', nargs='+',
                        help='input Legion Prof log filenames')

    args = parser.parse_args()

    if args.ascii_parser:
        deserializer = LegionProfASCIIDeserializer(Dummy(), callbacks)
    else:
        deserializer = LegionProfBinaryDeserializer(None, callbacks)

    if args.filenames is None:
        print("You must pass in a logfile!")
        exit(-1)
    has_matches = False 
    for file_name in args.filenames:
        matches = deserializer.parse(file_name, True)
        has_matches = has_matches or matches > 0

    if not has_matches:
        print('No matches found! Exiting...')
        return

    # now search through the spans

    task0 = args.tasks[0]
    task1 = args.tasks[1]
    skip0 = args.skip[0]
    skip1 = args.skip[1]

    matching_tasks0 = []
    matching_tasks1 = []
    
    for task_name in task_spans.keys():
        if re.search(task0, task_name):
            matching_tasks0.append(task_name)
        if re.search(task1, task_name):
            matching_tasks1.append(task_name)

    err = False
    for (task, matches) in [(task0, matching_tasks0), (task1, matching_tasks1)]:
        if len(matches) == 0:
            print("No matching tasks for " + task + " found!")
            err = True
        
        if len(matches) > 1:
            print("Task " + task + " had disambiguous matches. Pick the task name you want from the following list:")
            print(matches)
            err = True

    if err:
        exit(-1)

    match0 = matching_tasks0[0]
    match1 = matching_tasks1[0]
    span0 = sorted(task_spans[match0], key=lambda r: r[0])
    span1 = sorted(task_spans[match1], key=lambda r: r[0])

    span0 = span0[skip0:]
    if skip1 > 0:
        span1 = span1[0:-skip1]

    start = min([s[0] for s in span0])
    stop = max([s[1] for s in span1])
    task_span = (start, stop)
    print("Range between " + task0 + " and " + task1 + " was " + \
            str(task_span[1] - task_span[0]) + "us (start: " + \
            str(task_span[0]) + "us, stop: " + str(task_span[1]) + "us)")

if __name__ == "__main__":
    main()
