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

import sys, os, re, argparse, itertools

scriptPath = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.append(scriptPath + '../')
from legion_serializer import LegionProfBinaryDeserializer

noop = lambda **kwargs: None

# these will be maps of ids to strings
task_names = {}
meta_task_names = {}

task_kinds = {}

# these will be maps of tasks to time spans
task_spans = {}


def read_time(string):
    return long(string)/1000

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

def log_task_info(op_id, variant_id, proc_id, create, ready, start, stop):
    assert variant_id in task_names
    task_name = task_names[variant_id]
    time_range = (read_time(start), read_time(stop))
    if task_name in task_spans:
        prev_start, prev_stop = task_spans[task_name]
        time_range = (min(start, prev_start), max(stop, prev_stop))
    task_spans[task_name] = time_range

def log_meta_info(op_id, lg_id, proc_id, create, ready, start, stop):
    assert lg_id in meta_task_names
    task_name = meta_task_names[lg_id]
    time_range = (start, stop)
    if task_name in task_spans:
        prev_start, prev_stop = task_spans[task_name]
        time_range = (min(start, prev_start), max(stop, prev_stop))
    task_spans[task_name] = time_range

callbacks = {
    "MessageDesc": noop,
    "MapperCallDesc": noop,
    "RuntimeCallDesc": noop,
    "MetaDesc": log_meta_desc,
    "OpDesc": noop,
    "ProcDesc": noop,
    "MemDesc": noop,
    "TaskKind": log_kind,
    "TaskVariant": log_variant,
    "OperationInstance": noop,
    "MultiTask": noop,
    "SliceOwner": noop,
    "TaskWaitInfo": noop,
    "MetaWaitInfo": noop,
    "TaskInfo": log_task_info,
    "MetaInfo": log_meta_info,
    "CopyInfo": noop,
    "FillInfo": noop,
    "InstCreateInfo": noop,
    "InstUsageInfo": noop,
    "InstTimelineInfo": noop,
    "MessageInfo": noop,
    "MapperCallInfo": noop,
    "RuntimeCallInfo": noop,
    "ProfTaskInfo": noop
}

def main():
    parser = argparse.ArgumentParser()

    # usage: python span_example.py <task0> <task1> <logfile0> <logfile1>
    parser.add_argument("-t", "--tasks", nargs=2, type=str,
                        help="The two task names you want to find the span between. These are interpreted as regexes")
    parser.add_argument(dest='filenames', nargs='+',
                        help='input Legion Prof log filenames')

    args = parser.parse_args()

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

    matching_tasks0 = []
    matching_tasks1 = []
    
    for task_name in task_spans.iterkeys():
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
    span0 = task_spans[match0]
    span1 = task_spans[match1]

    task_span = (min(span0[0], span1[0]), max(span0[1], span1[1]))
    print("Range between " + task0 + " and " + task1 + " was " + str(task_span))

    if span0[0] < span1[1] or span1[0] < span0[1]:
        print("Warning: Task Spans overlap!")

if __name__ == "__main__":
    main()
