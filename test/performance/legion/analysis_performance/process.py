#!/usr/bin/env python
#
# Copyright 2017 NVIDIA Corporation
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

import argparse
from legion_serializer import LegionProfASCIIDeserializer

noop = lambda **kwargs: None

id_to_task_group = {}
task_groups = {}
num_task_groups = 0
summaries = []
versioning_ops = set([])
last_op_id = None

class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def length(self):
        return self.end - self.start

    def __repr__(self):
        return "[" + str(self.start) + ", " + str(self.end) + "] : " + \
                str(self.length()) + "us"

    def __add__(self, other):
        return Interval(min(self.start, other.start), max(self.end, other.end))

def itv_sum(l):
    return reduce(lambda x, y: x + y, l)

class Summary(object):
    def __init__(self, span):
        self.span = span
        self.num_tasks = 0
        self.logical_analysis = 0
        self.physical_analysis = 0
        self.post_end = 0
        self.prepipeline = 0
        self.versioning = 0
        self.sum = 0
        self.ctx_switch = 0

    def cache(self):
        self.sum = self.logical_analysis + self.physical_analysis + \
                self.post_end + self.prepipeline + self.versioning
        self.ctx_switch = self.span.length() - self.sum

    def __add__(self, other):
        result = Summary(None)
        result.num_tasks = self.num_tasks + other.num_tasks
        result.logical_analysis = self.logical_analysis + other.logical_analysis
        result.physical_analysis = self.physical_analysis + other.physical_analysis
        result.post_end = self.post_end + other.post_end
        result.prepipeline = self.prepipeline + other.prepipeline
        result.versioning = self.versioning + other.versioning
        result.span = self.span + other.span
        result.sum = self.sum + other.sum
        result.ctx_switch = self.ctx_switch + other.ctx_switch
        return result

    def __repr__(self):
        num_tasks = float(self.num_tasks)
        s = "* total overhead: " + str(self.span.length() / num_tasks) + "\n"
        s = s + "* number of tasks: " + str(self.num_tasks) + "\n"
        s = s + "* logical analysis: " + str(self.logical_analysis / num_tasks) + "\n"
        s = s + "* physical analysis: " + str(self.physical_analysis / num_tasks) + "\n"
        s = s + "* post end task: " + str(self.post_end / num_tasks) + "\n"
        s = s + "* prepipeline: " + str(self.prepipeline / num_tasks) + "\n"
        s = s + "* close/open/advance: " + str(self.versioning / num_tasks) + "\n"
        s = s + "* context switch: " + str(self.ctx_switch / num_tasks)
        return s

class TaskGroupInfo(object):
    def __init__(self):
        self.tasks = set([])
        self.logical_analysis = set([])
        self.physical_analysis = set([])
        self.post_end = set([])
        self.prepipeline = set([])
        self.versioning_ops = set([])
        self.last_interval = None
        self.variant_id = None

    def add_task(self, task_id):
        self.tasks.add(task_id)

    def add_interval(self, interval, op_kind):
        if op_kind == 17 or op_kind == 14 or op_kind == 15:
            self.physical_analysis.add(interval)
        elif op_kind == 1:
            self.post_end.add(interval)
        elif op_kind == 12:
            self.logical_analysis.add(interval)
        elif op_kind == 11:
            self.prepipeline.add(interval)
        else:
            return
        self.last_interval = interval

    def add_wait_interval(self, wait_interval, op_kind):
        assert(self.last_interval != None)
        if op_kind == 17 or op_kind == 14 or op_kind == 15:
            target = self.physical_analysis
        elif op_kind == 1:
            target = self.post_end
        elif op_kind == 12:
            target = self.logical_analysis
        elif op_kind == 11:
            target = self.prepipeline
        else:
            return
        target.remove(self.last_interval)
        before_wait = Interval(self.last_interval.start, wait_interval.start)
        after_wait = Interval(wait_interval.end, self.last_interval.end)
        if before_wait.length() > 0:
            target.add(before_wait)
        if after_wait.length() > 0:
            target.add(after_wait)
        self.last_interval = after_wait

    def add_versioning_op(self, interval):
        self.versioning_ops.add(interval)

    def get_summary(self):
        span = itv_sum(self.logical_analysis) + itv_sum(self.physical_analysis) + \
                itv_sum(self.post_end) + itv_sum(self.prepipeline)
        summary = Summary(span)
        summary.num_tasks = len(self.tasks)
        summary.logical_analysis = \
                sum([i.length() for i in self.logical_analysis])
        summary.physical_analysis = \
                sum([i.length() for i in self.physical_analysis])
        summary.post_end = \
                sum([i.length() for i in self.post_end])
        summary.prepipeline = \
                sum([i.length() for i in self.prepipeline])
        summary.versioning = \
                sum([i.length() for i in self.versioning_ops])
        summary.cache()
        return summary

def add_new_task_group(task_id):
    global num_task_groups
    task_group = TaskGroupInfo()
    num_task_groups = num_task_groups + 1
    task_groups[num_task_groups] = task_group
    id_to_task_group[task_id] = task_group
    return task_group

def gather_slice_owners(parent_id, op_id):
    if parent_id not in id_to_task_group:
        task_group = add_new_task_group(parent_id)
    else:
        task_group = id_to_task_group[parent_id]

    id_to_task_group[op_id] = task_group

def gather_meta_info(op_id, lg_id, proc_id, create, ready, start, stop):
    global last_op_id
    if op_id in versioning_ops:
        assert(last_op_id != None)
        task_group = id_to_task_group[last_op_id]
        task_group.add_versioning_op(Interval(start, stop))
    elif op_id == 0 or op_id not in id_to_task_group:
        return
    else:
        task_group = id_to_task_group[op_id]
        task_group.add_interval(Interval(start, stop), lg_id)
        last_op_id = op_id

def gather_meta_wait_info(op_id, lg_id, wait_start, wait_ready, wait_end):
    if op_id == 0 or op_id not in id_to_task_group:
        return
    task_group = id_to_task_group[op_id]
    task_group.add_wait_interval(Interval(wait_start, wait_end), lg_id)

def gather_task_info(op_id, variant_id, proc_id, create, ready, start, stop):
    if op_id not in id_to_task_group:
        task_group = add_new_task_group(op_id)
    else:
        task_group = id_to_task_group[op_id]
    task_group.add_task(op_id)
    task_group.variant_id = variant_id

def mark_versioning_ops(op_id, kind):
    if 5 <= kind and kind <= 8:
        versioning_ops.add(op_id)

callbacks = {
	"MessageDesc": noop,
	"MapperCallDesc": noop,
	"RuntimeCallDesc": noop,
	"MetaDesc": noop,
	"OpDesc": noop,
	"ProcDesc": noop,
	"MemDesc": noop,
	"TaskKind": noop,
	"TaskVariant": noop,
	"OperationInstance": mark_versioning_ops,
	"MultiTask": noop,
	"SliceOwner": gather_slice_owners,
	"TaskWaitInfo": noop,
	"MetaWaitInfo": gather_meta_wait_info,
	"TaskInfo": gather_task_info,
	"MetaInfo": gather_meta_info,
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

class Dummy(object):
    def __init__(self):
        self.has_spy_data = False

def main():
    global task_groups
    global summaries
    parser = argparse.ArgumentParser()
    # usage: python process.py <logfiles>
    parser.add_argument(dest='filenames', nargs='+',
            help='input Legion Prof log filenames')
    args = parser.parse_args()
    deserializer = LegionProfASCIIDeserializer(Dummy(), callbacks)

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

    # filter only the tasks of interest
    task_group_ids = set(task_groups.iterkeys())
    for i in task_group_ids:
        if task_groups[i].variant_id != 2:
            del task_groups[i]

    for i, grp in task_groups.iteritems():
        summaries.append(grp.get_summary())

    # filter out the first and last 5% as they are mostly abnormal
    outliers = len(summaries) / 10
    summaries = summaries[outliers:-outliers]

    print(itv_sum(summaries))

if __name__ == "__main__":
    main()
