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

from __future__ import print_function
import inspect
import re
import legion_spy



TASK_INFO_ENC         = 0
META_INFO_ENC         = 1
COPY_INFO_ENC         = 2
FILL_INFO_ENC         = 3
INST_CREATE_ENC       = 4
INST_USAGE_ENC        = 5
INST_TIMELINE_ENC     = 6
USER_INFO_ENC         = 7
TASK_WAIT_INFO_ENC    = 8
META_WAIT_INFO_ENC    = 9
TASK_KIND_ENC         = 10
TASK_KIND_OVER_ENC    = 11
TASK_VARIANT_ENC      = 12
OPERATION_ENC         = 13
MULTI_ENC             = 14
SLICE_OWNER_ENC       = 15
META_DESC_ENC         = 16
OP_DESC_ENC           = 17
PROC_DESC_ENC         = 18
MEM_DESC_ENC          = 19
MESSAGE_DESC_ENC      = 20
MESSAGE_INFO_ENC      = 21
MAPPER_CALL_DESC_ENC  = 22
MAPPER_CALL_INFO_ENC  = 23
RUNTIME_CALL_DESC_ENC = 24
RUNTIME_CALL_INFO_ENC = 25
PROFTASK_INFO_ENC     = 26

ENCODINGS = {
    "Task Info":          {"ENC": TASK_INFO_ENC,         "NUM_ARGS": 7},
    "Meta Info":          {"ENC": META_INFO_ENC,         "NUM_ARGS": 7},
    "Copy Info":          {"ENC": COPY_INFO_ENC,         "NUM_ARGS": 8},
    "Fill Info":          {"ENC": FILL_INFO_ENC,         "NUM_ARGS": 6},
    "Inst Create":        {"ENC": INST_CREATE_ENC,       "NUM_ARGS": 3},
    "Inst Usage" :        {"ENC": INST_USAGE_ENC,        "NUM_ARGS": 4},
    "Inst Timeline":      {"ENC": INST_TIMELINE_ENC,     "NUM_ARGS": 4},
    "User Info" :         {"ENC": USER_INFO_ENC,         "NUM_ARGS": 4},
    "Task Wait Info" :    {"ENC": TASK_WAIT_INFO_ENC,    "NUM_ARGS": 5},
    "Meta Wait Info" :    {"ENC": META_WAIT_INFO_ENC,    "NUM_ARGS": 5},
    "Task Kind" :         {"ENC": TASK_KIND_ENC,         "NUM_ARGS": 3},
    "Task Kind Over" :    {"ENC": TASK_KIND_OVER_ENC,    "NUM_ARGS": 3},
    "Task Variant" :      {"ENC": TASK_VARIANT_ENC,      "NUM_ARGS": 3},
    "Operation" :         {"ENC": OPERATION_ENC,         "NUM_ARGS": 2},
    "Multi" :             {"ENC": MULTI_ENC,             "NUM_ARGS": 2},
    "Slice Owner" :       {"ENC": SLICE_OWNER_ENC,       "NUM_ARGS": 2},
    "Meta Desc" :         {"ENC": META_DESC_ENC,         "NUM_ARGS": 2},
    "Op Desc" :           {"ENC": OP_DESC_ENC,           "NUM_ARGS": 2},
    "Proc Desc" :         {"ENC": PROC_DESC_ENC,         "NUM_ARGS": 2},
    "Mem Desc" :          {"ENC": MEM_DESC_ENC,          "NUM_ARGS": 3},
    "Message Desc" :      {"ENC": MESSAGE_DESC_ENC,      "NUM_ARGS": 2},
    "Message Info" :      {"ENC": MESSAGE_INFO_ENC,      "NUM_ARGS": 4},
    "Mapper Call Desc" :  {"ENC": MAPPER_CALL_DESC_ENC,  "NUM_ARGS": 2},
    "Mapper Call Info" :  {"ENC": MAPPER_CALL_INFO_ENC,  "NUM_ARGS": 5},
    "Runtime Call Desc" : {"ENC": RUNTIME_CALL_DESC_ENC, "NUM_ARGS": 2},
    "Runtime Call Info" : {"ENC": RUNTIME_CALL_INFO_ENC, "NUM_ARGS": 4},
    "ProfTask Info" :     {"ENC": PROFTASK_INFO_ENC,     "NUM_ARGS": 4}
}
REQUIRED_CALLBACKS = ENCODINGS.keys()
REQUIRED_CALLBACKS_SET = set(REQUIRED_CALLBACKS)

# Encodings for the different types of profiling. MAKE SURE THIS IS UP TO DATE WITH XXX


#         assert(len(callbacks) == len(REQUIRED_CALLBACKS))
# 
#         callbacks_valid = True
#         for callback_name, callback in callbacks.iteritems():
#             cur_valid = callback_name in REQUIRED_CALLBACKS_SET and \
#                         callable(callback) and \
#                         len(inspect.getargspec(callback).args) == ENCODINGS[callback_name]["NUM_ARGS"]
#             callbacks_valid = callbacks_valid and cur_valid
#         assert callbacks_valid


class LegionDeserializer(object):
    """This is a generic class for our deserializer"""
    def __init__(self, state, callbacks):
        """The constructor for the initializer.

        @param[state]:     The state object for our callbacks
        @param[callbacks]: A dictionary containing the callbacks we should use
                           after deserializing each item. You must pass a callback
                           for 
        """
        self.state = state
        self.callbacks = callbacks

    def deserialize(self, filepath):
        """ The generic deserialize method

        @param[filename]: The path to the binary file we want to deserialize
        """
        raise NotImplementedError

def read_time(string):
    return long(string)/1000

class LegionProfRegexDeserializer(LegionDeserializer):
    """
    This is the serializer for the regex log files
    """

    #               node           thread
    prefix = r'\[(?:[0-9]+) - (?:[0-9a-f]+)\] \{\w+\}\{legion_prof\}: '

    patterns = {
        "Task Info": re.compile(prefix + r'Prof Task Info (?P<op_id>[0-9]+) (?P<vid>[0-9]+) (?P<pid>[a-f0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "Meta Info": re.compile(prefix + r'Prof Meta Info (?P<op_id>[0-9]+) (?P<hlr>[0-9]+) (?P<pid>[a-f0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "Copy Info": re.compile(prefix + r'Prof Copy Info (?P<op_id>[0-9]+) (?P<src>[a-f0-9]+) (?P<dst>[a-f0-9]+) (?P<size>[0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "Fill Info": re.compile(prefix + r'Prof Fill Info (?P<op_id>[0-9]+) (?P<dst>[a-f0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "Inst Create": re.compile(prefix + r'Prof Inst Create (?P<op_id>[0-9]+) (?P<inst>[a-f0-9]+) (?P<create>[0-9]+)'),
        "Inst Usage": re.compile(prefix + r'Prof Inst Usage (?P<op_id>[0-9]+) (?P<inst>[a-f0-9]+) (?P<mem>[a-f0-9]+) (?P<size>[0-9]+)'),
        "Inst Timeline": re.compile(prefix + r'Prof Inst Timeline (?P<op_id>[0-9]+) (?P<inst>[a-f0-9]+) (?P<create>[0-9]+) (?P<destroy>[0-9]+)'),
        "User Info": re.compile(prefix + r'Prof User Info (?P<pid>[a-f0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+) (?P<name>[$()a-zA-Z0-9_]+)'),
        "Task Wait Info": re.compile(prefix + r'Prof Task Wait Info (?P<op_id>[0-9]+) (?P<vid>[0-9]+) (?P<start>[0-9]+) (?P<ready>[0-9]+) (?P<end>[0-9]+)'),
        "Meta Wait Info": re.compile(prefix + r'Prof Meta Wait Info (?P<op_id>[0-9]+) (?P<hlr>[0-9]+) (?P<start>[0-9]+) (?P<ready>[0-9]+) (?P<end>[0-9]+)'),
        "Task Kind": re.compile(prefix + r'Prof Task Kind (?P<tid>[0-9]+) (?P<name>[$()a-zA-Z0-9_<>.]+)'),
        "Task Kind Over": re.compile(prefix + r'Prof Task Kind (?P<tid>[0-9]+) (?P<name>[$()a-zA-Z0-9_<>.]+) (?P<overwrite>[0-1])'),
        "Task Variant": re.compile(prefix + r'Prof Task Variant (?P<tid>[0-9]+) (?P<vid>[0-9]+) (?P<name>[$()a-zA-Z0-9_<>.]+)'),
        "Operation": re.compile(prefix + r'Prof Operation (?P<op_id>[0-9]+) (?P<kind>[0-9]+)'),
        "Multi": re.compile(prefix + r'Prof Multi (?P<op_id>[0-9]+) (?P<tid>[0-9]+)'),
        "Slice Owner": re.compile(prefix + r'Prof Slice Owner (?P<parent_id>[0-9]+) (?P<op_id>[0-9]+)'),
        "Meta Desc": re.compile(prefix + r'Prof Meta Desc (?P<hlr>[0-9]+) (?P<name>[a-zA-Z0-9_ ]+)'),
        "Op Desc": re.compile(prefix + r'Prof Op Desc (?P<kind>[0-9]+) (?P<name>[a-zA-Z0-9_ ]+)'),
        "Prof Desc": re.compile(prefix + r'Prof Proc Desc (?P<pid>[a-f0-9]+) (?P<kind>[0-9]+)'),
        "Mem Desc": re.compile(prefix + r'Prof Mem Desc (?P<mem_id>[a-f0-9]+) (?P<kind>[0-9]+) (?P<size>[0-9]+)'),
        "Message Desc": re.compile(prefix + r'Prof Message Desc (?P<kind>[0-9]+) (?P<desc>[a-zA-Z0-9_ ]+)'),
        "Message Info": re.compile(prefix + r'Prof Message Info (?P<kind>[0-9]+) (?P<pid>[a-f0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "Mapper Call Desc": re.compile(prefix + r'Prof Mapper Call Desc (?P<kind>[0-9]+) (?P<desc>[a-zA-Z0-9_ ]+)'),
        "Mapper Call Info": re.compile(prefix + r'Prof Mapper Call Info (?P<kind>[0-9]+) (?P<pid>[a-f0-9]+) (?P<op_id>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "Runtime Call Desc": re.compile(prefix + r'Prof Runtime Call Desc (?P<kind>[0-9]+) (?P<desc>[a-zA-Z0-9_ ]+)'),
        "Runtime Call Info": re.compile(prefix + r'Prof Runtime Call Info (?P<kind>[0-9]+) (?P<pid>[a-f0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "Proftask Info": re.compile(prefix + r'Prof ProfTask Info (?P<pid>[a-f0-9]+) (?P<op_id>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)')
    }
    parse_callbacks = {
        "op_id": long,
        "parent_id": long,
        "size": long,
        "vid": int,
        "hlr": int,
        "uid": int,
        "overwrite": int,
        "tid": int,
        "kind": int,
        "opkind": int,
        "pid": lambda x: int(x, 16),
        "mem_id": lambda x: int(x, 16),
        "src": lambda x: int(x, 16),
        "dst": lambda x: int(x, 16),
        "inst": lambda x: int(x, 16),
        "mem": lambda x: int(x, 16),
        "create": read_time,
        "destroy": read_time,
        "ready": read_time,
        "start": read_time,
        "stop": read_time,
        "end": read_time,
        "name": lambda x: x,
        "desc": lambda x: x
    }

    def __init__(self, state, callbacks):
        LegionDeserializer.__init__(self, state, callbacks)

        assert len(callbacks) == len(LegionProfRegexDeserializer.patterns)

        callbacks_valid = True
        for callback_name, callback in callbacks.iteritems():
            cur_valid = callback_name in LegionProfRegexDeserializer.patterns and \
                        callable(callback)
            callbacks_valid = callbacks_valid and cur_valid
        assert callbacks_valid

    def parse_regex_matches(self, m):
        kwargs = m.groupdict()

        for key, arg in kwargs.iteritems():
            kwargs[key] = LegionProfRegexDeserializer.parse_callbacks[key](arg)
        return kwargs

    def parse(self, filepath, verbose):
        skipped = 0
        with open(filepath, 'rb') as log:
            matches = 0
            # Keep track of the first and last times
            first_time = 0L
            last_time = 0L
            for line in log:
                if not self.state.has_spy_data and \
                    (legion_spy.config_pat.match(line) or \
                     legion_spy.detailed_config_pat.match(line)):
                    self.state.has_spy_data = True
                matched = False

                for prof_event, pattern in LegionProfRegexDeserializer.patterns.iteritems():
                    m = pattern.match(line)
                    if m is not None:
                        callback = self.callbacks[prof_event]
                        kwargs = self.parse_regex_matches(m)
                        callback(**kwargs)
                        matched = True
                        break
                if matched:
                    matches += 1
                else:
                    skipped += 1
                    if verbose:
                        print('Skipping line: %s' % line.strip())
        if skipped > 0:
            print('WARNING: Skipped %d lines in %s' % (skipped, filepath))
        return matches

