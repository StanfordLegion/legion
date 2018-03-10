#!/usr/bin/env python

# Copyright 2018 Stanford University, NVIDIA Corporation
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
import struct
import legion_spy
import gzip
import io

binary_filetype_pat = re.compile(r"FileType: BinaryLegionProf v: (?P<version>\d+(\.\d+)?)")

def getFileObj(filename, compressed=False, buffer_size=32768):
    if compressed:
        return io.BufferedReader(gzip.open(filename, mode='rb'), buffer_size=buffer_size)
    else:
        return open(filename, mode='rb', buffering=buffer_size)

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

    def deserialize(self, filename):
        """ The generic deserialize method

        @param[filename]: The path to the binary file we want to deserialize
        """
        raise NotImplementedError

def read_time(string):
    return long(string)/1000

class LegionProfASCIIDeserializer(LegionDeserializer):
    """
    This is the serializer for the regex log files
    """

    #               node           thread
    prefix = r'\[(?:[0-9]+) - (?:[0-9a-f]+)\] \{\w+\}\{legion_prof\}: '

    patterns = {
        "MessageDesc": re.compile(prefix + r'Prof Message Desc (?P<kind>[0-9]+) (?P<name>[a-zA-Z0-9_ ]+)'),
        "MapperCallDesc": re.compile(prefix + r'Prof Mapper Call Desc (?P<kind>[0-9]+) (?P<name>[a-zA-Z0-9_ ]+)'),
        "RuntimeCallDesc": re.compile(prefix + r'Prof Runtime Call Desc (?P<kind>[0-9]+) (?P<name>[a-zA-Z0-9_ ]+)'),
        "MetaDesc": re.compile(prefix + r'Prof Meta Desc (?P<kind>[0-9]+) (?P<name>[a-zA-Z0-9_ ]+)'),
        "OpDesc": re.compile(prefix + r'Prof Op Desc (?P<kind>[0-9]+) (?P<name>[a-zA-Z0-9_ ]+)'),
        "ProcDesc": re.compile(prefix + r'Prof Proc Desc (?P<proc_id>[a-f0-9]+) (?P<kind>[0-9]+)'),
        "MemDesc": re.compile(prefix + r'Prof Mem Desc (?P<mem_id>[a-f0-9]+) (?P<kind>[0-9]+) (?P<capacity>[0-9]+)'),
        "TaskKind": re.compile(prefix + r'Prof Task Kind (?P<task_id>[0-9]+) (?P<name>[$()a-zA-Z0-9_<>.]+) (?P<overwrite>[0-1])'),
        "TaskVariant": re.compile(prefix + r'Prof Task Variant (?P<task_id>[0-9]+) (?P<variant_id>[0-9]+) (?P<name>[$()a-zA-Z0-9_<>.]+)'),
        "OperationInstance": re.compile(prefix + r'Prof Operation (?P<op_id>[0-9]+) (?P<kind>[0-9]+)'),
        "MultiTask": re.compile(prefix + r'Prof Multi (?P<op_id>[0-9]+) (?P<task_id>[0-9]+)'),
        "SliceOwner": re.compile(prefix + r'Prof Slice Owner (?P<parent_id>[0-9]+) (?P<op_id>[0-9]+)'),
        "TaskWaitInfo": re.compile(prefix + r'Prof Task Wait Info (?P<op_id>[0-9]+) (?P<task_id>[0-9]+) (?P<variant_id>[0-9]+) (?P<wait_start>[0-9]+) (?P<wait_ready>[0-9]+) (?P<wait_end>[0-9]+)'),
        "MetaWaitInfo": re.compile(prefix + r'Prof Meta Wait Info (?P<op_id>[0-9]+) (?P<lg_id>[0-9]+) (?P<wait_start>[0-9]+) (?P<wait_ready>[0-9]+) (?P<wait_end>[0-9]+)'),
        "TaskInfo": re.compile(prefix + r'Prof Task Info (?P<op_id>[0-9]+) (?P<task_id>[0-9]+) (?P<variant_id>[0-9]+) (?P<proc_id>[a-f0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "MetaInfo": re.compile(prefix + r'Prof Meta Info (?P<op_id>[0-9]+) (?P<lg_id>[0-9]+) (?P<proc_id>[a-f0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "CopyInfo": re.compile(prefix + r'Prof Copy Info (?P<op_id>[0-9]+) (?P<src>[a-f0-9]+) (?P<dst>[a-f0-9]+) (?P<size>[0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "FillInfo": re.compile(prefix + r'Prof Fill Info (?P<op_id>[0-9]+) (?P<dst>[a-f0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "InstCreateInfo": re.compile(prefix + r'Prof Inst Create (?P<op_id>[0-9]+) (?P<inst_id>[a-f0-9]+) (?P<create>[0-9]+)'),
        "InstUsageInfo": re.compile(prefix + r'Prof Inst Usage (?P<op_id>[0-9]+) (?P<inst_id>[a-f0-9]+) (?P<mem_id>[a-f0-9]+) (?P<size>[0-9]+)'),
        "InstTimelineInfo": re.compile(prefix + r'Prof Inst Timeline (?P<op_id>[0-9]+) (?P<inst_id>[a-f0-9]+) (?P<create>[0-9]+) (?P<destroy>[0-9]+)'),
        "PartitionInfo": re.compile(prefix + r'Prof Partition Timeline (?P<op_id>[0-9]+) (?P<part_op>[0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "MessageInfo": re.compile(prefix + r'Prof Message Info (?P<kind>[0-9]+) (?P<proc_id>[a-f0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "MapperCallInfo": re.compile(prefix + r'Prof Mapper Call Info (?P<kind>[0-9]+) (?P<proc_id>[a-f0-9]+) (?P<op_id>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "RuntimeCallInfo": re.compile(prefix + r'Prof Runtime Call Info (?P<kind>[0-9]+) (?P<proc_id>[a-f0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "ProfTaskInfo": re.compile(prefix + r'Prof ProfTask Info (?P<proc_id>[a-f0-9]+) (?P<op_id>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)')
        # "UserInfo": re.compile(prefix + r'Prof User Info (?P<proc_id>[a-f0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+) (?P<name>[$()a-zA-Z0-9_]+)')
    }
    parse_callbacks = {
        "op_id": long,
        "parent_id": long,
        "size": long,
        "capacity": long,
        "variant_id": int,
        "lg_id": int,
        "uid": int,
        "overwrite": int,
        "task_id": int,
        "kind": int,
        "opkind": int,
        "part_op": int,
        "proc_id": lambda x: int(x, 16),
        "mem_id": lambda x: int(x, 16),
        "src": lambda x: int(x, 16),
        "dst": lambda x: int(x, 16),
        "inst_id": lambda x: int(x, 16),
        "create": read_time,
        "destroy": read_time,
        "start": read_time,
        "ready": read_time,
        "stop": read_time,
        "end": read_time,
        "wait_start": read_time,
        "wait_ready": read_time,
        "wait_end": read_time,
        "name": lambda x: x,
        "desc": lambda x: x
    }

    def __init__(self, state, callbacks):
        LegionDeserializer.__init__(self, state, callbacks)

        assert len(callbacks) == len(LegionProfASCIIDeserializer.patterns)

        callbacks_valid = True
        for callback_name, callback in callbacks.iteritems():
            cur_valid = callback_name in LegionProfASCIIDeserializer.patterns and \
                        callable(callback)
            callbacks_valid = callbacks_valid and cur_valid
        assert callbacks_valid

    def parse_regex_matches(self, m):
        kwargs = m.groupdict()

        for key, arg in kwargs.iteritems():
            kwargs[key] = LegionProfASCIIDeserializer.parse_callbacks[key](arg)
        return kwargs

    def search_for_spy_data(self, filename):
        with open(filename, 'rb') as log:
            for line in log:
                if legion_spy.config_pat.match(line) or \
                   legion_spy.detailed_config_pat.match(line):
                   self.state.has_spy_data = True
                   break

    def parse(self, filename, verbose):
        skipped = 0
        with open(filename, 'rb') as log:
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

                for prof_event, pattern in LegionProfASCIIDeserializer.patterns.iteritems():
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
            print('WARNING: Skipped %d lines in %s' % (skipped, filename))
        return matches

class LegionProfBinaryDeserializer(LegionDeserializer):

    preamble_regex = re.compile(r'(?P<name>\w+) {id:(?P<id>\d+)(?P<params>.*)}')
    params_regex = re.compile(r', (?P<param_name>[^:]+):(?P<param_type>[^:]+):(?P<param_bytes>-?\d+)')

    preamble_data = {}
    name_to_id = {}

    # XXX: Make sure these are consistent with legion_profiling.h and legion_types.h!
    fmt_dict = {
        "ProcID":             "Q", # unsigned long long
        "MemID":              "Q", # unsigned long long
        "InstID":             "Q", # unsigned long long
        "UniqueID":           "Q", # unsigned long long
        "TaskID":             "I", # unsigned int
        "bool":               "?", # bool
        "VariantID":          "L", # unsigned long
        "unsigned":           "I", # unsigned int
        "timestamp_t":        "Q", # unsigned long long
        "unsigned long long": "Q", # unsigned long long
        "ProcKind":           "i", # int (really an enum so this depends)
        "MemKind":            "i", # int (really an enum so this depends)
        "MessageKind":        "i", # int (really an enum so this depends)
        "MappingCallKind":    "i", # int (really an enum so this depends)
        "RuntimeCallKind":    "i", # int (really an enum so this depends)
        "DepPartOpKind":      "i", # int (really an enum so this depends)
    }

    def __init__(self, state, callbacks):
        LegionDeserializer.__init__(self, state, callbacks)
        self.callbacks_translated = False

    @staticmethod
    def create_type_reader(num_bytes, param_type):
        if param_type == "string":
            def string_reader(log):
                string = ""
                char = log.read(1)
                while ord(char) != 0:
                    string += char
                    char = log.read(1)
                return string
            return string_reader
        else:
            fmt = LegionProfBinaryDeserializer.fmt_dict[param_type]
            def reader(log):
                raw_val = log.read(num_bytes)
                val = struct.unpack(fmt, raw_val)[0]
                if param_type == "timestamp_t":
                    val = val / 1000
                return val
            return reader

    def parse_preamble(self, log):
        log.readline() # filetype
        while(True):
            line = log.readline()
            if line == "\n":
                break

            m = LegionProfBinaryDeserializer.preamble_regex.match(line)
            if not m:
                print("Malformed binary log file. Must contain a valid preamble!")
                print("Malformed line: '" + line + "'")
                exit(-1)

            name = m.group('name')
            _id = int(m.group('id'))
            params = m.group('params')
            param_data = []
            
            for param_m in LegionProfBinaryDeserializer.params_regex.finditer(params):
                param_name = param_m.group('param_name')
                param_type = param_m.group('param_type')
                param_bytes = int(param_m.group('param_bytes'))

                reader = LegionProfBinaryDeserializer.create_type_reader(param_bytes, param_type)

                param_data.append((param_name, reader))

            LegionProfBinaryDeserializer.preamble_data[_id] = param_data
            LegionProfBinaryDeserializer.name_to_id[name] = _id


        # change the callbacks to be by id
        if not self.callbacks_translated:
            new_callbacks = {LegionProfBinaryDeserializer.name_to_id[name]: callback 
                               for name, callback in self.callbacks.iteritems()
                               if name in LegionProfBinaryDeserializer.name_to_id}
            self.callbacks = new_callbacks
            self.callbacks_translated = True


        # callbacks_valid = True
        # for callback_name, callback in callbacks.iteritems():
        #     cur_valid = callback_name in LegionProfASCIIDeserializer.patterns and \
        #                 callable(callback)
        #     callbacks_valid = callbacks_valid and cur_valid
        # assert callbacks_valid

    def parse(self, filename, verbose):
        print("parsing " + str(filename))
        def parse_file(log):
            matches = 0
            self.parse_preamble(log)
            _id_raw = log.read(4)
            while _id_raw:
                matches += 1
                _id = int(struct.unpack('i', _id_raw)[0])
                param_data = LegionProfBinaryDeserializer.preamble_data[_id]
                kwargs = {}
                for (param_name, reader) in param_data:
                    val = reader(log)
                    kwargs[param_name] = val
                self.callbacks[_id](**kwargs)
                _id_raw = log.read(4)
            return matches
        try:
            # Try it as a gzip file first
            with getFileObj(filename,compressed=True) as log:
                return parse_file(log)    
        except IOError:
            # If its not a gzip file try a normal file
            with getFileObj(filename,compressed=False) as log:
                return parse_file(log)

def GetFileTypeInfo(filename):
    def parse_file(log):
        filetype = None
        version = None
        line = log.readline().rstrip()
        m = binary_filetype_pat.match(line)
        if m is not None:
            filetype = "binary"
            version = m.group("version")
        else:
            filetype = "ascii" # assume if not binary, it's ascii
        return filetype,version
    try:
        # Try it as a gzip file first
        with getFileObj(filename,compressed=True) as log:
            return parse_file(log)
    except IOError:
        with getFileObj(filename,compressed=False) as log:
            return parse_file(log)
