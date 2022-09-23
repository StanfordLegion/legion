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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import re
import struct
import gzip
import io
import sys
from abc import ABC
from typing import Union, Dict, List, Tuple, Callable, Type, ItemsView, Any

import legion_spy
from legion_util import typeassert, typecheck
from legion_prof import State

long_type = int

binary_filetype_pat = re.compile(b"FileType: BinaryLegionProf v: (?P<version>\d+(\.\d+)?)")

max_dim_val = 0

#TODO: fix return type
@typeassert(filename=str, compressed=bool, buffer_size=int)
def getFileObj(filename: str, compressed: bool =False, buffer_size: int =32768) -> Any: # type: ignore
    if compressed:
        return io.BufferedReader(gzip.open(filename, mode='rb'), buffer_size=buffer_size) #type: ignore
    else:
        return open(filename, mode='rb', buffering=buffer_size)

class LegionDeserializer(ABC):
    """This is a generic class for our deserializer"""
    __slots__ = ["state", "callbacks"]

    def __init__(self, state: State, callbacks: Dict[str, Callable]):
        """The constructor for the initializer.

        @param[state]:     The state object for our callbacks
        @param[callbacks]: A dictionary containing the callbacks we should use
                           after deserializing each item. You must pass a callback
                           for 
        """
        self.state = state
        self.callbacks: Dict[Any, Callable] = callbacks # type: ignore # Any is str or int

    def deserialize(self, filename: str) -> None:
        """ The generic deserialize method

        @param[filename]: The path to the binary file we want to deserialize
        """
        raise NotImplementedError

@typecheck
def read_time(string: str) -> float:
    return long_type(string) / 1000

@typecheck
def read_max_dim(string: str) -> int:
    global max_dim_val
    max_dim_val = int(string)
    return int(string)

decimal_pat = re.compile("\-?[0-9]+")
@typecheck
def read_array(string: str) -> List:
    values = decimal_pat.findall(string)
    return values

class LegionProfASCIIDeserializer(LegionDeserializer):
    """
    This is the serializer for the regex log files
    """

    #               node           thread
    prefix = r'\[(?:[0-9]+) - (?:[0-9a-f]+)\](?:\s+[0-9]+\.[0-9]+)? \{\w+\}\{legion_prof\}: '

    patterns = {
        "MapperCallDesc": re.compile(prefix + r'Prof Mapper Call Desc (?P<kind>[0-9]+) (?P<name>[a-zA-Z0-9_ ]+)'),
        "RuntimeCallDesc": re.compile(prefix + r'Prof Runtime Call Desc (?P<kind>[0-9]+) (?P<name>[a-zA-Z0-9_ ]+)'),
        "MetaDesc": re.compile(prefix + r'Prof Meta Desc (?P<kind>[0-9]+) (?P<message>[0-1]) (?P<ordered_vc>[0-1]) (?P<name>[a-zA-Z0-9_ ]+)'),
        "OpDesc": re.compile(prefix + r'Prof Op Desc (?P<kind>[0-9]+) (?P<name>[a-zA-Z0-9_ ]+)'),
        "ProcDesc": re.compile(prefix + r'Prof Proc Desc (?P<proc_id>[a-f0-9]+) (?P<kind>[0-9]+)'),
        "MemDesc": re.compile(prefix + r'Prof Mem Desc (?P<mem_id>[a-f0-9]+) (?P<kind>[0-9]+) (?P<capacity>[0-9]+)'),
        "ProcMDesc": re.compile(prefix + r'Prof Mem Proc Affinity Desc (?P<proc_id>[a-f0-9]+) (?P<mem_id>[a-f0-9]+) (?P<bandwidth>[0-9]+) (?P<latency>[0-9]+)'),
        "MaxDimDesc": re.compile(prefix + r'Max Dim Desc (?P<max_dim>[0-9]+)'),
        "IndexSpacePointDesc": re.compile(prefix + r'Index Space Point Desc (?P<unique_id>[0-9]+) (?P<dim>[0-9]+) (?P<rem>.*)'),
        "IndexSpaceRectDesc": re.compile(prefix + r'Index Space Rect Desc (?P<unique_id>[0-9]+) (?P<dim>[0-9]+) (?P<rem>.*)'),
        "IndexSpaceEmptyDesc": re.compile(prefix + r'Index Space Empty Desc (?P<unique_id>[0-9]+)'),
        "FieldDesc": re.compile(prefix + r'Field Name Desc (?P<unique_id>[0-9]+) (?P<field_id>[0-9]+) (?P<size>[0-9]+) (?P<name>[$()a-zA-Z0-9_<>.]+)'),
        "FieldSpaceDesc": re.compile(prefix + r'Field Space Name Desc (?P<unique_id>[0-9]+) (?P<name>[$()a-zA-Z0-9_<>.]+)'),
        "IndexSpaceDesc": re.compile(prefix + r'Index Space Name Desc (?P<unique_id>[a-f0-9]+) (?P<name>[$()a-zA-Z0-9_<>.]+)'),
        "PartDesc": re.compile(prefix + r'Index Part Name Desc (?P<unique_id>[0-9]+) (?P<name>[a-zA-Z0-9_ ]+)'),
        "IndexPartitionDesc": re.compile(prefix + r'Index Partition Desc (?P<parent_id>[0-9]+) (?P<unique_id>[0-9]+) (?P<disjoint>[0-1]+) (?P<point0>[0-9]+)'),
        "IndexSubSpaceDesc": re.compile(prefix + r'Index Sub Space Desc (?P<parent_id>[a-f0-9]+) (?P<unique_id>[0-9]+)'),
        "LogicalRegionDesc": re.compile(prefix + r'Logical Region Desc (?P<ispace_id>[0-9]+) (?P<fspace_id>[0-9]+) (?P<tree_id>[0-9]+) (?P<name>[a-zA-Z0-9_ ]+)'),
        "PhysicalInstRegionDesc": re.compile(prefix + r'Physical Inst Region Desc (?P<op_id>[0-9]+) (?P<inst_id>[a-f0-9]+) (?P<ispace_id>[0-9]+) (?P<fspace_id>[0-9]+) (?P<tree_id>[0-9]+)'),
        "PhysicalInstLayoutDesc": re.compile(prefix + r'Physical Inst Layout Desc (?P<op_id>[0-9]+) (?P<inst_id>[a-f0-9]+) (?P<field_id>[0-9]+) (?P<fspace_id>[0-9]+) (?P<has_align>[0-1]) (?P<eqk>[0-9]+) (?P<align_desc>[0-9]+)'),
        "PhysicalInstDimOrderDesc": re.compile(prefix + r'Physical Inst Dim Order Desc (?P<op_id>[0-9]+) (?P<inst_id>[a-f0-9]+) (?P<dim>[0-9]+) (?P<dim_kind>[0-9]+)'),
        "IndexSpaceSizeDesc": re.compile(prefix + r'Index Space Size Desc (?P<unique_id>[0-9]+) (?P<dense_size>[0-9]+) (?P<sparse_size>[0-9]+) (?P<is_sparse>[0-1])'),
        "TaskKind": re.compile(prefix + r'Prof Task Kind (?P<task_id>[0-9]+) (?P<name>[$()a-zA-Z0-9_<>., ]+) (?P<overwrite>[0-1])'),
        "TaskVariant": re.compile(prefix + r'Prof Task Variant (?P<task_id>[0-9]+) (?P<variant_id>[0-9]+) (?P<name>[$()a-zA-Z0-9_<>., ]+)'),
        "OperationInstance": re.compile(prefix + r'Prof Operation (?P<op_id>[0-9]+) (?P<parent_id>[0-9]+) (?P<kind>[0-9]+) (?P<provenance>[a-zA-Z0-9_ ]*)'),
        "MultiTask": re.compile(prefix + r'Prof Multi (?P<op_id>[0-9]+) (?P<task_id>[0-9]+)'),
        "SliceOwner": re.compile(prefix + r'Prof Slice Owner (?P<parent_id>[0-9]+) (?P<op_id>[0-9]+)'),
        "TaskWaitInfo": re.compile(prefix + r'Prof Task Wait Info (?P<op_id>[0-9]+) (?P<task_id>[0-9]+) (?P<variant_id>[0-9]+) (?P<wait_start>[0-9]+) (?P<wait_ready>[0-9]+) (?P<wait_end>[0-9]+)'),
        "MetaWaitInfo": re.compile(prefix + r'Prof Meta Wait Info (?P<op_id>[0-9]+) (?P<lg_id>[0-9]+) (?P<wait_start>[0-9]+) (?P<wait_ready>[0-9]+) (?P<wait_end>[0-9]+)'),
        "TaskInfo": re.compile(prefix + r'Prof Task Info (?P<op_id>[0-9]+) (?P<task_id>[0-9]+) (?P<variant_id>[0-9]+) (?P<proc_id>[a-f0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "GPUTaskInfo": re.compile(prefix + r'Prof GPU Task Info (?P<op_id>[0-9]+) (?P<task_id>[0-9]+) (?P<variant_id>[0-9]+) (?P<proc_id>[a-f0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+) (?P<gpu_start>[0-9]+) (?P<gpu_stop>[0-9]+)'),
        "MetaInfo": re.compile(prefix + r'Prof Meta Info (?P<op_id>[0-9]+) (?P<lg_id>[0-9]+) (?P<proc_id>[a-f0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "CopyInfo": re.compile(prefix + r'Prof Copy Info (?P<op_id>[0-9]+) (?P<src>[a-f0-9]+) (?P<dst>[a-f0-9]+) (?P<size>[0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+) (?P<fevent>[a-f0-9]+) (?P<num_requests>[0-9]+)'),
        "CopyInstInfo": re.compile(prefix + r'Prof Copy Inst Info (?P<op_id>[0-9]+) (?P<src_inst>[a-f0-9]+) (?P<dst_inst>[a-f0-9]+) (?P<fevent>[a-f0-9]+) (?P<num_fields>[0-9]+) (?P<request_type>[0-9]+) (?P<num_hops>[0-9]+)'),
        "FillInfo": re.compile(prefix + r'Prof Fill Info (?P<op_id>[0-9]+) (?P<dst>[a-f0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "InstCreateInfo": re.compile(prefix + r'Prof Inst Create (?P<op_id>[0-9]+) (?P<inst_id>[a-f0-9]+) (?P<create>[0-9]+)'),
        "InstUsageInfo": re.compile(prefix + r'Prof Inst Usage (?P<op_id>[0-9]+) (?P<inst_id>[a-f0-9]+) (?P<mem_id>[a-f0-9]+) (?P<size>[0-9]+)'),
        "InstTimelineInfo": re.compile(prefix + r'Prof Inst Timeline (?P<op_id>[0-9]+) (?P<inst_id>[a-f0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<destroy>[0-9]+)'),
        "PartitionInfo": re.compile(prefix + r'Prof Partition Timeline (?P<op_id>[0-9]+) (?P<part_op>[0-9]+) (?P<create>[0-9]+) (?P<ready>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "MapperCallInfo": re.compile(prefix + r'Prof Mapper Call Info (?P<kind>[0-9]+) (?P<proc_id>[a-f0-9]+) (?P<op_id>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "RuntimeCallInfo": re.compile(prefix + r'Prof Runtime Call Info (?P<kind>[0-9]+) (?P<proc_id>[a-f0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)'),
        "ProfTaskInfo": re.compile(prefix + r'Prof ProfTask Info (?P<proc_id>[a-f0-9]+) (?P<op_id>[0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+)')
        # "UserInfo": re.compile(prefix + r'Prof User Info (?P<proc_id>[a-f0-9]+) (?P<start>[0-9]+) (?P<stop>[0-9]+) (?P<name>[$()a-zA-Z0-9_]+)')
    }
    parse_callbacks = {
        "op_id": long_type,
        "parent_id": long_type,
        "size": long_type,
        "capacity": long_type,
        "variant_id": int,
        "lg_id": int,
        "uid": int,
        "overwrite": int,
        "task_id": int,
        "kind": int,
        "opkind": int,
        "part_op": int,
        "point": int,
        "bandwidth": int,
        "latency": int,
        "point0": long_type,
        "point1": long_type,
        "point2": long_type,
        "dim": int,
        "field_id": int,
        "fspace_id": int,
        "ispace_id": long_type,
        "unique_id": long_type,
        "disjoint": bool,
        "has_align": int,
        "is_sparse": int,
        "tree_id": int,
        "max_dim": read_max_dim,
        "rem": read_array,
        "proc_id": lambda x: int(x, 16),
        "mem_id": lambda x: int(x, 16),
        "src": lambda x: int(x, 16),
        "dst": lambda x: int(x, 16),
        "src_inst": lambda x: int(x, 16),
        "dst_inst": lambda x: int(x, 16),
        "inst_id": lambda x: int(x, 16),
        "fevent": lambda x: int(x, 16),
        "create": read_time,
        "destroy": read_time,
        "start": read_time,
        "ready": read_time,
        "stop": read_time,
        "end": read_time,
        "gpu_start": read_time,
        "gpu_stop": read_time,
        "wait_start": read_time,
        "wait_ready": read_time,
        "wait_end": read_time,
        "align_desc": int,
        "eqk": int,
        "dim_kind": int,
        "dense_size": long_type,
        "sparse_size": long_type,
        "name": lambda x: x,
        "num_fields": int,
        "num_requests": int,
        "request_type": int,
        "num_hops": int,
        "message" : bool,
        "ordered_vc" : bool,
        "desc": lambda x: x,
        "provenance": lambda x: x
    }

    def __init__(self, state: State, callbacks: Dict[str, Callable]) -> None:
        LegionDeserializer.__init__(self, state, callbacks)
        assert len(callbacks) == len(LegionProfASCIIDeserializer.patterns)
        callbacks_valid = True
        for callback_name, callback in callbacks.items():
            cur_valid = callback_name in LegionProfASCIIDeserializer.patterns and \
                        callable(callback)
            callbacks_valid = callbacks_valid and cur_valid
        assert callbacks_valid

    # @typeassert(m=re.Match)
    def parse_regex_matches(self, m) -> Dict[str, Any]: # type: ignore
        kwargs = m.groupdict()

        for key, arg in kwargs.items():
            kwargs[key] = LegionProfASCIIDeserializer.parse_callbacks[key](arg)
        return kwargs

    @typecheck
    def search_for_spy_data(self, filename: str) -> None:
        with open(filename, 'rb') as log:
            for line_bytes in log:
                line = line_bytes.decode('utf-8')
                if legion_spy.config_pat.match(line) or \
                   legion_spy.detailed_config_pat.match(line):
                   self.state.has_spy_data = True
                   break

    @typecheck
    def parse(self, filename: str, verbose: bool) -> int:
        skipped = 0
        with open(filename, 'rb') as log:
            matches = 0
            # Keep track of the first and last times
            first_time = 0
            last_time = 0
            for line_bytes in log:
                line = line_bytes.decode('utf-8')
                if not self.state.has_spy_data and \
                    (legion_spy.config_pat.match(line) or \
                     legion_spy.detailed_config_pat.match(line)):
                    self.state.has_spy_data = True
                matched = False

                for prof_event, pattern in LegionProfASCIIDeserializer.patterns.items():
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

    preamble_data: Dict[int, List[Tuple[str, Callable]]] = {}
    name_to_id: Dict[str, int] = {}

    # XXX: Make sure these are consistent with legion_profiling.h and legion_types.h!
    fmt_dict = {
        "ProcID":             "Q", # unsigned long long
        "MemID":              "Q", # unsigned long long
        "InstID":             "Q", # unsigned long long
        "UniqueID":           "Q", # unsigned long long
        "IDType":             "Q", # unsigned long long
        "TaskID":             "I", # unsigned int
        "bool":               "?", # bool
        "VariantID":          "I", # unsigned int
        "unsigned":           "I", # unsigned int
        "timestamp_t":        "Q", # unsigned long long
        "maxdim":             "i", # int
        "unsigned long long": "Q", # unsigned long long
        "long long":          "q", # long long
        "array":              "Q", # unsigned long long
        "point":              "Q", # unsigned long long
        "int":                "i", # int
        "ProcKind":           "i", # int (really an enum so this depends)
        "MemKind":            "i", # int (really an enum so this depends)
        "MappingCallKind":    "i", # int (really an enum so this depends)
        "RuntimeCallKind":    "i", # int (really an enum so this depends)
        "DepPartOpKind":      "i", # int (really an enum so this depends)
    }

    def __init__(self, state: State, callbacks: Dict[str, Callable]) -> None:
        LegionDeserializer.__init__(self, state, callbacks)
        self.callbacks_translated = False

    @staticmethod
    @typecheck
    def create_type_reader(num_bytes: int, param_type: str
    ) -> Callable[[io.BufferedReader], Union[str, List]]:
        if param_type == "string":
            def string_reader(log: io.BufferedReader) -> str:
                string = ""
                char = log.read(1).decode('utf-8')
                while ord(char) != 0:
                    string += char
                    char = log.read(1).decode('utf-8')
                return string
            return string_reader
        if param_type == "point":
            fmt = LegionProfBinaryDeserializer.fmt_dict[param_type]
            def point_reader(log: io.BufferedReader) -> List:
                global max_dim_val
                values = []
                for index in range(max_dim_val):
                    raw_val = log.read(num_bytes)
                    value = struct.unpack(fmt, raw_val)[0]
                    values.append(value)
                return values
            return point_reader
        if param_type == "array":
            fmt = LegionProfBinaryDeserializer.fmt_dict[param_type]
            def array_reader(log: io.BufferedReader) -> List:
                global max_dim_val
                values = []
                for index in range(max_dim_val*2):
                    raw_val = log.read(num_bytes)
                    value = struct.unpack(fmt, raw_val)[0]
                    values.append(value)
                return values
            return array_reader
        else:
            fmt = LegionProfBinaryDeserializer.fmt_dict[param_type]
            def reader(log: io.BufferedReader) -> str:
                global max_dim_val
                raw_val = log.read(num_bytes)
                val = struct.unpack(fmt, raw_val)[0]
                if param_type == "timestamp_t":
                    val = val / 1000
                if param_type == "maxdim":
                    max_dim_val = val
                return val
            return reader

    @typecheck
    def parse_preamble(self, log: io.BufferedReader) -> None:
        log.readline() # filetype
        while(True):
            line = log.readline().decode('utf-8')
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
                               for name, callback in self.callbacks.items()
                               if name in LegionProfBinaryDeserializer.name_to_id}
            self.callbacks = new_callbacks
            self.callbacks_translated = True


        # callbacks_valid = True
        # for callback_name, callback in callbacks.items():
        #     cur_valid = callback_name in LegionProfASCIIDeserializer.patterns and \
        #                 callable(callback)
        #     callbacks_valid = callbacks_valid and cur_valid
        # assert callbacks_valid

    @typecheck
    def parse(self, filename: str, verbose: bool) -> int:
        print("parsing " + str(filename))
        def parse_file(log: io.BufferedReader) -> int:
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

@typecheck
def GetFileTypeInfo(filename: str) -> Tuple[str, Union[bytes, None]]:
    def parse_file(log: io.BufferedReader) -> Tuple[str, Union[bytes, None]]:
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
