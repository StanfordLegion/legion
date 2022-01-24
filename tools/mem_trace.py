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
#

import re
import sys
from enum import IntEnum

malloc_pat = re.compile("malloc\((?P<size>[0-9]+)\) = 0x(?P<ptr>[0-9a-f]+)")
free_pat = re.compile("free\((?P<ptr>0x[0-9a-f]+)\)")
nill_pat = re.compile("free\(\(nil\)\)")
calloc_pat = re.compile("calloc\((?P<align>[0-9]+), (?P<size>[0-9]+)\) = 0x(?P<ptr>[0-9a-f]+)")
realloc_pat = re.compile("realloc\(0x(?P<p1>[0-9a-f]+), (?P<size>[0-9]+)\) = 0x(?P<p2>[0-9a-f]+)")
memalign_pat = re.compile("memalign\((?P<align>[0-9]+), (?P<size>[0-9]+)\) = 0x(?P<ptr>[0-9a-f]+)")
posixmemalign_pat = re.compile("posix_memalign\(0x(?P<p1>[0-9a-f]+), (?P<align>[0-9]+), (?P<size>[0-9]+)\) = 0x(?P<p2>[0-9a-f]+)")
valloc_pat = re.compile("malloc\((?P<size>[0-9]+)\) = 0x(?P<ptr>[0-9a-f]+)") 

class AllocKind(IntEnum):
    MALLOC_KIND = 1
    CALLOC_KIND = 2
    REALLOC_KIND = 3
    MEMALIGN_KIND = 4
    POSIXMEMALIGN_KIND = 5
    VALLOC_KIND = 6

class Allocation(object):
    def __init__(self, ptr, size, kind, alignment = 16):
        self.ptr = ptr
        self.size = size
        self.kind = kind
        self.alignment = alignment

def check_leaks(file):
    allocations = dict()
    for line in file:
        m = malloc_pat.match(line)
        if m is not None:
            ptr = int(m.group('ptr'),16)
            allocations[ptr] = Allocation(ptr, int(m.group('size')), AllocKind.MALLOC_KIND)
            continue
        m = free_pat.match(line)
        if m is not None:
            ptr = int(m.group('ptr'),16)
            try:
                del allocations[ptr]
            except KeyError:
                pass
            continue
        m = nill_pat.match(line)
        if m is not None:
            continue
        m = posixmemalign_pat.match(line)
        if m is not None:
            ptr = int(m.group('p2'),16)
            allocations[ptr] = Allocation(ptr, int(m.group('size')), AllocKind.POSIXMEMALIGN_KIND, int(m.group('align')))
            continue
        m = calloc_pat.match(line)
        if m is not None:
            ptr = int(m.group('ptr'),16)
            allocations[ptr] = Allocation(ptr, int(m.group('size')), AllocKind.CALLOC_KIND, int(m.group('align')))
            continue
        m = realloc_pat.match(line)
        if m is not None:
            ptr = int(m.group('p2'),16)
            allocations[ptr] = Allocation(ptr, int(m.group('size')), AllocKind.REALLOC_KIND)
            continue
        m = memalign_pat.match(line)
        if m is not None:
            ptr = int(m.group('ptr'),16)
            allocations[ptr] = Allocation(ptr, int(m.group('size')), AllocKind.MEMALIGN_KIND, int(m.group('align')))
            continue
        m = valloc_pat.match(line)
        if m is not None:
            ptr = int(m.group('ptr'),16)
            allocations[ptr] = Allocation(ptr, int(m.group('size')), AllocKind.VALLOC_KIND)
            continue
        print("WARNING! Unable to match line: "+line.strip())
    print("There are "+str(len(allocations))+" live allocations")
    # Group allocations by sizes
    groups = dict()
    for alloc in allocations.values():
        try:
            groups[alloc.size].append(alloc)
        except KeyError:
            group = list()
            group.append(alloc)
            groups[alloc.size] = group
    group_sizes = [(size,size*len(group)) for size,group in groups.items()]
    for size,total in reversed(sorted(group_sizes, key=lambda x: x[1])):
        print("Allocation Size: "+str(size))
        print("  Total live allocations: "+str(len(groups[size])))
        print("  Total live memory: "+str(total))

if __name__ == "__main__":
    assert len(sys.argv) > 1
    print("Checking "+sys.argv[1]+" for leaks")
    with open(sys.argv[1], 'r') as file:
        check_leaks(file)
