#!/usr/bin/env python3

# Copyright 2023 Stanford University, NVIDIA Corporation
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

# Normalize mapper logs, for easier comparison across runs.
# TODO: normalize field IDs, instance names

from ast import literal_eval
from fileinput import input
from re import search, sub
from typing import Dict, List, Union, Tuple

# Collect blocks, strip off log line header (if present)
Block = List[str]
blocks: List[Block] = []
for line in input():
    parts = line[:-1].split("}: ")
    assert len(parts) <= 2
    line = parts[-1]
    if line.startswith(" "):
        assert len(blocks) > 0
        assert len(blocks[-1]) > 0
        blocks[-1].append(line)
    else:
        blocks.append([line])

# Sort blocks by operation ID
def get_op_id(block: Block) -> int:
    m = search(r"<([0-9]+)>", block[0])
    return 0 if m is None else int(m.group(1))
blocks.sort(key=get_op_id)

# Sort by index_point within blocks of point operations
def get_callback_name(block: Block) -> str:
    return block[0].split()[0]
OperationString = Tuple[str, Union[str, None]]
def get_op_string(block: Block) -> OperationString:
    m = search(r"for ([0-9a-zA-Z_:]+)", block[0])
    if m is None:
        return ("", "")
    op_name = m.group(1)
    parts = block[0].split("@")
    assert len(parts) <= 2
    provenance = None if len(parts) == 1 else parts[1][1:]
    return (op_name, provenance)
groups: List[List[Block]] = []
latest_gid_for_op: Dict[OperationString, int] = {}
for block in blocks:
    op_string = get_op_string(block)
    callback_name = get_callback_name(block)
    gid = latest_gid_for_op.get(op_string)
    if (
        gid is None
        or callback_name == "SELECT_SHARDING_FUNCTOR"
        or (
            callback_name == "SLICE_TASK"
            and any(
                get_callback_name(block) == "SLICE_TASK"
                for block in groups[gid]
            )
        )
    ):
        gid = len(groups)
        groups.append([])
        latest_gid_for_op[op_string] = gid
    groups[gid].append(block)
def get_sort_key(block: Block) -> Tuple[Tuple[int,...], str]:
    m = search(r"\(index_point=(\([0-9,]+\))\)", block[0])
    index_point = (0,)
    if m is not None:
        p = literal_eval(m.group(1))
        index_point = (p,) if type(p) == int else p
    callback_name = get_callback_name(block)
    return (index_point, callback_name)
for group in groups:
    group.sort(key=get_sort_key)
blocks = [block for group in groups for block in group]

# Remove identifiers that are not stable between runs
def remove_run_specific_ids(line: str) -> str:
    if line.startswith(" "):
        # remove region triples
        line = sub(
            r"region=\([0-9]+,\([0-9]+,[0-9]+\),[0-9]+\)",
            "region=()",
            line
        )
        line = sub(r"region=\([0-9]+,\*,[0-9]+\)", "region=()", line)
    else:
        # remove operation IDs
        line = sub(r"<[0-9]+>", "<>", line)
    return line
blocks = [
    [remove_run_specific_ids(line) for line in block]
    for block in blocks
]

# Print final result
for block in blocks:
    for line in block:
        print(line)
