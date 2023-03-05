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

import argparse
from typing import List

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument("file", type=argparse.FileType("r"))
parser.add_argument("pattern", nargs="+")
args = parser.parse_args()

# Collect blocks, strip off log line header (if present)
Block = List[str]
blocks: List[Block] = []
for line in args.file:
    parts = line[:-1].split("}: ")
    assert len(parts) <= 2
    line = parts[-1]
    if line.startswith(" "):
        assert len(blocks) > 0
        assert len(blocks[-1]) > 0
        blocks[-1].append(line)
    else:
        blocks.append([line])

# Print out blocks that contain all patterns
for block in blocks:
    found_all: bool = True
    for pat in args.pattern:
        found: bool = False
        for line in block:
            if pat in line:
                found = True
                break
        if not found:
            found_all = False
            break
    if found_all:
        for line in block:
            print(line)
