#!/usr/bin/env python

# Copyright 2013 Stanford University and Los Alamos National Security, LLC
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

###
### Compiler Passes
###

from . import imports, leaf_task_analysis, parse as _parse, region_analysis, trans, type_check

_default_search_path = ('/usr/include', '/usr/local/include')

def parse(input_file):
    parser = _parse.Parser()
    program = parser.parse(input_file)
    return program

def check(program, opts):
    opts = opts.with_search_path(
        _default_search_path +
        ((opts.legion_runtime_dir,) if opts.legion_runtime_dir is not None else ()) +
        opts.search_path)
    imports.augment_imports(program, opts)
    type_map, constraints, foreign_types = type_check.type_check(program, opts)
    return type_map, constraints, foreign_types

def compile(program, opts):
    type_map, constraints, foreign_types = check(program, opts)
    region_usage = region_analysis.region_analysis(program, opts, type_map)
    leaf_tasks = leaf_task_analysis.leaf_task_analysis(program, opts, type_map)
    return trans.trans(program, opts, type_map, constraints, foreign_types, region_usage, leaf_tasks)
