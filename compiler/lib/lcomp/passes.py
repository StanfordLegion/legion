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

from . import imports, lower_expressions, parse as _parse, region_analysis, trans, type_check

_default_search_path = ('/usr/include', '/usr/local/include')

def parse(input_file):
    parser = _parse.Parser()
    program = parser.parse(input_file)
    return program

def check(program, search_path):
    imports.augment_imports(program, _default_search_path + search_path)
    type_map, constraints = type_check.type_check(program)
    return type_map, constraints

def lower(program, search_path):
    type_map, constraints = check(program, search_path)
    return lower_expressions.lower(program, type_map)

def compile(program, search_path):
    type_map, constraints = check(program, search_path)
    region_usage = region_analysis.region_analysis(program, type_map)
    return trans.trans(program, type_map, constraints, region_usage)
