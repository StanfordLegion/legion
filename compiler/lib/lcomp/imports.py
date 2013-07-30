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
### Augments imports with ASTs of foreign code
###

import os
from . import ast
from .clang import parse

def augment_node(node, search_path):
    if isinstance(node, ast.Program):
        for definition in node.defs:
            augment_node(definition, search_path)
        return
    if isinstance(node, ast.Import):
        filename = None
        for search_dir in search_path:
            test = os.path.join(search_dir, node.filename)
            if os.path.exists(test):
                filename = test
                break
        if filename is None:
            raise Exception('Failed to locate file %s in search path %s' % (
                    node.filename, search_path))
        parser = parse.Parser()
        node.ast = parser.parse(filename)
        return
    if isinstance(node, ast.Struct):
        return
    if isinstance(node, ast.Function):
        return
    raise Exception('Failed to augment imports for %s' % node)

def augment_imports(node, search_path):
    augment_node(node, search_path)
