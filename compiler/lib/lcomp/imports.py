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

# Backport of singledispatch to Python 2.x.
try:
    from functools import singledispatch
except ImportError:
    from singledispatch import singledispatch

import os
from . import ast
from .clang import parse

@singledispatch
def augment_node(node, opts):
    raise Exception('Failed to augment imports for %s' % node)

@augment_node.register(ast.Program)
def _(node, opts):
    augment_node(node.definitions, opts)
    return

@augment_node.register(ast.Definitions)
def _(node, opts):
    for definition in node.definitions:
        augment_node(definition, opts)
    return

@augment_node.register(ast.Import)
def _(node, opts):
    filename = None
    for search_dir in opts.search_path:
        test = os.path.join(search_dir, node.filename)
        if os.path.exists(test):
            filename = test
            break
    if filename is None:
        raise Exception('Failed to locate file %s in search path %s' % (
                node.filename, opts.search_path))
    parser = parse.Parser()
    node.ast = parser.parse(filename, opts)
    return

@augment_node.register(ast.Struct)
def _(node, opts):
    return

@augment_node.register(ast.Function)
def _(node, opts):
    return

def augment_imports(program, opts):
    augment_node(program, opts)
