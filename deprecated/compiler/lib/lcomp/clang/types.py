#!/usr/bin/env python

# Copyright 2014 Stanford University and Los Alamos National Security, LLC
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
### Legion Types for C++
###

# Work around for OrderedDict missing in Python 2.6.
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import sys
from . import ast
from .. import types

class UnknownForeignTypeException(Exception):
    pass

def foreign_type(node, opts):
    if isinstance(node, ast.Program):
        def_types = []
        for definition in node.definitions:
            try:
                def_types.append(foreign_type(definition, opts))
            except UnknownForeignTypeException:
                # Skip any definitions that cannot be translated.
                name = definition.name
                if opts.allow_warning() and len(name) > 0 and not name.startswith('_'):
                    sys.stderr.write('WARNING: Skipping import of %s\n' % name)
                    sys.stderr.flush()
        return types.Module(OrderedDict(def_types))
    if isinstance(node, ast.Function):
        return (node.name, foreign_type(node.type, opts))
    if isinstance(node, ast.Typedef):
        return (node.name, types.Kind(foreign_type(node.type, opts)))
    if isinstance(node, ast.Struct):
        return (node.name, types.Kind(foreign_type(node.type, opts)))

    # boolean
    if isinstance(node, ast.TypeBool):
        return types.Bool()

    # floating-point
    if isinstance(node, ast.TypeDouble):
        return types.Double()
    if isinstance(node, ast.TypeFloat):
        return types.Float()

    # machine-dependent signed integers
    if isinstance(node, ast.TypeIntPtr):
        return types.Int()

    # machine-dependent unsigned integers
    if isinstance(node, ast.TypeUIntPtr):
        return types.UInt()

    # fixed-size signed integers
    if isinstance(node, ast.TypeInt8):
        return types.Int8()
    if isinstance(node, ast.TypeInt16):
        return types.Int16()
    if isinstance(node, ast.TypeInt32):
        return types.Int32()
    if isinstance(node, ast.TypeInt64):
        return types.Int64()

    # fixed-size unsigned integers
    if isinstance(node, ast.TypeUInt8):
        return types.UInt8()
    if isinstance(node, ast.TypeUInt16):
        return types.UInt16()
    if isinstance(node, ast.TypeUInt32):
        return types.UInt32()
    if isinstance(node, ast.TypeUInt64):
        return types.UInt64()

    # special Legion constructs
    if isinstance(node, ast.TypeLegionColoring):
        return types.ForeignColoring()
    if isinstance(node, ast.TypeLegionContext):
        return types.ForeignContext()
    if isinstance(node, ast.TypeLegionPointer):
        return types.ForeignPointer()

    # functions
    if isinstance(node, ast.TypeFunction):
        foreign_param_types = [foreign_type(param, opts) for param in node.param_types]
        param_types = [param for param in foreign_param_types
                       if not (types.is_foreign_context(param) or
                               types.is_foreign_runtime(param))]
        return_type = foreign_type(node.return_type, opts)
        function_type = types.ForeignFunction(foreign_param_types, param_types, [], return_type)
        return function_type

    # pointers
    if isinstance(node, ast.TypePointer):
        # special Legion constructs
        if isinstance(node.points_to_type, ast.TypeLegionRegion):
            return types.ForeignRegion()
        if isinstance(node.points_to_type, ast.TypeLegionRuntime):
            return types.ForeignRuntime()

    # arrays
    if isinstance(node, ast.TypeArray):
        # special Legion constructs
        if isinstance(node.element_type, ast.TypeLegionRegion):
            return types.ForeignRegion()

    # structs
    if isinstance(node, ast.TypeStruct):
        name = node.name
        fields = []
        for field, field_type in node.fields:
            # Skip importing methods of the struct.
            if isinstance(field_type, ast.TypeFunction):
                continue
            fields.append((field, foreign_type(field_type, opts)))
        field_map = OrderedDict(fields)
        return types.Struct(name, [], [], set(), field_map)

    if isinstance(node, ast.TypeVoid):
        return types.Void()
    raise UnknownForeignTypeException('Failed to translate C++ type %s' % node)
