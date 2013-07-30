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
### Legion Types for C++
###

# Work around for OrderedDict missing in Python 2.6.
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from . import ast
from .. import types

class UnknownForeignTypeException(Exception):
    pass

def foreign_type(node):
    if isinstance(node, ast.Program):
        def_types = []
        for definition in node.definitions:
            try:
                def_types.append(foreign_type(definition))
            except UnknownForeignTypeException:
                # skip any definitions that cannot be translated
                print 'WARNING: Skipping import of %s' % definition
        return types.Module(OrderedDict(def_types))
    if isinstance(node, ast.Function):
        return (node.name, foreign_type(node.type))
    if isinstance(node, ast.TypeFunction):
        param_types = [foreign_type(param) for param in node.param_types]
        return_type = foreign_type(node.return_type)
        function_type = types.Function(param_types, [], return_type)
        return function_type

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

    if isinstance(node, ast.TypeVoid):
        return types.Void()
    raise UnknownForeignTypeException('Failed to translate C++ type %s' % node)
