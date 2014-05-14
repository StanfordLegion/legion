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
### AST for C++
###

def equivalent(a, b):
    if a == b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        for a_elt, b_elt in zip(a, b):
            if not equivalent(a_elt, b_elt):
                return False
        return True
    if isinstance(a, ASTNode) and isinstance(b, ASTNode):
        if a.__class__ is not b.__class__:
            return False
        for k in a.__dict__.iterkeys():
            if k not in b.__dict__ or not equivalent(a.__dict__[k], b.__dict__[k]):
                return False
        return True
    raise Exception('Equivalence undefined for type %s %s' % type(a))

class ASTNode:
    def __repr__(self):
        return '%s(%s)' % (
            self.__class__.__name__,
            ', '.join('%s: %s' % (key, value) for key, value in self.__dict__.iteritems()
                      if not key.startswith('_'))
            )

class Program(ASTNode):
    def __init__(self, definitions):
        self.definitions = definitions

class Function(ASTNode):
    def __init__(self, name, type):
        self.name = name
        self.type = type

class Typedef(ASTNode):
    def __init__(self, name, type):
        self.name = name
        self.type = type

class Struct(ASTNode):
    def __init__(self, name, type):
        self.name = name
        self.type = type

# boolean
class TypeBool(ASTNode): pass

# machine-dependent signed integers
class TypeChar(ASTNode): pass
class TypeShort(ASTNode): pass
class TypeInt(ASTNode): pass
class TypeLong(ASTNode): pass
class TypeLongLong(ASTNode): pass
class TypeIntPtr(ASTNode): pass

# fixed-size signed integers
class TypeInt8(ASTNode): pass
class TypeInt16(ASTNode): pass
class TypeInt32(ASTNode): pass
class TypeInt64(ASTNode): pass
class TypeInt128(ASTNode): pass

# machine-dependent unsigned integers
class TypeUChar(ASTNode): pass
class TypeUShort(ASTNode): pass
class TypeUInt(ASTNode): pass
class TypeULong(ASTNode): pass
class TypeULongLong(ASTNode): pass
class TypeUIntPtr(ASTNode): pass

# fixed-size unsigned integers
class TypeUInt8(ASTNode): pass
class TypeUInt16(ASTNode): pass
class TypeUInt32(ASTNode): pass
class TypeUInt64(ASTNode): pass
class TypeUInt128(ASTNode): pass

# machine-dependent floating point
class TypeFloat(ASTNode): pass
class TypeDouble(ASTNode): pass
class TypeLongDouble(ASTNode): pass

# Legion-defined types
class TypeLegionColoring(ASTNode): pass
class TypeLegionContext(ASTNode): pass
class TypeLegionPointer(ASTNode): pass
class TypeLegionRegion(ASTNode): pass
class TypeLegionRuntime(ASTNode): pass

class TypeFunction(ASTNode):
    def __init__(self, param_types, return_type):
        self.param_types = param_types
        self.return_type = return_type

class TypePointer(ASTNode):
    def __init__(self, points_to_type):
        self.points_to_type = points_to_type

class TypeArray(ASTNode):
    def __init__(self, element_type, size):
        self.element_type = element_type
        self.size = size

class TypeStruct(ASTNode):
    def __init__(self, name, fields):
        self.name = name
        self.fields = fields

class TypeVoid(ASTNode): pass
