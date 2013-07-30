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
### Parser for C++
###

try:
    import clang.cindex
except ImportError:
    # If Clang is missing, throw an error on first use rather than
    # throw an error when loading.
    class ClangMissingGuard:
        @property
        def cindex(self):
            raise Exception('Unable to find Clang bindings for Python')
    clang = ClangMissingGuard()

import os
from . import ast

class ForeignParseException(Exception):
    pass

def parse_translation_unit(node):
    definitions = []
    for child in node.get_children():
        try:
            if child.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                definitions.append(parse_function_decl(child))
            elif child.kind == clang.cindex.CursorKind.TYPEDEF_DECL:
                definitions.append(parse_typedef_decl(child))
        except ForeignParseException as e:
            # skip any definitions that cannot be parsed
            pass
    return ast.Program(definitions = definitions)

def parse_function_decl(node):
    function_type = parse_type(node.type)
    return ast.Function(name = node.spelling, type = function_type)

def parse_typedef_decl(node):
    typedef_type = parse_type(node.type)
    return ast.Typedef(name = node.spelling, type = typedef_type)

def parse_type(node):
    # machine-dependent signed integers
    if node.kind == clang.cindex.TypeKind.CHAR_S:
        return ast.TypeChar()
    if node.kind == clang.cindex.TypeKind.SCHAR:
        return ast.TypeChar()
    if node.kind == clang.cindex.TypeKind.SHORT:
        return ast.TypeShort()
    if node.kind == clang.cindex.TypeKind.INT:
        return ast.TypeInt()
    if node.kind == clang.cindex.TypeKind.LONG:
        return ast.TypeLong()
    if node.kind == clang.cindex.TypeKind.LONGLONG:
        return ast.TypeLongLong()

    # fixed-size signed integers
    if node.kind == clang.cindex.TypeKind.INT128:
        return ast.TypeInt128()

    # machine-dependent unsigned integers
    if node.kind == clang.cindex.TypeKind.CHAR_U:
        return ast.TypeUChar()
    if node.kind == clang.cindex.TypeKind.UCHAR:
        return ast.TypeUChar()
    if node.kind == clang.cindex.TypeKind.USHORT:
        return ast.TypeUShort()
    if node.kind == clang.cindex.TypeKind.UINT:
        return ast.TypeUInt()
    if node.kind == clang.cindex.TypeKind.ULONG:
        return ast.TypeULong()
    if node.kind == clang.cindex.TypeKind.ULONGLONG:
        return ast.TypeULongLong()

    # fixed-size unsigned integers
    if node.kind == clang.cindex.TypeKind.UINT128:
        return ast.TypeUInt128()

    # machine-dependent floating point
    if node.kind == clang.cindex.TypeKind.FLOAT:
        return ast.TypeFloat()
    if node.kind == clang.cindex.TypeKind.DOUBLE:
        return ast.TypeDouble()
    if node.kind == clang.cindex.TypeKind.LONGDOUBLE:
        return ast.TypeLongDouble()

    # composite types
    if node.kind == clang.cindex.TypeKind.FUNCTIONPROTO:
        return ast.TypeFunction(
            [parse_type(arg) for arg in node.argument_types()],
            parse_type(node.get_result()))
    if node.kind == clang.cindex.TypeKind.FUNCTIONNOPROTO:
        return ast.TypeFunction(
            [],
            parse_type(node.get_result()))
    if node.kind == clang.cindex.TypeKind.POINTER:
        points_to_type = parse_type(node.get_pointee())
        return ast.TypePointer(points_to_type = points_to_type)
    if node.kind == clang.cindex.TypeKind.TYPEDEF:
        # Hack: In stdint.h, the fixed-size integer types (int8_t,
        # etc.) are defined via typedefs on the standard C types
        # (signed char, etc.). However, we can't rely on the
        # properties of these types because we don't know what
        # compiler will eventually be used. So we have to translate
        # those types here, before they are lowered to anything else.
        #
        # As a safety check, we only do this if the file that declared
        # the types was named stdint.h.

        decl = node.get_declaration()
        filename = decl.location.file
        is_stdint = False
        if filename is not None:
            is_stdint = (os.path.basename(filename.name) == 'stdint.h')

        if is_stdint:
            name = decl.spelling

            # machine-dependent signed integers
            if name == 'intptr_t':
                return ast.TypeIntPtr()

            # fixed-size signed integers
            if name == 'int8_t':
                return ast.TypeInt8()
            if name == 'int16_t':
                return ast.TypeInt16()
            if name == 'int32_t':
                return ast.TypeInt32()
            if name == 'int64_t':
                return ast.TypeInt64()

            # machine-dependent signed integers
            if name == 'uintptr_t':
                return ast.TypeUIntPtr()

            # fixed-size unsigned integers
            if name == 'uint8_t':
                return ast.TypeUInt8()
            if name == 'uint16_t':
                return ast.TypeUInt16()
            if name == 'uint32_t':
                return ast.TypeUInt32()
            if name == 'uint64_t':
                return ast.TypeUInt64()

        return parse_type(node.get_canonical())

    # miscellaneous
    if node.kind == clang.cindex.TypeKind.VOID:
        return ast.TypeVoid()
    raise ForeignParseException('Failed to parse %s' % node.kind)

class Parser:
    def __init__(self):
        self.index = clang.cindex.Index.create()

    def parse(self, src):
        translation_unit = self.index.parse(src)
        return parse_translation_unit(translation_unit.cursor)
