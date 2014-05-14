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

import os, sys
from . import ast

_typedefs = {
    # legion_types.h
    'Coloring': ast.TypeLegionColoring(),
    'Context': ast.TypeLegionContext(),

    # stdint.h
    'int8_t': ast.TypeInt8(),
    'int16_t': ast.TypeInt16(),
    'int32_t': ast.TypeInt32(),
    'int64_t': ast.TypeInt64(),
    'intptr_t': ast.TypeIntPtr(),
    'uint8_t': ast.TypeUInt8(),
    'uint16_t': ast.TypeUInt16(),
    'uint32_t': ast.TypeUInt32(),
    'uint64_t': ast.TypeUInt64(),
    'uintptr_t': ast.TypeUIntPtr(),
}

_structs = {
    # 'common.h'
    'ptr_t': ast.TypeLegionPointer(),

    # 'legion.h'
    'HighLevelRuntime': ast.TypeLegionRuntime(),
    'PhysicalRegion': ast.TypeLegionRegion(),
}

class ForeignParseException(Exception):
    pass

def parse_translation_unit(node, opts):
    defs = []
    for decl in node.get_children():
        defs.extend(parse_decl(decl, opts))
    return ast.Program(definitions = defs)

def parse_decl(node, opts):
    try:
        if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            return [parse_function_decl(node)]
        elif node.kind == clang.cindex.CursorKind.TYPEDEF_DECL:
            return [parse_typedef_decl(node)]
        elif node.kind == clang.cindex.CursorKind.STRUCT_DECL:
            return [parse_struct_decl(node)]
        elif node.kind == clang.cindex.CursorKind.UNEXPOSED_DECL:
            defs = []
            for decl in node.get_children():
                defs.extend(parse_decl(decl, opts))
            return defs
        else:
            raise ForeignParseException('Failed to parse %s' % node.spelling)
    except ForeignParseException as e:
        # skip any definitions that cannot be parsed
        name = node.spelling
        if opts.allow_warning() and len(name) > 0 and not name.startswith('_'):
            sys.stderr.write('WARNING: Skipping import of %s\n' % name)
            sys.stderr.flush()
        return []

def parse_function_decl(node):
    function_type = parse_type(node.type)
    return ast.Function(name = node.spelling, type = function_type)

def parse_struct_decl(node):
    struct_type = parse_type(node.type)
    return ast.Struct(name = node.spelling, type = struct_type)

def parse_typedef_decl(node):
    typedef_type = parse_type(node.type)
    return ast.Typedef(name = node.spelling, type = typedef_type)

def parse_type(node):
    # boolean
    if node.kind == clang.cindex.TypeKind.BOOL:
        return ast.TypeBool()

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
    if node.kind == clang.cindex.TypeKind.CONSTANTARRAY:
        element_type = parse_type(node.get_array_element_type())
        return ast.TypeArray(element_type, node.get_array_size())
    if node.kind == clang.cindex.TypeKind.TYPEDEF:
        # Hack: In stdint.h, the fixed-size integer types (int*_t) are
        # defined via typedefs on the standard C types (signed char,
        # etc.). However, we can't rely on the properties of these
        # types because we don't know what compiler will eventually be
        # used. So we have to translate those types here, before they
        # are lowered to anything else.
        #
        # Unfortunately, when we see a typedef for int*_t, we cannot
        # even verify that the type was defined in stdint.h, because
        # the header files are implementation dependent. Therefore we
        # just trust the user not to cause any name conflicts with
        # int*_t in the global namespace.

        name = node.get_declaration().spelling
        if name in _typedefs:
            return _typedefs[name]

        return parse_type(node.get_canonical())
    if node.kind == clang.cindex.TypeKind.RECORD:
        return parse_type_struct(node.get_declaration())
    if node.kind == clang.cindex.TypeKind.UNEXPOSED:
        # For some reason Clang seems to have multiple ways of
        # declaring struct types, one of which results in an UNEXPOSED
        # type. If the definition looks like a struct, parse it as a
        # struct.
        definition = node.get_declaration()
        if definition.kind == clang.cindex.CursorKind.STRUCT_DECL:
            return parse_type_struct(node.get_declaration())

    # miscellaneous
    if node.kind == clang.cindex.TypeKind.VOID:
        return ast.TypeVoid()

    raise ForeignParseException('Failed to parse %s' % node.kind)

_struct_cache = {}
def parse_type_struct(node):
    name = node.spelling

    if name in _structs:
        return _structs[name]

    if name in _struct_cache:
        return _struct_cache[name]

    struct_type = ast.TypeStruct(node.spelling, [])
    _struct_cache[name] = struct_type

    fields = []
    for field in node.get_children():
        if field.spelling is not None:
            fields.append((field.spelling, parse_type(field.type)))
    struct_type.fields = fields
    return struct_type

class Parser:
    def __init__(self):
        self.index = clang.cindex.Index.create()

    def parse(self, src, opts):
        args = [arg for path in opts.search_path for arg in ('-I', path)] + ['-x', 'c++-header']
        translation_unit = self.index.parse(src, args)
        if opts.allow_warning():
            for diagnostic in translation_unit.diagnostics:
                sys.stderr.write('WARNING: %s:%s:%s: %s\n' % (
                    diagnostic.location.file.name,
                    diagnostic.location.line,
                    diagnostic.location.column,
                    diagnostic.spelling))
        return parse_translation_unit(translation_unit.cursor, opts)
