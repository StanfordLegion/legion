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
### Pretty Printer
###

# Backport of singledispatch to Python 2.x.
try:
    from functools import singledispatch
except ImportError:
    from singledispatch import singledispatch

import copy
from . import ast, parse

class Context:
    def __init__(self):
        self.indent = None
    def new_block_scope(self):
        cx = copy.copy(self)
        cx.indent += 1
        return cx
    def new_program_scope(self):
        cx = copy.copy(self)
        cx.indent = 0
        return cx

@singledispatch
def pretty_print_node(node, cx):
    raise Exception('Pretty printing failed at %s' % node)

@pretty_print_node.register(ast.Program)
def _(node, cx):
    cx = cx.new_program_scope()
    return pretty_print_node(node.definitions, cx)

@pretty_print_node.register(ast.Definitions)
def _(node, cx):
    return '\n\n'.join(
        pretty_print_node(definition, cx)
        for definition in node.definitions)

@pretty_print_node.register(ast.Import)
def _(node, cx):
    return 'import "%s";' % node.filename

@pretty_print_node.register(ast.Struct)
def _(node, cx):
    return 'struct %s%s%s%s %s' % (
        pretty_print_node(node.name, cx),
        pretty_print_node(node.params, cx),
        pretty_print_node(node.regions, cx),
        pretty_print_node(node.constraints, cx),
        pretty_print_node(node.field_decls, cx))

@pretty_print_node.register(ast.StructName)
def _(node, cx):
    return node.name

@pretty_print_node.register(ast.StructParams)
def _(node, cx):
    if len(node.params) == 0:
        return ''
    return '<%s>' % ', '.join(
        pretty_print_node(param, cx)
        for param in node.params)

@pretty_print_node.register(ast.StructParam)
def _(node, cx):
    return '%s: %s' % (
        node.name,
        pretty_print_node(node.type, cx))

@pretty_print_node.register(ast.StructRegions)
def _(node, cx):
    if len(node.regions) == 0:
        return ''
    return '[%s]' % ', '.join(
        pretty_print_node(region, cx)
        for region in node.regions)

@pretty_print_node.register(ast.StructRegion)
def _(node, cx):
    return '%s: %s' % (
        node.name,
        pretty_print_node(node.type, cx))

@pretty_print_node.register(ast.StructConstraints)
def _(node, cx):
    if len(node.constraints) == 0:
        return ''
    return ', %s' % ', '.join(
        pretty_print_node(constraint, cx)
        for constraint in node.constraints)

@pretty_print_node.register(ast.StructConstraint)
def _(node, cx):
    return '%s %s %s' % (
        pretty_print_node(node.lhs, cx),
        node.op,
        pretty_print_node(node.rhs, cx))

@pretty_print_node.register(ast.StructConstraintRegion)
def _(node, cx):
    return node.name

@pretty_print_node.register(ast.FieldDecls)
def _(node, cx):
    block_cx = cx.new_block_scope()
    return '{\n%s\n%s}' % (
        ',\n'.join('  '*block_cx.indent + pretty_print_node(field_decl, block_cx)
                   for field_decl in node.field_decls),
        '  '*cx.indent
        )

@pretty_print_node.register(ast.FieldDecl)
def _(node, cx):
    return '%s: %s' % (
        node.name,
        pretty_print_node(node.field_type, cx))

@pretty_print_node.register(ast.Function)
def _(node, cx):
    return 'task %s%s%s%s %s' % (
        pretty_print_node(node.name, cx),
        pretty_print_node(node.params, cx),
        pretty_print_node(node.return_type, cx),
        pretty_print_node(node.privileges, cx),
        pretty_print_node(node.block, cx))

@pretty_print_node.register(ast.FunctionName)
def _(node, cx):
    return node.name

@pretty_print_node.register(ast.FunctionParams)
def _(node, cx):
    return '(%s)' % ', '.join(
        pretty_print_node(param, cx)
        for param in node.params)

@pretty_print_node.register(ast.FunctionParam)
def _(node, cx):
    return '%s: %s' % (node.name, pretty_print_node(node.declared_type, cx))

@pretty_print_node.register(ast.FunctionReturnType)
def _(node, cx):
    if isinstance(node.declared_type, ast.TypeVoid):
        return ''
    return ': %s' % pretty_print_node(node.declared_type, cx)

@pretty_print_node.register(ast.FunctionPrivileges)
def _(node, cx):
    return ''.join(
        pretty_print_node(privilege, cx)
        for privilege in node.privileges)

@pretty_print_node.register(ast.FunctionPrivilege)
def _(node, cx):
    return ', %s' % pretty_print_node(node.privilege, cx)

@pretty_print_node.register(ast.TypeVoid)
def _(node, cx):
    return 'void'

@pretty_print_node.register(ast.TypeBool)
def _(node, cx):
    return 'bool'

@pretty_print_node.register(ast.TypeDouble)
def _(node, cx):
    return 'double'

@pretty_print_node.register(ast.TypeFloat)
def _(node, cx):
    return 'float'

@pretty_print_node.register(ast.TypeInt)
def _(node, cx):
    return 'int'

@pretty_print_node.register(ast.TypeUInt)
def _(node, cx):
    return 'uint'

@pretty_print_node.register(ast.TypeInt8)
def _(node, cx):
    return 'int8'

@pretty_print_node.register(ast.TypeInt16)
def _(node, cx):
    return 'int16'

@pretty_print_node.register(ast.TypeInt32)
def _(node, cx):
    return 'int32'

@pretty_print_node.register(ast.TypeInt64)
def _(node, cx):
    return 'int64'

@pretty_print_node.register(ast.TypeUInt8)
def _(node, cx):
    return 'uint8'

@pretty_print_node.register(ast.TypeUInt16)
def _(node, cx):
    return 'uint16'

@pretty_print_node.register(ast.TypeUInt32)
def _(node, cx):
    return 'uint32'

@pretty_print_node.register(ast.TypeUInt64)
def _(node, cx):
    return 'uint64'

@pretty_print_node.register(ast.TypeColoring)
def _(node, cx):
    return 'coloring<%s>' % pretty_print_node(node.region, cx)

@pretty_print_node.register(ast.TypeColoringRegion)
def _(node, cx):
    return node.name

@pretty_print_node.register(ast.TypeID)
def _(node, cx):
    return '%s%s' % (
        node.name,
        pretty_print_node(node.args, cx))

@pretty_print_node.register(ast.TypeArgs)
def _(node, cx):
    if len(node.args) == 0:
        return ''
    return '<%s>' % ', '.join(pretty_print_node(arg, cx) for arg in node.args)

@pretty_print_node.register(ast.TypeArg)
def _(node, cx):
    return node.name

@pretty_print_node.register(ast.TypeArgWild)
def _(node, cx):
    return '?'

@pretty_print_node.register(ast.TypePointer)
def _(node, cx):
    return '%s@%s' % (
        pretty_print_node(node.points_to_type, cx),
        pretty_print_node(node.regions, cx))

@pretty_print_node.register(ast.TypePointerRegions)
def _(node, cx):
    if len(node.regions) == 1:
        return pretty_print_node(node.regions[0], cx)
    return '(%s)' % ', '.join(pretty_print_node(region, cx) for region in node.regions)

@pretty_print_node.register(ast.TypeRegion)
def _(node, cx):
    return node.name

@pretty_print_node.register(ast.TypeRegionWild)
def _(node, cx):
    return '?'

@pretty_print_node.register(ast.TypeRegionKind)
def _(node, cx):
    if node.contains_type is None:
        return 'region'
    return 'region<%s>' % pretty_print_node(node.contains_type, cx)

@pretty_print_node.register(ast.TypeArrayKind)
def _(node, cx):
    return 'array<%s, %s>' % (
        pretty_print_node(node.ispace, cx),
        pretty_print_node(node.contains_type, cx))

@pretty_print_node.register(ast.TypeIspace)
def _(node, cx):
    return node.name

@pretty_print_node.register(ast.TypeIspaceKind)
def _(node, cx):
    return 'ispace<%s>' % pretty_print_node(node.index_type, cx)

@pretty_print_node.register(ast.Privilege)
def _(node, cx):
    if node.op is not None:
        return '%s<%s>(%s)' % (
            node.privilege,
            node.op,
            pretty_print_node(node.regions, cx))
    return '%s(%s)' % (
        node.privilege,
        pretty_print_node(node.regions, cx))

@pretty_print_node.register(ast.PrivilegeRegions)
def _(node, cx):
    return ', '.join(
        pretty_print_node(region, cx)
        for region in node.regions)

@pretty_print_node.register(ast.PrivilegeRegion)
def _(node, cx):
    return '%s%s' % (
        node.name,
        pretty_print_node(node.fields, cx))

@pretty_print_node.register(ast.PrivilegeRegionFields)
def _(node, cx):
    if len(node.fields) == 0:
        return ''
    if len(node.fields) == 1:
        return '.%s' % pretty_print_node(node.fields[0], cx)
    return '.{%s}' % ', '.join(
        pretty_print_node(field, cx)
        for field in node.fields)

@pretty_print_node.register(ast.PrivilegeRegionField)
def _(node, cx):
    if node.fields is None:
        return node.name
    return '%s%s' % (
        node.name,
        pretty_print_node(node.fields, cx))

@pretty_print_node.register(ast.Block)
def _(node, cx):
    block_cx = cx.new_block_scope()
    return '{\n%s\n%s}' % (
        '\n'.join('  '*block_cx.indent + pretty_print_node(statement, block_cx)
                  for statement in node.block),
        '  '*cx.indent
        )

@pretty_print_node.register(ast.StatementAssert)
def _(node, cx):
    return 'assert %s;' % pretty_print_node(node.expr, cx)

@pretty_print_node.register(ast.StatementExpr)
def _(node, cx):
    return '%s;' % pretty_print_node(node.expr, cx)

@pretty_print_node.register(ast.StatementIf)
def _(node, cx):
    if node.else_block is None:
        return 'if (%s) %s' % (
            pretty_print_node(node.condition, cx),
            pretty_print_node(node.then_block, cx))
    return 'if (%s) %s else %s' % (
        pretty_print_node(node.condition, cx),
        pretty_print_node(node.then_block, cx),
        pretty_print_node(node.else_block, cx))

@pretty_print_node.register(ast.StatementFor)
def _(node, cx):
    if len(node.indices.indices) == 1:
        return 'for %s in %s %s' % (
            pretty_print_node(node.indices, cx),
            pretty_print_node(node.regions, cx),
            pretty_print_node(node.block, cx))
    return 'for %s in zip(%s) %s' % (
        pretty_print_node(node.indices, cx),
        pretty_print_node(node.regions, cx),
        pretty_print_node(node.block, cx))

@pretty_print_node.register(ast.ForIndices)
def _(node, cx):
    return ', '.join(pretty_print_node(index, cx) for index in node.indices)

@pretty_print_node.register(ast.ForIndex)
def _(node, cx):
    if node.type is None:
        return node.name
    return '%s: %s' % (node.name, pretty_print_node(node.type, cx))

@pretty_print_node.register(ast.ForRegions)
def _(node, cx):
    return ', '.join(
        pretty_print_node(region, cx)
        for region in node.regions)

@pretty_print_node.register(ast.ForRegion)
def _(node, cx):
    return node.name

@pretty_print_node.register(ast.StatementLet)
def _(node, cx):
    if node.type is None:
        return 'let %s = %s;' % (
            node.name,
            pretty_print_node(node.expr, cx))
    return 'let %s: %s = %s;' % (
        node.name,
        pretty_print_node(node.type, cx),
        pretty_print_node(node.expr, cx))

@pretty_print_node.register(ast.StatementLetRegion)
def _(node, cx):
    if node.region_kind is None:
        return 'let %s = region<%s>(%s);' % (
            node.name,
            pretty_print_node(node.element_type, cx),
            pretty_print_node(node.size_expr, cx))
    return 'let %s: %s = region<%s>(%s);' % (
        node.name,
        pretty_print_node(node.region_kind, cx),
        pretty_print_node(node.element_type, cx),
        pretty_print_node(node.size_expr, cx))

@pretty_print_node.register(ast.StatementLetArray)
def _(node, cx):
    if node.region_kind is None:
        return 'let %s = array<%s, %s>();' % (
            node.name,
            pretty_print_node(node.ispace_type, cx),
            pretty_print_node(node.element_type, cx))
    return 'let %s: %s = array<%s, %s>();' % (
        node.name,
        pretty_print_node(node.region_kind, cx),
        pretty_print_node(node.ispace_type, cx),
        pretty_print_node(node.element_type, cx))

@pretty_print_node.register(ast.StatementLetIspace)
def _(node, cx):
    if node.ispace_kind is None:
        return 'let %s = ispace<%s>(%s);' % (
            node.name,
            pretty_print_node(node.index_type, cx),
            pretty_print_node(node.size_expr, cx))
    return 'let %s: %s = ispace<%s, %s>();' % (
        node.name,
        pretty_print_node(node.ispace_kind, cx),
        pretty_print_node(node.index_type, cx),
        pretty_print_node(node.size_expr, cx))

@pretty_print_node.register(ast.StatementLetPartition)
def _(node, cx):
    if node.partition_kind is None:
        return 'let %s = partition<%s, %s>(%s);' % (
            node.name,
            pretty_print_node(node.region_type, cx),
            pretty_print_node(node.mode, cx),
            pretty_print_node(node.coloring_expr, cx))
    return 'let %s: %s = partition<%s, %s>(%s);' % (
        node.name,
        pretty_print_node(node.partition_kind, cx),
        pretty_print_node(node.region_type, cx),
        pretty_print_node(node.mode, cx),
        pretty_print_node(node.coloring_expr, cx))

@pretty_print_node.register(ast.PartitionMode)
def _(node, cx):
    return node.mode

@pretty_print_node.register(ast.StatementReturn)
def _(node, cx):
    return 'return %s;' % pretty_print_node(node.expr, cx)

@pretty_print_node.register(ast.StatementUnpack)
def _(node, cx):
    return 'unpack %s as %s: %s%s;' % (
        pretty_print_node(node.expr, cx),
        node.name,
        pretty_print_node(node.type, cx),
        pretty_print_node(node.regions, cx))

@pretty_print_node.register(ast.UnpackRegions)
def _(node, cx):
    return '[%s]' % ', '.join(
        pretty_print_node(region, cx)
        for region in node.regions)

@pretty_print_node.register(ast.UnpackRegion)
def _(node, cx):
    return node.name

@pretty_print_node.register(ast.StatementVar)
def _(node, cx):
    if node.type is None:
        return 'var %s = %s;' % (
            node.name,
            pretty_print_node(node.expr, cx))
    return 'var %s: %s = %s;' % (
        node.name,
        pretty_print_node(node.type, cx),
        pretty_print_node(node.expr, cx))

@pretty_print_node.register(ast.StatementWhile)
def _(node, cx):
    return 'while (%s) %s' % (
        pretty_print_node(node.condition, cx),
        pretty_print_node(node.block, cx))

@pretty_print_node.register(ast.ExprID)
def _(node, cx):
    return node.name

@pretty_print_node.register(ast.ExprAssignment)
def _(node, cx):
    return '(%s = %s)' % (
        pretty_print_node(node.lval, cx),
        pretty_print_node(node.rval, cx))

@pretty_print_node.register(ast.ExprUnaryOp)
def _(node, cx):
    return '(%s%s)' % (
        node.op,
        pretty_print_node(node.arg, cx))

@pretty_print_node.register(ast.ExprBinaryOp)
def _(node, cx):
    return '(%s %s %s)' % (
        pretty_print_node(node.lhs, cx),
        node.op,
        pretty_print_node(node.rhs, cx))

@pretty_print_node.register(ast.ExprReduceOp)
def _(node, cx):
    return '(%s %s= %s)' % (
        pretty_print_node(node.lhs, cx),
        node.op,
        pretty_print_node(node.rhs, cx))

@pretty_print_node.register(ast.ExprCast)
def _(node, cx):
    return '(%s(%s))' % (
        pretty_print_node(node.cast_to_type, cx),
        pretty_print_node(node.expr, cx))

@pretty_print_node.register(ast.ExprNull)
def _(node, cx):
    return '(null<%s>())' % pretty_print_node(node.pointer_type, cx)

@pretty_print_node.register(ast.ExprIsnull)
def _(node, cx):
    return '(isnull(%s))' % pretty_print_node(node.pointer_expr, cx)

@pretty_print_node.register(ast.ExprNew)
def _(node, cx):
    return '(new<%s>())' % pretty_print_node(node.pointer_type, cx)

@pretty_print_node.register(ast.ExprRead)
def _(node, cx):
    return '(read(%s))' % pretty_print_node(node.pointer_expr, cx)

@pretty_print_node.register(ast.ExprWrite)
def _(node, cx):
    return '(write(%s, %s))' % (
        pretty_print_node(node.pointer_expr, cx),
        pretty_print_node(node.value_expr, cx))

@pretty_print_node.register(ast.ExprReduce)
def _(node, cx):
    return '(reduce<%s>(%s, %s))' % (
        node.op,
        pretty_print_node(node.pointer_expr, cx),
        pretty_print_node(node.value_expr, cx))

@pretty_print_node.register(ast.ExprDereference)
def _(node, cx):
    return '(*%s)' % pretty_print_node(node.pointer_expr, cx)

@pretty_print_node.register(ast.ExprArrayAccess)
def _(node, cx):
    return '(%s[%s])' % (
        pretty_print_node(node.array_expr, cx),
        pretty_print_node(node.index_expr, cx))

@pretty_print_node.register(ast.ExprFieldAccess)
def _(node, cx):
    return '(%s.%s)' % (
        pretty_print_node(node.struct_expr, cx),
        node.field_name)

@pretty_print_node.register(ast.ExprFieldDereference)
def _(node, cx):
    return '(%s->%s)' % (
        pretty_print_node(node.pointer_expr, cx),
        node.field_name)

@pretty_print_node.register(ast.ExprFieldValues)
def _(node, cx):
    return '(%s)' % pretty_print_node(node.field_values, cx)

@pretty_print_node.register(ast.FieldValues)
def _(node, cx):
    return '{%s}' % ', '.join(
        pretty_print_node(field_value, cx)
        for field_value in node.field_values)

@pretty_print_node.register(ast.FieldValue)
def _(node, cx):
    return '%s: %s' % (
        node.field_name,
        pretty_print_node(node.value_expr, cx))

@pretty_print_node.register(ast.ExprFieldUpdates)
def _(node, cx):
    return '(%s%s)' % (
        pretty_print_node(node.struct_expr, cx),
        pretty_print_node(node.field_updates, cx))

@pretty_print_node.register(ast.FieldUpdates)
def _(node, cx):
    return '{%s}' % ', '.join(
        pretty_print_node(field_update, cx)
        for field_update in node.field_updates)

@pretty_print_node.register(ast.FieldUpdate)
def _(node, cx):
    return '%s: %s' % (
        node.field_name,
        pretty_print_node(node.update_expr, cx))

@pretty_print_node.register(ast.ExprColoring)
def _(node, cx):
    return '(coloring<%s>())' % pretty_print_node(node.region, cx)

@pretty_print_node.register(ast.ColoringRegion)
def _(node, cx):
    return node.name

@pretty_print_node.register(ast.ExprColor)
def _(node, cx):
    return '(color(%s, %s, %s))' % (
        pretty_print_node(node.coloring_expr, cx),
        pretty_print_node(node.pointer_expr, cx),
        pretty_print_node(node.color_expr, cx))

@pretty_print_node.register(ast.ExprUpregion)
def _(node, cx):
    return '(upregion%s(%s))' % (
        pretty_print_node(node.regions, cx),
        pretty_print_node(node.expr, cx))

@pretty_print_node.register(ast.UpregionRegions)
def _(node, cx):
    return '<%s>' % ', '.join(
        pretty_print_node(region, cx)
        for region in node.regions)

@pretty_print_node.register(ast.UpregionRegion)
def _(node, cx):
    return node.name

@pretty_print_node.register(ast.ExprDownregion)
def _(node, cx):
    return '(downregion%s(%s))' % (
        pretty_print_node(node.regions, cx),
        pretty_print_node(node.expr, cx))

@pretty_print_node.register(ast.DownregionRegions)
def _(node, cx):
    return '<%s>' % ', '.join(
        pretty_print_node(region, cx)
        for region in node.regions)

@pretty_print_node.register(ast.DownregionRegion)
def _(node, cx):
    return node.name

@pretty_print_node.register(ast.ExprPack)
def _(node, cx):
    return '(pack %s as %s%s)' % (
        pretty_print_node(node.expr, cx),
        pretty_print_node(node.type, cx),
        pretty_print_node(node.regions, cx))

@pretty_print_node.register(ast.PackRegions)
def _(node, cx):
    return '[%s]' % ', '.join(
        pretty_print_node(region, cx)
        for region in node.regions)

@pretty_print_node.register(ast.PackRegion)
def _(node, cx):
    return node.name

@pretty_print_node.register(ast.ExprCall)
def _(node, cx):
    return '(%s%s)' % (
        pretty_print_node(node.function, cx),
        pretty_print_node(node.args, cx))

@pretty_print_node.register(ast.Args)
def _(node, cx):
    return '(%s)' % ', '.join(
        pretty_print_node(arg, cx)
        for arg in node.args)

@pretty_print_node.register(ast.ExprConstBool)
def _(node, cx):
    return str(node.value).lower()

@pretty_print_node.register(ast.ExprConstDouble)
def _(node, cx):
    return node.value

@pretty_print_node.register(ast.ExprConstFloat)
def _(node, cx):
    return '%sf' % node.value

@pretty_print_node.register(ast.ExprConstInt)
def _(node, cx):
    return str(node.value)

@pretty_print_node.register(ast.ExprConstUInt)
def _(node, cx):
    return '%su' % str(node.value)

def pretty_print(node):
    cx = Context()
    return pretty_print_node(node, cx)
