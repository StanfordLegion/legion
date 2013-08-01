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

from . import ast, parse

def pretty_print_node(node, indent):
    if isinstance(node, ast.Program):
        return '\n\n'.join(pretty_print_node(definition, indent) for definition in node.defs)
    if isinstance(node, ast.Import):
        return 'import "%s";' % node.filename
    if isinstance(node, ast.Struct):
        return 'struct %s%s%s%s %s' % (
            node.name,
            pretty_print_node(node.params, indent),
            pretty_print_node(node.regions, indent),
            pretty_print_node(node.constraints, indent),
            pretty_print_node(node.field_decls, indent))
    if isinstance(node, ast.StructParams):
        if len(node.params) == 0:
            return ''
        return '<%s>' % ', '.join(pretty_print_node(param, indent) for param in node.params)
    if isinstance(node, ast.StructParam):
        return '%s: %s' % (node.name, pretty_print_node(node.type, indent))
    if isinstance(node, ast.StructRegions):
        if len(node.regions) == 0:
            return ''
        return '[%s]' % ', '.join(pretty_print_node(region, indent) for region in node.regions)
    if isinstance(node, ast.StructRegion):
        return '%s: %s' % (node.name, pretty_print_node(node.type, indent))
    if isinstance(node, ast.StructConstraints):
        if len(node.constraints) == 0:
            return ''
        return ', %s' % ', '.join(pretty_print_node(constraint, indent) for constraint in node.constraints)
    if isinstance(node, ast.StructConstraint):
        return '%s %s %s' % (
            pretty_print_node(node.lhs, indent),
            node.op,
            pretty_print_node(node.rhs, indent))
    if isinstance(node, ast.StructConstraintRegion):
        return node.name
    if isinstance(node, ast.FieldDecls):
        return '{\n%s\n%s}' % (
            ',\n'.join('  '*(indent + 1) + pretty_print_node(field_decl, indent + 1)
                       for field_decl in node.field_decls),
            '  '*indent
            )
    if isinstance(node, ast.FieldDecl):
        return '%s: %s' % (
            node.name,
            pretty_print_node(node.field_type, indent)
            )
    if isinstance(node, ast.Function):
        return 'task %s%s: %s%s %s' % (
            node.name,
            pretty_print_node(node.params, indent),
            pretty_print_node(node.return_type, indent),
            pretty_print_node(node.privileges, indent),
            pretty_print_node(node.block, indent))
    if isinstance(node, ast.Params):
        return '(%s)' % ', '.join(pretty_print_node(param, indent) for param in node.params)
    if isinstance(node, ast.Param):
        return '%s: %s' % (node.name, pretty_print_node(node.declared_type, indent))
    if isinstance(node, ast.Privileges):
        if len(node.privileges) == 0:
            return ''
        return ', %s' % ', '.join(pretty_print_node(privilege, indent) for privilege in node.privileges)
    if isinstance(node, ast.Privilege):
        if node.op is not None:
            return '%s<%s>(%s)' % (node.privilege, node.op, pretty_print_node(node.regions, indent))
        return '%s(%s)' % (node.privilege, pretty_print_node(node.regions, indent))
    if isinstance(node, ast.PrivilegeRegions):
        return ', '.join(pretty_print_node(region, indent) for region in node.regions)
    if isinstance(node, ast.PrivilegeRegion):
        return '%s%s' % (node.name, pretty_print_node(node.fields, indent))
    if isinstance(node, ast.PrivilegeRegionFields):
        if len(node.fields) == 0:
            return ''
        if len(node.fields) == 1:
            return '.%s' % pretty_print_node(node.fields[0], indent)
        return '.{%s}' % ', '.join(pretty_print_node(field, indent) for field in node.fields)
    if isinstance(node, ast.PrivilegeRegionField):
        if node.fields is None:
            return node.name
        return '%s%s' % (node.name, pretty_print_node(node.fields, indent))
    if isinstance(node, ast.TypeVoid):
        return 'void'
    if isinstance(node, ast.TypeBool):
        return 'bool'
    if isinstance(node, ast.TypeDouble):
        return 'double'
    if isinstance(node, ast.TypeFloat):
        return 'float'
    if isinstance(node, ast.TypeInt):
        return 'int'
    if isinstance(node, ast.TypeUInt):
        return 'uint'
    if isinstance(node, ast.TypeInt8):
        return 'int8'
    if isinstance(node, ast.TypeInt16):
        return 'int16'
    if isinstance(node, ast.TypeInt32):
        return 'int32'
    if isinstance(node, ast.TypeInt64):
        return 'int64'
    if isinstance(node, ast.TypeUInt8):
        return 'uint8'
    if isinstance(node, ast.TypeUInt16):
        return 'uint16'
    if isinstance(node, ast.TypeUInt32):
        return 'uint32'
    if isinstance(node, ast.TypeUInt64):
        return 'uint64'
    if isinstance(node, ast.TypeColoring):
        return 'coloring<%s>' % pretty_print_node(node.region, indent)
    if isinstance(node, ast.TypeColoringRegion):
        return node.name
    if isinstance(node, ast.TypeID):
        return '%s%s' % (
            node.name,
            pretty_print_node(node.args, indent))
    if isinstance(node, ast.TypeArgs):
        if len(node.args) == 0:
            return ''
        return '<%s>' % ', '.join(pretty_print_node(arg, indent) for arg in node.args)
    if isinstance(node, ast.TypeArg):
        return node.name
    if isinstance(node, ast.TypeArgWild):
        return '?'
    if isinstance(node, ast.TypePointer):
        return '%s@%s' % (
            pretty_print_node(node.points_to_type, indent),
            pretty_print_node(node.regions, indent))
    if isinstance(node, ast.TypePointerRegions):
        if len(node.regions) == 1:
            return pretty_print_node(node.regions[0], indent)
        return '(%s)' % ', '.join(pretty_print_node(region, indent) for region in node.regions)
    if isinstance(node, ast.TypeRegion):
        return node.name
    if isinstance(node, ast.TypeRegionWild):
        return '?'
    if isinstance(node, ast.TypeRegionKind):
        if node.contains_type is None:
            return 'region'
        return 'region<%s>' % pretty_print_node(node.contains_type, indent)
    if isinstance(node, ast.TypeArrayKind):
        return 'array<%s, %s>' % (
            pretty_print_node(node.ispace, indent),
            pretty_print_node(node.contains_type, indent))
    if isinstance(node, ast.TypeIspace):
        return node.name
    if isinstance(node, ast.TypeIspaceKind):
        return 'ispace<%s>' % pretty_print_node(node.index_type, indent)
    if isinstance(node, ast.Block):
        return '{\n%s\n%s}' % (
            '\n'.join('  '*(indent + 1) + pretty_print_node(statement, indent + 1)
                      for statement in node.block),
            '  '*indent
            )
    if isinstance(node, ast.StatementAssert):
        return 'assert %s;' % pretty_print_node(node.expr, indent)
    if isinstance(node, ast.StatementExpr):
        return '%s;' % pretty_print_node(node.expr, indent)
    if isinstance(node, ast.StatementIf):
        if node.else_block is None:
            return 'if (%s) %s' % (
                pretty_print_node(node.condition, indent),
                pretty_print_node(node.then_block, indent))
        return 'if (%s) %s else %s' % (
            pretty_print_node(node.condition, indent),
            pretty_print_node(node.then_block, indent),
            pretty_print_node(node.else_block, indent))
    if isinstance(node, ast.StatementFor):
        if len(node.indices.indices) == 1:
            return 'for %s in %s %s' % (
                pretty_print_node(node.indices, indent),
                pretty_print_node(node.regions, indent),
                pretty_print_node(node.block, indent))
        return 'for %s in zip(%s) %s' % (
            pretty_print_node(node.indices, indent),
            pretty_print_node(node.regions, indent),
            pretty_print_node(node.block, indent))
    if isinstance(node, ast.ForIndices):
        return ', '.join(pretty_print_node(index, indent) for index in node.indices)
    if isinstance(node, ast.ForIndex):
        if node.type is None:
            return node.name
        return '%s: %s' % (node.name, pretty_print_node(node.type, indent))
    if isinstance(node, ast.ForRegions):
        return ', '.join(pretty_print_node(region, indent) for region in node.regions)
    if isinstance(node, ast.ForRegion):
        return node.name
    if isinstance(node, ast.StatementLet):
        if node.type is None:
            return 'let %s = %s;' % (
                node.name,
                pretty_print_node(node.expr, indent))
        return 'let %s: %s = %s;' % (
            node.name,
            pretty_print_node(node.type, indent),
            pretty_print_node(node.expr, indent))
    if isinstance(node, ast.StatementLetRegion):
        if node.region_kind is None:
            return 'let %s = region<%s>(%s);' % (
                node.name,
                pretty_print_node(node.element_type, indent),
                pretty_print_node(node.size_expr, indent))
        return 'let %s: %s = region<%s>(%s);' % (
            node.name,
            pretty_print_node(node.region_kind, indent),
            pretty_print_node(node.element_type, indent),
            pretty_print_node(node.size_expr, indent))
    if isinstance(node, ast.StatementLetArray):
        if node.region_kind is None:
            return 'let %s = array<%s, %s>();' % (
                node.name,
                pretty_print_node(node.ispace_type, indent),
                pretty_print_node(node.element_type, indent))
        return 'let %s: %s = array<%s, %s>();' % (
            node.name,
            pretty_print_node(node.region_kind, indent),
            pretty_print_node(node.ispace_type, indent),
            pretty_print_node(node.element_type, indent))
    if isinstance(node, ast.StatementLetIspace):
        if node.ispace_kind is None:
            return 'let %s = ispace<%s>(%s);' % (
                node.name,
                pretty_print_node(node.index_type, indent),
                pretty_print_node(node.size_expr, indent))
        return 'let %s: %s = ispace<%s, %s>();' % (
            node.name,
            pretty_print_node(node.ispace_kind, indent),
            pretty_print_node(node.index_type, indent),
            pretty_print_node(node.size_expr, indent))
    if isinstance(node, ast.StatementLetPartition):
        if node.partition_kind is None:
            return 'let %s = partition<%s, %s>(%s);' % (
                node.name,
                pretty_print_node(node.region_type, indent),
                pretty_print_node(node.mode, indent),
                pretty_print_node(node.coloring_expr, indent))
        return 'let %s: %s = partition<%s, %s>(%s);' % (
            node.name,
            pretty_print_node(node.partition_kind, indent),
            pretty_print_node(node.region_type, indent),
            pretty_print_node(node.mode, indent),
            pretty_print_node(node.coloring_expr, indent))
    if isinstance(node, ast.PartitionMode):
        return node.mode
    if isinstance(node, ast.StatementReturn):
        return 'return %s;' % pretty_print_node(node.expr, indent)
    if isinstance(node, ast.StatementUnpack):
        return 'unpack %s as %s: %s%s;' % (
            pretty_print_node(node.expr, indent),
            node.name,
            pretty_print_node(node.type, indent),
            pretty_print_node(node.regions, indent))
    if isinstance(node, ast.UnpackRegions):
        return '[%s]' % ', '.join(pretty_print_node(region, indent) for region in node.regions)
    if isinstance(node, ast.UnpackRegion):
        return node.name
    if isinstance(node, ast.StatementVar):
        if node.type is None:
            return 'var %s = %s;' % (
                node.name,
                pretty_print_node(node.expr, indent))
        return 'var %s: %s = %s;' % (
            node.name,
            pretty_print_node(node.type, indent),
            pretty_print_node(node.expr, indent))
    if isinstance(node, ast.StatementWhile):
        return 'while (%s) %s' % (
            pretty_print_node(node.condition, indent),
            pretty_print_node(node.block, indent))
    if isinstance(node, ast.ExprID):
        return node.name
    if isinstance(node, ast.ExprAssignment):
        return '(%s = %s)' % (
            pretty_print_node(node.lval, indent),
            pretty_print_node(node.rval, indent))
    if isinstance(node, ast.ExprUnaryOp):
        return '(%s%s)' % (
            node.op,
            pretty_print_node(node.arg, indent))
    if isinstance(node, ast.ExprBinaryOp):
        return '(%s %s %s)' % (
            pretty_print_node(node.lhs, indent),
            node.op,
            pretty_print_node(node.rhs, indent))
    if isinstance(node, ast.ExprReduceOp):
        return '(%s %s= %s)' % (
            pretty_print_node(node.lhs, indent),
            node.op,
            pretty_print_node(node.rhs, indent))
    if isinstance(node, ast.ExprCast):
        return '(%s(%s))' % (
            pretty_print_node(node.cast_to_type, indent),
            pretty_print_node(node.expr, indent))
    if isinstance(node, ast.ExprNull):
        return '(null<%s>())' % pretty_print_node(node.pointer_type, indent)
    if isinstance(node, ast.ExprIsnull):
        return '(isnull(%s))' % pretty_print_node(node.pointer_expr, indent)
    if isinstance(node, ast.ExprNew):
        return '(new<%s>())' % pretty_print_node(node.pointer_type, indent)
    if isinstance(node, ast.ExprRead):
        return '(read(%s))' % pretty_print_node(node.pointer_expr, indent)
    if isinstance(node, ast.ExprWrite):
        return '(write(%s, %s))' % (
            pretty_print_node(node.pointer_expr, indent),
            pretty_print_node(node.value_expr, indent))
    if isinstance(node, ast.ExprReduce):
        return '(reduce<%s>(%s, %s))' % (
            node.op,
            pretty_print_node(node.pointer_expr, indent),
            pretty_print_node(node.value_expr, indent))
    if isinstance(node, ast.ExprDereference):
        return '(*%s)' % pretty_print_node(node.pointer_expr, indent)
    if isinstance(node, ast.ExprArrayAccess):
        return '(%s[%s])' % (
            pretty_print_node(node.array_expr, indent),
            pretty_print_node(node.index_expr, indent))
    if isinstance(node, ast.ExprFieldAccess):
        return '(%s.%s)' % (
            pretty_print_node(node.struct_expr, indent),
            node.field_name)
    if isinstance(node, ast.ExprFieldDereference):
        return '(%s->%s)' % (
            pretty_print_node(node.pointer_expr, indent),
            node.field_name)
    if isinstance(node, ast.ExprFieldValues):
        return '(%s)' % pretty_print_node(node.field_values, indent)
    if isinstance(node, ast.FieldValues):
        return '{%s}' % ', '.join(
            pretty_print_node(field_value, indent)
            for field_value in node.field_values)
    if isinstance(node, ast.FieldValue):
        return '%s: %s' % (
            node.field_name,
            pretty_print_node(node.value_expr, indent))
    if isinstance(node, ast.ExprFieldUpdates):
        return '(%s%s)' % (
            pretty_print_node(node.struct_expr, indent),
            pretty_print_node(node.field_updates, indent))
    if isinstance(node, ast.FieldUpdates):
        return '{%s}' % ', '.join(
            pretty_print_node(field_update, indent)
            for field_update in node.field_updates)
    if isinstance(node, ast.FieldUpdate):
        return '%s: %s' % (
            node.field_name,
            pretty_print_node(node.update_expr, indent))
    if isinstance(node, ast.ExprColoring):
        return '(coloring<%s>())' % pretty_print_node(node.region, indent)
    if isinstance(node, ast.ColoringRegion):
        return node.name
    if isinstance(node, ast.ExprColor):
        return '(color(%s, %s, %s))' % (
            pretty_print_node(node.coloring_expr, indent),
            pretty_print_node(node.pointer_expr, indent),
            pretty_print_node(node.color_expr, indent))
    if isinstance(node, ast.ExprUpregion):
        return '(upregion%s(%s))' % (
            pretty_print_node(node.regions, indent),
            pretty_print_node(node.expr, indent))
    if isinstance(node, ast.UpregionRegions):
        return '<%s>' % ', '.join(pretty_print_node(region, indent) for region in node.regions)
    if isinstance(node, ast.UpregionRegion):
        return node.name
    if isinstance(node, ast.ExprDownregion):
        return '(downregion%s(%s))' % (
            pretty_print_node(node.regions, indent),
            pretty_print_node(node.expr, indent))
    if isinstance(node, ast.DownregionRegions):
        return '<%s>' % ', '.join(pretty_print_node(region, indent) for region in node.regions)
    if isinstance(node, ast.DownregionRegion):
        return node.name
    if isinstance(node, ast.ExprPack):
        return '(pack %s as %s%s)' % (
            pretty_print_node(node.expr, indent),
            pretty_print_node(node.type, indent),
            pretty_print_node(node.regions, indent))
    if isinstance(node, ast.PackRegions):
        return '[%s]' % ', '.join(pretty_print_node(region, indent) for region in node.regions)
    if isinstance(node, ast.PackRegion):
        return node.name
    if isinstance(node, ast.ExprCall):
        return '(%s%s)' % (
            pretty_print_node(node.function, indent),
            pretty_print_node(node.args, indent))
    if isinstance(node, ast.Args):
        return '(%s)' % ', '.join(pretty_print_node(arg, indent) for arg in node.args)
    if isinstance(node, ast.ExprConstBool):
        return str(node.value).lower()
    if isinstance(node, ast.ExprConstDouble):
        return node.value
    if isinstance(node, ast.ExprConstFloat):
        return '%sf' % node.value
    if isinstance(node, ast.ExprConstInt):
        return str(node.value)
    if isinstance(node, ast.ExprConstUInt):
        return '%su' % str(node.value)
    raise Exception('Pretty printing failed at %s' % node)

def pretty_print(node):
    return pretty_print_node(node, 0)
