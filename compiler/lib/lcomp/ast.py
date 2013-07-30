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
### AST
###

from ply.lex import LexToken

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
    if isinstance(a, Span) and isinstance(b, Span):
        return True
    if isinstance(a, ASTNode) and isinstance(b, ASTNode):
        if a.__class__ is not b.__class__:
            return False
        for k in a.__dict__.iterkeys():
            if k not in b.__dict__ or not equivalent(a.__dict__[k], b.__dict__[k]):
                return False
        return True
    raise Exception('Equivalence undefined for type %s %s' % (
            type(a), type(b)))

class SpanPosition:
    def __init__(self, position, line, column):
        self.position = position
        self.line = line
        self.column = column
    def __repr__(self):
        return '%s:%s' % (self.line, self.column)
    @staticmethod
    def from_token_start(token):
        return SpanPosition(token.lexpos, token.lineno, token.column)
    @staticmethod
    def from_token_end(token):
        return SpanPosition(token.lexpos + token.length,
                            token.lineno,
                            token.column + token.length)

class Span:
    def __init__(self, source, start, end):
        self.source = source
        self.start = start
        self.end = end
    def __repr__(self):
        return '%s:%s' % (self.source, self.start)
    @staticmethod
    def from_slice(production):
        source = production.lexer.source

        nodes = [
            node
            for node in production.slice[1:]
            if node is not None and (
                isinstance(node, LexToken) or
                node.value is not None)]
        if len(nodes) < 1:
            return Span(source,
                        SpanPosition(0, 0, 0),
                        SpanPosition(0, 0, 0))

        node_start, node_end = nodes[0], nodes[-1]

        if isinstance(node_start, LexToken):
            position_start = SpanPosition.from_token_start(node_start)
        else:
            position_start = node_start.value.span.start

        if isinstance(node_end, LexToken):
            position_end = SpanPosition.from_token_end(node_end)
        else:
            position_end = node_end.value.span.end

        return Span(source, position_start, position_end)

class ASTNode:
    def __init__(self, span):
        self.span = span
    def __repr__(self):
        return '%s(%s)' % (
            self.__class__.__name__,
            ', '.join(
                '%s: %s' % (key, value)
                for key, value in self.__dict__.iteritems()
                if not key.startswith('_')))

class Program(ASTNode):
    def __init__(self, span, program = None, definition = None):
        ASTNode.__init__(self, span)
        self.defs = program.defs if program is not None else []
        self.defs += [definition] if definition is not None else []

class Import(ASTNode):
    def __init__(self, span, filename):
        ASTNode.__init__(self, span)
        self.filename = filename
        self.ast = None

class Struct(ASTNode):
    def __init__(self, span, name, params, regions, constraints, field_decls):
        ASTNode.__init__(self, span)
        self.name = name
        self.params = params
        self.regions = regions
        self.constraints = constraints
        self.field_decls = field_decls

class StructParams(ASTNode):
    def __init__(self, span, params = None, param = None):
        ASTNode.__init__(self, span)
        self.params = params.params if params is not None else []
        self.params += [param] if param is not None else []

class StructParam(ASTNode):
    def __init__(self, span, name, type):
        ASTNode.__init__(self, span)
        self.name = name
        self.type = type

class StructRegions(ASTNode):
    def __init__(self, span, regions = None, region = None):
        ASTNode.__init__(self, span)
        self.regions = regions.regions if regions is not None else []
        self.regions += [region] if region is not None else []

class StructRegion(ASTNode):
    def __init__(self, span, name, type):
        ASTNode.__init__(self, span)
        self.name = name
        self.type = type

class StructConstraints(ASTNode):
    def __init__(self, span, constraints = None, constraint = None):
        ASTNode.__init__(self, span)
        self.constraints = constraints.constraints if constraints is not None else []
        self.constraints += [constraint] if constraint is not None else []

class StructConstraint(ASTNode):
    def __init__(self, span, op, lhs, rhs):
        ASTNode.__init__(self, span)
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

class StructConstraintRegion(ASTNode):
    def __init__(self, span, name):
        ASTNode.__init__(self, span)
        self.name = name

class FieldDecls(ASTNode):
    def __init__(self, span, field_decls = None, field_decl = None):
        ASTNode.__init__(self, span)
        self.field_decls = field_decls.field_decls if field_decls is not None else []
        self.field_decls += [field_decl] if field_decl is not None else []

class FieldDecl(ASTNode):
    def __init__(self, span, name, field_type):
        ASTNode.__init__(self, span)
        self.name = name
        self.field_type = field_type

class Function(ASTNode):
    def __init__(self, span, name, params, return_type, privileges, block):
        ASTNode.__init__(self, span)
        self.name = name
        self.params = params
        self.return_type = return_type
        self.privileges = privileges
        self.block = block

class Params(ASTNode):
    def __init__(self, span, params = None, param = None):
        ASTNode.__init__(self, span)
        self.params = params.params if params is not None else []
        self.params += [param] if param is not None else []

class Param(ASTNode):
    def __init__(self, span, name, declared_type):
        ASTNode.__init__(self, span)
        self.name = name
        self.declared_type = declared_type

class Privileges(ASTNode):
    def __init__(self, span, privileges = None, privilege = None):
        ASTNode.__init__(self, span)
        self.privileges = privileges.privileges if privileges is not None else []
        self.privileges += [privilege] if privilege is not None else []

class Privilege(ASTNode):
    def __init__(self, span, privilege, regions, op = None):
        ASTNode.__init__(self, span)
        self.privilege = privilege
        self.regions = regions
        self.op = op

class PrivilegeRegions(ASTNode):
    def __init__(self, span, regions = None, region = None):
        ASTNode.__init__(self, span)
        self.regions = regions.regions if regions is not None else []
        self.regions += [region] if region is not None else []

class PrivilegeRegion(ASTNode):
    def __init__(self, span, name, fields):
        ASTNode.__init__(self, span)
        self.name = name
        self.fields = fields

class PrivilegeRegionFields(ASTNode):
    def __init__(self, span, fields = None, field = None):
        ASTNode.__init__(self, span)
        self.fields = fields.fields if fields is not None else []
        self.fields += [field] if field is not None else []

class PrivilegeRegionField(ASTNode):
    def __init__(self, span, name, fields):
        ASTNode.__init__(self, span)
        self.name = name
        self.fields = fields

class TypeVoid(ASTNode): pass
class TypeBool(ASTNode): pass
class TypeDouble(ASTNode): pass
class TypeFloat(ASTNode): pass

class TypeInt(ASTNode): pass
class TypeUInt(ASTNode): pass

class TypeInt8(ASTNode): pass
class TypeInt16(ASTNode): pass
class TypeInt32(ASTNode): pass
class TypeInt64(ASTNode): pass
class TypeUInt8(ASTNode): pass
class TypeUInt16(ASTNode): pass
class TypeUInt32(ASTNode): pass
class TypeUInt64(ASTNode): pass

class TypeColoring(ASTNode):
    def __init__(self, span, region):
        ASTNode.__init__(self, span)
        self.region = region

class TypeColoringRegion(ASTNode):
    def __init__(self, span, name):
        ASTNode.__init__(self, span)
        self.name = name

class TypeID(ASTNode):
    def __init__(self, span, name, args):
        ASTNode.__init__(self, span)
        self.name = name
        self.args = args

class TypeArgs(ASTNode):
    def __init__(self, span, args = None, arg = None):
        ASTNode.__init__(self, span)
        self.args = args.args if args is not None else []
        self.args += [arg] if arg is not None else []

class TypeArg(ASTNode):
    def __init__(self, span, name):
        ASTNode.__init__(self, span)
        self.name = name

class TypeArgWild(ASTNode): pass

class TypePointer(ASTNode):
    def __init__(self, span, points_to_type, regions):
        ASTNode.__init__(self, span)
        self.points_to_type = points_to_type
        self.regions = regions

class TypePointerRegions(ASTNode):
    def __init__(self, span, regions = None, region = None):
        ASTNode.__init__(self, span)
        self.regions = regions.regions if regions is not None else []
        self.regions += [region] if region is not None else []

class TypeRegion(ASTNode):
    def __init__(self, span, name):
        ASTNode.__init__(self, span)
        self.name = name

class TypeRegionWild(ASTNode): pass

class TypeRegionKind(ASTNode):
    def __init__(self, span, contains_type):
        ASTNode.__init__(self, span)
        self.contains_type = contains_type

class TypeArrayKind(ASTNode):
    def __init__(self, span, ispace, contains_type):
        ASTNode.__init__(self, span)
        self.ispace = ispace
        self.contains_type = contains_type

class TypeIspace(ASTNode):
    def __init__(self, span, name):
        ASTNode.__init__(self, span)
        self.name = name

class TypeIspaceKind(ASTNode):
    def __init__(self, span, index_type):
        ASTNode.__init__(self, span)
        self.index_type = index_type

class Block(ASTNode):
    def __init__(self, span, block = None, statement = None):
        ASTNode.__init__(self, span)
        self.block = block.block if block is not None else []
        self.block += [statement] if statement is not None else []

class StatementAssert(ASTNode):
    def __init__(self, span, expr):
        ASTNode.__init__(self, span)
        self.expr = expr

class StatementExpr(ASTNode):
    def __init__(self, span, expr):
        ASTNode.__init__(self, span)
        self.expr = expr

class StatementIf(ASTNode):
    def __init__(self, span, condition, then_block, else_block = None):
        ASTNode.__init__(self, span)
        self.condition = condition
        self.then_block = then_block
        self.else_block = else_block

class StatementFor(ASTNode):
    def __init__(self, span, indices, regions, block):
        ASTNode.__init__(self, span)
        self.indices = indices
        self.regions = regions
        self.block = block

class ForIndices(ASTNode):
    def __init__(self, span, indices = None, index = None):
        ASTNode.__init__(self, span)
        self.indices = indices.indices if indices is not None else []
        self.indices += [index] if index is not None else []

class ForIndex(ASTNode):
    def __init__(self, span, name, type):
        ASTNode.__init__(self, span)
        self.name = name
        self.type = type

class ForRegions(ASTNode):
    def __init__(self, span, regions = None, region = None):
        ASTNode.__init__(self, span)
        self.regions = regions.regions if regions is not None else []
        self.regions += [region] if region is not None else []

class ForRegion(ASTNode):
    def __init__(self, span, name):
        ASTNode.__init__(self, span)
        self.name = name

class StatementLet(ASTNode):
    def __init__(self, span, name, type, expr):
        ASTNode.__init__(self, span)
        self.name = name
        self.type = type
        self.expr = expr

class StatementLetRegion(ASTNode):
    def __init__(self, span, name, region_kind, element_type, size_expr):
        ASTNode.__init__(self, span)
        self.name = name
        self.region_kind = region_kind
        self.element_type = element_type
        self.size_expr = size_expr

class StatementLetArray(ASTNode):
    def __init__(self, span, name, region_kind, ispace_type, element_type):
        ASTNode.__init__(self, span)
        self.name = name
        self.region_kind = region_kind
        self.ispace_type = ispace_type
        self.element_type = element_type

class StatementLetIspace(ASTNode):
    def __init__(self, span, name, ispace_kind, index_type, size_expr):
        ASTNode.__init__(self, span)
        self.name = name
        self.ispace_kind = ispace_kind
        self.index_type = index_type
        self.size_expr = size_expr

class StatementLetPartition(ASTNode):
    def __init__(self, span, name, partition_kind, region_type, mode, coloring_expr):
        ASTNode.__init__(self, span)
        self.name = name
        self.partition_kind = partition_kind
        self.region_type = region_type
        self.mode = mode
        self.coloring_expr = coloring_expr

class PartitionMode(ASTNode):
    def __init__(self, span, mode):
        ASTNode.__init__(self, span)
        self.mode = mode

class StatementReturn(ASTNode):
    def __init__(self, span, expr):
        ASTNode.__init__(self, span)
        self.expr = expr

class StatementUnpack(ASTNode):
    def __init__(self, span, expr, name, type, regions):
        ASTNode.__init__(self, span)
        self.expr = expr
        self.name = name
        self.type = type
        self.regions = regions

class UnpackRegions(ASTNode):
    def __init__(self, span, regions = None, region = None):
        ASTNode.__init__(self, span)
        self.regions = regions.regions if regions is not None else []
        self.regions += [region] if region is not None else []

class UnpackRegion(ASTNode):
    def __init__(self, span, name):
        ASTNode.__init__(self, span)
        self.name = name

class StatementVar(ASTNode):
    def __init__(self, span, name, type, expr):
        ASTNode.__init__(self, span)
        self.name = name
        self.type = type
        self.expr = expr

class StatementWhile(ASTNode):
    def __init__(self, span, condition, block):
        ASTNode.__init__(self, span)
        self.condition = condition
        self.block = block

class ExprID(ASTNode):
    def __init__(self, span, name):
        ASTNode.__init__(self, span)
        self.name = name

class ExprAssignment(ASTNode):
    def __init__(self, span, lval, rval):
        ASTNode.__init__(self, span)
        self.lval = lval
        self.rval = rval

class ExprUnaryOp(ASTNode):
    def __init__(self, span, op, arg):
        ASTNode.__init__(self, span)
        self.op = op
        self.arg = arg

class ExprBinaryOp(ASTNode):
    def __init__(self, span, op, lhs, rhs):
        ASTNode.__init__(self, span)
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

class ExprReduceOp(ASTNode):
    def __init__(self, span, op, lhs, rhs):
        ASTNode.__init__(self, span)
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

class ExprCast(ASTNode):
    def __init__(self, span, cast_to_type, expr):
        ASTNode.__init__(self, span)
        self.cast_to_type = cast_to_type
        self.expr = expr

class ExprNull(ASTNode):
    def __init__(self, span, pointer_type):
        ASTNode.__init__(self, span)
        self.pointer_type = pointer_type

class ExprIsnull(ASTNode):
    def __init__(self, span, pointer_expr):
        ASTNode.__init__(self, span)
        self.pointer_expr = pointer_expr

class ExprNew(ASTNode):
    def __init__(self, span, pointer_type):
        ASTNode.__init__(self, span)
        self.pointer_type = pointer_type

class ExprRead(ASTNode):
    def __init__(self, span, pointer_expr):
        ASTNode.__init__(self, span)
        self.pointer_expr = pointer_expr

class ExprWrite(ASTNode):
    def __init__(self, span, pointer_expr, value_expr):
        ASTNode.__init__(self, span)
        self.pointer_expr = pointer_expr
        self.value_expr = value_expr

class ExprReduce(ASTNode):
    def __init__(self, span, op, pointer_expr, value_expr):
        ASTNode.__init__(self, span)
        self.op = op
        self.pointer_expr = pointer_expr
        self.value_expr = value_expr

class ExprDereference(ASTNode):
    def __init__(self, span, pointer_expr):
        ASTNode.__init__(self, span)
        self.pointer_expr = pointer_expr

class ExprArrayAccess(ASTNode):
    def __init__(self, span, array_expr, index_expr):
        ASTNode.__init__(self, span)
        self.array_expr = array_expr
        self.index_expr = index_expr

class ExprFieldAccess(ASTNode):
    def __init__(self, span, struct_expr, field_name):
        ASTNode.__init__(self, span)
        self.struct_expr = struct_expr
        self.field_name = field_name

class ExprFieldDereference(ASTNode):
    def __init__(self, span, pointer_expr, field_name):
        ASTNode.__init__(self, span)
        self.pointer_expr = pointer_expr
        self.field_name = field_name

class ExprFieldValues(ASTNode):
    def __init__(self, span, field_values):
        ASTNode.__init__(self, span)
        self.field_values = field_values

class FieldValues(ASTNode):
    def __init__(self, span, field_values = None, field_value = None):
        ASTNode.__init__(self, span)
        self.field_values = field_values.field_values if field_values is not None else []
        self.field_values += [field_value] if field_value is not None else []

class FieldValue(ASTNode):
    def __init__(self, span, field_name, value_expr):
        ASTNode.__init__(self, span)
        self.field_name = field_name
        self.value_expr = value_expr

class ExprFieldUpdates(ASTNode):
    def __init__(self, span, struct_expr, field_updates):
        ASTNode.__init__(self, span)
        self.struct_expr = struct_expr
        self.field_updates = field_updates

class FieldUpdates(ASTNode):
    def __init__(self, span, field_updates = None, field_update = None):
        ASTNode.__init__(self, span)
        self.field_updates = field_updates.field_updates if field_updates is not None else []
        self.field_updates += [field_update] if field_update is not None else []

class FieldUpdate(ASTNode):
    def __init__(self, span, field_name, update_expr):
        ASTNode.__init__(self, span)
        self.field_name = field_name
        self.update_expr = update_expr

class ExprColoring(ASTNode):
    def __init__(self, span, region):
        ASTNode.__init__(self, span)
        self.region = region

class ColoringRegion(ASTNode):
    def __init__(self, span, name):
        ASTNode.__init__(self, span)
        self.name = name

class ExprColor(ASTNode):
    def __init__(self, span, coloring_expr, pointer_expr, color_expr):
        ASTNode.__init__(self, span)
        self.coloring_expr = coloring_expr
        self.pointer_expr = pointer_expr
        self.color_expr = color_expr

class ExprUpregion(ASTNode):
    def __init__(self, span, regions, expr):
        ASTNode.__init__(self, span)
        self.regions = regions
        self.expr = expr

class UpregionRegions(ASTNode):
    def __init__(self, span, regions = None, region = None):
        ASTNode.__init__(self, span)
        self.regions = regions.regions if regions is not None else []
        self.regions += [region] if region is not None else []

class UpregionRegion(ASTNode):
    def __init__(self, span, name):
        ASTNode.__init__(self, span)
        self.name = name

class ExprDownregion(ASTNode):
    def __init__(self, span, regions, expr):
        ASTNode.__init__(self, span)
        self.regions = regions
        self.expr = expr

class DownregionRegions(ASTNode):
    def __init__(self, span, regions = None, region = None):
        ASTNode.__init__(self, span)
        self.regions = regions.regions if regions is not None else []
        self.regions += [region] if region is not None else []

class DownregionRegion(ASTNode):
    def __init__(self, span, name):
        ASTNode.__init__(self, span)
        self.name = name

class ExprPack(ASTNode):
    def __init__(self, span, expr, type, regions):
        ASTNode.__init__(self, span)
        self.expr = expr
        self.type = type
        self.regions = regions

class PackRegions(ASTNode):
    def __init__(self, span, regions = None, region = None):
        ASTNode.__init__(self, span)
        self.regions = regions.regions if regions is not None else []
        self.regions += [region] if region is not None else []

class PackRegion(ASTNode):
    def __init__(self, span, name):
        ASTNode.__init__(self, span)
        self.name = name

class ExprCall(ASTNode):
    def __init__(self, span, function, args):
        ASTNode.__init__(self, span)
        self.function = function
        self.args = args

class Args(ASTNode):
    def __init__(self, span, args = None, arg = None):
        ASTNode.__init__(self, span)
        self.args = args.args if args is not None else []
        self.args += [arg] if arg is not None else []

class ExprConstBool(ASTNode):
    def __init__(self, span, value):
        ASTNode.__init__(self, span)
        self.value = value

class ExprConstDouble(ASTNode):
    def __init__(self, span, value):
        ASTNode.__init__(self, span)
        self.value = value

class ExprConstInt(ASTNode):
    def __init__(self, span, value):
        ASTNode.__init__(self, span)
        self.value = value

class ExprConstUInt(ASTNode):
    def __init__(self, span, value):
        ASTNode.__init__(self, span)
        self.value = value
