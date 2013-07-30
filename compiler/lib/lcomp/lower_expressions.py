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
### Lower Expressions
###
### This pass transforms the expression AST into an RTL-like form. For
### example, compound expressions like the following
###
###     var x = (y + z) + w;
###
### will be transformed into
###
###     let _1 = y + z;
###     let _2 = _1 + w;
###     var x = _2;
###
### When this pass is run, all compound expressions will be
### transformed into a series of statements, where each subexpression
### is let-bound to a unique name. Note that field accesses are not
### touched by this pass, because the language needs to know what
### fields are being accessed at the time of the pointer
### dereference. So for example, the following
###
###     x = *(y->next) = 1;
###
### will be transformed into
###
###     let _3 = 1;
###     let _4 = y->next;
###     let _5 = (*_4 = _3);
###     let _6 = (x = _5);
###

import copy
from . import ast, types

_temp_name_counter = 0
def temp_name():
    global _temp_name_counter
    _temp_name_counter += 1
    return '_%s' % _temp_name_counter

class Context:
    def __init__(self, type_map):
        self.type_map = type_map

class Block:
    def __init__(self, block):
        self.block = block

class Statement:
    def __init__(self, actions):
        self.actions = actions

class Expr:
    def __init__(self, expr, actions):
        self.expr = expr
        self.actions = actions

class Value:
    def __init__(self, expr, actions):
        self._expr = expr
        self._actions = actions
    def read(self):
        name = temp_name()
        return Expr(
            ast.ExprID(
                span = self._expr.span,
                name = name),
            self._actions +
            [ast.StatementLet(
                    span = self._expr.span,
                    name = name,
                    type = None,
                    expr = self._expr)])
    def write(self):
        raise Exception('Expression is not an l-value')
    def get_field(self, field_name):
        name = temp_name()
        return StackReference(
            ast.ExprID(
                span = self._expr.span,
                name = name),
            self._actions +
            [ast.StatementLet(
                    span = self._expr.span,
                    name = name,
                    type = None,
                    expr = self._expr)],
            [field_name])

class Effect:
    def __init__(self, expr, actions):
        self._expr = expr
        self._actions = actions
    def read(self):
        return Expr(
            None,
            self._actions +
            [ast.StatementExpr(
                    span = self._expr.span,
                    expr = self._expr)])
    def write(self):
        raise Exception('Expression is not an l-value')

def wrap_field_access(node, field_name):
    return ast.ExprFieldAccess(
        span = node.span,
        struct_expr = node,
        field_name = field_name)

class Reference:
    def __init__(self, expr, actions, field_path = []):
        self._expr = expr
        self._actions = actions
        self.field_path = field_path
    def read(self):
        name = temp_name()
        return Expr(
            ast.ExprID(
                span = self._expr.span,
                name = name),
            self._actions +
            [ast.StatementLet(
                    span = self._expr.span,
                    name = name,
                    type = None,
                    expr = reduce(
                        wrap_field_access,
                        self.field_path,
                        self._expr))])
    def write(self):
        return Expr(
            reduce(wrap_field_access, self.field_path, self._expr),
            self._actions)
    def get_field(self, field_name):
        return Reference(self._expr, self._actions, self.field_path + [field_name])

class StackReference:
    def __init__(self, expr, actions, field_path = []):
        self._expr = expr
        self._actions = actions
        self.field_path = field_path
    def read(self):
        return Expr(
            reduce(
                wrap_field_access,
                self.field_path,
                self._expr),
            self._actions)
    def write(self):
        return self.read()
    def get_field(self, field_name):
        return StackReference(self._expr, self._actions, self.field_path + [field_name])

def lower_helper(node, cx):
    if isinstance(node, ast.Program):
        ll_program = ast.Program(span = node.span)
        for definition in node.defs:
            ll_def = lower_helper(definition, cx)
            ll_program.defs.extend(ll_def.actions)
        return ll_program
    if isinstance(node, ast.Import):
        return Statement([node])
    if isinstance(node, ast.Struct):
        return Statement([node])
    if isinstance(node, ast.Function):
        ll_block = lower_helper(node.block, cx).block
        ll_function = ast.Function(
            span = node.span,
            name = node.name,
            params = node.params,
            return_type = node.return_type,
            privileges = node.privileges,
            block = ll_block)
        return Statement([ll_function])
    if isinstance(node, ast.Block):
        ll_block = ast.Block(node.span)
        for statement in node.block:
            ll_statements = lower_helper(statement, cx)
            ll_block.block.extend(ll_statements.actions)
        return Block(ll_block)
    if isinstance(node, ast.StatementAssert):
        ll_expr = lower_helper(node.expr, cx).read()
        return Statement(
            ll_expr.actions +
            [ast.StatementAssert(
                    span = node.span,
                    expr = ll_expr.expr)])
    if isinstance(node, ast.StatementExpr):
        return Statement(
            lower_helper(node.expr, cx).read().actions)
    if isinstance(node, ast.StatementIf):
        ll_condition = lower_helper(node.condition, cx).read()
        then_block = lower_helper(node.then_block, cx).block
        else_block = (lower_helper(node.else_block, cx).block
                      if node.else_block is not None else None)
        return Statement(
            ll_condition.actions +
            [ast.StatementIf(
                    span = node.span,
                    condition = ll_condition.expr,
                    then_block = then_block,
                    else_block = else_block)])
    if isinstance(node, ast.StatementFor):
        ll_block = lower_helper(node.block, cx).block
        return Statement(
            [ast.StatementFor(
                    span = node.span,
                    indices = node.indices,
                    regions = node.regions,
                    block = ll_block)])
    if isinstance(node, ast.StatementLet):
        ll_expr = lower_helper(node.expr, cx).read()
        return Statement(
            ll_expr.actions +
            [ast.StatementLet(
                    span = node.span,
                    name = node.name,
                    type = node.type,
                    expr = ll_expr.expr)])
    if isinstance(node, ast.StatementLetRegion):
        ll_size_expr = lower_helper(node.size_expr, cx).read()
        return Statement(
            ll_size_expr.actions +
            [ast.StatementLetRegion(
                    span = node.span,
                    name = node.name,
                    region_kind = node.region_kind,
                    element_type = node.element_type,
                    size_expr = ll_size_expr.expr)])
    if isinstance(node, ast.StatementLetArray):
        return Statement([node])
    if isinstance(node, ast.StatementLetIspace):
        ll_size_expr = lower_helper(node.size_expr, cx).read()
        return Statement(
            ll_size_expr.actions +
            [ast.StatementLetIspace(
                    span = node.span,
                    name = node.name,
                    ispace_kind = node.ispace_kind,
                    index_type = node.index_type,
                    size_expr = ll_size_expr.expr)])
    if isinstance(node, ast.StatementPartition):
        ll_expr = lower_helper(node.expr, cx).read()
        return Statement(
            ll_expr.actions +
            [ast.StatementPartition(
                    span = node.span,
                    parent = node.parent,
                    expr = ll_expr.expr,
                    regions = node.regions)])
    if isinstance(node, ast.StatementReturn):
        ll_expr = lower_helper(node.expr, cx).read()
        return Statement(
            ll_expr.actions +
            [ast.StatementReturn(
                    span = node.span,
                    expr = ll_expr.expr)])
    if isinstance(node, ast.StatementUnpack):
        ll_expr = lower_helper(node.expr, cx).read()
        return Statement(
            ll_expr.actions +
            [ast.StatementUnpack(
                    span = node.span,
                    expr = ll_expr.expr,
                    name = node.name,
                    type = node.type,
                    regions = node.regions)])
    if isinstance(node, ast.StatementVar):
        ll_expr = lower_helper(node.expr, cx).read()
        return Statement(
            ll_expr.actions +
            [ast.StatementVar(
                    span = node.span,
                    name = node.name,
                    type = node.type,
                    expr = ll_expr.expr)])
    if isinstance(node, ast.StatementWhile):
        ll_condition_top = lower_helper(node.condition, cx).read()
        ll_condition_bot = lower_helper(node.condition, cx).read()
        temp_condition = temp_name()
        ll_block = lower_helper(node.block, cx).block
        for action in ll_condition_bot.actions:
            ll_block = ast.Block(
                span = ll_block.span,
                block = ll_block,
                statement = action)
        ll_block = ast.Block(
            span = ll_block.span,
            block = ll_block,
            statement = ast.StatementLet(
                span = node.span,
                name = temp_name(),
                type = None,
                expr = ast.ExprAssignment(
                    span = node.span,
                    lval = ast.ExprID(
                        span = node.span,
                        name = temp_condition),
                    rval = ll_condition_bot.expr)))
        return Statement(
            ll_condition_top.actions +
            [ast.StatementVar(
                    span = node.span,
                    name = temp_condition,
                    type = None,
                    expr = ll_condition_top.expr),
             ast.StatementWhile(
                    span = node.span,
                    condition = ast.ExprID(
                        span = node.span,
                        name = temp_condition),
                    block = ll_block)])
    if isinstance(node, ast.ExprID):
        return StackReference(
            ast.ExprID(
                span = node.span,
                name = node.name),
            [])
    if isinstance(node, ast.ExprAssignment):
        ll_lval = lower_helper(node.lval, cx).write()
        ll_rval = lower_helper(node.rval, cx).read()
        return Value(
            ast.ExprAssignment(
                span = node.span,
                lval = ll_lval.expr,
                rval = ll_rval.expr),
            ll_lval.actions + ll_rval.actions)
    if isinstance(node, ast.ExprUnaryOp):
        ll_arg = lower_helper(node.arg, cx).read()
        return Value(
            ast.ExprUnaryOp(
                span = node.span,
                op = node.op,
                arg = ll_arg.expr),
            ll_arg.actions)
    if isinstance(node, ast.ExprBinaryOp):
        ll_lhs = lower_helper(node.lhs, cx).read()
        ll_rhs = lower_helper(node.rhs, cx).read()
        return Value(
            ast.ExprBinaryOp(
                span = node.span,
                op = node.op,
                lhs = ll_lhs.expr,
                rhs = ll_rhs.expr),
            ll_lhs.actions + ll_rhs.actions)
    if isinstance(node, ast.ExprReduceOp):
        ll_lhs = lower_helper(node.lhs, cx).write()
        ll_rhs = lower_helper(node.rhs, cx).read()
        return Effect(
            ast.ExprReduceOp(
                span = node.span,
                op = node.op,
                lhs = ll_lhs.expr,
                rhs = ll_rhs.expr),
            ll_rhs.actions + ll_lhs.actions)
    if isinstance(node, ast.ExprCast):
        ll_expr = lower_helper(node.expr, cx).read()
        return Value(
            ast.ExprCast(
                span = node.span,
                cast_to_type = node.cast_to_type,
                expr = ll_expr.expr),
            ll_expr.actions)
    if isinstance(node, ast.ExprNull):
        return Value(node, [])
    if isinstance(node, ast.ExprIsnull):
        ll_pointer_expr = lower_helper(node.pointer_expr, cx).read()
        return Value(
            ast.ExprIsnull(
                span = node.span,
                pointer_expr = ll_pointer_expr.expr),
            ll_pointer_expr.actions)
    if isinstance(node, ast.ExprNew):
        return Value(node, [])
    if isinstance(node, ast.ExprRead):
        ll_pointer_expr = lower_helper(node.pointer_expr, cx).read()
        return Value(
            ast.ExprRead(
                span = node.span,
                pointer_expr = ll_pointer_expr.expr),
            ll_pointer_expr.actions)
    if isinstance(node, ast.ExprWrite):
        ll_pointer_expr = lower_helper(node.pointer_expr, cx).read()
        ll_value_expr = lower_helper(node.value_expr, cx).read()
        return Effect(
            ast.ExprWrite(
                span = node.span,
                pointer_expr = ll_pointer_expr.expr,
                value_expr = ll_value_expr.expr),
            ll_pointer_expr.actions + ll_value_expr.actions)
    if isinstance(node, ast.ExprReduce):
        ll_pointer_expr = lower_helper(node.pointer_expr, cx).read()
        ll_value_expr = lower_helper(node.value_expr, cx).read()
        return Effect(
            ast.ExprReduce(
                span = node.span,
                op = node.op,
                pointer_expr = ll_pointer_expr.expr,
                value_expr = ll_value_expr.expr),
            ll_pointer_expr.actions + ll_value_expr.actions)
    if isinstance(node, ast.ExprDereference):
        ll_pointer_expr = lower_helper(node.pointer_expr, cx).read()
        return Reference(
            ast.ExprDereference(
                span = node.span,
                pointer_expr = ll_pointer_expr.expr),
            ll_pointer_expr.actions)
    if isinstance(node, ast.ExprArrayAccess):
        ll_array_expr = lower_helper(node.array_expr, cx).read()
        ll_index_expr = lower_helper(node.index_expr, cx).read()
        return Reference(
            ast.ExprArrayAccess(
                span = node.span,
                array_expr = ll_array_expr.expr,
                index_expr = ll_index_expr.expr),
            ll_array_expr.actions + ll_index_expr.actions)
    if isinstance(node, ast.ExprFieldAccess):
        ll_struct_expr = lower_helper(node.struct_expr, cx)
        return ll_struct_expr.get_field(node.field_name)
    if isinstance(node, ast.ExprFieldDereference):
        ll_pointer_expr = lower_helper(node.pointer_expr, cx).read()
        return Reference(
            ast.ExprFieldDereference(
                span = node.span,
                pointer_expr = ll_pointer_expr.expr,
                field_name = node.field_name),
            ll_pointer_expr.actions)
    if isinstance(node, ast.ExprFieldValues):
        ll_field_values = lower_helper(node.field_values, cx)
        return Value(
            ast.ExprFieldValues(
                span = node.span,
                field_values = ll_field_values.expr),
            ll_field_values.actions)
    if isinstance(node, ast.FieldValues):
        ll_field_values = ast.FieldValues(
            span = node.span)
        ll_field_actions = []
        for field_value in node.field_values:
            ll_field_value = lower_helper(field_value, cx)
            ll_field_values = ast.FieldValues(
                span = node.span,
                field_values = ll_field_values,
                field_value = ll_field_value.expr)
            ll_field_actions.extend(ll_field_value.actions)
        return Expr(ll_field_values, ll_field_actions)
    if isinstance(node, ast.FieldValue):
        ll_value_expr = lower_helper(node.value_expr, cx).read()
        return Expr(
            ast.FieldValue(
                span = node.span,
                field_name = node.field_name,
                value_expr = ll_value_expr.expr),
            ll_value_expr.actions)
    if isinstance(node, ast.ExprFieldUpdates):
        ll_struct_expr = lower_helper(node.struct_expr, cx).read()
        ll_field_updates = lower_helper(node.field_updates, cx)
        return Value(
            ast.ExprFieldUpdates(
                span = node.span,
                struct_expr = ll_struct_expr.expr,
                field_updates = ll_field_updates.expr),
            ll_struct_expr.actions + ll_field_updates.actions)
    if isinstance(node, ast.FieldUpdates):
        ll_field_updates = ast.FieldUpdates(
            span = node.span)
        ll_field_actions = []
        for field_update in node.field_updates:
            ll_field_update = lower_helper(field_update, cx)
            ll_field_updates = ast.FieldUpdates(
                span = node.span,
                field_updates = ll_field_updates,
                field_update = ll_field_update.expr)
            ll_field_actions.extend(ll_field_update.actions)
        return Expr(ll_field_updates, ll_field_actions)
    if isinstance(node, ast.FieldUpdate):
        ll_update_expr = lower_helper(node.update_expr, cx).read()
        return Expr(
            ast.FieldUpdate(
                span = node.span,
                field_name = node.field_name,
                update_expr = ll_update_expr.expr),
            ll_update_expr.actions)
    if isinstance(node, ast.ExprColoring):
        return Value(node, [])
    if isinstance(node, ast.ExprColor):
        ll_coloring_expr = lower_helper(node.coloring_expr, cx).read()
        ll_pointer_expr = lower_helper(node.pointer_expr, cx).read()
        ll_color_expr = lower_helper(node.color_expr, cx).read()
        return Value(
            ast.ExprColor(
                span = node.span,
                coloring_expr = ll_coloring_expr.expr,
                pointer_expr = ll_pointer_expr.expr,
                color_expr = ll_color_expr.expr),
            (ll_coloring_expr.actions +
             ll_pointer_expr.actions +
             ll_color_expr.actions))
    if isinstance(node, ast.ExprUpregion):
        ll_expr = lower_helper(node.expr, cx).read()
        return Value(
            ast.ExprUpregion(
                span = node.span,
                regions = node.regions,
                expr = ll_expr.expr),
            ll_expr.actions)
    if isinstance(node, ast.ExprDownregion):
        ll_expr = lower_helper(node.expr, cx).read()
        return Value(
            ast.ExprDownregion(
                span = node.span,
                regions = node.regions,
                expr = ll_expr.expr),
            ll_expr.actions)
    if isinstance(node, ast.ExprPack):
        ll_expr = lower_helper(node.expr, cx).read()
        return Value(
            ast.ExprPack(
                span = node.span,
                expr = ll_expr.expr,
                type = node.type,
                regions = node.regions),
            ll_expr.actions)
    if isinstance(node, ast.ExprCall):
        ll_function = lower_helper(node.function, cx).read()
        ll_args = lower_helper(node.args, cx)

        factory = Value
        if types.is_void(cx.type_map[node.function].return_type):
            factory = Effect
        return factory(
            ast.ExprCall(
                span = node.span,
                function = ll_function.expr,
                args = ll_args.expr),
            ll_function.actions + ll_args.actions)
    if isinstance(node, ast.Args):
        ll_args = ast.Args(
            span = node.span)
        ll_args_actions = []
        for arg in node.args:
            ll_arg = lower_helper(arg, cx).read()
            ll_args = ast.Args(
                span = node.span,
                args = ll_args,
                arg = ll_arg.expr)
            ll_args_actions.extend(ll_arg.actions)
        return Expr(ll_args, ll_args_actions)
    if isinstance(node, ast.ExprConstBool):
        return Value(node, [])
    if isinstance(node, ast.ExprConstDouble):
        return Value(node, [])
    if isinstance(node, ast.ExprConstInt):
        return Value(node, [])
    raise Exception('Expression lowering failed at %s' % node)

def lower(program, type_map):
    cx = Context(type_map)
    return lower_helper(program, cx)
