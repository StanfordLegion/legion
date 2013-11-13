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
### Leaf Task Analysis
###
### Leaf task analysis computes, for each task, whether that task can
### be declared as a leaf task or not. Leaf tasks must not make any
### calls into the Legion runtime. At the language level, this means
### leaf tasks must avoid:
###
###   * creating regions, index spaces, arrays, or partitions
###   * calling tasks
###   * downregion (generates a call to safe_cast)
###   * new (generates a call to create_index_allocator)
###
### When aggressive leaf task optimization is enabled, this pass
### allows the following features in leaf tasks, despite being
### potentially unsafe. The user must disable debug mode when using
### this optimization, in order to avoid assertion failures from
### illegal runtime calls.
###
###   * for (generates a call to get_index_space_domain)
###

# Backport of singledispatch to Python 2.x.
try:
    from functools import singledispatch
except ImportError:
    from singledispatch import singledispatch

import copy
from . import ast, types

class Context:
    def __init__(self, opts, type_map):
        self.opts = opts
        self.type_map = type_map
        self.leaf_tasks = {}

# Returns true if a function is safe to call in the context of a leaf
# task. Functions are considered safe if:
#
#   * The function is a foreign function.
#   * The function does not take a ForeignContext parameter, or
#     aggressive leaf task optimization is enabled.
def is_function_safe(function_type, cx):
    if not types.is_foreign_function(function_type):
        return False
    if not cx.opts.leaf_task_optimization:
        for arg in function_type.foreign_param_types:
            if types.is_foreign_context(arg):
                return False
    return True

@singledispatch
def leaf_task_analysis_node(node, cx):
    raise Exception('Leaf task analysis failed at %s' % node)

@leaf_task_analysis_node.register(ast.Program)
def _(node, cx):
    leaf_task_analysis_node(node.definitions, cx)
    return

@leaf_task_analysis_node.register(ast.Definitions)
def _(node, cx):
    for definition in node.definitions:
        leaf_task_analysis_node(definition, cx)
    return

@leaf_task_analysis_node.register(ast.Import)
def _(node, cx):
    return

@leaf_task_analysis_node.register(ast.Struct)
def _(node, cx):
    return

@leaf_task_analysis_node.register(ast.Function)
def _(node, cx):
    leaf_task = leaf_task_analysis_node(node.block, cx)
    assert leaf_task in (True, False)
    cx.leaf_tasks[node] = leaf_task
    return

@leaf_task_analysis_node.register(ast.Block)
def _(node, cx):
    return all(
        leaf_task_analysis_node(expr, cx)
        for expr in node.block)

@leaf_task_analysis_node.register(ast.StatementAssert)
def _(node, cx):
    return leaf_task_analysis_node(node.expr, cx)

@leaf_task_analysis_node.register(ast.StatementExpr)
def _(node, cx):
    return leaf_task_analysis_node(node.expr, cx)

@leaf_task_analysis_node.register(ast.StatementIf)
def _(node, cx):
    return (
        leaf_task_analysis_node(node.condition, cx) and
        leaf_task_analysis_node(node.then_block, cx) and
        (leaf_task_analysis_node(node.else_block, cx)
         if node.else_block is not None else True))

@leaf_task_analysis_node.register(ast.StatementFor)
def _(node, cx):
    if not cx.opts.leaf_task_optimization:
        return False
    return leaf_task_analysis_node(node.block, cx)

@leaf_task_analysis_node.register(ast.StatementLet)
def _(node, cx):
    return leaf_task_analysis_node(node.expr, cx)

@leaf_task_analysis_node.register(ast.StatementLetRegion)
def _(node, cx):
    return False

@leaf_task_analysis_node.register(ast.StatementLetArray)
def _(node, cx):
    return False

@leaf_task_analysis_node.register(ast.StatementLetIspace)
def _(node, cx):
    return False

@leaf_task_analysis_node.register(ast.StatementLetPartition)
def _(node, cx):
    return False

@leaf_task_analysis_node.register(ast.StatementReturn)
def _(node, cx):
    return leaf_task_analysis_node(node.expr, cx)

@leaf_task_analysis_node.register(ast.StatementUnpack)
def _(node, cx):
    return leaf_task_analysis_node(node.expr, cx)

@leaf_task_analysis_node.register(ast.StatementVar)
def _(node, cx):
    return leaf_task_analysis_node(node.expr, cx)

@leaf_task_analysis_node.register(ast.StatementWhile)
def _(node, cx):
    return (
        leaf_task_analysis_node(node.condition, cx) and
        leaf_task_analysis_node(node.block, cx))

@leaf_task_analysis_node.register(ast.ExprID)
def _(node, cx):
    return True

@leaf_task_analysis_node.register(ast.ExprAssignment)
def _(node, cx):
    return (
        leaf_task_analysis_node(node.lval, cx) and
        leaf_task_analysis_node(node.rval, cx))

@leaf_task_analysis_node.register(ast.ExprUnaryOp)
def _(node, cx):
    return leaf_task_analysis_node(node.arg, cx)

@leaf_task_analysis_node.register(ast.ExprBinaryOp)
def _(node, cx):
    return (
        leaf_task_analysis_node(node.lhs, cx) and
        leaf_task_analysis_node(node.rhs, cx))

@leaf_task_analysis_node.register(ast.ExprReduceOp)
def _(node, cx):
    return (
        leaf_task_analysis_node(node.lhs, cx) and
        leaf_task_analysis_node(node.rhs, cx))

@leaf_task_analysis_node.register(ast.ExprCast)
def _(node, cx):
    return leaf_task_analysis_node(node.expr, cx)

@leaf_task_analysis_node.register(ast.ExprNull)
def _(node, cx):
    return True

@leaf_task_analysis_node.register(ast.ExprIsnull)
def _(node, cx):
    return leaf_task_analysis_node(node.pointer_expr, cx)

@leaf_task_analysis_node.register(ast.ExprNew)
def _(node, cx):
    return False

@leaf_task_analysis_node.register(ast.ExprRead)
def _(node, cx):
    return leaf_task_analysis_node(node.pointer_expr, cx)

@leaf_task_analysis_node.register(ast.ExprWrite)
def _(node, cx):
    return (
        leaf_task_analysis_node(node.pointer_expr, cx) and
        leaf_task_analysis_node(node.value_expr, cx))

@leaf_task_analysis_node.register(ast.ExprReduce)
def _(node, cx):
    return (
        leaf_task_analysis_node(node.pointer_expr, cx) and
        leaf_task_analysis_node(node.value_expr, cx))

@leaf_task_analysis_node.register(ast.ExprDereference)
def _(node, cx):
    return leaf_task_analysis_node(node.pointer_expr, cx)

@leaf_task_analysis_node.register(ast.ExprArrayAccess)
def _(node, cx):
    return (
        leaf_task_analysis_node(node.array_expr, cx) and
        leaf_task_analysis_node(node.index_expr, cx))

@leaf_task_analysis_node.register(ast.ExprFieldAccess)
def _(node, cx):
    return leaf_task_analysis_node(node.struct_expr, cx)

@leaf_task_analysis_node.register(ast.ExprFieldDereference)
def _(node, cx):
    return leaf_task_analysis_node(node.pointer_expr, cx)

@leaf_task_analysis_node.register(ast.ExprFieldValues)
def _(node, cx):
    return leaf_task_analysis_node(node.field_values, cx)

@leaf_task_analysis_node.register(ast.FieldValues)
def _(node, cx):
    return all(leaf_task_analysis_node(field_value, cx)
               for field_value in node.field_values)

@leaf_task_analysis_node.register(ast.FieldValue)
def _(node, cx):
    return leaf_task_analysis_node(node.value_expr, cx)

@leaf_task_analysis_node.register(ast.ExprFieldUpdates)
def _(node, cx):
    return (
        leaf_task_analysis_node(node.struct_expr, cx) and
        leaf_task_analysis_node(node.field_updates, cx))

@leaf_task_analysis_node.register(ast.FieldUpdates)
def _(node, cx):
    return all(leaf_task_analysis_node(field_update, cx)
               for field_update in node.field_updates)

@leaf_task_analysis_node.register(ast.FieldUpdate)
def _(node, cx):
    return leaf_task_analysis_node(node.update_expr, cx)

@leaf_task_analysis_node.register(ast.ExprColoring)
def _(node, cx):
    return True

@leaf_task_analysis_node.register(ast.ExprColor)
def _(node, cx):
    return (
        leaf_task_analysis_node(node.coloring_expr, cx) and
        leaf_task_analysis_node(node.pointer_expr, cx) and
        leaf_task_analysis_node(node.color_expr, cx))

@leaf_task_analysis_node.register(ast.ExprUpregion)
def _(node, cx):
    return leaf_task_analysis_node(node.expr, cx)

@leaf_task_analysis_node.register(ast.ExprDownregion)
def _(node, cx):
    return False

@leaf_task_analysis_node.register(ast.ExprPack)
def _(node, cx):
    return leaf_task_analysis_node(node.expr, cx)

@leaf_task_analysis_node.register(ast.ExprCall)
def _(node, cx):
    if not is_function_safe(cx.type_map[node.function], cx):
        return False
    return leaf_task_analysis_node(node.args, cx)

@leaf_task_analysis_node.register(ast.Args)
def _(node, cx):
    return all(leaf_task_analysis_node(arg, cx)
               for arg in node.args)

@leaf_task_analysis_node.register(ast.ExprConstBool)
def _(node, cx):
    return True

@leaf_task_analysis_node.register(ast.ExprConstDouble)
def _(node, cx):
    return True

@leaf_task_analysis_node.register(ast.ExprConstFloat)
def _(node, cx):
    return True

@leaf_task_analysis_node.register(ast.ExprConstInt)
def _(node, cx):
    return True

@leaf_task_analysis_node.register(ast.ExprConstUInt)
def _(node, cx):
    return True

def leaf_task_analysis(program, opts, type_map):
    cx = Context(opts, type_map)
    leaf_task_analysis_node(program, cx)
    return cx.leaf_tasks
