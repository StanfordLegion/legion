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
### Region Analysis
###
### This pass analyzes region usage to determine, for each region,
### whether that region will be read, written to, or allocated. This
### is useful in code gen where the usage modes of each region need to
### be known in order to create accessors of the appropriate types.
###

# Backport of singledispatch to Python 2.x.
try:
    from functools import singledispatch
except ImportError:
    from singledispatch import singledispatch

import copy
from . import ast, types

class Context:
    def __init__(self, type_map):
        self.type_map = type_map
        self.access_modes = {}

ALL_FIELDS = ()

class Mode:
    def __init__(self, mode, op = None):
        self.mode = mode
        self.op = op
    def __eq__(self, other):
        return types.is_same_class(self, other) and self.mode == other.mode and \
            (self.op == other.op if self.op is not None and other.op is not None else True)
    def __hash__(self):
        return hash(('mode', self.mode))
    def __call__(self, args):
        assert self.op is None
        assert len(args) == 1
        return Mode(self.mode, args[0])
    def __repr__(self):
        if self.op is None:
            return self.mode
        return '%s<%s>' % (self.mode, self.op)

READ = Mode('reads')
WRITE = Mode('writes')
REDUCE = Mode('reduces')
ALLOC = Mode('allocates')
# FFI mode denotes when a region is passed into a foreign function.
FFI = Mode('uses FFI')

def mark_usage(region, field_path, mode, cx):
    assert types.is_region(region)
    if region not in cx.access_modes:
        cx.access_modes[region] = {}
    if field_path not in cx.access_modes[region]:
        cx.access_modes[region][field_path] = set()
    cx.access_modes[region][field_path].add(mode)

class Value:
    def read(self, cx):
        return self
    def write(self, cx):
        return self
    def reduce(self, op, cx):
        return self
    def get_field(self, field_name):
        return self

class Reference:
    def __init__(self, refers_to_type, regions, field_path):
        self.refers_to_type = refers_to_type
        self.regions = regions
        self.field_path = tuple(field_path)
    def read(self, cx):
        for region in self.regions:
            mark_usage(region, self.field_path, READ, cx)
        return self
    def write(self, cx):
        for region in self.regions:
            mark_usage(region, self.field_path, WRITE, cx)
        return self
    def reduce(self, op, cx):
        for region in self.regions:
            mark_usage(region, self.field_path, REDUCE(op), cx)
        return self
    def get_field(self, field_name):
        return Reference(self.refers_to_type, self.regions, self.field_path + (field_name,))

@singledispatch
def region_analysis_node(node, cx):
    raise Exception('Region analysis failed at %s' % node)

@region_analysis_node.register(ast.Program)
def _(node, cx):
    region_analysis_node(node.definitions, cx)
    return

@region_analysis_node.register(ast.Definitions)
def _(node, cx):
    for definition in node.definitions:
        region_analysis_node(definition, cx)
    return

@region_analysis_node.register(ast.Import)
def _(node, cx):
    return

@region_analysis_node.register(ast.Struct)
def _(node, cx):
    return

@region_analysis_node.register(ast.Function)
def _(node, cx):
    region_analysis_node(node.block, cx)
    return

@region_analysis_node.register(ast.Block)
def _(node, cx):
    for expr in node.block:
        region_analysis_node(expr, cx)
    return

@region_analysis_node.register(ast.StatementAssert)
def _(node, cx):
    region_analysis_node(node.expr, cx).read(cx)
    return

@region_analysis_node.register(ast.StatementExpr)
def _(node, cx):
    region_analysis_node(node.expr, cx).read(cx)
    return

@region_analysis_node.register(ast.StatementIf)
def _(node, cx):
    region_analysis_node(node.condition, cx).read(cx)
    region_analysis_node(node.then_block, cx)
    if node.else_block is not None:
        region_analysis_node(node.else_block, cx)
    return

@region_analysis_node.register(ast.StatementFor)
def _(node, cx):
    region_analysis_node(node.block, cx)
    return

@region_analysis_node.register(ast.StatementLet)
def _(node, cx):
    region_analysis_node(node.expr, cx).read(cx)
    return

@region_analysis_node.register(ast.StatementLetRegion)
def _(node, cx):
    region_analysis_node(node.size_expr, cx).read(cx)
    return

@region_analysis_node.register(ast.StatementLetArray)
def _(node, cx):
    return

@region_analysis_node.register(ast.StatementLetIspace)
def _(node, cx):
    region_analysis_node(node.size_expr, cx).read(cx)
    return

@region_analysis_node.register(ast.StatementLetPartition)
def _(node, cx):
    region_analysis_node(node.coloring_expr, cx).read(cx)
    return

@region_analysis_node.register(ast.StatementReturn)
def _(node, cx):
    region_analysis_node(node.expr, cx).read(cx)
    return

@region_analysis_node.register(ast.StatementUnpack)
def _(node, cx):
    region_analysis_node(node.expr, cx).read(cx)
    return

@region_analysis_node.register(ast.StatementVar)
def _(node, cx):
    region_analysis_node(node.expr, cx).read(cx)
    return

@region_analysis_node.register(ast.StatementWhile)
def _(node, cx):
    region_analysis_node(node.condition, cx).read(cx)
    region_analysis_node(node.block, cx)
    return

@region_analysis_node.register(ast.ExprID)
def _(node, cx):
    return Value()

@region_analysis_node.register(ast.ExprAssignment)
def _(node, cx):
    region_analysis_node(node.lval, cx).write(cx)
    return region_analysis_node(node.rval, cx).read(cx)

@region_analysis_node.register(ast.ExprUnaryOp)
def _(node, cx):
    region_analysis_node(node.arg, cx).read(cx)
    return Value()

@region_analysis_node.register(ast.ExprBinaryOp)
def _(node, cx):
    region_analysis_node(node.lhs, cx).read(cx)
    region_analysis_node(node.rhs, cx).read(cx)
    return Value()

@region_analysis_node.register(ast.ExprReduceOp)
def _(node, cx):
    region_analysis_node(node.lhs, cx).reduce(node.op, cx)
    region_analysis_node(node.rhs, cx).read(cx)
    return Value()

@region_analysis_node.register(ast.ExprCast)
def _(node, cx):
    region_analysis_node(node.expr, cx).read(cx)
    return Value()

@region_analysis_node.register(ast.ExprNull)
def _(node, cx):
    return Value()

@region_analysis_node.register(ast.ExprIsnull)
def _(node, cx):
    region_analysis_node(node.pointer_expr, cx).read(cx)
    return Value()

@region_analysis_node.register(ast.ExprNew)
def _(node, cx):
    pointer_type = cx.type_map[node].as_read()
    for region in pointer_type.regions:
        mark_usage(region, ALL_FIELDS, ALLOC, cx)
    return Value()

@region_analysis_node.register(ast.ExprRead)
def _(node, cx):
    region_analysis_node(node.pointer_expr, cx).read(cx)
    pointer_type = cx.type_map[node.pointer_expr].as_read()
    for region in pointer_type.regions:
        mark_usage(region, ALL_FIELDS, READ, cx)
    return Value()

@region_analysis_node.register(ast.ExprWrite)
def _(node, cx):
    region_analysis_node(node.pointer_expr, cx).read(cx)
    region_analysis_node(node.value_expr, cx).read(cx)
    pointer_type = cx.type_map[node.pointer_expr].as_read()
    for region in pointer_type.regions:
        mark_usage(region, ALL_FIELDS, WRITE, cx)
    return Value()

@region_analysis_node.register(ast.ExprReduce)
def _(node, cx):
    region_analysis_node(node.pointer_expr, cx).read(cx)
    region_analysis_node(node.value_expr, cx).read(cx)
    pointer_type = cx.type_map[node.pointer_expr].as_read()
    for region in pointer_type.regions:
        mark_usage(region, ALL_FIELDS, REDUCE(node.op), cx)
    return Value()

@region_analysis_node.register(ast.ExprDereference)
def _(node, cx):
    region_analysis_node(node.pointer_expr, cx).read(cx)
    pointer_type = cx.type_map[node.pointer_expr].as_read()
    return Reference(pointer_type.points_to_type, pointer_type.regions, [])

@region_analysis_node.register(ast.ExprArrayAccess)
def _(node, cx):
    region_analysis_node(node.array_expr, cx).read(cx)
    region_analysis_node(node.index_expr, cx).read(cx)
    if types.is_partition(cx.type_map[node.array_expr]):
        return Value()
    region_type = cx.type_map[node.array_expr].as_read()
    return Reference(region_type.kind.contains_type, [region_type], [])

@region_analysis_node.register(ast.ExprFieldAccess)
def _(node, cx):
    return region_analysis_node(node.struct_expr, cx).get_field(node.field_name)

@region_analysis_node.register(ast.ExprFieldDereference)
def _(node, cx):
    region_analysis_node(node.pointer_expr, cx).read(cx)
    pointer_type = cx.type_map[node.pointer_expr].as_read()
    return Reference(pointer_type.points_to_type, pointer_type.regions, [node.field_name])

@region_analysis_node.register(ast.ExprFieldValues)
def _(node, cx):
    region_analysis_node(node.field_values, cx)
    return Value()

@region_analysis_node.register(ast.FieldValues)
def _(node, cx):
    for field_value in node.field_values:
        region_analysis_node(field_value, cx)
    return

@region_analysis_node.register(ast.FieldValue)
def _(node, cx):
    region_analysis_node(node.value_expr, cx).read(cx)
    return

@region_analysis_node.register(ast.ExprFieldUpdates)
def _(node, cx):
    region_analysis_node(node.struct_expr, cx).read(cx)
    region_analysis_node(node.field_updates, cx)
    return Value()

@region_analysis_node.register(ast.FieldUpdates)
def _(node, cx):
    for field_update in node.field_updates:
        region_analysis_node(field_update, cx)
    return

@region_analysis_node.register(ast.FieldUpdate)
def _(node, cx):
    region_analysis_node(node.update_expr, cx).read(cx)
    return

@region_analysis_node.register(ast.ExprColoring)
def _(node, cx):
    return Value()

@region_analysis_node.register(ast.ExprColor)
def _(node, cx):
    region_analysis_node(node.coloring_expr, cx).read(cx)
    region_analysis_node(node.pointer_expr, cx).read(cx)
    region_analysis_node(node.color_expr, cx).read(cx)
    return Value()

@region_analysis_node.register(ast.ExprUpregion)
def _(node, cx):
    region_analysis_node(node.expr, cx).read(cx)
    return Value()

@region_analysis_node.register(ast.ExprDownregion)
def _(node, cx):
    region_analysis_node(node.expr, cx).read(cx)
    return Value()

@region_analysis_node.register(ast.ExprPack)
def _(node, cx):
    region_analysis_node(node.expr, cx).read(cx)
    return Value()

@region_analysis_node.register(ast.ExprCall)
def _(node, cx):
    region_analysis_node(node.function, cx).read(cx)
    region_analysis_node(node.args, cx)

    # Foreign functions must be assumed to touch all arguments.
    if types.is_foreign_function(cx.type_map[node.function]):
        for arg in node.args.args:
            arg_type = cx.type_map[arg]
            if types.is_region(arg_type):
                mark_usage(arg_type, ALL_FIELDS, FFI, cx)

    return Value()

@region_analysis_node.register(ast.Args)
def _(node, cx):
    for arg in node.args:
        region_analysis_node(arg, cx).read(cx)
    return

@region_analysis_node.register(ast.ExprConstBool)
def _(node, cx):
    return Value()

@region_analysis_node.register(ast.ExprConstDouble)
def _(node, cx):
    return Value()

@region_analysis_node.register(ast.ExprConstFloat)
def _(node, cx):
    return Value()

@region_analysis_node.register(ast.ExprConstInt)
def _(node, cx):
    return Value()

@region_analysis_node.register(ast.ExprConstUInt)
def _(node, cx):
    return Value()

def region_analysis(program, opts, type_map):
    cx = Context(type_map)
    region_analysis_node(program, cx)
    return cx.access_modes

def summarize_modes_helper(constraint_graph, region_usage, region, visited):
    access_modes = region_usage

    if region in visited:
        return ({}, set())

    modes = {}
    regions = set()
    if region in access_modes:
        modes.update(access_modes[region])
    regions.add(region)
    if region in constraint_graph:
        visited.add(region)
        for next_region in constraint_graph[region]:
            new_modes, new_regions = summarize_modes_helper(
                constraint_graph, region_usage, next_region, visited)
            modes.update(new_modes)
            regions.update(new_regions)
    return modes, regions

def summarize_modes(constraints, region_usage, region):
    access_modes = region_usage
    constraint_graph = types.make_constraint_graph(constraints, types.Constraint.SUBREGION, inverse = True)

    visited = set()
    modes, regions = summarize_modes_helper(constraint_graph, region_usage, region, visited)
    return modes

def summarize_subregions(constraints, region_usage, region):
    access_modes = region_usage
    constraint_graph = types.make_constraint_graph(constraints, types.Constraint.SUBREGION, inverse = True)

    visited = set()
    modes, regions = summarize_modes_helper(constraint_graph, region_usage, region, visited)
    return regions
