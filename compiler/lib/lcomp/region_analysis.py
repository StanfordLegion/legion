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

from . import ast, types

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

def mark_usage(points_to_type, region, field_path, mode, context):
    type_map, access_modes = context

    if region not in access_modes:
        access_modes[region] = {}
    if field_path not in access_modes[region]:
        access_modes[region][field_path] = set()
    access_modes[region][field_path].add(mode)

class Value:
    def read(self, context):
        return self
    def write(self, context):
        return self
    def reduce(self, op, context):
        return self
    def get_field(self, field_name):
        return self

class Reference:
    def __init__(self, refers_to_type, regions, field_path):
        self.refers_to_type = refers_to_type
        self.regions = regions
        self.field_path = tuple(field_path)
    def read(self, context):
        for region in self.regions:
            mark_usage(self.refers_to_type, region, self.field_path, READ, context)
        return self
    def write(self, context):
        for region in self.regions:
            mark_usage(self.refers_to_type, region, self.field_path, WRITE, context)
        return self
    def reduce(self, op, context):
        for region in self.regions:
            mark_usage(self.refers_to_type, region, self.field_path, REDUCE(op), context)
        return self
    def get_field(self, field_name):
        return Reference(self.refers_to_type, self.regions, self.field_path + (field_name,))

def region_analysis_helper(node, context):
    type_map, access_modes = context
    if isinstance(node, ast.Program):
        for definition in node.defs:
            region_analysis_helper(definition, context)
        return
    if isinstance(node, ast.Import):
        return
    if isinstance(node, ast.Struct):
        return
    if isinstance(node, ast.Function):
        region_analysis_helper(node.block, context)
        return
    if isinstance(node, ast.Block):
        for expr in node.block:
            region_analysis_helper(expr, context)
        return
    if isinstance(node, ast.StatementAssert):
        region_analysis_helper(node.expr, context).read(context)
        return
    if isinstance(node, ast.StatementExpr):
        region_analysis_helper(node.expr, context)
        return
    if isinstance(node, ast.StatementIf):
        region_analysis_helper(node.condition, context).read(context)
        region_analysis_helper(node.then_block, context)
        if node.else_block is not None:
            region_analysis_helper(node.else_block, context)
        return
    if isinstance(node, ast.StatementFor):
        region_analysis_helper(node.block, context)
        return
    if isinstance(node, ast.StatementLet):
        region_analysis_helper(node.expr, context).read(context)
        return
    if isinstance(node, ast.StatementLetRegion):
        region_analysis_helper(node.size_expr, context).read(context)
        return
    if isinstance(node, ast.StatementLetArray):
        return
    if isinstance(node, ast.StatementLetIspace):
        region_analysis_helper(node.size_expr, context).read(context)
        return
    if isinstance(node, ast.StatementLetPartition):
        region_analysis_helper(node.coloring_expr, context).read(context)
        return
    if isinstance(node, ast.StatementReturn):
        region_analysis_helper(node.expr, context).read(context)
        return
    if isinstance(node, ast.StatementUnpack):
        region_analysis_helper(node.expr, context).read(context)
        return
    if isinstance(node, ast.StatementVar):
        region_analysis_helper(node.expr, context).read(context)
        return
    if isinstance(node, ast.StatementWhile):
        region_analysis_helper(node.condition, context).read(context)
        region_analysis_helper(node.block, context)
        return
    if isinstance(node, ast.ExprID):
        return Value()
    if isinstance(node, ast.ExprAssignment):
        region_analysis_helper(node.lval, context).write(context)
        return region_analysis_helper(node.rval, context).read(context)
    if isinstance(node, ast.ExprUnaryOp):
        region_analysis_helper(node.arg, context).read(context)
        return Value()
    if isinstance(node, ast.ExprBinaryOp):
        region_analysis_helper(node.lhs, context).read(context)
        region_analysis_helper(node.rhs, context).read(context)
        return Value()
    if isinstance(node, ast.ExprReduceOp):
        region_analysis_helper(node.lhs, context).reduce(node.op, context)
        region_analysis_helper(node.rhs, context).read(context)
        return Value()
    if isinstance(node, ast.ExprCast):
        region_analysis_helper(node.expr, context).read(context)
        return Value()
    if isinstance(node, ast.ExprNull):
        return Value()
    if isinstance(node, ast.ExprIsnull):
        region_analysis_helper(node.pointer_expr, context).read(context)
        return Value()
    if isinstance(node, ast.ExprNew):
        pointer_type = type_map[node].as_read()
        for region in pointer_type.regions:
            mark_usage(pointer_type.points_to_type, region, ALL_FIELDS, ALLOC, context)
        return Value()
    if isinstance(node, ast.ExprRead):
        region_analysis_helper(node.pointer_expr, context).read(context)
        pointer_type = type_map[node.pointer_expr].as_read()
        for region in pointer_type.regions:
            mark_usage(pointer_type.points_to_type, region, ALL_FIELDS, READ, context)
        return Value()
    if isinstance(node, ast.ExprWrite):
        region_analysis_helper(node.pointer_expr, context).read(context)
        region_analysis_helper(node.value_expr, context).read(context)
        pointer_type = type_map[node.pointer_expr].as_read()
        for region in pointer_type.regions:
            mark_usage(pointer_type.points_to_type, region, ALL_FIELDS, WRITE, context)
        return Value()
    if isinstance(node, ast.ExprReduce):
        region_analysis_helper(node.pointer_expr, context).read(context)
        region_analysis_helper(node.value_expr, context).read(context)
        pointer_type = type_map[node.pointer_expr].as_read()
        for region in pointer_type.regions:
            mark_usage(pointer_type.points_to_type, region, ALL_FIELDS, REDUCE(node.op), context)
        return Value()
    if isinstance(node, ast.ExprDereference):
        region_analysis_helper(node.pointer_expr, context).read(context)
        pointer_type = type_map[node.pointer_expr].as_read()
        return Reference(pointer_type.points_to_type, pointer_type.regions, [])
    if isinstance(node, ast.ExprArrayAccess):
        region_analysis_helper(node.array_expr, context).read(context)
        region_analysis_helper(node.index_expr, context).read(context)
        if types.is_partition(type_map[node.array_expr]):
            return Value()
        region_type = type_map[node.array_expr].as_read()
        return Reference(region_type.kind.contains_type, [region_type], [])
    if isinstance(node, ast.ExprFieldAccess):
        return region_analysis_helper(node.struct_expr, context).get_field(node.field_name)
    if isinstance(node, ast.ExprFieldDereference):
        region_analysis_helper(node.pointer_expr, context).read(context)
        pointer_type = type_map[node.pointer_expr].as_read()
        return Reference(pointer_type.points_to_type, pointer_type.regions, [node.field_name])
    if isinstance(node, ast.ExprFieldValues):
        region_analysis_helper(node.field_values, context)
        return Value()
    if isinstance(node, ast.FieldValues):
        for field_value in node.field_values:
            region_analysis_helper(field_value, context)
        return
    if isinstance(node, ast.FieldValue):
        region_analysis_helper(node.value_expr, context).read(context)
        return
    if isinstance(node, ast.ExprFieldUpdates):
        region_analysis_helper(node.struct_expr, context).read(context)
        region_analysis_helper(node.field_updates, context)
        return Value()
    if isinstance(node, ast.FieldUpdates):
        for field_update in node.field_updates:
            region_analysis_helper(field_update, context)
        return
    if isinstance(node, ast.FieldUpdate):
        region_analysis_helper(node.update_expr, context).read(context)
        return
    if isinstance(node, ast.ExprColoring):
        return Value()
    if isinstance(node, ast.ExprColor):
        region_analysis_helper(node.coloring_expr, context).read(context)
        region_analysis_helper(node.pointer_expr, context).read(context)
        region_analysis_helper(node.color_expr, context).read(context)
        return Value()
    if isinstance(node, ast.ExprUpregion):
        region_analysis_helper(node.expr, context).read(context)
        return Value()
    if isinstance(node, ast.ExprDownregion):
        region_analysis_helper(node.expr, context).read(context)
        return Value()
    if isinstance(node, ast.ExprPack):
        region_analysis_helper(node.expr, context).read(context)
        return Value()
    if isinstance(node, ast.ExprCall):
        region_analysis_helper(node.function, context).read(context)
        region_analysis_helper(node.args, context)
        return Value()
    if isinstance(node, ast.Args):
        for arg in node.args:
            region_analysis_helper(arg, context).read(context)
        return
    if isinstance(node, ast.ExprConstBool):
        return Value()
    if isinstance(node, ast.ExprConstDouble):
        return Value()
    if isinstance(node, ast.ExprConstInt):
        return Value()
    if isinstance(node, ast.ExprConstUInt):
        return Value()
    raise Exception('Region analysis failed at %s' % node)

def region_analysis(program, type_map):
    access_modes = {}
    context = type_map, access_modes
    region_analysis_helper(program, context)
    return access_modes

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
