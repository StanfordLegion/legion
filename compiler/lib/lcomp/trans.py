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
### Code Generator
###
### This pass transforms the AST into C++ source code.
###

# Work around for OrderedDict missing in Python 2.6.
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import copy
from cStringIO import StringIO
from . import ast, parse, region_analysis, symbol_table, types

reduction_op_table = {
    '*': {'name': 'times',        'identity': '1',  'fold': '*'},
    '/': {'name': 'divide',       'identity': '1',  'fold': '*'},
    '+': {'name': 'plus',         'identity': '0',  'fold': '+'},
    '-': {'name': 'minus',        'identity': '0',  'fold': '+'},
    '&': {'name': 'bitwise_and',  'identity': '-1', 'fold': '&'},
    '^': {'name': 'bitwise_xor',  'identity': '0',  'fold': '^'},
    '|': {'name': 'bitwise_or',   'identity': '0',  'fold': '|'},
}
def mangle_reduction_op(op, element_type, cx):
    assert op in reduction_op_table
    op_name = reduction_op_table[op]['name']

    ll_element_type = trans_type(element_type, cx)
    return 'reduction_%s_%s' % (
        op_name, ll_element_type)

def mangle_reduction_op_enum(op, element_type, cx):
    return mangle_reduction_op(op, element_type, cx).upper()

def mangle_struct_name(name):
    return 'struct_%s' % name

def mangle_task_name(name):
    return 'task_%s' % name

def mangle_task_enum(name):
    return mangle_task_name(name).upper()

def mangle_task_params_struct(name):
    return '%s_params' % mangle_task_name(name)

def mangle_task_param(name):
    return 'param_%s' % name

class Counter:
    def __init__(self):
        self.value = 0
    def next(self):
        self.value += 1
        return self.value

_mangle_anonymous_struct_name_counter = Counter()
def mangle_anonymous_struct_name(node, cx):
    if node in cx.alpha_map:
        return cx.alpha_map[node]
    name = 'anon_struct_%s' % _mangle_anonymous_struct_name_counter.next()
    cx.alpha_map[node] = name
    return name

_mangle_for_counter = Counter()
def mangle_for(node, cx):
    if node in cx.alpha_map:
        return cx.alpha_map[node]
    name = 'loop_index_%s' % _mangle_for_counter.next()
    cx.alpha_map[node] = name
    return name

_mangle_let_counter = Counter()
def mangle_let(node, cx):
    if node in cx.alpha_map:
        return cx.alpha_map[node]
    name = 'local_%s_%s' % (_mangle_let_counter.next(), node.name)
    cx.alpha_map[node] = name
    return name

_mangle_region_counter = Counter()
def mangle_region(node, cx):
    if node in cx.alpha_map:
        return cx.alpha_map[node]
    name = 'region_%s_%s' % (_mangle_region_counter.next(), node.name)
    cx.alpha_map[node] = name
    return name

_mangle_unpack_counter = Counter()
def mangle_unpack(node, cx):
    if node in cx.alpha_map:
        return cx.alpha_map[node]
    name = 'unpack_%s_%s' % (_mangle_unpack_counter.next(), node.name)
    cx.alpha_map[node] = name
    return name

def mangle(node, cx):
    if isinstance(node, ast.Struct):
        return mangle_struct_name(node.name)
    if isinstance(node, ast.Function):
        return mangle_task_name(node.name)
    if isinstance(node, ast.Param):
        return mangle_task_param(node.name)
    if isinstance(node, ast.StatementFor):
        return mangle_for(node, cx)
    if isinstance(node, ast.StatementLet):
        return mangle_let(node, cx)
    if isinstance(node, ast.StatementLetRegion):
        return mangle_region(node, cx)
    if isinstance(node, ast.StatementLetArray):
        return mangle_region(node, cx)
    if isinstance(node, ast.StatementLetIspace):
        return mangle_let(node, cx)
    if isinstance(node, ast.StatementVar):
        return mangle_let(node, cx)
    if isinstance(node, ast.StatementUnpack):
        return mangle_unpack(node, cx)
    if isinstance(node, ast.UnpackRegion):
        return mangle_region(node, cx)
    raise Exception('Unable to mangle node %s' % node)

def mangle_type(t, cx):
    if types.is_struct(t):
        if t.name is None:
            return mangle_anonymous_struct_name(t, cx)
        return mangle_struct_name(t.name)
    raise Exception('Unable to mangle type %s' % t)

_mangle_temp_counter = Counter()
def mangle_temp():
    name = 'temp_%s' % _mangle_temp_counter.next()
    return name

class Block:
    def __init__(self, actions):
        self.actions = actions
    def render(self):
        return ['  ' + line
                for action in self.actions
                for line in (action.render() if hasattr(action, 'render') else [action])]

class Statement:
    def __init__(self, actions):
        self.actions = actions
    def render(self):
        return [line
                for action in self.actions
                for line in (action.render() if hasattr(action, 'render') else [action])]

class Expr:
    def __init__(self, value, actions):
        self.value = value
        self.actions = actions
    def render(self):
        return Statement(self.actions).render()

class Value:
    def __init__(self, node, expr, type):
        self.node = node
        self.expr = expr
        self.type = type
    def read(self, cx):
        return self.expr
    def write(self, update_value, cx):
        raise Exception('Value is not an lval')
    def reduce(self, op, update_value, cx):
        raise Exception('Value is not an lval')

class ReductionOp(Value):
    def __init__(self, node, expr, type, op, element_type):
        Value.__init__(self, node, expr, type)
        self.op = op
        self.element_type = element_type

class Function(Value):
    def __init__(self, node, expr, type, params):
        Value.__init__(self, node, expr, type)
        self.params = params

class Builtin(Value):
    def __init__(self, expr, type):
        Value.__init__(self, None, expr, type)

class Reference(Value):
    def __init__(self, node, pointer_expr, pointer_type, field_path = ()):
        assert isinstance(field_path, tuple)
        type = types.Reference(pointer_type.points_to_type, pointer_type.regions, field_path)
        Value.__init__(self, node, pointer_expr, type)
        self.pointer_type = pointer_type
        self.field_path = field_path
    def read(self, cx):
        pointer_value = Value(self.node, self.expr, self.pointer_type)
        return trans_read_helper(
            self.node, pointer_value, self.field_path, cx).read(cx)
    def write(self, update_value, cx):
        pointer_value = Value(self.node, self.expr, self.pointer_type)
        return trans_write_helper(
            self.node, pointer_value, self.field_path, update_value,
            cx).read(cx)
    def reduce(self, op, update_value, cx):
        pointer_value = Value(self.node, self.expr, self.pointer_type)
        return trans_reduce_helper(
            self.node, pointer_value, self.field_path, op, update_value,
            cx).read(cx)
    def get_field(self, node, field_name):
        return Reference(node, self.expr, self.pointer_type, self.field_path + (field_name,))

class StackReference(Value):
    def __init__(self, node, expr, type, field_path = ()):
        assert isinstance(field_path, tuple)
        Value.__init__(self, node, expr, type)
        self.node = node
        self.field_path = field_path
    def read(self, cx):
        value = self.expr.value
        for field_name in self.field_path:
            value = '(%s.%s)' % (value, field_name)
        return Expr(value, self.expr.actions)
    def write(self, update_value, cx):
        update_expr = update_value.read(cx)
        value = self.expr.value
        for field_name in self.field_path:
            value = '(%s.%s)' % (value, field_name)
        value = '(%s = %s)' % (value, update_expr.value)
        return Expr(value, self.expr.actions + update_expr.actions)
    def reduce(self, op, update_value, cx):
        update_expr = update_value.read(cx)
        value = self.expr.value
        for field_name in self.field_path:
            value = '(%s.%s)' % (value, field_name)
        value = '(%s %s= %s)' % (value, op, update_expr.value)
        return Expr(value, self.expr.actions + update_expr.actions)
    def get_field(self, node, field_name):
        type = self.type.get_field(field_name)
        return StackReference(node, self.expr, type, self.field_path + (field_name,))

class Region(Value):
    def __init__(self, node, expr, type, region_root, region_requirement,
                 physical_region, ispace, fspace, field_map, field_type_map,
                 allocator, accessor_map):
        Value.__init__(self, node, expr, type)
        self.region_root = region_root
        self.region_requirement = region_requirement
        self.physical_region = physical_region
        self.ispace = ispace
        self.fspace = fspace
        self.field_map = field_map
        self.field_type_map = field_type_map
        self.allocator = allocator
        self.accessor_map = accessor_map

class Partition(Value):
    def __init__(self, node, expr, type, region):
        Value.__init__(self, node, expr, type)
        self.region = region

# Instances of FieldID mascarade as types because there are no
# high-level types which describe what a FieldID is. Used primarily in
# trans_function and friends.
class FieldID:
    def key(self):
        return id(self)

def is_fieldid(t):
    return isinstance(t, FieldID)

class Context:
    def __init__(self, type_set, type_map, constraints, region_usage):
        self.env = None
        self.alpha_map = None
        self.regions = None
        self.mapped_regions = None
        self.ll_type_map = None
        self.ll_field_map = None
        self.reduction_ops = None
        self.type_set = type_set
        self.type_map = type_map
        self.constraints = constraints
        self.region_usage = region_usage
    def increase_indent(self):
        cx = copy.copy(self)
        return cx
    def new_block_scope(self):
        cx = copy.copy(self)
        cx.env = cx.env.new_scope()
        cx.regions = copy.copy(cx.regions)
        cx.mapped_regions = cx.mapped_regions + (set(),)
        return cx
    def new_function_scope(self):
        cx = copy.copy(self)
        cx.env = cx.env.new_scope()
        cx.regions = {}
        cx.mapped_regions = (set(),)
        return cx
    def new_struct_scope(self):
        cx = copy.copy(self)
        cx.env = cx.env.new_scope()
        return cx
    def new_global_scope(self):
        cx = copy.copy(self)
        cx.env = symbol_table.SymbolTable()
        cx.alpha_map = {}
        cx.ll_type_map = {}
        cx.ll_field_map = {}
        cx.reduction_ops = {}
        return cx

def trans_program(program, cx):
    cx = cx.new_global_scope()
    type_list = [d for d in program.defs if types.is_kind(cx.type_map[d])]
    other_list = [d for d in program.defs if not types.is_kind(cx.type_map[d])]

    type_defs = trans_type_defs(type_list, cx)
    reduction_op_defs = trans_reduction_ops(cx)
    other_defs = [trans_node(definition, cx) for definition in other_list]

    task_list = [d for d in program.defs if isinstance(d, ast.Function)]
    top_level_task = check_top_level_task(task_list, cx)

    sep = '\n'
    buffer = StringIO()
    buffer.write(trans_header())
    buffer.write(sep)
    buffer.write(trans_reduction_op_list(reduction_op_defs, cx))
    buffer.write(sep)
    buffer.write(sep)
    buffer.write(trans_task_list(task_list))
    buffer.write(sep)
    buffer.write(sep)
    buffer.write(sep.join(line for d in type_defs for line in d.read(cx).render() + ['']))
    buffer.write(sep)
    buffer.write(sep.join(line for d in reduction_op_defs for line in d.read(cx).render() + ['']))
    buffer.write(sep)
    buffer.write(sep.join(line for d in other_defs for line in d.read(cx).render()))
    buffer.write(sep)
    buffer.write(trans_main(reduction_op_defs, task_list, top_level_task, cx))
    return buffer.getvalue()

def check_top_level_task(task_list, cx):
    top_level_tasks = [task for task in task_list if task.name == 'main']
    assert len(top_level_tasks) == 1
    top_level_task = top_level_tasks[0]
    task_type = cx.type_map[top_level_task]
    assert task_type == types.Function([], set(), types.Void())
    return top_level_task

def trans_header():
    return '''
#include <cassert>
#include <cstdint>

#include "legion.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

LegionRuntime::Logger::Category log_app("app");
'''

def trans_reduction_op_list(reduction_op_defs, cx):
    return 'enum {\n%s\n};' % (
        '\n'.join(
            '  %s = %s,' % (
                mangle_reduction_op_enum(redop.op, redop.element_type, cx),
                index)
            for redop, index in zip(reduction_op_defs,
                                    xrange(1, len(reduction_op_defs)+1))))

def trans_task_list(task_list):
    return 'enum {\n%s\n};' % (
        '\n'.join('  %s,' % mangle_task_enum(task.name)
                  for task in task_list))

def trans_main(reduction_op_defs, task_list, top_level_task, cx):
    reduction_op_registrations = '\n  '.join(
        'HighLevelRuntime::register_reduction_op<%s>(%s);' % (
            mangle_reduction_op(redop.op, redop.element_type, cx),
            mangle_reduction_op_enum(redop.op, redop.element_type, cx))
        for redop in reduction_op_defs)
    task_registrations = '\n  '.join(
        'HighLevelRuntime::register_single_task<%s%s>(%s, Processor::LOC_PROC, false, "%s");' % (
            ('%s,' % trans_return_type(cx.type_map[task].return_type, cx)
             if not types.is_void(cx.type_map[task].return_type) else ''),
            mangle(task, cx), mangle_task_enum(task.name), task.name)
        for task in task_list)
    set_top_level_task = 'HighLevelRuntime::set_top_level_task_id(%s);' % mangle_task_enum(top_level_task.name)
    return '''
void create_mappers(Machine *machine, HighLevelRuntime *runtime,
                    const std::set<Processor> &local_procs) {
}

int main(int argc, char **argv) {
  HighLevelRuntime::set_registration_callback(create_mappers);
  %s
  %s
  %s

  return HighLevelRuntime::start(argc, argv);
}
''' % (reduction_op_registrations, task_registrations, set_top_level_task)

class TypeTranslationFailedException(Exception):
    pass

def trans_type_defs(kind_defs, cx):
    remaining = cx.type_set

    type_defs = []
    while len(remaining) > 0:
        succeeded = set()
        for t in remaining:
            try:
                type_def = trans_type_def(t, cx)
                if type_def is not None:
                    cx.ll_type_map[t.key()] = type_def.expr.value
                    type_defs.append(type_def)
                types.add_key(succeeded, t)
            except TypeTranslationFailedException:
                pass
        remaining = types.unwrap(types.wrap(remaining) - succeeded)
        if len(succeeded) == 0:
            raise Exception('Unable to translate all types')
    return type_defs

def trans_type_def(t, cx):
    if types.is_simple(t):
        return None
    if types.is_region(t):
        return None
    if types.is_coloring(t):
        return None
    if types.is_pointer(t):
        return None
    if types.is_function(t):
        return None
    if types.is_struct_instance(t):
        return None
    if types.is_struct(t):
        struct_definition = (
            ['// type def for %s' % t.pretty(),
             'struct %s {' % (
                    mangle_type(t, cx)),
             Block(['%s %s;' % (trans_type(region, cx), region.name)
                    for region in t.regions] +
                   ['%s %s;' % (trans_type(field_type, cx), field_name)
                    for field_name, field_type in t.field_map.iteritems()]),
             '};'])
        return Value(
            t,
            Expr(mangle_type(t, cx), struct_definition),
            types.Kind(t))

def trans_reduction_ops(cx):
    # Produce a list of reduction ops that need to be defined, and for
    # each reduction op, what types it will be applied to.
    reduction_ops = {}
    for region, region_usage in cx.region_usage.iteritems():
        field_types = trans_fields(region.kind.contains_type, cx)
        for field_path, modes in region_usage.iteritems():
            for mode in modes:
                if mode == region_analysis.REDUCE:
                    if mode.op not in reduction_ops:
                        reduction_ops[mode.op] = types.wrap([])
                    types.add_key(reduction_ops[mode.op], field_types[field_path])
    reduction_ops = dict((k, types.unwrap(v)) for k, v in reduction_ops.iteritems())

    reduction_classes = []
    for op, element_types in reduction_ops.iteritems():
        for element_type in element_types:
            reduction_class = trans_reduction_op(op, element_type, cx)
            reduction_classes.append(reduction_class)
            cx.reduction_ops[(op, element_type.key())] = Value(
                None,
                Expr(reduction_class.read(cx).value, []),
                None)
    return reduction_classes

def trans_reduction_op(op, element_type, cx):
    ll_name = mangle_reduction_op(op, element_type, cx)

    ll_element_type = trans_type(element_type, cx)

    assert op in reduction_op_table
    identity = reduction_op_table[op]['identity']
    apply_op = op
    fold_op = reduction_op_table[op]['fold']

    reduction_class = [
        'class %s {' % ll_name,
        'public:',
        Block(['typedef %s LHS;' % ll_element_type,
               'typedef %s RHS;' % ll_element_type,
               'static const RHS identity;',
               'template <bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs) { lhs %s= rhs; }' % apply_op,
               'template <bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2) { rhs1 %s= rhs2; }' % fold_op]),
        '};',
        '',
        'const %s::RHS %s::identity = %s;' % (ll_name, ll_name, identity)]
    return ReductionOp(
        node = None,
        expr = Expr(ll_name, reduction_class),
        type = None,
        op = op,
        element_type = element_type)

def trans_import(node, cx):
    module_type = cx.type_map[node]
    for def_name, def_type in module_type.def_types.iteritems():
        if types.is_function(def_type):
            cx.env.insert(def_name, Builtin(Expr(def_name, []), def_type))
    return Value(
        node,
        Expr(None, ['#include "%s"' % node.filename]),
        cx.type_map[node])

def trans_struct(node, cx):
    # struct already defined in trans_type_def
    return Value(
        node,
        Expr(mangle(node, cx), []),
        cx.type_map[node])

def trans_param_type_size(task, size, param, param_type, cx):
    if types.is_coloring(param_type):
        serializer = mangle_temp()
        return (
            ['ColoringSerializer %s(%s);' % (serializer, param),
             'result += %s.legion_buffer_size();' % serializer])
    return [
        'result += sizeof(%s);' % trans_type(param_type, cx)]

def trans_param_type_serialize(task, buffer, param, param_type, cx):
    if types.is_coloring(param_type):
        serializer = mangle_temp()
        size = mangle_temp()
        return (
            ['ColoringSerializer %s(%s);' % (serializer, param),
             'size_t %s = %s.legion_serialize((void *) ptr);' % (size, serializer),
             'ptr += %s;' % size])
    return [
        '*(%s *)ptr = %s;' % (trans_type(param_type, cx), param),
        'ptr += sizeof(%s);' % trans_type(param_type, cx)]

def trans_param_type_deserialize(task, buffer, param, param_type, cx):
    if types.is_coloring(param_type):
        serializer = mangle_temp()
        size = mangle_temp()
        return (
            ['ColoringSerializer %s;' % serializer,
             'size_t %s = %s.legion_deserialize((void *) ptr);' % (size, serializer),
             '%s = %s.ref();' % (param, serializer),
             'ptr += %s;' % size])
    return [
        '%s = *(%s *)ptr;' % (param, trans_type(param_type, cx)),
        'ptr += sizeof(%s);' % trans_type(param_type, cx)]

def trans_params_size(task, params, cx):
    return [
        'size_t legion_buffer_size() const {',
        Block(
            ['size_t result = 0;'] +
            [line
             for param_type, param in params
             for line in trans_param_type_size(task, 'result', param, param_type, cx)] +
            ['return result;']),
        '}']

def trans_params_serialize(task, params, cx):
    return [
        'void legion_serialize(void *buffer) const {',
        Block(['char *ptr = (char *)buffer;'] +
              [line
               for param_type, param in params
               for line in trans_param_type_serialize(task, 'ptr', param, param_type, cx)]),
        '}']

def trans_params_deserialize(task, params, cx):
    return [
        'size_t legion_deserialize(const void *buffer) {',
        Block(['const char *ptr = (const char *)buffer;'] +
              [line
               for param_type, param in params
               for line in trans_param_type_deserialize(task, 'ptr', param, param_type, cx)]),
        '}']

def trans_params(task, params, cx):
    struct_name = mangle_task_params_struct(task.name)
    struct_def = (
        ['struct %s' % struct_name,
         '{',
         Block(['%s %s;' % (
                        trans_type(param_type, cx),
                        param)
                for param_type, param in params] +
               trans_params_size(task, params, cx) +
               trans_params_serialize(task, params, cx) +
               trans_params_deserialize(task, params, cx)),
         '};'])
    return Expr(struct_name, struct_def)

def trans_cleanup(index, cx):
    cleanup = []
    for scope in cx.mapped_regions[index:]:
        for region in scope:
            assert region.physical_region is not None
            cleanup.append('runtime->unmap_region(ctx, %s);' % (
                    region.physical_region))
    return cleanup

def trans_function(task, cx):
    original_cx = cx
    cx = cx.new_function_scope()

    task_name = mangle(task, cx)
    task_inputs = ', '.join([
        'const void *input',
        'size_t input_size',
        'const std::vector<RegionRequirement> &reqs',
        'const std::vector<PhysicalRegion> &regions',
        'Context ctx',
        'HighLevelRuntime *runtime',
        ])
    task_type = cx.type_map[task]
    ll_return_type = trans_return_type(task_type.return_type, cx)

    # Normal parameters are passed via TaskArgument.
    param_nodes = [
        param for param, param_type in zip(task.params.params, task_type.param_types)
        if not types.is_region(param_type)]
    param_types = [
        param_type.as_read() for param_type in task_type.param_types
        if not types.is_region(param_type)]
    params = [mangle(param, cx) for param in param_nodes]

    # Region parameters are passed via RegionRequirement.
    all_region_nodes = [
        param for param, param_type in zip(task.params.params, task_type.param_types)
        if types.is_region(param_type)]
    all_region_types = [
        param_type for param_type in task_type.param_types
        if types.is_region(param_type)]
    all_regions = [mangle_region(region, cx) for region in all_region_nodes]

    # For regions with no privileges requested, do not pass a
    # RegionRequirement.
    privileges_requested = [
        trans_privilege_mode(region_type, task_type.privileges, cx)
        for region_type in all_region_types]

    region_needs_physical = [
        len(privileges) > 0
        for privileges in privileges_requested]

    physical_regions = [mangle_temp() for region in all_regions]
    index = iter(xrange(len(all_regions)))
    physical_indexes = [next(index) if needs_physical else None
                        for needs_physical in region_needs_physical]

    ispaces = [mangle_temp() for region in all_regions]
    fspaces = [mangle_temp() for region in all_regions]
    allocators = [mangle_temp() for region in all_regions]

    field_types = [trans_fields(region_type.kind.contains_type, cx)
                   for region_type in all_region_types]

    ll_field_types = [
        OrderedDict(
            (field, trans_type(field_type, cx))
            for field, field_type in region_field_types.iteritems())
        for region_field_types in field_types]
    fields = [
        OrderedDict(
            (field, 'field_%s' % ('_'.join((region,) + field)))
            for field in region_field_types.iterkeys())
        for region, region_field_types in zip(all_regions, field_types)]
    region_params = (
        [(region_type, region) for region, region_type in zip(all_regions, all_region_types)] +
        [(FieldID(), field) for region_fields in fields for field in region_fields.itervalues()])

    # For regions that need read, write, or allocate access, create
    # the appropriate accessors and/or allocators.
    region_access_modes = [
        region_analysis.summarize_modes(cx.constraints, cx.region_usage, region_type)
        for region_type in all_region_types]
    region_needs_allocator = [
        any(region_analysis.ALLOC in modes
            for modes in field_modes.itervalues())
        for field_modes in region_access_modes]
    accessor_fields_inner = [
        [field for field, modes in field_modes.iteritems()
         if region_analysis.READ in modes or
         region_analysis.WRITE in modes or
         region_analysis.REDUCE in modes]
        for field_modes in region_access_modes]
    accessor_fields = [
        OrderedDict(
            (field_leaf, ())
            for field in region_fields
            for field_leaf in region_field_types.iterkeys()
            if is_prefix(field_leaf, field)).keys()
        for region_fields, region_field_types in zip(accessor_fields_inner, field_types)]

    privilege_fields = [
        OrderedDict(
            (field, privilege)
            for privilege, fields in privileges.iteritems()
            for field in fields)
        for privileges in privileges_requested]

    region_needs_field_accessor = [
        [region_privilege_fields[field] in ('READ_WRITE', 'READ_ONLY', 'WRITE_ONLY')
         for field in region_accessor_fields]
        for region_privilege_fields, region_accessor_fields in zip(
            privilege_fields, accessor_fields)]

    accessors = [
        OrderedDict((field, mangle_temp()) for field in region_fields)
        for region_fields in accessor_fields]

    # Build a struct to hold the task's parameters.
    task_params_struct = trans_params(
        task,
        region_params +
        [(param_type, param)
         for param, param_type in zip(params, param_types)],
        cx)

    task_header = (
        ['log_app.info("In %s...");' % task.name] +
        (['%s params;' % (task_params_struct.value),
          'params.legion_deserialize(input);']
         if len(task.params.params) > 0 else []) +
        ['%s %s = params.%s;' % (
                trans_type(region_type, cx),
                region, region)
         for region_type, region in region_params] +
        ['%s %s = params.%s;' % (
                trans_type(param_type, cx),
                param,
                param)
         for param, param_type in zip(params, param_types)] +
        ['PhysicalRegion %s = regions[%s];' % (region, index)
         for region, index in zip(physical_regions, physical_indexes)
         if index is not None] +
        ['IndexSpace %s = %s.get_index_space();' % (ispace , region)
         for region, ispace in zip(all_regions, ispaces)] +
        ['FieldSpace %s = %s.get_field_space();' % (fspace , region)
         for region, fspace in zip(all_regions, fspaces)] +
        ['IndexAllocator %s = runtime->create_index_allocator(ctx, %s);' % (
                ispace_alloc, ispace)
         for needs_allocator, ispace_alloc, ispace in zip(
                region_needs_allocator, allocators, ispaces)
         if needs_allocator] +
        ['RegionAccessor<AccessorType::Generic, %s> %s = %s.get_%saccessor(%s).typeify<%s>();' % (
                ll_field_type[field], accessor, physical_region,
                ('field_' if field_accessor else ''),
                (region_fields[field] if field_accessor else ''),
                ll_field_type[field])
         for ll_field_type, region_fields, region_accessors, physical_region, needs_field_accessor in zip(
                ll_field_types, fields, accessors, physical_regions, region_needs_field_accessor)
         for (field, accessor), field_accessor in zip(region_accessors.iteritems(), needs_field_accessor)])

    # insert function name into global scope
    task_value = Function(
        node = task,
        expr = Expr(mangle(task, cx), []),
        type = task_type,
        params = task_params_struct.value)
    original_cx.env.insert(task.name, task_value)
    cx = cx.new_function_scope()
    # insert function regions and params into local scope
    for region_node, region, region_type, ispace, fspace, \
            region_fields, region_field_types, physical_region, \
            physical_index, needs_allocator, allocator, \
            region_accessors in \
            zip(all_region_nodes, all_regions, all_region_types, ispaces,
                fspaces, fields, field_types, physical_regions,
                physical_indexes, region_needs_allocator, allocators,
                accessors):

            region_value = Region(
                node = region_node,
                expr = Expr(region, []),
                type = region_type,
                region_root = region_type,
                region_requirement = physical_index,
                physical_region = physical_region if physical_index is not None else None,
                ispace = ispace,
                fspace = fspace,
                field_map = region_fields,
                field_type_map = region_field_types,
                allocator = allocator if needs_allocator else None,
                accessor_map = region_accessors)

            cx.env.insert(region_node.name, region_value)
            cx.regions[region_type] = region_value

            if physical_index is not None:
                cx.mapped_regions[-1].add(region_value)

    for param_node, param, param_type in zip(param_nodes, params, param_types):
        value = Value(param_node, Expr(param, []), param_type)
        if types.allows_var_binding(param_type):
            value = StackReference(param_node, Expr(param, []), types.StackReference(param_type))
        cx.env.insert(param_node.name, value)

    task_body = trans_node(task.block, cx)
    task_cleanup = trans_cleanup(0, cx)

    task_definition = [
        '%s %s(%s)' % (
            ll_return_type,
            task_name,
            task_inputs),
        '{',
        Block(task_header + [task_body] + task_cleanup),
        '}']

    task_definition = Function(
        node = task,
        expr = Expr(
            mangle(task, cx),
            task_params_struct.actions + [''] + task_definition),
        type = task_type,
        params = task_params_struct.value)
    return task_definition

def trans_for(node, cx):
    cx = cx.new_block_scope()

    index_name = mangle(node, cx)
    index_type = cx.type_map[node.indices.indices[0]]
    for index in node.indices.indices:
        cx.env.insert(index.name, Value(node, Expr(index_name, []), index_type))

    ll_index_type = trans_type(index_type, cx)
    region = cx.env.lookup(node.regions.regions[0].name)
    region_type = region.type
    ll_region = region.read(cx)

    ll_ispace = ll_region
    if types.is_region(region_type):
        ll_ispace = Expr(region.ispace, ll_region.actions)

    temp_domain = mangle_temp()
    temp_iterator = mangle_temp()
    block = (
        ll_ispace.actions +
        ['Domain %s = runtime->get_index_space_domain(ctx, %s);' % (
                temp_domain, ll_ispace.value),
         'for (Domain::DomainPointIterator %s(%s); %s; %s++) {' % (
                temp_iterator, temp_domain, temp_iterator, temp_iterator),
         Block(['%s %s(%s.p.get_index());' % (
                        ll_index_type,
                        index_name,
                        temp_iterator),
                trans_node(node.block, cx)]),
         '}'])
    return Block(block)

def trans_copy_helper(node, name, cx):
    copy_value = trans_node(node.expr, cx)
    copy_expr = copy_value.read(cx)
    copy_type = copy_value.type.as_read()
    ll_copy_type = trans_type(copy_type, cx)

    if types.is_region(copy_value.type):
        region_type = cx.type_map[node]
        region_value = Region(
            node = node,
            expr = Expr(name, []),
            type = region_type,
            region_root = copy_value.region_root,
            region_requirement = None,
            physical_region = None,
            ispace = copy_value.ispace,
            fspace = copy_value.fspace,
            field_map = copy_value.field_map,
            field_type_map = copy_value.field_type_map,
            allocator = copy_value.allocator,
            accessor_map = copy_value.accessor_map)
        cx.env.insert(node.name, region_value)
        cx.regions[region_type] = region_value
        return Statement(
            copy_expr.actions +
            ['%s %s = %s;' % (ll_copy_type, name, copy_expr.value)])

    cx.env.insert(
        node.name,
        Value(
            node,
            Expr(name, []),
            copy_type),
        shadow = True)
    return Statement(
        copy_expr.actions +
        ['%s %s = %s;' % (ll_copy_type, name, copy_expr.value)])

def trans_let(node, cx):
    name = mangle(node, cx)
    return trans_copy_helper(node, name, cx)

def trans_region(node, cx):
    ll_region = mangle(node, cx)
    region_type = cx.type_map[node]
    ll_size_expr = trans_node(node.size_expr, cx).read(cx)

    field_types = trans_fields(region_type.kind.contains_type, cx)
    ll_field_types = [trans_type(field_type, cx) for field_type in field_types.itervalues()]

    ispace = mangle_temp()
    ispace_alloc = mangle_temp()
    fspace = mangle_temp()
    fspace_alloc = mangle_temp()
    fields = OrderedDict((field, mangle_temp()) for field in field_types.iterkeys())
    ispaces = mangle_temp()
    fspaces = mangle_temp()
    field_set = mangle_temp()
    region_requirement = mangle_temp()
    privilege = 'READ_WRITE'
    physical_region = mangle_temp()
    accessors = OrderedDict((field, mangle_temp()) for field in field_types.iterkeys())
    create_region = (
        ll_size_expr.actions +
        ['IndexSpace %s = runtime->create_index_space(ctx, %s);' % (
                ispace, ll_size_expr.value),
         'IndexAllocator %s = runtime->create_index_allocator(ctx, %s);' % (
                ispace_alloc, ispace),
         'FieldSpace %s = runtime->create_field_space(ctx);' % fspace,
         'FieldAllocator %s = runtime->create_field_allocator(ctx, %s);' % (fspace_alloc, fspace)] +
        ['FieldID %s = %s.allocate_field(sizeof(%s));' % (
                field, fspace_alloc, ll_field_type)
         for field, ll_field_type in zip(fields.itervalues(), ll_field_types)] +
        ['LogicalRegion %s = runtime->create_logical_region(ctx, %s, %s);' % (
                ll_region, ispace, fspace),
         'std::vector<IndexSpaceRequirement> %s;' % ispaces,
         '%s.push_back(IndexSpaceRequirement(%s, ALLOCABLE, %s));' % (ispaces, ispace, ispace),
         'std::vector<FieldSpaceRequirement> %s;' % fspaces,
         '%s.push_back(FieldSpaceRequirement(%s, ALLOCABLE));' % (fspaces, fspace),
         'std::vector<FieldID> %s;' % field_set] +
        ['%s.push_back(%s);' % (field_set, field)
         for field in fields.itervalues()] +
        ['RegionRequirement %s(%s, std::set<FieldID>(%s.begin(), %s.end()), %s, %s, EXCLUSIVE, %s);' % (
                region_requirement, ll_region, field_set, field_set, field_set, privilege, ll_region),
         'PhysicalRegion %s = runtime->map_region(ctx, %s);' % (
                physical_region, region_requirement),
         '%s.wait_until_valid();' % physical_region] +
        ['RegionAccessor<AccessorType::Generic, %s> %s = %s.get_field_accessor(%s).typeify<%s>();' % (
                ll_field_type, accessor, physical_region, field, ll_field_type)
         for accessor, field, ll_field_type in zip(
                accessors.itervalues(), fields.itervalues(), ll_field_types)])

    region_value = Region(
        node = node,
        expr = Expr(ll_region, []),
        type = region_type,
        region_root = region_type,
        region_requirement = region_requirement,
        physical_region = physical_region,
        ispace = ispace,
        fspace = fspace,
        field_map = fields,
        field_type_map = field_types,
        allocator = ispace_alloc,
        accessor_map = accessors)

    cx.env.insert(node.name, region_value)
    cx.regions[region_type] = region_value
    cx.mapped_regions[-1].add(region_value)

    return Statement(create_region)

def trans_array(node, cx):
    ll_region = mangle(node, cx)
    region_type = cx.type_map[node]
    ispace_expr = cx.env.lookup(cx.type_map[node.ispace_type].name).read(cx)

    field_types = trans_fields(region_type.kind.contains_type, cx)
    ll_field_types = [trans_type(field_type, cx) for field_type in field_types.itervalues()]

    ispace = mangle_temp()
    fspace = mangle_temp()
    fspace_alloc = mangle_temp()
    fields = OrderedDict((field, mangle_temp()) for field in field_types.iterkeys())
    ispaces = mangle_temp()
    fspaces = mangle_temp()
    field_set = mangle_temp()
    region_requirement = mangle_temp()
    privilege = 'READ_WRITE'
    physical_region = mangle_temp()
    accessors = OrderedDict((field, mangle_temp()) for field in field_types.iterkeys())
    create_region = (
        ispace_expr.actions +
        ['IndexSpace %s = %s;' % (
                ispace, ispace_expr.value),
         'FieldSpace %s = runtime->create_field_space(ctx);' % fspace,
         'FieldAllocator %s = runtime->create_field_allocator(ctx, %s);' % (fspace_alloc, fspace)] +
        ['FieldID %s = %s.allocate_field(sizeof(%s));' % (
                field, fspace_alloc, ll_field_type)
         for field, ll_field_type in zip(fields.itervalues(), ll_field_types)] +
        ['LogicalRegion %s = runtime->create_logical_region(ctx, %s, %s);' % (
                ll_region, ispace, fspace),
         'std::vector<IndexSpaceRequirement> %s;' % ispaces,
         '%s.push_back(IndexSpaceRequirement(%s, ALLOCABLE, %s));' % (ispaces, ispace, ispace),
         'std::vector<FieldSpaceRequirement> %s;' % fspaces,
         '%s.push_back(FieldSpaceRequirement(%s, ALLOCABLE));' % (fspaces, fspace),
         'std::vector<FieldID> %s;' % field_set] +
        ['%s.push_back(%s);' % (field_set, field)
         for field in fields.itervalues()] +
        ['RegionRequirement %s(%s, std::set<FieldID>(%s.begin(), %s.end()), %s, %s, EXCLUSIVE, %s);' % (
                region_requirement, ll_region, field_set, field_set, field_set, privilege, ll_region),
         'PhysicalRegion %s = runtime->map_region(ctx, %s);' % (
                physical_region, region_requirement),
         '%s.wait_until_valid();' % physical_region] +
        ['RegionAccessor<AccessorType::Generic, %s> %s = %s.get_field_accessor(%s).typeify<%s>();' % (
                ll_field_type, accessor, physical_region, field, ll_field_type)
         for accessor, field, ll_field_type in zip(
                accessors.itervalues(), fields.itervalues(), ll_field_types)])

    region_value = Region(
        node = node,
        expr = Expr(ll_region, []),
        type = region_type,
        region_root = region_type,
        region_requirement = region_requirement,
        physical_region = physical_region,
        ispace = ispace,
        fspace = fspace,
        field_map = fields,
        field_type_map = field_types,
        allocator = None,
        accessor_map = accessors)

    cx.env.insert(node.name, region_value)
    cx.regions[region_type] = region_value
    cx.mapped_regions[-1].add(region_value)

    return Statement(create_region)

def trans_ispace(node, cx):
    ll_ispace = mangle(node, cx)
    cx.env.insert(node.name, Value(node, Expr(ll_ispace, []), cx.type_map[node]))
    ll_size_expr = trans_node(node.size_expr, cx).read(cx)

    ispace_alloc = mangle_temp()
    create_ispace = (
        ll_size_expr.actions +
        ['IndexSpace %s = runtime->create_index_space(ctx, %s);' % (
                ll_ispace, ll_size_expr.value),
         'IndexAllocator %s = runtime->create_index_allocator(ctx, %s);' % (
                ispace_alloc, ll_ispace),
         '%s.alloc(%s);' % (
                ispace_alloc, ll_size_expr.value)])

    return Statement(create_ispace)

def trans_partition(node, cx):
    partition_type = cx.type_map[node]
    parent_region = cx.env.lookup(node.region_type.name)
    parent_region_expr = parent_region.read(cx)
    coloring_expr = trans_node(node.coloring_expr, cx).read(cx)
    temp_coloring_name = mangle_temp()
    index_partition = mangle_temp()
    logical_partition = mangle_temp()

    subregions = partition_type.static_subregions

    ll_regions = [mangle_temp() for region in subregions]
    ispaces = [mangle_temp() for region in subregions]
    allocators = [mangle_temp() for region in subregions]
    region_types = [region_type for region_type in subregions.itervalues()]

    # Handle index spaces:
    if types.is_ispace(partition_type.kind.region):
        create_partition = (
            parent_region_expr.actions +
            coloring_expr.actions +
            ['Coloring %s = %s;' % (temp_coloring_name, coloring_expr.value)] +
            ['%s[%s];' % (temp_coloring_name, color)
             for color in subregions.keys()] +
            ['IndexPartition %s = runtime->create_index_partition(ctx, %s, %s, %s);' % (
                    index_partition, parent_region_expr.value, temp_coloring_name,
                    'true' if partition_type.kind.mode == types.Partition.DISJOINT else 'false')] +
            ['IndexSpace %s = runtime->get_index_subspace(ctx, %s, Color(%s));' % (
                    ispace,
                    index_partition,
                    color)
             for ispace, color in zip(ispaces, subregions.keys())] +
            ['IndexAllocator %s = runtime->create_index_allocator(ctx, %s);' % (
                    ispace_alloc, ispace)
             for ispace_alloc, ispace in zip(allocators, ispaces)])

        partition_value = Partition(
            node = node,
            expr = Expr(index_partition, []),
            type = partition_type,
            region = parent_region)
        cx.env.insert(node.name, partition_value)

        for ispace_type, ispace, allocator in zip(
            region_types, ispaces, allocators):

            ispace_value = Value(
                node = node,
                expr = Expr(ispace, []),
                type = ispace_type)
            cx.regions[ispace_type] = ispace_value

        return Statement(create_partition)

    # Handle regions:
    parent_region = cx.env.lookup(node.region_type.name)
    parent_region_expr = parent_region.read(cx)
    coloring_expr = trans_node(node.coloring_expr, cx).read(cx)
    temp_coloring_name = mangle_temp()
    index_partition = mangle_temp()
    logical_partition = mangle_temp()

    subregions = partition_type.static_subregions

    ll_regions = [mangle_temp() for region in subregions]
    ispaces = [mangle_temp() for region in subregions]
    allocators = [mangle_temp() for region in subregions]
    region_types = [region_type for region_type in subregions.itervalues()]

    create_partition = (
        parent_region_expr.actions +
        coloring_expr.actions +
        ['Coloring %s = %s;' % (temp_coloring_name, coloring_expr.value)] +
        ['%s[%s];' % (temp_coloring_name, color)
         for color in subregions.keys()] +
        ['IndexPartition %s = runtime->create_index_partition(ctx, %s, %s, %s);' % (
                index_partition, parent_region.ispace, temp_coloring_name,
                'true' if partition_type.kind.mode == types.Partition.DISJOINT else 'false'),
         'LogicalPartition %s = runtime->get_logical_partition(ctx, %s, %s);' % (
                logical_partition, parent_region_expr.value, index_partition)] +
        ['LogicalRegion %s = runtime->get_logical_subregion_by_color(ctx, %s, Color(%s));' % (
                ll_region,
                logical_partition,
                color)
         for ll_region, color in zip(ll_regions, subregions.keys())] +
        ['IndexSpace %s = %s.get_index_space();' % (ispace, ll_region)
         for ispace, ll_region in zip(ispaces, ll_regions)] +
        ['IndexAllocator %s = runtime->create_index_allocator(ctx, %s);' % (
                ispace_alloc, ispace)
         for ispace_alloc, ispace in zip(allocators, ispaces)])

    partition_value = Partition(
        node = node,
        expr = Expr(logical_partition, []),
        type = partition_type,
        region = parent_region)
    cx.env.insert(node.name, partition_value)

    for ll_region, region_type, ispace, allocator in zip(
        ll_regions, region_types, ispaces, allocators):

        region_value = Region(
            node = node,
            expr = Expr(ll_region, []),
            type = region_type,
            region_root = parent_region.region_root,
            region_requirement = None,
            physical_region = None,
            ispace = ispace,
            fspace = parent_region.fspace,
            field_map = parent_region.field_map,
            field_type_map = parent_region.field_type_map,
            allocator = allocator,
            accessor_map = parent_region.accessor_map)

        cx.regions[region_type] = region_value

    return Statement(create_partition)

def trans_unpack(node, cx):
    actions = []

    struct = trans_node(node.expr, cx)
    ll_struct = struct.read(cx)
    ll_struct_type = trans_type(struct.type.as_read(), cx)
    ll_name = mangle(node, cx)

    actions.extend(ll_struct.actions)
    actions.append('%s %s = %s;' % (ll_struct_type, ll_name, ll_struct.value))
    cx.env.insert(node.name, Value(node, Expr(ll_name, []), struct.type.as_read()))

    region_params = cx.type_map[node.expr].as_read().regions
    region_args = cx.type_map[node]

    for index, region_node, region_param, region_arg in zip(
        xrange(len(region_args)), node.regions.regions, region_params, region_args):

        ll_region = mangle(region_node, cx)
        ispace = mangle_temp()

        # We extract the LogicalRegion contained in the region
        # relation, and we could even inline map it to get a
        # PhysicalRegion. But we almost never actually need it because
        # the region is contained within some larger parent region, so
        # we can just access it through that.

        region_root = None
        for available_region in cx.regions.itervalues():
            subregions = region_analysis.summarize_subregions(
                cx.constraints, cx.region_usage, available_region.type)
            if region_arg in subregions:
                region_root = available_region
        assert region_root is not None

        extract_region = (
            ['LogicalRegion %s = %s.%s;' % (
                    ll_region,
                    ll_name,
                    region_param.name),
             'IndexSpace %s = %s.get_index_space();' % (
                    ispace, ll_region)])
        actions.extend(extract_region)

        region_value = Region(
            node = region_node,
            expr = Expr(ll_region, []),
            type = region_arg,
            region_root = region_root.type,
            region_requirement = None,
            physical_region = None,
            ispace = ispace,
            fspace = region_root.fspace,
            field_map = region_root.field_map,
            field_type_map = region_root.field_type_map,
            allocator = None,
            accessor_map = region_root.accessor_map)

        cx.env.insert(region_node.name, region_value)
        cx.regions[region_arg] = region_value

    return Statement(actions)

def trans_return(node, cx):
    expr = trans_node(node.expr, cx)
    ll_expr = expr.read(cx)
    ll_expr_type = trans_return_type(expr.type.as_read(), cx)
    temp_name = mangle_temp()

    return_value = (
        ll_expr.actions +
        ['%s %s(%s);' % (
                ll_expr_type, temp_name,
                ll_expr.value)])

    cleanup = trans_cleanup(0, cx)

    return_statement = 'return %s;' % temp_name
    return Statement(
        return_value + cleanup + [return_statement])

# FIXME: All of the following functinos need to be made so that their
# values are idempotent.
def trans_new(node, cx):
    pointer_type = cx.type_map[node]
    assert len(pointer_type.regions) == 1
    region_type = pointer_type.regions[0]
    ll_ispace_alloc = cx.regions[region_type].allocator
    assert ll_ispace_alloc is not None

    actions = []

    temp = mangle_temp()
    actions.append('ptr_t %s = %s.alloc();' % (
            temp, ll_ispace_alloc))
    return Value(node, Expr(temp, actions), pointer_type)

def is_prefix(seq, fragment):
    return seq[:len(fragment)] == fragment

def safe_read(ll_pointer, ll_result, field_path, cx):
    def helper(otherwise, region_type):
        ll_temp = mangle_temp()

        region = cx.regions[region_type]
        region_expr = region.read(cx)

        accessors = OrderedDict(
            (field, accessor)
            for field, accessor in region.accessor_map.iteritems()
            if is_prefix(field, field_path))
        assert len(accessors) > 0

        actions = []
        actions.extend(region_expr.actions)
        actions.extend(
            ['ptr_t %s = runtime->safe_cast(ctx, %s, %s);' % (
                    ll_temp, ll_pointer, region_expr.value),
             'if (!%s.is_null()) {' % ll_temp,
             Block(['%s = %s.read(%s);' % (
                            '.'.join((ll_result,) + field[len(field_path):]), accessor, ll_temp)
                    for field, accessor in accessors.iteritems()]),
             '} else {',
             Block(otherwise),
             '}'])
        return actions
    return helper

def safe_write(ll_pointer, ll_update, field_path, cx):
    def helper(otherwise, region_type):
        ll_temp = mangle_temp()

        region = cx.regions[region_type]
        region_expr = region.read(cx)

        accessors = OrderedDict(
            (field, accessor)
            for field, accessor in region.accessor_map.iteritems()
            if is_prefix(field, field_path))
        assert len(accessors) > 0

        actions = []
        actions.extend(region_expr.actions)
        actions.extend(
            ['ptr_t %s = runtime->safe_cast(ctx, %s, %s);' % (
                    ll_temp, ll_pointer, region_expr.value),
             'if (!%s.is_null()) {' % ll_temp,
             Block(['%s.write(%s, %s);' % (
                            accessor, ll_temp, '.'.join((ll_update,) + field[len(field_path):]))
                    for field, accessor in accessors.iteritems()]),
             '} else {',
             Block(otherwise),
             '}'])
        return actions
    return helper

def safe_reduce(ll_pointer, ll_update, op, field_path, cx):
    def helper(otherwise, region_type):
        ll_temp = mangle_temp()

        region = cx.regions[region_type]
        region_expr = region.read(cx)

        field_types = trans_fields(region.type.kind.contains_type, cx)


        def lookup_reduction_op(op, field, cx):
            field_type = field_types[field]
            return cx.reduction_ops[(op, field_type.key())].read(cx).value

        accessors = OrderedDict(
            (field, (accessor, lookup_reduction_op(op, field, cx)))
            for field, accessor in region.accessor_map.iteritems()
            if is_prefix(field, field_path))
        assert len(accessors) > 0

        actions = []
        actions.extend(region_expr.actions)
        actions.extend(
            ['ptr_t %s = runtime->safe_cast(ctx, %s, %s);' % (
                    ll_temp, ll_pointer, region_expr.value),
             'if (!%s.is_null()) {' % ll_temp,
             Block(['%s.reduce<%s>(%s, %s);' % (
                            accessor,
                            reduction_op,
                            ll_temp,
                            '.'.join((ll_update,) + field[len(field_path):]))
                    for field, (accessor, reduction_op) in accessors.iteritems()]),
             '} else {',
             Block(otherwise),
             '}'])
        return actions
    return helper

def trans_read_helper(node, pointer_value, field_path, cx):
    pointer_expr = pointer_value.read(cx)
    pointer_type = pointer_value.type.as_read()

    actions = []
    actions.extend(pointer_expr.actions)

    ll_result = mangle_temp()
    ll_result_type = trans_type(cx.type_map[node].as_read(), cx)

    actions.append('%s %s;' % (ll_result_type, ll_result))
    actions.extend(
        reduce(safe_read(pointer_expr.value, ll_result, field_path, cx),
               reversed(pointer_type.regions),
               ['assert(0);']))
    return Value(node, Expr(ll_result, actions), cx.type_map[node])

def trans_read(node, cx):
    pointer_value = trans_node(node.pointer_expr, cx)
    return trans_read_helper(node, pointer_value, (), cx)

def trans_write_helper(node, pointer_value, field_path, update_value, cx):
    pointer_expr = pointer_value.read(cx)
    pointer_type = pointer_value.type.as_read()

    update_expr = update_value.read(cx)
    update_type = update_value.type.as_read()
    ll_update_type = trans_type(update_type, cx)

    actions = []
    actions.extend(pointer_expr.actions)
    actions.extend(update_expr.actions)
    actions.extend(
        reduce(
            safe_write(pointer_expr.value, update_expr.value, field_path, cx),
            reversed(pointer_type.regions),
            ['assert(0);']))
    return Value(node, Expr(update_expr.value, actions), cx.type_map[node])

def trans_write(node, cx):
    pointer_value = trans_node(node.pointer_expr, cx)
    update_value = trans_node(node.value_expr, cx)
    return trans_write_helper(node, pointer_value, (), update_value, cx)

def trans_reduce_helper(node, pointer_value, field_path, op, update_value, cx):
    pointer_expr = pointer_value.read(cx)
    pointer_type = pointer_value.type.as_read()

    update_expr = update_value.read(cx)
    update_type = update_value.type.as_read()
    ll_update_type = trans_type(update_type, cx)

    actions = []
    actions.extend(pointer_expr.actions)
    actions.extend(update_expr.actions)

    actions.extend(
        reduce(
            safe_reduce(pointer_expr.value, update_expr.value, op, field_path, cx),
            reversed(pointer_type.regions),
            ['assert(0);']))
    return Value(node, Expr(update_expr.value, actions), cx.type_map[node])

def trans_reduce(node, cx):
    pointer_value = trans_node(node.pointer_expr, cx)
    update_value = trans_node(node.value_expr, cx)
    return trans_reduce_helper(node, pointer_value, (), node.op, update_value, cx)

def trans_dereference(node, cx):
    pointer = trans_node(node.pointer_expr, cx)
    ll_pointer = pointer.read(cx)
    pointer_type = pointer.type.as_read()
    return Reference(node, ll_pointer, pointer_type)

def trans_array_access(node, cx):
    array_value = trans_node(node.array_expr, cx)
    array_expr = array_value.read(cx)
    array_type = array_value.type.as_read()
    index_value = trans_node(node.index_expr, cx)
    index_expr = index_value.read(cx)
    index_type = index_value.type.as_read()

    # Handle partitions:
    if types.is_partition(array_type):
        region_type = cx.type_map[node]

        # If the subregion already exists (e.g. because it is static),
        # reuse it.
        if region_type in cx.regions:
            return cx.regions[region_type]

        # Otherwise generate a fresh subregion.
        parent_region = array_value.region

        ll_region = mangle_temp()
        ispace = mangle_temp()
        allocator = mangle_temp()

        # Handle index spaces:
        if types.is_ispace(array_type.kind.region):
            create_ispace = (
                array_expr.actions +
                index_expr.actions +
                ['IndexSpace %s = runtime->get_index_subspace(ctx, %s, Color(%s));' % (
                        ispace,
                        array_expr.value,
                        index_expr.value),
                 'IndexAllocator %s = runtime->create_index_allocator(ctx, %s);' % (
                        allocator, ispace)])

            ispace_value = Value(
                node = node,
                expr = Expr(ispace, []),
                type = region_type)

            cx.regions[region_type] = ispace_value

            ispace_value = copy.copy(ispace_value)
            ispace_value.expr = Expr(ispace, create_ispace)
            return ispace_value

        # Handle regions:
        create_region = (
            array_expr.actions +
            index_expr.actions +
            ['LogicalRegion %s = runtime->get_logical_subregion_by_color(ctx, %s, Color(%s));' % (
                    ll_region,
                    array_expr.value,
                    index_expr.value),
             'IndexSpace %s = %s.get_index_space();' % (ispace, ll_region),
             'IndexAllocator %s = runtime->create_index_allocator(ctx, %s);' % (
                    allocator, ispace)])

        region_value = Region(
            node = node,
            expr = Expr(ll_region, []),
            type = region_type,
            region_root = parent_region.region_root,
            region_requirement = None,
            physical_region = None,
            ispace = ispace,
            fspace = parent_region.fspace,
            field_map = parent_region.field_map,
            field_type_map = parent_region.field_type_map,
            allocator = allocator,
            accessor_map = parent_region.accessor_map)

        cx.regions[region_type] = region_value

        region_value = copy.copy(region_value)
        region_value.expr = Expr(ll_region, create_region)
        return region_value

    # Handle array slicing:
    if types.is_region(array_type) and types.is_ispace(index_type):
        region_type = cx.type_map[node]

        # If the subregion already exists (e.g. because it is static),
        # reuse it.
        if region_type in cx.regions:
            return cx.regions[region_type]

        # Otherwise generate a fresh subregion.
        region_value = cx.regions[array_type]

        ll_region = mangle_temp()
        ispace = mangle_temp()
        allocator = mangle_temp()

        create_region = (
            array_expr.actions +
            index_expr.actions +
            ['LogicalRegion %s = runtime->get_logical_subregion_by_tree(ctx, %s, %s.get_field_space(), %s.get_tree_id());' % (
                    ll_region,
                    index_expr.value,
                    array_expr.value,
                    array_expr.value),
             'IndexSpace %s = %s.get_index_space();' % (ispace, ll_region),
             'IndexAllocator %s = runtime->create_index_allocator(ctx, %s);' % (
                    allocator, ispace)])

        region_value = Region(
            node = node,
            expr = Expr(ll_region, []),
            type = region_type,
            region_root = array_value.region_root,
            region_requirement = None,
            physical_region = None,
            ispace = ispace,
            fspace = array_value.fspace,
            field_map = array_value.field_map,
            field_type_map = array_value.field_type_map,
            allocator = allocator,
            accessor_map = array_value.accessor_map)

        cx.regions[region_type] = region_value

        region_value = copy.copy(region_value)
        region_value.expr = Expr(ll_region, create_region)
        return region_value

    # Handle arrays:
    assert types.is_region(array_type) and types.is_int(index_type)
    pointer_expr = Expr('(ptr_t(%s))' % index_expr.value, index_expr.actions)

    pointer_type = types.Pointer(array_type.kind.contains_type, [array_type])
    return Reference(node, pointer_expr, pointer_type)

def trans_dereference_field(node, cx):
    pointer = trans_node(node.pointer_expr, cx)
    ll_pointer = pointer.read(cx)
    pointer_type = pointer.type.as_read()
    return Reference(node, ll_pointer, pointer_type, (node.field_name,))

def trans_call(node, cx):
    task = trans_node(node.function, cx)
    all_args = trans_node(node.args, cx)
    return_type = cx.type_map[node]

    # Hack: Call builtin functions directly.
    if isinstance(task, Builtin):
        function = task.read(cx)
        args = [arg.read(cx) for arg in all_args]
        return Value(
            node,
            Expr(
                '%s(%s)' % (
                    function.value,
                    ', '.join(arg.value for arg in args)),
                function.actions + [action for arg in args for action in arg.actions]),
            return_type)

    # FIXME: This code is too brittle to handle dynamic task calls.
    assert isinstance(task, Function)

    actions = []

    # Normal parameters are passed via TaskArgument.
    params = [mangle(param_node, cx)
              for param_node, param_type in zip(task.node.params.params, task.type.param_types)
              if not types.is_region(param_type)]
    arg_values = [arg.read(cx)
            for arg, arg_node in zip(all_args, node.args.args)
            if not types.is_region(cx.type_map[arg_node])]
    args = [arg.value for arg in arg_values]
    for arg in arg_values:
        actions.extend(arg.actions)
    assert len(params) == len(args)

    # Region parameters are (mostly) passed via RegionRequirement.
    region_param_types = [
        param_type
        for param_type in task.type.param_types
        if types.is_region(param_type)]
    region_params = [
        mangle_region(param_node, cx)
        for param_node, param_type in zip(task.node.params.params, task.type.param_types)
        if types.is_region(param_type)]
    assert len(region_param_types) == len(region_params)

    region_arg_types = [
        cx.type_map[arg_node]
        for arg_node in node.args.args
        if types.is_region(cx.type_map[arg_node])]
    region_args = [
        arg
        for arg_node, arg in zip(node.args.args, all_args)
        if types.is_region(cx.type_map[arg_node])]
    for region_arg in region_args:
        actions.extend(region_arg.read(cx).actions)
    assert len(region_arg_types) == len(region_args)
    assert len(region_params) == len(region_args)

    # For regions with no privileges requested, do not pass a
    # RegionRequirement.
    privileges_requested = [
        trans_privilege_mode(region_type, task.type.privileges, cx)
        for region_type in region_param_types]

    region_roots = [
        cx.regions[cx.regions[region_type].region_root]
        for region_type in region_arg_types]

    task_params_struct_name = task.params
    task_result = mangle_temp()
    ispaces = [
        (region.ispace, parent.ispace)
        for region_type, region, parent
        in zip(region_param_types, region_args, region_roots)]
    field_sets = [
        [mangle_temp() for privilege in privileges]
        for privileges in privileges_requested]
    fields = [
        [[region_arg.field_map[field_path] for field_path in field_paths]
         for field_paths in privileges.itervalues()]
        for region_arg, privileges in zip(region_args, privileges_requested)]
    region_meta = (
        [(region_param, region_arg.read(cx).value)
         for region_param, region_arg in zip(region_params, region_args)] +
        [('field_%s' % ('_'.join((region_param,) + field_path)), field)
         for region_param, region_arg in zip(region_params, region_args)
         for field_path, field in region_arg.field_map.iteritems()])

    actions.append('Future %s;' % task_result)

    # Calculate the set of regions currently in use that are needed
    # for the child task call and unmap them.
    unmap_regions = OrderedDict((cx.regions[cx.regions[region_type].region_root], ())
                                for region_type, privileges in zip(region_arg_types, privileges_requested)
                                if len(privileges) > 0).keys()

    task_call_body = (
        ['std::vector<IndexSpaceRequirement> ispaces;'] +
        ['ispaces.push_back(IndexSpaceRequirement(%s, ALLOCABLE, %s));' % (ispace, parent)
         for (ispace, parent) in ispaces] +
        ['std::vector<FieldSpaceRequirement> fspaces;'] +
        ['std::vector<FieldID> %s;' % field_set
         for region_field_sets in field_sets
         for field_set in region_field_sets] +
        ['%s.push_back(%s);' % (field_set, field)
         for region_field_sets, region_fields in zip(field_sets, fields)
         for field_set, field_set_fields in zip(region_field_sets, region_fields)
         for field in field_set_fields] +
        ['std::vector<RegionRequirement> regions;'] +
        ['regions.push_back(RegionRequirement(%s, std::set<FieldID>(%s.begin(), %s.end()), %s, %s, EXCLUSIVE, %s));' % (
                region.read(cx).value, field_set, field_set, field_set, privilege, region_root.read(cx).value)
         for region, region_root, region_field_sets, privileges in zip(
                region_args, region_roots, field_sets, privileges_requested)
         for field_set, privilege in zip(region_field_sets, privileges.iterkeys())] +
        ['%s args;' % task_params_struct_name] +
        ['args.%s = %s;' % meta for meta in region_meta] +
        ['args.%s = %s;' % (param, arg)
         for param, arg in zip(params, args)] +
        ['size_t buffer_size = args.legion_buffer_size();',
         'void *buffer = malloc(buffer_size);',
         'assert(buffer);',
         'args.legion_serialize(buffer);'] +
        ['runtime->unmap_region(ctx, %s);' % (region.physical_region)
         for region in unmap_regions] +
        [('%s = runtime->execute_task(ctx, %s, ispaces, fspaces, regions,' +
          'TaskArgument(buffer, buffer_size), Predicate::TRUE_PRED, false);') %
         (task_result, mangle_task_enum(task.node.name))] +
        ['%s = runtime->map_region(ctx, %s);' % (region.physical_region, region.region_requirement)
         for region in unmap_regions] +
        ['%s.wait_until_valid();' % (region.physical_region)
         for region in unmap_regions])
    # FIXME: Re-establish accessors when re-mapping.
    actions.extend(['{', Block(task_call_body), '}'])

    task_return_expr = '(%s.get_result<%s>())' % (
        task_result,
        trans_return_type(return_type, cx))
    if types.is_void(return_type):
        task_return_expr = '((void)0)'
    if types.is_coloring(return_type):
        task_return_expr = '(%s.ref())' % task_return_expr

    return Value(node, Expr(task_return_expr, actions), return_type)

def trans_type(t, cx):
    # For simple types, translate them directly:
    if types.is_simple(t):
        return types.type_name(t)
    if types.is_integral(t):
        return t.cname
    if types.is_ispace(t):
        return 'IndexSpace'
    if types.is_region(t):
        return 'LogicalRegion'
    if types.is_coloring(t):
        return 'Coloring'
    if types.is_pointer(t):
        return 'ptr_t'
    if is_fieldid(t):
        return 'FieldID'

    # For complex types, lookup a translation:
    k = t.key()
    if k in cx.ll_type_map:
        return cx.ll_type_map[k]
    raise TypeTranslationFailedException('Code generation failed at %s' % t)

def trans_return_type(t, cx):
    # Some types need to be passed via wrapper objects. These types
    # are listed below.
    if types.is_coloring(t):
        return 'ColoringSerializer'

    return trans_type(t, cx)

def trans_fields(t, cx):
    if t in cx.ll_field_map:
        return cx.ll_field_map[t]

    if types.is_struct(t):
        field_map = OrderedDict()
        for field_type in t.regions:
            field_name = field_type.name
            nested_field_map = trans_fields(field_type, cx)
            for nested_field_path, nested_field_type in nested_field_map.iteritems():
                field_path = (field_name,) + nested_field_path
                field_map[field_path] = nested_field_type
        for field_name, field_type in t.field_map.iteritems():
            nested_field_map = trans_fields(field_type, cx)
            for nested_field_path, nested_field_type in nested_field_map.iteritems():
                field_path = (field_name,) + nested_field_path
                field_map[field_path] = nested_field_type
        cx.ll_field_map[t] = field_map
        return field_map

    field_map = OrderedDict([((), t)])
    cx.ll_field_map[t] = field_map
    return field_map

def image_map(d):
    return dict((v, set(k2 for k2, v2 in d.iteritems() if v2 == v))
                for v in d.itervalues())

def trans_privilege_mode(region, requested_privileges, cx):
    field_privileges = {}
    for privilege in requested_privileges:
        if privilege.region == region:
            if privilege.field_path not in field_privileges:
                field_privileges[privilege.field_path] = set()
            field_privileges[privilege.field_path].add(privilege)

    field_types = trans_fields(region.kind.contains_type, cx)
    nested_field_privileges = {}
    for field_path, privileges in field_privileges.iteritems():
        for nested_field_path in field_types.iterkeys():
            if is_prefix(nested_field_path, field_path):
                if nested_field_path not in nested_field_privileges:
                    nested_field_privileges[nested_field_path] = set()
                nested_field_privileges[nested_field_path].update(privileges)

    ll_nested_field_privileges = {}
    for nested_field_path, privileges in nested_field_privileges.iteritems():
        reads = False
        writes = False
        reduce_op = None
        element_type = None
        for privilege in privileges:
            if types.Privilege.READ == privilege.privilege:
                reads = True
            elif types.Privilege.WRITE == privilege.privilege:
                writes = True
            elif types.Privilege.REDUCE == privilege.privilege:
                assert reduce_op is None and element_type is None
                reduce_op = privilege.op
                field_types = trans_fields(privilege.region.kind.contains_type, cx)
                element_type = field_types[nested_field_path]
            else:
                assert False

        if reduce_op is not None:
            assert not reads and not writes

        ll_privilege = None
        if reads and writes:
            ll_privilege = 'READ_WRITE'
        elif reads:
            ll_privilege = 'READ_ONLY'
        elif writes:
            ll_privilege = 'WRITE_ONLY'
        elif reduce_op is not None:
            ll_privilege = mangle_reduction_op_enum(reduce_op, element_type, cx)
        else:
            assert False

        if ll_privilege is not None:
            ll_nested_field_privileges[nested_field_path] = ll_privilege

    return image_map(ll_nested_field_privileges)

def trans_node(node, cx):
    if isinstance(node, ast.Program):
        return trans_program(node, cx)
    if isinstance(node, ast.Import):
        return trans_import(node, cx)
    if isinstance(node, ast.Struct):
        return trans_struct(node, cx)
    if isinstance(node, ast.Function):
        return trans_function(node, cx)
    if isinstance(node, ast.Params):
        return ', '.join(trans_node(param, cx) for param in node.params)
    if isinstance(node, ast.Param):
        return '%s %s' % (trans_node(node.declared_type, cx), node.name)
    if isinstance(node, ast.TypeVoid):
        return trans_type(cx.type_map[node].type, cx)
    if isinstance(node, ast.TypeBool):
        return trans_type(cx.type_map[node].type, cx)
    if isinstance(node, ast.TypeDouble):
        return trans_type(cx.type_map[node].type, cx)
    if isinstance(node, ast.TypeFloat):
        return trans_type(cx.type_map[node].type, cx)
    if isinstance(node, ast.TypeInt):
        return trans_type(cx.type_map[node].type, cx)
    if isinstance(node, ast.TypeUInt):
        return trans_type(cx.type_map[node].type, cx)
    if isinstance(node, ast.TypeInt8):
        return trans_type(cx.type_map[node].type, cx)
    if isinstance(node, ast.TypeInt16):
        return trans_type(cx.type_map[node].type, cx)
    if isinstance(node, ast.TypeInt32):
        return trans_type(cx.type_map[node].type, cx)
    if isinstance(node, ast.TypeInt64):
        return trans_type(cx.type_map[node].type, cx)
    if isinstance(node, ast.TypeUInt8):
        return trans_type(cx.type_map[node].type, cx)
    if isinstance(node, ast.TypeUInt16):
        return trans_type(cx.type_map[node].type, cx)
    if isinstance(node, ast.TypeUInt32):
        return trans_type(cx.type_map[node].type, cx)
    if isinstance(node, ast.TypeUInt64):
        return trans_type(cx.type_map[node].type, cx)
    if isinstance(node, ast.TypeColoring):
        return trans_type(cx.type_map[node].type, cx)
    if isinstance(node, ast.TypeID):
        return trans_type(cx.type_map[node].type, cx)
    if isinstance(node, ast.TypePointer):
        return trans_type(cx.type_map[node].type, cx)
    if isinstance(node, ast.Block):
        cx = cx.new_block_scope()
        block = [trans_node(statement, cx) for statement in node.block]
        cleanup = trans_cleanup(-1, cx)
        return Statement(['{', Block(block + cleanup), '}'])
    if isinstance(node, ast.StatementAssert):
        ll_expr = trans_node(node.expr, cx).read(cx)
        return Statement(
            ll_expr.actions +
            ['assert(%s);' % ll_expr.value])
    if isinstance(node, ast.StatementExpr):
        ll_expr = trans_node(node.expr, cx).read(cx)
        return Statement(
            ll_expr.actions +
            ['%s;' % ll_expr.value])
    if isinstance(node, ast.StatementIf):
        ll_condition = trans_node(node.condition, cx).read(cx)
        if node.else_block is None:
            return Statement(
                ll_condition.actions +
                ['if (%s)' % ll_condition.value,
                 trans_node(node.then_block, cx)])
        return Statement(
            ll_condition.actions +
            ['if (%s)' % ll_condition.value,
             trans_node(node.then_block, cx),
             'else',
             trans_node(node.else_block, cx)])
    if isinstance(node, ast.StatementFor):
        return trans_for(node, cx)
    if isinstance(node, ast.StatementLet):
        return trans_let(node, cx)
    if isinstance(node, ast.StatementLetRegion):
        return trans_region(node, cx)
    if isinstance(node, ast.StatementLetArray):
        return trans_array(node, cx)
    if isinstance(node, ast.StatementLetIspace):
        return trans_ispace(node, cx)
    if isinstance(node, ast.StatementLetPartition):
        return trans_partition(node, cx)
    if isinstance(node, ast.StatementReturn):
        return trans_return(node, cx)
    if isinstance(node, ast.StatementUnpack):
        return trans_unpack(node, cx)
    if isinstance(node, ast.StatementVar):
        expr = trans_node(node.expr, cx)
        ll_expr = expr.read(cx)
        ll_expr_type = trans_type(expr.type.as_read(), cx)
        ll_name = mangle(node, cx)
        cx.env.insert(
            node.name,
            StackReference(
                node,
                Expr(ll_name, []),
                types.StackReference(expr.type.as_read())),
            shadow = True)
        return Statement(
            ll_expr.actions +
            ['%s %s = %s;' % (ll_expr_type, ll_name, ll_expr.value)])
    if isinstance(node, ast.StatementWhile):
        condition_top = trans_node(node.condition, cx)
        ll_condition_type = trans_type(condition_top.type, cx)
        ll_condition_top = condition_top.read(cx)
        ll_condition_bot = trans_node(node.condition, cx).read(cx)
        ll_condition_name = mangle_temp()
        block = trans_node(node.block, cx)
        return Statement(
            ll_condition_top.actions +
            ['%s %s = %s;' % (
                    ll_condition_type,
                    ll_condition_name,
                    ll_condition_top.value),
             'while (%s) {' % ll_condition_name,
             Block([block] +
                    ll_condition_bot.actions +
                    ['%s = %s;' % (
                            ll_condition_name,
                            ll_condition_bot.value)]),
             '}'])
    if isinstance(node, ast.ExprID):
        value = cx.env.lookup(node.name)
        assert value is not None
        return value
    if isinstance(node, ast.ExprAssignment):
        lval = trans_node(node.lval, cx)
        rval = trans_node(node.rval, cx)
        return Value(
            node,
            lval.write(rval, cx),
            cx.type_map[node])
    if isinstance(node, ast.ExprUnaryOp):
        arg = trans_node(node.arg, cx).read(cx)
        return Value(
            node,
            Expr(
                '(%s%s)' % (
                    node.op,
                    arg.value),
                arg.actions),
            cx.type_map[node])
    if isinstance(node, ast.ExprBinaryOp):
        lhs = trans_node(node.lhs, cx).read(cx)
        rhs = trans_node(node.rhs, cx).read(cx)
        return Value(
            node,
            Expr(
                '(%s %s %s)' % (
                    lhs.value,
                    node.op,
                    rhs.value),
                lhs.actions + rhs.actions),
            cx.type_map[node])
    if isinstance(node, ast.ExprReduceOp):
        lval = trans_node(node.lhs, cx)
        rval = trans_node(node.rhs, cx)
        return Value(
            node,
            lval.reduce(node.op, rval, cx),
            cx.type_map[node])
    if isinstance(node, ast.ExprCast):
        expr = trans_node(node.expr, cx).read(cx)
        return Value(
            node,
            Expr(
                '((%s)%s)' % (
                    trans_type(cx.type_map[node], cx),
                    expr.value),
                expr.actions),
            cx.type_map[node])
    if isinstance(node, ast.ExprNull):
        return Value(
            node,
            Expr('(ptr_t::nil())', []),
            cx.type_map[node])
    if isinstance(node, ast.ExprIsnull):
        ll_pointer_expr = trans_node(node.pointer_expr, cx).read(cx)
        return Value(
            node,
            Expr('(!%s)' % ll_pointer_expr.value, ll_pointer_expr.actions),
            cx.type_map[node])
    if isinstance(node, ast.ExprNew):
        return trans_new(node, cx)
    if isinstance(node, ast.ExprRead):
        return trans_read(node, cx)
    if isinstance(node, ast.ExprWrite):
        return trans_write(node, cx)
    if isinstance(node, ast.ExprReduce):
        return trans_reduce(node, cx)
    if isinstance(node, ast.ExprDereference):
        return trans_dereference(node, cx)
    if isinstance(node, ast.ExprArrayAccess):
        return trans_array_access(node, cx)
    if isinstance(node, ast.ExprFieldAccess):
        struct = trans_node(node.struct_expr, cx)
        if types.is_reference(struct.type):
            return struct.get_field(node, node.field_name)
        ll_struct = struct.read(cx)
        return Value(
            node,
            Expr('(%s.%s)' % (ll_struct.value, node.field_name),
                 ll_struct.actions),
            cx.type_map[node])
    if isinstance(node, ast.ExprFieldDereference):
        return trans_dereference_field(node, cx)
    if isinstance(node, ast.ExprFieldValues):
        cx = cx.increase_indent()
        ll_result_type = trans_type(cx.type_map[node], cx)
        ll_result = mangle_temp()

        actions = []
        actions.append('%s %s;' % (ll_result_type, ll_result))
        actions.extend(
            action
            for field_value in trans_node(node.field_values, cx)
            for action in field_value.actions + [field_value.value % ll_result])
        return Value(
            node,
            Expr(ll_result, actions),
            cx.type_map[node])
    if isinstance(node, ast.FieldValues):
        return [trans_node(field_value, cx)
                for field_value in node.field_values]
    if isinstance(node, ast.FieldValue):
        field_value = trans_node(node.value_expr, cx).read(cx)
        return Expr(
            '%%s.%s = %s;' % (
                node.field_name.replace('%', '%%'),
                field_value.value.replace('%', '%%')),
            field_value.actions)
    if isinstance(node, ast.ExprFieldUpdates):
        cx = cx.increase_indent()

        struct = trans_node(node.struct_expr, cx)
        ll_struct = struct.read(cx)

        ll_result_type = trans_type(cx.type_map[node], cx)
        ll_result = mangle_temp()

        actions = []
        actions.extend(ll_struct.actions)

        actions.append('%s %s;' % (ll_result_type, ll_result))

        actions.extend(
            '%s.%s = %s.%s;' % (ll_result, field_name, ll_struct.value, field_name)
            for field_name in struct.type.as_read().field_map.iterkeys())
        actions.extend(
            action
            for field_update in trans_node(node.field_updates, cx)
            for action in field_update.actions + [field_update.value % ll_result])

        return Value(
            node,
            Expr(ll_result, actions),
            cx.type_map[node])
    if isinstance(node, ast.FieldUpdates):
        return [trans_node(field_update, cx)
                for field_update in node.field_updates]
    if isinstance(node, ast.FieldUpdate):
        field_update = trans_node(node.update_expr, cx).read(cx)
        return Expr(
            '%%s.%s = %s;' % (
                node.field_name.replace('%', '%%'),
                field_update.value.replace('%', '%%')),
            field_update.actions)
    if isinstance(node, ast.ExprColoring):
        ll_coloring_type = trans_type(cx.type_map[node], cx)
        ll_temp = mangle_temp()
        actions = ['%s %s;' % (ll_coloring_type, ll_temp)]
        return Value(node, Expr(ll_temp, actions), cx.type_map[node])
    if isinstance(node, ast.ExprColor):
        cx = cx.increase_indent()

        coloring = trans_node(node.coloring_expr, cx)
        ll_coloring = coloring.read(cx)
        pointer = trans_node(node.pointer_expr, cx)
        ll_pointer = pointer.read(cx)
        if types.is_ispace(coloring.type.as_read().region):
            ll_pointer = Expr('ptr_t(%s)' % ll_pointer.value, ll_pointer.actions)
        color = trans_node(node.color_expr, cx)
        ll_color = color.read(cx)

        ll_coloring_type = trans_type(coloring.type.as_read(), cx)
        ll_color_type = trans_type(color.type.as_read(), cx)

        ll_temp_coloring_name = mangle_temp()
        ll_temp_color_name = mangle_temp()
        ll_temp_points_name = mangle_temp()

        actions = []
        actions.append('%s %s;' % (
                ll_coloring_type,
                ll_temp_coloring_name))

        actions.extend(
            ll_coloring.actions +
            ll_color.actions +
            ll_pointer.actions +
            ['%s = %s;' % (ll_temp_coloring_name, ll_coloring.value),
             '%s %s = %s;' % (ll_color_type, ll_temp_color_name, ll_color.value),
             'std::set<ptr_t> %s = %s[%s].points;' % (
                    ll_temp_points_name,
                    ll_temp_coloring_name,
                    ll_temp_color_name),
             '%s.insert(%s);' % (ll_temp_points_name, ll_pointer.value),
             '%s[%s].points = %s;' % (
                    ll_temp_coloring_name,
                    ll_temp_color_name,
                    ll_temp_points_name)])

        return Value(
            node,
            Expr(
                ll_temp_coloring_name,
                actions),
            cx.type_map[node])
    if isinstance(node, ast.ExprUpregion):
        return trans_node(node.expr, cx)
    if isinstance(node, ast.ExprDownregion):
        regions = trans_node(node.regions, cx)
        ll_pointer = trans_node(node.expr, cx).read(cx)

        actions = []
        actions.extend(ll_pointer.actions)

        ll_result = mangle_temp()
        actions.append('ptr_t %s = ptr_t::nil();' % ll_result)

        def safe_cast(otherwise, region):
            _actions = []
            ll_temp = mangle_temp()
            _actions.extend(
                ['ptr_t %s = runtime->safe_cast(ctx, %s, %s);' % (
                        ll_temp, ll_pointer.value, region.value),
                 'if (!%s.is_null()) {' % ll_temp,
                 Block(['%s = %s;' % (ll_result, ll_temp)]),
                 '} else {',
                 Block(otherwise),
                 '}'])
            return _actions

        actions.extend(reduce(safe_cast, reversed(regions), []))
        return Value(
            node,
            Expr(ll_result, actions),
            cx.type_map[node])
    if isinstance(node, ast.DownregionRegions):
        return [trans_node(region, cx).read(cx)
                for region in node.regions]
    if isinstance(node, ast.DownregionRegion):
        return cx.env.lookup(node.name)
    if isinstance(node, ast.ExprPack):
        cx = cx.increase_indent()

        struct_value = trans_node(node.expr, cx)
        struct_expr = struct_value.read(cx)
        ll_struct_type = trans_type(struct_value.type.as_read(), cx)

        regions = trans_node(node.regions, cx)

        result_type = cx.type_map[node]
        ll_result_type = trans_type(result_type, cx)

        ll_temp_in = mangle_temp()
        ll_temp_out = mangle_temp()

        actions = []

        actions.extend(struct_expr.actions)
        for region in regions:
            actions.extend(region.actions)

        actions.append('%s %s;' % (ll_struct_type, ll_temp_in))
        actions.append('%s %s;' % (ll_result_type, ll_temp_out))

        actions.append('%s = %s;' % (ll_temp_in, struct_expr.value))
        actions.extend(
            '%s.%s = %s.%s;' % (ll_temp_out, field_name, ll_temp_in, field_name)
            for field_name in result_type.as_read().field_map.iterkeys())
        actions.extend(
            '%s.%s = %s;' % (ll_temp_out, region_type.name, region.value)
            for region_type, region in zip(result_type.regions, regions))

        return Value(
            node,
            Expr(ll_temp_out, actions),
            result_type)
    if isinstance(node, ast.PackRegions):
        return [trans_node(region, cx)
                for region in node.regions]
    if isinstance(node, ast.PackRegion):
        return cx.env.lookup(node.name).read(cx)
    if isinstance(node, ast.ExprCall):
        return trans_call(node, cx)
    if isinstance(node, ast.Args):
        return [trans_node(arg, cx) for arg in node.args]
    if isinstance(node, ast.ExprConstBool):
        return Value(node, Expr(str(node.value).lower(), []), cx.type_map[node])
    if isinstance(node, ast.ExprConstDouble):
        return Value(node, Expr(node.value, []), cx.type_map[node])
    if isinstance(node, ast.ExprConstFloat):
        return Value(
            node,
            Expr('%sf' % node.value, []),
            cx.type_map[node])
    if isinstance(node, ast.ExprConstInt):
        return Value(
            node,
            Expr('((intptr_t)INTMAX_C(%s))' % str(node.value), []),
            cx.type_map[node])
    if isinstance(node, ast.ExprConstUInt):
        return Value(
            node,
            Expr('((uintptr_t)UINTMAX_C(%s))' % str(node.value), []),
            cx.type_map[node])
    raise Exception('Code generation failed at %s' % node)

def trans(program, type_map, constraints, region_usage):
    type_set = types.unwrap(types.union(t.component_types()
                                        for t in type_map.itervalues()
                                        if types.is_type(t)))
    cx = Context(type_set, type_map, constraints, region_usage)
    return trans_node(program, cx)
