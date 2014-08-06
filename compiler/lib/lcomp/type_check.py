#!/usr/bin/env python

# Copyright 2014 Stanford University and Los Alamos National Security, LLC
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
### Type Checker
###

# Backport of singledispatch to Python 2.x.
try:
    from functools import singledispatch
except ImportError:
    from singledispatch import singledispatch

# Work around for OrderedDict missing in Python 2.6.
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from . import ast, types
from .clang import types as ctypes

def is_eq(t): return types.is_POD(t) or types.is_pointer(t)
def returns_same_type(*ts): return ts[0]
def returns_bool(*_ignored): return types.Bool()
unary_operator_table = {
    '-': (types.is_numeric, returns_same_type),
    '!': (types.is_bool, returns_bool),
    '~': (types.is_integral, returns_same_type),
}
binary_operator_table = {
    '*':  (types.is_numeric, returns_same_type),
    '/':  (types.is_numeric, returns_same_type),
    '%':  (types.is_integral, returns_same_type),
    '+':  (types.is_numeric, returns_same_type),
    '-':  (types.is_numeric, returns_same_type),
    '>>': (types.is_integral, returns_same_type),
    '<<': (types.is_integral, returns_same_type),
    '<':  (types.is_numeric, returns_bool),
    '<=': (types.is_numeric, returns_bool),
    '>':  (types.is_numeric, returns_bool),
    '>=': (types.is_numeric, returns_bool),
    '==': (is_eq, returns_bool),
    '!=': (is_eq, returns_bool),
    '&':  (types.is_integral, returns_same_type),
    '^':  (types.is_integral, returns_same_type),
    '|':  (types.is_integral, returns_same_type),
    '&&': (types.is_bool, returns_bool),
    '||': (types.is_bool, returns_bool),
}
reduce_operator_table = {
    '*':  types.is_numeric,
    '/':  types.is_numeric,
    '%':  types.is_integral,
    '+':  types.is_numeric,
    '-':  types.is_numeric,
    '>>': types.is_integral,
    '<<': types.is_integral,
    '&':  types.is_integral,
    '^':  types.is_integral,
    '|':  types.is_integral,
}

# A method combination around wrapper a la Common Lisp.
class DispatchAround:
    def __init__(self, inner_fn, outer_fn):
        self.inner_fn = inner_fn
        self.outer_fn = outer_fn
    def __call__(self, *args, **kwargs):
        return self.outer_fn(self.inner_fn, *args, **kwargs)
    def __getattr__(self, name):
        return getattr(self.inner_fn, name)

def store_result_in_type_map(fn):
    def helper(fn, node, cx):
        node_type = fn(node, cx)
        cx.type_map[node] = node_type
        return node_type
    return DispatchAround(fn, helper)

@store_result_in_type_map
@singledispatch
def type_check_node(node, cx):
    raise Exception('Type checking failed at %s' % node)

@type_check_node.register(ast.Program)
def _(node, cx):
    cx = cx.new_global_scope()
    def_types = type_check_node(node.definitions, cx)
    return types.Program(def_types)

@type_check_node.register(ast.Definitions)
def _(node, cx):
    def_types = []
    for definition in node.definitions:
        def_types.append(type_check_node(definition, cx))
    return def_types

@type_check_node.register(ast.Import)
def _(node, cx):
    module_type = ctypes.foreign_type(node.ast, cx.opts)
    for foreign_name, foreign_type in module_type.def_types.iteritems():
        cx.insert(node, foreign_name, foreign_type)
        cx.foreign_types.append(foreign_type)
    return module_type

@type_check_node.register(ast.Struct)
def _(node, cx):
    original_cx = cx
    cx = cx.new_struct_scope()

    # Initially create empty struct type.
    struct_name = type_check_node(node.name, cx)
    param_types = [
        cx.region_forest.add(
            types.Region(param.name, types.RegionKind(None, None)))
        for param in node.params.params]
    region_types = [
        cx.region_forest.add(
            types.Region(region.name, types.RegionKind(None, None)))
        for region in node.regions.regions]
    struct_constraints = []
    empty_field_map = OrderedDict()
    struct_type = types.Struct(struct_name, param_types, region_types, struct_constraints, empty_field_map)
    def_struct_type = types.Kind(type = struct_type)

    # Insert the struct name into global scope.
    original_cx.insert(node, struct_name, def_struct_type)

    # Figure out the actual types for params and regions and
    # insert them into struct scope.
    for param, param_type in zip(node.params.params, param_types):
        cx.insert(node, param.name, param_type)
        param_type.kind = type_check_node(param.type, cx)
        if not param_type.validate_regions():
            raise types.TypeError(node, 'Region type is inconsistent with itself: %s' % param_type.pretty_kind())
    for region, region_type in zip(node.regions.regions, region_types):
        cx.insert(node, region.name, region_type)
        region_type.kind = type_check_node(region.type, cx)
        if not region_type.validate_regions():
            raise types.TypeError(node, 'Region type is inconsistent with itself: %s' % region_type.pretty_kind())

    struct_constraints = type_check_node(node.constraints, cx)
    struct_type.constraints = struct_constraints

    field_map = type_check_node(node.field_decls, cx)
    struct_type.field_map = field_map

    # Note: This simple check only works as long as mutual
    # recursion is disallowed on structs.
    for field_type in field_map.itervalues():
        if field_type == struct_type:
            raise types.TypeError(node, 'Struct may not contain itself')

    return def_struct_type

@type_check_node.register(ast.StructName)
def _(node, cx):
    return node.name

@type_check_node.register(ast.StructConstraints)
def _(node, cx):
    return [type_check_node(constraint, cx) for constraint in node.constraints]

@type_check_node.register(ast.StructConstraint)
def _(node, cx):
    lhs = type_check_node(node.lhs, cx)
    rhs = type_check_node(node.rhs, cx)
    if lhs.kind.contains_type != rhs.kind.contains_type:
        raise types.TypeError(node, 'Type mismatch in region element types for constraint: %s and %s' % (
            lhs.kind.contains_type, rhs.kind.contains_type))
    constraint = types.Constraint(node.op, lhs, rhs)
    return constraint

@type_check_node.register(ast.StructConstraintRegion)
def _(node, cx):
    region_type = cx.lookup(node, node.name)
    assert types.is_region(region_type)
    return region_type

@type_check_node.register(ast.FieldDecls)
def _(node, cx):
    return OrderedDict([
            type_check_node(field_decl, cx)
            for field_decl in node.field_decls])

@type_check_node.register(ast.FieldDecl)
def _(node, cx):
    field_kind = type_check_node(node.field_type, cx)
    return (node.name, field_kind.type)

@type_check_node.register(ast.Function)
def _(node, cx):
    original_cx = cx
    cx = cx.new_function_scope()

    fn_name = type_check_node(node.name, cx)
    param_types = type_check_node(node.params, cx)
    cx.privileges = type_check_node(node.privileges, cx)
    return_kind = type_check_node(node.return_type, cx)
    assert types.is_kind(return_kind)
    return_type = return_kind.type
    fn_type = types.Function(param_types, cx.privileges, return_type)

    # Insert function name into global scope. Second insert
    # prevents parameters from shadowing function name.
    original_cx.insert(node, fn_name, fn_type)
    cx.insert(node, fn_name, fn_type)

    type_check_node(node.block, cx.with_return_type(return_type))

    return fn_type

@type_check_node.register(ast.FunctionName)
def _(node, cx):
    return node.name

@type_check_node.register(ast.FunctionParams)
def _(node, cx):
    return [type_check_node(param, cx)
            for param in node.params]

@type_check_node.register(ast.FunctionParam)
def _(node, cx):
    if isinstance(node.declared_type, ast.TypeRegionKind):
        # Region types may be self-referential. Insert regions
        # into scope early to handle recursive types.
        region_type = types.Region(node.name, types.RegionKind(None, None))
        cx.region_forest.add(region_type)
        cx.insert(node, node.name, region_type)

        region_kind = type_check_node(node.declared_type, cx)
        region_type.kind = region_kind
        if not region_type.validate_regions():
            raise types.TypeError(node, 'Region type is inconsistent with itself: %s' % region_type.pretty_kind())
        return region_type
    if isinstance(node.declared_type, ast.TypeArrayKind):
        # Region types may be self-referential. Insert regions
        # into scope early to handle recursive types.
        region_type = types.Region(node.name, types.RegionKind(None, None))
        cx.region_forest.add(region_type)
        cx.insert(node, node.name, region_type)

        region_kind = type_check_node(node.declared_type, cx)
        region_type.kind = region_kind
        return region_type
    if isinstance(node.declared_type, ast.TypeIspaceKind):
        ispace_kind = type_check_node(node.declared_type, cx)
        ispace_type = types.Ispace(node.name, ispace_kind)
        cx.insert(node, node.name, ispace_type)
        return ispace_type

    # Handle non-region types:
    declared_kind = type_check_node(node.declared_type, cx)
    assert types.is_kind(declared_kind)
    declared_type = declared_kind.type

    if types.is_void(declared_type):
        raise types.TypeError(node, 'Task parameters are not allowed to be void')
    if not types.is_concrete(declared_type):
        raise types.TypeError(node, 'Task parameters are not allowed to contain wildcards')

    assert types.allows_var_binding(declared_type)
    reference_type = types.StackReference(declared_type)
    cx.insert(node, node.name, reference_type)

    return declared_type

@type_check_node.register(ast.FunctionReturnType)
def _(node, cx):
    return type_check_node(node.declared_type, cx)

@type_check_node.register(ast.FunctionPrivileges)
def _(node, cx):
    return cx.privileges | set(
        privilege
        for privilege_node in node.privileges
        for privilege in type_check_node(privilege_node, cx))

@type_check_node.register(ast.FunctionPrivilege)
def _(node, cx):
    return type_check_node(node.privilege, cx)

@type_check_node.register(ast.TypeVoid)
def _(node, cx):
    return types.Kind(types.Void())

@type_check_node.register(ast.TypeBool)
def _(node, cx):
    return types.Kind(types.Bool())

@type_check_node.register(ast.TypeDouble)
def _(node, cx):
    return types.Kind(types.Double())

@type_check_node.register(ast.TypeFloat)
def _(node, cx):
    return types.Kind(types.Float())

@type_check_node.register(ast.TypeInt)
def _(node, cx):
    return types.Kind(types.Int())

@type_check_node.register(ast.TypeUInt)
def _(node, cx):
    return types.Kind(types.UInt())

@type_check_node.register(ast.TypeInt8)
def _(node, cx):
    return types.Kind(types.Int8())

@type_check_node.register(ast.TypeInt16)
def _(node, cx):
    return types.Kind(types.Int16())

@type_check_node.register(ast.TypeInt32)
def _(node, cx):
    return types.Kind(types.Int32())

@type_check_node.register(ast.TypeInt64)
def _(node, cx):
    return types.Kind(types.Int64())

@type_check_node.register(ast.TypeUInt8)
def _(node, cx):
    return types.Kind(types.UInt8())

@type_check_node.register(ast.TypeUInt16)
def _(node, cx):
    return types.Kind(types.UInt16())

@type_check_node.register(ast.TypeUInt32)
def _(node, cx):
    return types.Kind(types.UInt32())

@type_check_node.register(ast.TypeUInt64)
def _(node, cx):
    return types.Kind(types.UInt64())

@type_check_node.register(ast.TypeColoring)
def _(node, cx):
    region = type_check_node(node.region, cx)
    if not (types.is_region(region) or types.is_ispace(region)):
        raise types.TypeError(node, 'Type mismatch in type %s: expected %s but got %s' % (
                'coloring', 'a region or ispace', region))
    return types.Kind(types.Coloring(region))

@type_check_node.register(ast.TypeColoringRegion)
def _(node, cx):
    return cx.lookup(node, node.name)

@type_check_node.register(ast.TypeID)
def _(node, cx):
    kind = cx.lookup(node, node.name)
    args = type_check_node(node.args, cx)
    if not types.is_kind(kind):
        raise types.TypeError(node, 'Type mismatch in type %s: expected a type but got %s' % (
                node.name, kind))

    if len(args) != len(kind.type.params):
        raise types.TypeError(node, 'Incorrect number of arguments for struct %s: expected %s but got %s' % (
                node.name, len(kind.type.params), len(args)))

    region_map = dict([
            (old_region, new_region)
            for old_region, new_region in zip(kind.type.params, args)])

    for param, arg in zip(kind.type.params, args):
        assert types.is_region(param)
        if types.is_region(arg):
            if param.kind.contains_type is not None and arg.kind.contains_type is not None:
                param_kind = param.kind.substitute_regions(region_map)
                arg_kind = arg.kind
                if param_kind != arg_kind:
                    raise types.TypeError(node, 'Type mismatch in type parameter to %s: expected %s but got %s' % (
                        node.name, param_kind, arg_kind))
        elif types.is_region_wild(arg):
            pass
        else:
            assert False

    return kind.instantiate_params(region_map)

@type_check_node.register(ast.TypeArgs)
def _(node, cx):
    return [type_check_node(arg, cx) for arg in node.args]

@type_check_node.register(ast.TypeArg)
def _(node, cx):
    arg = cx.lookup(node, node.name)
    if not types.is_region(arg):
        raise types.TypeError(node, 'Type mismatch in type %s: expected a region but got %s' % (
                node.name, arg))
    return arg

@type_check_node.register(ast.TypeArgWild)
def _(node, cx):
    return types.RegionWild()

@type_check_node.register(ast.TypePointer)
def _(node, cx):
    points_to_kind = type_check_node(node.points_to_type, cx)
    regions = type_check_node(node.regions, cx)
    assert types.is_kind(points_to_kind)
    points_to_type = points_to_kind.type

    for region in regions:
        if types.is_region(region):
            contains_type = region.kind.contains_type
            if contains_type is not None and contains_type != points_to_type:
                raise types.TypeError(node, 'Type mismatch in pointer type: expected %s but got %s' % (
                    contains_type, points_to_type))
        elif types.is_region_wild(region):
            pass
        else:
            if not types.is_kind(region):
                raise types.TypeError(node, 'Type mismatch in pointer type: expected a region but got %s' % (
                        region))
            raise types.TypeError(node, 'Type mismatch in pointer type: expected a region but got %s' % (
                    region.type))
    return types.Kind(types.Pointer(points_to_type, regions))

@type_check_node.register(ast.TypePointerRegions)
def _(node, cx):
    return [type_check_node(region, cx)
            for region in node.regions]

@type_check_node.register(ast.TypeRegion)
def _(node, cx):
    region_type = cx.lookup(node, node.name)
    return region_type

@type_check_node.register(ast.TypeRegionWild)
def _(node, cx):
    return types.RegionWild()

@type_check_node.register(ast.TypeRegionKind)
def _(node, cx):
    contains_type = None
    if node.contains_type is not None:
        contains_type = type_check_node(node.contains_type, cx).type
    return types.RegionKind(None, contains_type)

@type_check_node.register(ast.TypeArrayKind)
def _(node, cx):
    ispace = type_check_node(node.ispace, cx)
    contains_type = type_check_node(node.contains_type, cx).type
    return types.RegionKind(ispace, contains_type)

@type_check_node.register(ast.TypeIspace)
def _(node, cx):
    ispace_type = cx.lookup(node, node.name)
    return ispace_type

@type_check_node.register(ast.TypeIspaceKind)
def _(node, cx):
    index_type = type_check_node(node.index_type, cx).type
    return types.IspaceKind(index_type)

@type_check_node.register(ast.Privilege)
def _(node, cx):
    if node.privilege == 'reads':
        privilege = types.Privilege.READ
    elif node.privilege == 'writes':
        privilege = types.Privilege.WRITE
    elif node.privilege == 'reduces':
        privilege = types.Privilege.REDUCE
    else:
        assert False
    regions = type_check_node(node.regions, cx)
    return [
        types.Privilege(node, privilege, node.op, region, field_path)
        for region, field_path in regions]

@type_check_node.register(ast.PrivilegeRegions)
def _(node, cx):
    return [
        region
        for region_node in node.regions
        for region in type_check_node(region_node, cx)]

@type_check_node.register(ast.PrivilegeRegion)
def _(node, cx):
    region = cx.lookup(node, node.name)
    field_paths = type_check_node(node.fields, cx)
    return [(region, field_path) for field_path in field_paths]

@type_check_node.register(ast.PrivilegeRegionFields)
def _(node, cx):
    if len(node.fields) == 0:
        return [()]
    return [
        field_path
        for field_node in node.fields
        for field_path in type_check_node(field_node, cx)]

@type_check_node.register(ast.PrivilegeRegionField)
def _(node, cx):
    prefix = (node.name,)
    field_paths = type_check_node(node.fields, cx)
    return [prefix + field_path for field_path in field_paths]

@type_check_node.register(ast.Block)
def _(node, cx):
    cx = cx.new_block_scope()
    for expr in node.block:
        type_check_node(expr, cx)
    return types.Void()

@type_check_node.register(ast.StatementAssert)
def _(node, cx):
    expr_type = type_check_node(node.expr, cx).check_read(node.expr, cx)
    if not types.is_bool(expr_type):
        raise types.TypeError(node, 'Type mismatch in assert statement: expected %s but got %s' % (
                types.Bool(), expr_type))
    return types.Void()

@type_check_node.register(ast.StatementExpr)
def _(node, cx):
    type_check_node(node.expr, cx).check_read(node.expr, cx)
    return types.Void()

@type_check_node.register(ast.StatementIf)
def _(node, cx):
    condition_type = type_check_node(node.condition, cx).check_read(node.condition, cx)
    type_check_node(node.then_block, cx)
    if node.else_block is not None:
        type_check_node(node.else_block, cx)
    if not types.is_bool(condition_type):
        raise types.TypeError(node, 'If condition expression is not type bool')
    return types.Void()

@type_check_node.register(ast.StatementFor)
def _(node, cx):
    cx = cx.new_block_scope()
    index_types = type_check_node(node.indices, cx)
    region_types = type_check_node(node.regions, cx)
    if len(index_types) != len(region_types):
        raise types.TypeError(node, 'Incorrect number of indices in for statement: expected %s but got %s' % (
                len(region_types), len(index_types)))

    # Two forms of iteration are supported, over a single index
    # space, or over any number of regions. In the case where
    # multiple regions are being iterated, it is assumed the
    # regions have the same index space. At the moment this has to
    # be checked dynamically to be sound.
    if len(region_types) == 1 and types.is_ispace(region_types[0]):
        index_node = node.indices.indices[0]
        index_type = index_types[0]
        ispace_type = region_types[0]
        # We can infer the index type if unspecified.
        if index_type is None:
            index_type = ispace_type.kind.index_type
        if index_type != ispace_type.kind.index_type:
            raise types.TypeError(node, 'Type mismatch in for statement: expected %s but got %s' % (
                    index_type, ispace_type.kind.index_type))
        # Patch environment and type map to know about the inferred index type.
        cx.insert(node, index_node.name, index_type)
        cx.type_map[index_node] = index_type
    else:
        for index_node, index_type, region_type, index \
                in zip(node.indices.indices, index_types, region_types, xrange(len(index_types))):
            if not types.is_region(region_type):
                raise types.TypeError(node, 'Type mismatch on index %s of for statement: expected a region but got %s' % (
                        index, region_type))

            # We can infer the index type as long as the region is explicitly typed.
            if index_type is None:
                if region_type.kind.contains_type is None:
                    raise types.TypeError(node, 'Unable to infer type of index %s of for statement: region %s has no element type' % (
                            index, region_type))
                index_type = types.Pointer(region_type.kind.contains_type, [region_type])

            if not types.is_pointer(index_type):
                raise types.TypeError(node, 'Type mismatch on index %s of for statement: expected a pointer but got %s' % (
                    index, index_type))
            if len(index_type.regions) != 1 or index_type.regions[0] != region_type:
                raise types.TypeError(node, 'Type mismatch on index %s of for statement: expected %s but got %s' % (
                    index, index_type,
                    types.Pointer(region_type.kind.contains_type, [region_type])))
            # Patch environment and type map to know about the inferred index type.
            cx.insert(node, index_node.name, index_type)
            cx.type_map[index_node] = index_type

    type_check_node(node.block, cx)
    return types.Void()

@type_check_node.register(ast.ForIndices)
def _(node, cx):
    return [type_check_node(index, cx)
            for index in node.indices]

@type_check_node.register(ast.ForIndex)
def _(node, cx):
    if node.type is not None:
        declared_kind = type_check_node(node.type, cx)
        assert types.is_kind(declared_kind)
        return declared_kind.type
    return None

@type_check_node.register(ast.ForRegions)
def _(node, cx):
    return [type_check_node(region, cx)
            for region in node.regions]

@type_check_node.register(ast.ForRegion)
def _(node, cx):
    region_type = cx.lookup(node, node.name)
    return region_type

@type_check_node.register(ast.StatementLet)
def _(node, cx):
    declared_type = None
    if node.type is not None:
        declared_kind = type_check_node(node.type, cx)
        if types.is_region_kind(declared_kind):
            declared_type = types.Region(node.name, declared_kind)
            cx.region_forest.add(declared_type)
        if types.is_kind(declared_kind):
            declared_type = declared_kind.type
        else:
            assert False

    expr_type = type_check_node(node.expr, cx).check_read(node.expr, cx)

    # Hack: Rather full type inference, which gets ugly fast, just
    # implement "auto-style" inference by using the expression
    # type if no type declaration is provided.
    if declared_type is None:
        if types.is_region(expr_type):
            declared_type = types.Region(node.name, expr_type.kind)
            cx.region_forest.add(declared_type)
        else:
            declared_type = expr_type

    if not types.is_concrete(declared_type):
        raise types.TypeError(node, 'Let bound expressions are not allowed to contain wildcards')
    if types.is_void(declared_type):
        raise types.TypeError(node, 'Let bound expressions are not allowed to be void')
    if types.is_region(expr_type) and types.is_region(declared_type):
        if expr_type.kind != declared_type.kind:
            raise types.TypeError(node, 'Let bound expression of type %s does not match declared type %s' % (
                    expr_type.kind, declared_type.kind))
    else:
        if expr_type != declared_type:
            raise types.TypeError(node, 'Let bound expression of type %s does not match declared type %s' % (
                    expr_type, declared_type))
    cx.insert(node, node.name, declared_type, shadow = True)
    if types.is_region(expr_type):
        cx.region_forest.union(declared_type, expr_type)
        cx.constraints.add(
            types.Constraint(lhs = expr_type, op = types.Constraint.SUBREGION, rhs = declared_type))
        cx.constraints.add(
            types.Constraint(lhs = declared_type, op = types.Constraint.SUBREGION, rhs = expr_type))

    return declared_type

@type_check_node.register(ast.StatementLetRegion)
def _(node, cx):
    region_type = types.Region(node.name, types.RegionKind(None, None))
    cx.region_forest.add(region_type)
    # Insert region name into scope so that element type can refer to it.
    cx.insert(node, node.name, region_type)

    declared_region_kind = None
    if node.region_kind is not None:
        declared_region_kind = type_check_node(node.region_kind, cx)
    element_kind = type_check_node(node.element_type, cx)
    size_type = type_check_node(node.size_expr, cx).check_read(node.size_expr, cx)
    assert types.is_kind(element_kind) and not types.is_void(element_kind.type)

    if not types.is_int(size_type):
        raise types.TypeError(node, 'Type mismatch in region: expected %s but got %s' % (
                types.Int(), size_type))

    # Now patch region type so that it refers to the contained type.
    region_kind = types.RegionKind(None, element_kind.type)
    region_type.kind = region_kind
    if not region_type.validate_regions():
        raise types.TypeError(node, 'Region type is inconsistent with itself: %s' % region_type.pretty_kind())

    if declared_region_kind is None:
        declared_region_kind = region_kind

    if declared_region_kind != region_kind:
        raise types.TypeError(node, 'Let bound expression of type %s does not match declared type %s' % (
                region_kind, declared_region_kind))

    cx.privileges.add(types.Privilege(node, types.Privilege.READ, None, region_type, ()))
    cx.privileges.add(types.Privilege(node, types.Privilege.WRITE, None, region_type, ()))
    return region_type

@type_check_node.register(ast.StatementLetArray)
def _(node, cx):
    ispace_type = type_check_node(node.ispace_type, cx)
    region_type = types.Region(node.name, types.RegionKind(ispace_type, None))
    cx.region_forest.add(region_type)
    # insert region name into scope so that element type can refer to it
    cx.insert(node, node.name, region_type)

    declared_region_kind = None
    if node.region_kind is not None:
        declared_region_kind = type_check_node(node.region_kind, cx)
    element_kind = type_check_node(node.element_type, cx)
    assert types.is_kind(element_kind) and not types.is_void(element_kind.type)

    # now patch region type so that it refers to the contained type
    region_kind = types.RegionKind(ispace_type, element_kind.type)
    region_type.kind = region_kind

    if declared_region_kind is None:
        declared_region_kind = region_kind

    if declared_region_kind != region_kind:
        raise types.TypeError(node, 'Let bound expression of type %s does not match declared type %s' % (
                region_kind, declared_region_kind))

    cx.privileges.add(types.Privilege(node, types.Privilege.READ, None, region_type, ()))
    cx.privileges.add(types.Privilege(node, types.Privilege.WRITE, None, region_type, ()))
    return region_type

@type_check_node.register(ast.StatementLetIspace)
def _(node, cx):
    declared_ispace_kind = None
    if node.ispace_kind is not None:
        declared_ispace_kind = type_check_node(node.ispace_kind, cx)

    index_kind = type_check_node(node.index_type, cx)
    size_type = type_check_node(node.size_expr, cx).check_read(node.size_expr, cx)
    assert types.is_kind(index_kind) and types.is_int(index_kind.type)

    if not types.is_int(size_type):
        raise types.TypeError(node, 'Type mismatch in ispace: expected %s but got %s' % (
                types.Int(), size_type))

    ispace_kind = types.IspaceKind(index_kind.type)

    if declared_ispace_kind is None:
        declared_ispace_kind = ispace_kind

    if declared_ispace_kind != ispace_kind:
        raise types.TypeError(node, 'Let bound expression of type %s does not match declared type %s' % (
                ispace_kind, declared_ispace_kind))

    ispace_type = types.Ispace(node.name, ispace_kind)
    cx.insert(node, node.name, ispace_type)

    return types.Void()

@type_check_node.register(ast.StatementLetPartition)
def _(node, cx):
    region_type = type_check_node(node.region_type, cx).check_read(node.region_type, cx)
    mode = type_check_node(node.mode, cx)
    coloring_type = type_check_node(node.coloring_expr, cx).check_read(node.coloring_expr, cx)

    if not (types.is_region(region_type) or types.is_ispace(region_type)):
        raise types.TypeError(node, 'Type mismatch in partition: expected a region or ispace but got %s' % (
                region_type))
    expected_coloring_type = types.Coloring(region_type)
    if coloring_type != expected_coloring_type:
        raise types.TypeError(node, 'Type mismatch in partition: expected %s but got %s' % (
                expected_coloring_type, coloring_type))

    partition_kind = types.PartitionKind(region_type, mode)
    partition_type = types.Partition(node.name, partition_kind)
    cx.insert(node, node.name, partition_type)

    return partition_type

@type_check_node.register(ast.PartitionMode)
def _(node, cx):
    if node.mode == 'disjoint':
        return types.Partition.DISJOINT
    elif node.mode == 'aliased':
        return types.Partition.ALIASED
    assert False

@type_check_node.register(ast.StatementReturn)
def _(node, cx):
    expr_type = type_check_node(node.expr, cx).check_read(node.expr, cx)
    if expr_type != cx.return_type:
        raise types.TypeError(node, 'Returned expression of type %s does not match declared return type %s' % (
                expr_type, cx.return_type))
    return types.Void()

@type_check_node.register(ast.StatementUnpack)
def _(node, cx):
    expr_type = type_check_node(node.expr, cx).check_read(node.expr, cx)
    declared_kind = type_check_node(node.type, cx)
    assert types.is_kind(declared_kind)
    declared_type = declared_kind.type

    if not types.is_struct(expr_type):
        raise types.TypeError(node, 'Type mismatch in unpack: expected %s but got %s' % (
                'a struct', expr_type))

    region_types = type_check_node(node.regions, cx)
    for region, region_type in zip(node.regions.regions, region_types):
        cx.insert(node, region.name, region_type) # FIXME: handle shadowing
    region_map = dict(zip(declared_type.regions, region_types))
    actual_type = declared_type.instantiate_regions(region_map)
    # Patch regions so that they contain the correct type.
    for region_type, declared_region_type in zip(region_types, declared_type.regions):
        region_type.kind = declared_region_type.kind.substitute_regions(region_map)

    if expr_type != declared_type:
        raise types.TypeError(node, 'Type mismatch in unpack: expected %s but got %s' % (
                declared_type, expr_type))
    cx.insert(node, node.name, actual_type) # FIXME: handle shadowing
    cx.constraints.update(actual_type.constraints)

    return region_types

@type_check_node.register(ast.UnpackRegions)
def _(node, cx):
    return [type_check_node(region, cx) for region in node.regions]

@type_check_node.register(ast.UnpackRegion)
def _(node, cx):
    # Create regions with empty region_types initially, patch later.
    region_type = types.Region(node.name, types.RegionKind(None, None))
    cx.region_forest.add(region_type)
    return region_type

@type_check_node.register(ast.StatementVar)
def _(node, cx):
    declared_type = None
    if node.type is not None:
        declared_kind = type_check_node(node.type, cx)
        assert types.is_kind(declared_kind)
        declared_type = declared_kind.type

    expr_type = type_check_node(node.expr, cx).check_read(node.expr, cx)

    # Hack: Rather full type inference, which gets ugly fast, just
    # implement "auto-style" inference by using the expression
    # type if no type declaration is provided.
    if declared_type is None:
        declared_type = expr_type

    if not types.is_concrete(declared_type):
        raise types.TypeError(node, 'Variables are not allowed to contain wildcards')
    if expr_type != declared_type:
        raise types.TypeError(node, 'Variable initializer of type %s does not match declared type %s' % (
                expr_type, declared_type))
    assert types.allows_var_binding(declared_type)

    reference_type = types.StackReference(declared_type)
    cx.insert(node, node.name, reference_type, shadow = True)
    return types.Void()

@type_check_node.register(ast.StatementWhile)
def _(node, cx):
    condition_type = type_check_node(node.condition, cx).check_read(node.condition, cx)
    type_check_node(node.block, cx)
    if not types.is_bool(condition_type):
        raise types.TypeError(node, 'While condition expression is not type bool')
    return types.Void()

@type_check_node.register(ast.ExprID)
def _(node, cx):
    id_type = cx.lookup(node, node.name)
    return id_type

@type_check_node.register(ast.ExprAssignment)
def _(node, cx):
    lval_type = type_check_node(node.lval, cx).check_write(node.lval, cx)
    rval_type = type_check_node(node.rval, cx).check_read(node.rval, cx)
    if lval_type != rval_type:
        raise types.TypeError(node, 'Type mismatch in assignment: %s and %s' % (
                lval_type, rval_type))
    return rval_type

@type_check_node.register(ast.ExprUnaryOp)
def _(node, cx):
    arg_type = type_check_node(node.arg, cx).check_read(node.arg, cx)
    if not unary_operator_table[node.op][0](arg_type):
        raise types.TypeError(node, 'Type mismatch in operand to unary operator: %s' % (
                arg_type))
    return unary_operator_table[node.op][1](arg_type)

@type_check_node.register(ast.ExprBinaryOp)
def _(node, cx):
    lhs_type = type_check_node(node.lhs, cx).check_read(node.lhs, cx)
    rhs_type = type_check_node(node.rhs, cx).check_read(node.rhs, cx)
    if lhs_type != rhs_type:
        raise types.TypeError(node, 'Type mismatch in operands to binary operator: %s and %s' % (
                lhs_type, rhs_type))
    if not binary_operator_table[node.op][0](lhs_type):
        raise types.TypeError(node, 'Type mismatch in operand to binary operator: %s' % (
                lhs_type))
    if not binary_operator_table[node.op][0](rhs_type):
        raise types.TypeError(node, 'Type mismatch in operand to binary operator: %s' % (
                rhs_type))
    return binary_operator_table[node.op][1](lhs_type, rhs_type)

@type_check_node.register(ast.ExprReduceOp)
def _(node, cx):
    lhs_type = type_check_node(node.lhs, cx).check_reduce(node.lhs, node.op, cx)
    rhs_type = type_check_node(node.rhs, cx).check_read(node.rhs, cx)
    if lhs_type != rhs_type:
        raise types.TypeError(node, 'Type mismatch in operands to binary operator: %s and %s' % (
                lhs_type, rhs_type))
    if not reduce_operator_table[node.op](lhs_type):
        raise types.TypeError(node, 'Type mismatch in operand to binary operator: %s' % (
                lhs_type))
    if not reduce_operator_table[node.op](rhs_type):
        raise types.TypeError(node, 'Type mismatch in operand to binary operator: %s' % (
                rhs_type))
    return types.Void()

@type_check_node.register(ast.ExprCast)
def _(node, cx):
    cast_to_kind = type_check_node(node.cast_to_type, cx)
    assert types.is_kind(cast_to_kind) and types.is_numeric(cast_to_kind.type)
    expr_type = type_check_node(node.expr, cx).check_read(node.expr, cx)
    if not types.is_numeric(expr_type):
        raise types.TypeError(node, 'Type mismatch in cast: expected a number but got %s' % (
                expr_type))
    return cast_to_kind.type

@type_check_node.register(ast.ExprNull)
def _(node, cx):
    pointer_kind = type_check_node(node.pointer_type, cx)
    assert types.is_kind(pointer_kind) and types.is_pointer(pointer_kind.type)
    return pointer_kind.type

@type_check_node.register(ast.ExprIsnull)
def _(node, cx):
    pointer_type = type_check_node(node.pointer_expr, cx).check_read(node.pointer_expr, cx)
    if not types.is_pointer(pointer_type):
        raise types.TypeError(node, 'Type mismatch for argument %s in call to task %s: expected %s but got %s' % (
                0, 'isnull', 'a pointer', pointer_type))
    return types.Bool()

@type_check_node.register(ast.ExprNew)
def _(node, cx):
    pointer_kind = type_check_node(node.pointer_type, cx)
    assert types.is_kind(pointer_kind) and types.is_pointer(pointer_kind.type)
    pointer_type = pointer_kind.type

    if len(pointer_type.regions) != 1:
        raise types.TypeError(node, 'Type mismatch in new: cannot allocate pointer with more than one region %s' % (
            pointer_type))
    region_type = pointer_type.regions[0]

    if region_type.kind.ispace is not None:
        raise types.TypeError(node, 'Type mismatch in new: cannot allocate into array %s' %
                        region_type)
    return pointer_type

@type_check_node.register(ast.ExprRead)
def _(node, cx):
    pointer_type = type_check_node(node.pointer_expr, cx).check_read(node.pointer_expr, cx)

    if not types.is_pointer(pointer_type):
        raise types.TypeError(node, 'Type mismatch in read: expected a pointer but got %s' % (
                pointer_type))

    privileges_requested = [
        types.Privilege(node, types.Privilege.READ, None, region, ())
        for region in pointer_type.regions]
    success, failed_request = types.check_privileges(privileges_requested, cx)
    if not success:
        raise types.TypeError(node, 'Invalid privilege %s requested in read' % failed_request)
    value_type = pointer_type.points_to_type
    return value_type

@type_check_node.register(ast.ExprWrite)
def _(node, cx):
    pointer_type = type_check_node(node.pointer_expr, cx).check_read(node.pointer_expr, cx)
    value_type = type_check_node(node.value_expr, cx).check_read(node.value_expr, cx)

    if not types.is_pointer(pointer_type):
        raise types.TypeError(node, 'Type mismatch in write: expected a pointer but got %s' % (
                pointer_type))
    if pointer_type.points_to_type != value_type:
        raise types.TypeError(node, 'Type mismatch in write: expected %s but got %s' % (
                value_type, pointer_type.points_to_type))

    privileges_requested = [
        types.Privilege(node, types.Privilege.WRITE, None, region, ())
        for region in pointer_type.regions]
    success, failed_request = types.check_privileges(privileges_requested, cx)
    if not success:
        raise types.TypeError(node, 'Invalid privilege %s requested in write' % failed_request)
    return types.Void()

@type_check_node.register(ast.ExprReduce)
def _(node, cx):
    pointer_type = type_check_node(node.pointer_expr, cx).check_read(node.pointer_expr, cx)
    value_type = type_check_node(node.value_expr, cx).check_read(node.value_expr, cx)

    if not types.is_pointer(pointer_type):
        raise types.TypeError(node, 'Type mismatch in reduce: expected a pointer but got %s' % (
                pointer_type))
    if pointer_type.points_to_type != value_type:
        raise types.TypeError(node, 'Type mismatch in reduce: %s and %s' % (
                pointer_type.points_to_type, value_type))
    if not reduce_operator_table[node.op](pointer_type.points_to_type):
        raise types.TypeError(node, 'Type mismatch in reduce: %s' % (
                pointer_type.points_to_type))
    if not reduce_operator_table[node.op](value_type):
        raise types.TypeError(node, 'Type mismatch in reduce: %s' % (
                value_type))

    privileges_requested = [
        types.Privilege(node, types.Privilege.REDUCE, node.op, region, ())
        for region in pointer_type.regions]
    success, failed_request = types.check_privileges(privileges_requested, cx)
    if not success:
        raise types.TypeError(node, 'Invalid privilege %s requested in reduce' % failed_request)
    return types.Void()

@type_check_node.register(ast.ExprDereference)
def _(node, cx):
    pointer_type = type_check_node(node.pointer_expr, cx).check_read(node.pointer_expr, cx)
    if not types.is_pointer(pointer_type):
        raise types.TypeError(node, 'Type mismatch in pointer dereference: expected a pointer but got %s' % (
                pointer_type))
    reference_type = types.Reference(
        refers_to_type = pointer_type.points_to_type,
        regions = pointer_type.regions)
    return reference_type

@type_check_node.register(ast.ExprArrayAccess)
def _(node, cx):
    array_type = type_check_node(node.array_expr, cx).check_read(node.array_expr, cx)
    index_type = type_check_node(node.index_expr, cx).check_read(node.index_expr, cx)

    # Handle partitions:
    if types.is_partition(array_type):
        if not types.is_int(index_type):
            raise types.TypeError(node, 'Type mismatch in index for partition access: expected %s but got %s' % (
                    types.Int(),
                    index_type))

        # Check whether the index expression is a compile-time
        # constant value. Add disjointness constraints for the
        # subregion if and only if the the index is constant.
        if isinstance(node.index_expr, ast.ExprConstInt):
            index = node.index_expr.value
            subregion_type = array_type.static_subregion(index, cx)
        else:
            index_expr = node.index_expr
            subregion_type = array_type.dynamic_subregion(index_expr, cx)

        return subregion_type

    # Handle array slicing:
    if types.is_region(array_type) and types.is_ispace(index_type):
        if array_type.kind.ispace is None:
            raise types.TypeError(node, 'Type mismatch in array slice: expected an array but got %s' % (
                    array_type))
        # Check constraints for the index space to make sure it is
        # a subset of the index space of the array.
        success, failed_request = types.check_constraints(
            [types.Constraint(lhs = index_type, op = types.Constraint.SUBREGION, rhs = array_type.kind.ispace)],
            cx.constraints)
        if not success:
            raise types.TypeError(node, 'Invalid constraint %s requested in array slice' % (
                '%s <= %s' % (index_type, array_type.kind.ispace)))

        array_kind = types.RegionKind(index_type, array_type.kind.contains_type)
        subarray_type = types.Region('%s[%s]' % (array_type, index_type), array_kind)
        cx.region_forest.union(subarray_type, array_type)

        return subarray_type

    # Handle arrays:
    if not types.is_region(array_type):
        raise types.TypeError(node, 'Type mismatch in array access: expected an array but got %s' % (
                array_type))
    ispace = array_type.kind.ispace
    if ispace is None:
        raise types.TypeError(node, 'Type mismatch in array access: expected an array but got %s' % (
                array_type.kind))
    if ispace.kind.index_type != index_type:
        raise types.TypeError(node, 'Type mismatch in index for array access: expected %s but got %s' % (
                ispace.kind.index_type,
                index_type))
    reference_type = types.Reference(
        refers_to_type = array_type.kind.contains_type,
        regions = [array_type])
    return reference_type

@type_check_node.register(ast.ExprFieldAccess)
def _(node, cx):
    wrapper_type = type_check_node(node.struct_expr, cx)

    struct_type = wrapper_type.as_read()
    if not types.is_struct(struct_type):
        raise types.TypeError(node, 'Type mismatch in struct field access: expected a struct but got %s' % (
            struct_type))
    if node.field_name not in struct_type.field_map:
        raise types.TypeError(node, 'Struct %s has no field named %s' % (
            struct_type, node.field_name))

    return wrapper_type.get_field(node.field_name)

@type_check_node.register(ast.ExprFieldDereference)
def _(node, cx):
    pointer_type = type_check_node(node.pointer_expr, cx).check_read(node.pointer_expr, cx)

    if not types.is_pointer(pointer_type):
        raise types.TypeError(node, 'Type mismatch in struct field dereference: expected a pointer to a struct but got %s' % (
            pointer_type))
    if not types.is_struct(pointer_type.points_to_type):
        raise types.TypeError(node, 'Type mismatch in struct field dereference: expected a pointer to a struct but got %s' % (
            pointer_type))
    if node.field_name not in pointer_type.points_to_type.field_map:
        raise types.TypeError(node, 'Struct %s has no field named %s' % (
            pointer_type.points_to_type, node.field_name))

    return types.Reference(pointer_type.points_to_type, pointer_type.regions).get_field(node.field_name)

@type_check_node.register(ast.ExprFieldValues)
def _(node, cx):
    field_values = type_check_node(node.field_values, cx)

    field_map = OrderedDict()
    for field_name, value_type in field_values:
        field_map[field_name] = value_type
    struct_type = types.Struct(None, [], [], set(), field_map)

    return struct_type

@type_check_node.register(ast.FieldValues)
def _(node, cx):
    return [type_check_node(field_value, cx)
            for field_value in node.field_values]

@type_check_node.register(ast.FieldValue)
def _(node, cx):
    return (
        node.field_name,
        type_check_node(node.value_expr, cx).check_read(node.value_expr, cx))

@type_check_node.register(ast.ExprFieldUpdates)
def _(node, cx):
    struct_type = type_check_node(node.struct_expr, cx).check_read(node.struct_expr, cx)
    field_updates = type_check_node(node.field_updates, cx)
    if not types.is_struct(struct_type):
        raise types.TypeError(node, 'Type mismatch in struct field updates: expected a struct but got %s' % (
            struct_type))

    all_fields_match = True
    for field_name, update_type in field_updates:
        assert field_name in struct_type.field_map
        if update_type != struct_type.field_map[field_name]:
            all_fields_match = False

    if all_fields_match:
        new_struct_type = struct_type
    else:
        new_field_map = struct_type.field_map.copy()
        for field_name, update_type in field_updates:
            new_field_map[field_name] = update_type
        new_struct_type = types.Struct(None, [], [], set(), new_field_map)

    return new_struct_type

@type_check_node.register(ast.FieldUpdates)
def _(node, cx):
    return [type_check_node(field_update, cx)
            for field_update in node.field_updates]

@type_check_node.register(ast.FieldUpdate)
def _(node, cx):
    return (
        node.field_name,
        type_check_node(node.update_expr, cx).check_read(node.update_expr, cx))

@type_check_node.register(ast.ExprColoring)
def _(node, cx):
    region_type = type_check_node(node.region, cx).check_read(node.region, cx)
    if not (types.is_region(region_type) or types.is_ispace(region_type)):
        raise types.TypeError(node, 'Type mismatch in coloring: expected a region or ispace but got %s' % (
            region_type))
    return types.Coloring(region_type)

@type_check_node.register(ast.ColoringRegion)
def _(node, cx):
    return cx.lookup(node, node.name)

@type_check_node.register(ast.ExprColor)
def _(node, cx):
    coloring_type = type_check_node(node.coloring_expr, cx).check_read(node.coloring_expr, cx)
    pointer_type = type_check_node(node.pointer_expr, cx).check_read(node.pointer_expr, cx)
    color_type = type_check_node(node.color_expr, cx).check_read(node.color_expr, cx)

    if not types.is_coloring(coloring_type):
        raise types.TypeError(node, 'Type mismatch for argument %s in call to task %s: expected %s but got %s' % (
            0, 'color', 'a coloring', coloring_type))
    if types.is_region(coloring_type.region):
        expected_pointer_type = types.Pointer(
            coloring_type.region.kind.contains_type,
            [coloring_type.region])
    elif types.is_ispace(coloring_type.region):
        expected_pointer_type = coloring_type.region.kind.index_type
    else:
        assert False
    if pointer_type != expected_pointer_type:
        raise types.TypeError(node, 'Type mismatch for argument %s in call to task %s: expected %s but got %s' % (
            1, 'color', expected_pointer_type, pointer_type))
    if not types.is_int(color_type):
        raise types.TypeError(node, 'Type mismatch for argument %s in call to task %s: expected %s but got %s' % (
            2, 'color', types.Int(), color_type))
    return coloring_type

@type_check_node.register(ast.ExprUpregion)
def _(node, cx):
    region_types = type_check_node(node.regions, cx)
    expr_type = type_check_node(node.expr, cx).check_read(node.expr, cx)

    for index, region_type in zip(xrange(len(region_types)), region_types):
        if not types.is_region(region_type):
            raise types.TypeError(node, 'Type mismatch for type argument %s in call to task %s: expected %s but got %s' % (
                index, 'upregion', 'a region', region_type))
    if not types.is_pointer(expr_type):
        raise types.TypeError(node, 'Type mismatch for argument %s in call to task %s: expected %s but got %s' % (
            index, 'upregion', 'a pointer', expr_type))

    for expr_region in expr_type.regions:
        subregion = False
        for region_type in region_types:
            success, failed_request = types.check_constraints(
                [types.Constraint(lhs = expr_region, op = types.Constraint.SUBREGION, rhs = region_type)],
                cx.constraints)
            if success:
                subregion = True
                break
        if not subregion:
            raise types.TypeError(node, 'Invalid constraint %s requested in upregion expression' % (
                '%s <= %s' % (expr_region, region_type)))
    return types.Pointer(expr_type.points_to_type, region_types)

@type_check_node.register(ast.UpregionRegions)
def _(node, cx):
    return [type_check_node(region, cx).check_read(region, cx)
            for region in node.regions]

@type_check_node.register(ast.UpregionRegion)
def _(node, cx):
    return cx.lookup(node, node.name)

@type_check_node.register(ast.ExprDownregion)
def _(node, cx):
    region_types = type_check_node(node.regions, cx)
    expr_type = type_check_node(node.expr, cx).check_read(node.expr, cx)

    for index, region_type in zip(xrange(len(region_types)), region_types):
        if not types.is_region(region_type):
            raise types.TypeError(node, 'Type mismatch for type argument %s in call to task %s: expected %s but got %s' % (
                index, 'downregion', 'a region', region_type))
    if not types.is_pointer(expr_type):
        raise types.TypeError(node, 'Type mismatch for argument %s in call to task %s: expected %s but got %s' % (
            index, 'downregion', 'a pointer', expr_type))

    return types.Pointer(expr_type.points_to_type, region_types)

@type_check_node.register(ast.DownregionRegions)
def _(node, cx):
    return [type_check_node(region, cx).check_read(region, cx)
            for region in node.regions]

@type_check_node.register(ast.DownregionRegion)
def _(node, cx):
    return cx.lookup(node, node.name)

@type_check_node.register(ast.ExprPack)
def _(node, cx):
    declared_kind = type_check_node(node.type, cx)
    assert types.is_kind(declared_kind)
    declared_type = declared_kind.type
    region_types = type_check_node(node.regions, cx)
    actual_type = declared_type.instantiate_regions(dict(zip(declared_type.regions, region_types)))

    expr_type = type_check_node(node.expr, cx).check_read(node.expr, cx)

    if expr_type != actual_type:
        raise types.TypeError(node, 'Type mismatch in pack: expected %s but got %s' % (
                actual_type, expr_type))
    success, failed_request = types.check_constraints(actual_type.constraints, cx.constraints)
    if not success:
        raise types.TypeError(node, 'Invalid constraint %s requested in pack expression' % (
                failed_request))

    return declared_type

@type_check_node.register(ast.PackRegions)
def _(node, cx):
    return [type_check_node(region, cx) for region in node.regions]

@type_check_node.register(ast.PackRegion)
def _(node, cx):
    region_type = cx.lookup(node, node.name)
    assert types.is_region(region_type)
    return region_type

@type_check_node.register(ast.ExprCall)
def _(node, cx):
    fn_type = type_check_node(node.function, cx).check_read(node.function, cx)
    assert types.is_function(fn_type)
    function_name = node.function.name

    arg_types = type_check_node(node.args, cx)

    region_map = dict(
        [(param, arg)
         for param, arg in zip(fn_type.param_types, arg_types)
         if (types.is_region(param) and types.is_region(arg))
         or (types.is_ispace(param) and types.is_ispace(arg))])

    param_types = [t.substitute_regions(region_map) for t in fn_type.param_types]
    privileges_requested = [t.substitute_regions(region_map) for t in fn_type.privileges]
    return_type = fn_type.return_type.substitute_regions(region_map)

    if len(param_types) != len(arg_types):
        raise types.TypeError(node, 'Incorrect number of arguments for call to task %s: expected %s but got %s' % (
                function_name, len(param_types), len(arg_types)))
    for param_type, arg_type, index in zip(param_types, arg_types, xrange(len(param_types))):
        if types.is_ispace(param_type):
            if not types.is_ispace(arg_type) or param_type.kind != arg_type.kind:
                raise types.TypeError(node, 'Type mismatch for argument %s in call to task %s: expected %s but got %s' % (
                        index, function_name, param_type.kind, arg_type))
        elif types.is_region(param_type):
            # First check that both are regions.
            if not types.is_region(arg_type):
                raise types.TypeError(node, 'Type mismatch for argument %s in call to task %s: expected %s but got %s' % (
                        index, function_name, param_type.kind, arg_type))
            # Then check that the regions contains compatible types.
            param_kind = param_type.kind.substitute_regions(region_map)
            arg_kind = arg_type.kind
            if param_kind != arg_kind:
                raise types.TypeError(node, 'Type mismatch for argument %s in call to task %s: expected %s but got %s' % (
                        index, function_name, param_kind, arg_kind))
        elif param_type != arg_type:
            raise types.TypeError(node, 'Type mismatch for argument %s in call to task %s: expected %s but got %s' % (
                    index, function_name, param_type, arg_type))
    success, failed_request = types.check_privileges(privileges_requested, cx)
    if not success:
        raise types.TypeError(node, 'Invalid privilege %s requested in call to task %s' % (
                failed_request, function_name))
    return return_type

@type_check_node.register(ast.Args)
def _(node, cx):
    return [type_check_node(arg, cx).check_read(arg, cx)
            for arg in node.args]

@type_check_node.register(ast.ExprConstBool)
def _(node, cx):
    return types.Bool()

@type_check_node.register(ast.ExprConstDouble)
def _(node, cx):
    return types.Double()

@type_check_node.register(ast.ExprConstFloat)
def _(node, cx):
    return types.Float()

@type_check_node.register(ast.ExprConstInt)
def _(node, cx):
    return types.Int()

@type_check_node.register(ast.ExprConstUInt)
def _(node, cx):
    return types.UInt()

def type_check(program, opts):
    cx = types.Context(opts)
    type_check_node(program, cx)
    return cx.type_map, cx.constraints, cx.foreign_types
