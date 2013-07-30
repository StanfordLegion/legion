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
### Type Checker
###

# Work around for OrderedDict missing in Python 2.6.
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from . import ast, types
from .clang import types as ctypes

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
    '==': (types.is_POD, returns_bool),
    '!=': (types.is_POD, returns_bool),
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

# All calls to type_check_node should enter via type_check_helper.
def type_check_helper(node, cx):
    node_type = type_check_node(node, cx)
    cx.type_map[node] = node_type
    return node_type

def type_check_node(node, cx):
    if isinstance(node, ast.Program):
        cx = cx.new_global_scope()

        def_types = []
        for definition in node.defs:
            def_types.append(type_check_helper(definition, cx))
        return types.Program(def_types)
    if isinstance(node, ast.Import):
        module_type = ctypes.foreign_type(node.ast)
        for foreign_name, foreign_type in module_type.def_types.iteritems():
            cx.insert(node, foreign_name, foreign_type)
        return module_type
    if isinstance(node, ast.Struct):
        original_cx = cx
        cx = cx.new_struct_scope()

        # Initially create empty struct type.
        struct_name = node.name
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
            param_type.kind = type_check_helper(param.type, cx)
            if not param_type.validate_regions():
                raise types.TypeError(node, 'Region type is inconsistent with itself: %s' % param_type.pretty_kind())
        for region, region_type in zip(node.regions.regions, region_types):
            cx.insert(node, region.name, region_type)
            region_type.kind = type_check_helper(region.type, cx)
            if not region_type.validate_regions():
                raise types.TypeError(node, 'Region type is inconsistent with itself: %s' % region_type.pretty_kind())

        struct_constraints = type_check_helper(node.constraints, cx)
        struct_type.constraints = struct_constraints

        field_map = type_check_helper(node.field_decls, cx)
        struct_type.field_map = field_map

        # Note: This simple check only works as long as mutual
        # recursion is disallowed on structs.
        for field_type in field_map.itervalues():
            if field_type == struct_type:
                raise types.TypeError(node, 'Struct may not contain itself')

        return def_struct_type
    if isinstance(node, ast.StructConstraints):
        return [type_check_helper(constraint, cx) for constraint in node.constraints]
    if isinstance(node, ast.StructConstraint):
        lhs = type_check_helper(node.lhs, cx)
        rhs = type_check_helper(node.rhs, cx)
        if lhs.kind.contains_type != rhs.kind.contains_type:
            raise types.TypeError(node, 'Type mismatch in region element types for constraint: %s and %s' % (
                lhs.kind.contains_type, rhs.kind.contains_type))
        constraint = types.Constraint(node.op, lhs, rhs)
        return constraint
    if isinstance(node, ast.StructConstraintRegion):
        region_type = cx.lookup(node, node.name)
        assert types.is_region(region_type)
        return region_type
    if isinstance(node, ast.FieldDecls):
        return OrderedDict([
                type_check_helper(field_decl, cx)
                for field_decl in node.field_decls])
    if isinstance(node, ast.FieldDecl):
        field_kind = type_check_helper(node.field_type, cx)
        return (node.name, field_kind.type)
    if isinstance(node, ast.Function):
        original_cx = cx
        cx = cx.new_function_scope()

        fn_name = node.name
        param_types = type_check_helper(node.params, cx)
        cx.privileges = type_check_helper(node.privileges, cx)
        return_kind = type_check_helper(node.return_type, cx)
        assert types.is_kind(return_kind)
        return_type = return_kind.type
        fn_type = types.Function(param_types, cx.privileges, return_type)

        # Insert function name into global scope. Second insert
        # prevents parameters from shadowing function name.
        original_cx.insert(node, fn_name, fn_type)
        cx.insert(node, fn_name, fn_type)

        type_check_helper(node.block, cx.with_return_type(return_type))

        return fn_type
    if isinstance(node, ast.Params):
        return [type_check_helper(param, cx)
                for param in node.params]
    if isinstance(node, ast.Param):
        if isinstance(node.declared_type, ast.TypeRegionKind):
            # Region types may be self-referential. Insert regions
            # into scope early to handle recursive types.
            region_type = types.Region(node.name, types.RegionKind(None, None))
            cx.region_forest.add(region_type)
            cx.insert(node, node.name, region_type)

            region_kind = type_check_helper(node.declared_type, cx)
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

            region_kind = type_check_helper(node.declared_type, cx)
            region_type.kind = region_kind
            return region_type
        if isinstance(node.declared_type, ast.TypeIspaceKind):
            ispace_kind = type_check_helper(node.declared_type, cx)
            ispace_type = types.Ispace(node.name, ispace_kind)
            cx.insert(node, node.name, ispace_type)
            return ispace_type

        # Handle non-region types:
        declared_kind = type_check_helper(node.declared_type, cx)
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
    if isinstance(node, ast.Privileges):
        return cx.privileges | set(
            privilege
            for privilege_node in node.privileges
            for privilege in type_check_helper(privilege_node, cx))
    if isinstance(node, ast.Privilege):
        if node.privilege == 'reads':
            privilege = types.Privilege.READ
        elif node.privilege == 'writes':
            privilege = types.Privilege.WRITE
        elif node.privilege == 'reduces':
            privilege = types.Privilege.REDUCE
        else:
            assert False
        regions = type_check_helper(node.regions, cx)
        return [
            types.Privilege(privilege, node.op, region, field_path)
            for region, field_path in regions]
    if isinstance(node, ast.PrivilegeRegions):
        return [
            region
            for region_node in node.regions
            for region in type_check_helper(region_node, cx)]
    if isinstance(node, ast.PrivilegeRegion):
        region = cx.lookup(node, node.name)
        field_paths = type_check_helper(node.fields, cx)
        return [(region, field_path) for field_path in field_paths]
    if isinstance(node, ast.PrivilegeRegionFields):
        if len(node.fields) == 0:
            return [()]
        return [
            field_path
            for field_node in node.fields
            for field_path in type_check_helper(field_node, cx)]
    if isinstance(node, ast.PrivilegeRegionField):
        prefix = (node.name,)
        field_paths = type_check_helper(node.fields, cx)
        return [prefix + field_path for field_path in field_paths]
    if isinstance(node, ast.TypeVoid):
        return types.Kind(types.Void())
    if isinstance(node, ast.TypeBool):
        return types.Kind(types.Bool())
    if isinstance(node, ast.TypeDouble):
        return types.Kind(types.Double())
    if isinstance(node, ast.TypeFloat):
        return types.Kind(types.Float())
    if isinstance(node, ast.TypeInt):
        return types.Kind(types.Int())
    if isinstance(node, ast.TypeUInt):
        return types.Kind(types.UInt())
    if isinstance(node, ast.TypeInt8):
        return types.Kind(types.Int8())
    if isinstance(node, ast.TypeInt16):
        return types.Kind(types.Int16())
    if isinstance(node, ast.TypeInt32):
        return types.Kind(types.Int32())
    if isinstance(node, ast.TypeInt64):
        return types.Kind(types.Int64())
    if isinstance(node, ast.TypeUInt8):
        return types.Kind(types.UInt8())
    if isinstance(node, ast.TypeUInt16):
        return types.Kind(types.UInt16())
    if isinstance(node, ast.TypeUInt32):
        return types.Kind(types.UInt32())
    if isinstance(node, ast.TypeUInt64):
        return types.Kind(types.UInt64())
    if isinstance(node, ast.TypeColoring):
        region = type_check_helper(node.region, cx)
        assert types.is_region(region)
        return types.Kind(types.Coloring(region))
    if isinstance(node, ast.TypeColoringRegion):
        return cx.lookup(node, node.name)
    if isinstance(node, ast.TypeID):
        kind = cx.lookup(node, node.name)
        args = type_check_helper(node.args, cx)
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
    if isinstance(node, ast.TypeArgs):
        return [type_check_helper(arg, cx) for arg in node.args]
    if isinstance(node, ast.TypeArg):
        arg = cx.lookup(node, node.name)
        if not types.is_region(arg):
            raise types.TypeError(node, 'Type mismatch in type %s: expected a region but got %s' % (
                    node.name, arg))
        return arg
    if isinstance(node, ast.TypeArgWild):
        return types.RegionWild()
    if isinstance(node, ast.TypePointer):
        points_to_kind = type_check_helper(node.points_to_type, cx)
        regions = type_check_helper(node.regions, cx)
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
    if isinstance(node, ast.TypePointerRegions):
        return [type_check_helper(region, cx)
                for region in node.regions]
    if isinstance(node, ast.TypeRegion):
        region_type = cx.lookup(node, node.name)
        return region_type
    if isinstance(node, ast.TypeRegionWild):
        return types.RegionWild()
    if isinstance(node, ast.TypeRegionKind):
        contains_type = None
        if node.contains_type is not None:
            contains_type = type_check_helper(node.contains_type, cx).type
        return types.RegionKind(None, contains_type)
    if isinstance(node, ast.TypeArrayKind):
        ispace = type_check_helper(node.ispace, cx)
        contains_type = type_check_helper(node.contains_type, cx).type
        return types.RegionKind(ispace, contains_type)
    if isinstance(node, ast.TypeIspace):
        ispace_type = cx.lookup(node, node.name)
        return ispace_type
    if isinstance(node, ast.TypeIspaceKind):
        index_type = type_check_helper(node.index_type, cx).type
        return types.IspaceKind(index_type)
    if isinstance(node, ast.Block):
        cx = cx.new_block_scope()
        for expr in node.block:
            type_check_helper(expr, cx)
        return types.Void()
    if isinstance(node, ast.StatementAssert):
        expr_type = type_check_helper(node.expr, cx).check_read(node.expr, cx)
        if not types.is_bool(expr_type):
            raise types.TypeError(node, 'Type mismatch in assert statement: expected %s but got %s' % (
                    types.Bool(), expr_type))
        return types.Void()
    if isinstance(node, ast.StatementExpr):
        type_check_helper(node.expr, cx)
        return types.Void()
    if isinstance(node, ast.StatementIf):
        condition_type = type_check_helper(node.condition, cx).check_read(node.condition, cx)
        type_check_helper(node.then_block, cx)
        if node.else_block is not None:
            type_check_helper(node.else_block, cx)
        if not types.is_bool(condition_type):
            raise types.TypeError(node, 'If condition expression is not type bool')
        return types.Void()
    if isinstance(node, ast.StatementFor):
        cx = cx.new_block_scope()
        index_types = type_check_helper(node.indices, cx)
        region_types = type_check_helper(node.regions, cx)
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

        type_check_helper(node.block, cx)
        return types.Void()
    if isinstance(node, ast.ForIndices):
        return [type_check_helper(index, cx)
                for index in node.indices]
    if isinstance(node, ast.ForIndex):
        if node.type is not None:
            declared_kind = type_check_helper(node.type, cx)
            assert types.is_kind(declared_kind)
            return declared_kind.type
        return None
    if isinstance(node, ast.ForRegions):
        return [type_check_helper(region, cx)
                for region in node.regions]
    if isinstance(node, ast.ForRegion):
        region_type = cx.lookup(node, node.name)
        return region_type
    if isinstance(node, ast.StatementLet):
        declared_type = None
        if node.type is not None:
            declared_kind = type_check_helper(node.type, cx)
            if types.is_region_kind(declared_kind):
                declared_type = types.Region(node.name, declared_kind)
                cx.region_forest.add(declared_type)
            if types.is_kind(declared_kind):
                declared_type = declared_kind.type
            else:
                assert False

        expr_type = type_check_helper(node.expr, cx).check_read(node.expr, cx)

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
    if isinstance(node, ast.StatementLetRegion):
        region_type = types.Region(node.name, types.RegionKind(None, None))
        cx.region_forest.add(region_type)
        # Insert region name into scope so that element type can refer to it.
        cx.insert(node, node.name, region_type)

        declared_region_kind = None
        if node.region_kind is not None:
            declared_region_kind = type_check_helper(node.region_kind, cx)
        element_kind = type_check_helper(node.element_type, cx)
        size_type = type_check_helper(node.size_expr, cx).check_read(node.size_expr, cx)
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

        cx.privileges.add(types.Privilege(types.Privilege.READ, None, region_type, ()))
        cx.privileges.add(types.Privilege(types.Privilege.WRITE, None, region_type, ()))
        return region_type
    if isinstance(node, ast.StatementLetArray):
        ispace_type = type_check_helper(node.ispace_type, cx)
        region_type = types.Region(node.name, types.RegionKind(ispace_type, None))
        cx.region_forest.add(region_type)
        # insert region name into scope so that element type can refer to it
        cx.insert(node, node.name, region_type)

        declared_region_kind = None
        if node.region_kind is not None:
            declared_region_kind = type_check_helper(node.region_kind, cx)
        element_kind = type_check_helper(node.element_type, cx)
        assert types.is_kind(element_kind) and not types.is_void(element_kind.type)

        # now patch region type so that it refers to the contained type
        region_kind = types.RegionKind(ispace_type, element_kind.type)
        region_type.kind = region_kind

        if declared_region_kind is None:
            declared_region_kind = region_kind

        if declared_region_kind != region_kind:
            raise types.TypeError(node, 'Let bound expression of type %s does not match declared type %s' % (
                    region_kind, declared_region_kind))

        cx.privileges.add(types.Privilege(types.Privilege.READ, None, region_type, ()))
        cx.privileges.add(types.Privilege(types.Privilege.WRITE, None, region_type, ()))
        return region_type
    if isinstance(node, ast.StatementLetIspace):
        declared_ispace_kind = None
        if node.ispace_kind is not None:
            declared_ispace_kind = type_check_helper(node.ispace_kind, cx)

        index_kind = type_check_helper(node.index_type, cx)
        size_type = type_check_helper(node.size_expr, cx).check_read(node.size_expr, cx)
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
    if isinstance(node, ast.StatementLetPartition):
        region_type = type_check_helper(node.region_type, cx).check_read(node.region_type, cx)
        mode = type_check_helper(node.mode, cx)
        coloring_type = type_check_helper(node.coloring_expr, cx).check_read(node.coloring_expr, cx)

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
    if isinstance(node, ast.PartitionMode):
        if node.mode == 'disjoint':
            return types.Partition.DISJOINT
        elif node.mode == 'aliased':
            return types.Partition.ALIASED
        assert False
    if isinstance(node, ast.StatementReturn):
        expr_type = type_check_helper(node.expr, cx).check_read(node.expr, cx)
        if expr_type != cx.return_type:
            raise types.TypeError(node, 'Returned expression of type %s does not match declared return type %s' % (
                    expr_type, cx.return_type))
        return types.Void()
    if isinstance(node, ast.StatementUnpack):
        expr_type = type_check_helper(node.expr, cx).check_read(node.expr, cx)
        declared_kind = type_check_helper(node.type, cx)
        assert types.is_kind(declared_kind)
        declared_type = declared_kind.type

        if not types.is_struct(expr_type):
            raise types.TypeError(node, 'Type mismatch in unpack: expected %s but got %s' % (
                    'a struct', expr_type))

        region_types = type_check_helper(node.regions, cx)
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
    if isinstance(node, ast.UnpackRegions):
        return [type_check_helper(region, cx) for region in node.regions]
    if isinstance(node, ast.UnpackRegion):
        # Create regions with empty region_types initially, patch later.
        region_type = types.Region(node.name, types.RegionKind(None, None))
        cx.region_forest.add(region_type)
        return region_type
    if isinstance(node, ast.StatementVar):
        declared_type = None
        if node.type is not None:
            declared_kind = type_check_helper(node.type, cx)
            assert types.is_kind(declared_kind)
            declared_type = declared_kind.type

        expr_type = type_check_helper(node.expr, cx).check_read(node.expr, cx)

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
    if isinstance(node, ast.StatementWhile):
        condition_type = type_check_helper(node.condition, cx).check_read(node.condition, cx)
        type_check_helper(node.block, cx)
        if not types.is_bool(condition_type):
            raise types.TypeError(node, 'While condition expression is not type bool')
        return types.Void()
    if isinstance(node, ast.ExprID):
        id_type = cx.lookup(node, node.name)
        return id_type
    if isinstance(node, ast.ExprAssignment):
        lval_type = type_check_helper(node.lval, cx).check_write(node.lval, cx)
        rval_type = type_check_helper(node.rval, cx).check_read(node.rval, cx)
        if lval_type != rval_type:
            raise types.TypeError(node, 'Type mismatch in assignment: %s and %s' % (
                    lval_type, rval_type))
        return rval_type
    if isinstance(node, ast.ExprUnaryOp):
        arg_type = type_check_helper(node.arg, cx).check_read(node.arg, cx)
        if not unary_operator_table[node.op][0](arg_type):
            raise types.TypeError(node, 'Type mismatch in operand to unary operator: %s' % (
                    arg_type))
        return unary_operator_table[node.op][1](arg_type)
    if isinstance(node, ast.ExprBinaryOp):
        lhs_type = type_check_helper(node.lhs, cx).check_read(node.lhs, cx)
        rhs_type = type_check_helper(node.rhs, cx).check_read(node.rhs, cx)
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
    if isinstance(node, ast.ExprReduceOp):
        lhs_type = type_check_helper(node.lhs, cx).check_reduce(node.lhs, node.op, cx)
        rhs_type = type_check_helper(node.rhs, cx).check_read(node.rhs, cx)
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
    if isinstance(node, ast.ExprCast):
        cast_to_kind = type_check_helper(node.cast_to_type, cx)
        assert types.is_kind(cast_to_kind) and types.is_numeric(cast_to_kind.type)
        expr_type = type_check_helper(node.expr, cx).check_read(node.expr, cx)
        if not types.is_numeric(expr_type):
            raise types.TypeError(node, 'Type mismatch in cast: expected a number but got %s' % (
                    expr_type))
        return cast_to_kind.type
    if isinstance(node, ast.ExprNull):
        pointer_kind = type_check_helper(node.pointer_type, cx)
        assert types.is_kind(pointer_kind) and types.is_pointer(pointer_kind.type)
        return pointer_kind.type
    if isinstance(node, ast.ExprIsnull):
        pointer_type = type_check_helper(node.pointer_expr, cx).check_read(node.pointer_expr, cx)
        if not types.is_pointer(pointer_type):
            raise types.TypeError(node, 'Type mismatch for argument %s in call to task %s: expected %s but got %s' % (
                    0, 'isnull', 'a pointer', pointer_type))
        return types.Bool()
    if isinstance(node, ast.ExprNew):
        pointer_kind = type_check_helper(node.pointer_type, cx)
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
    if isinstance(node, ast.ExprRead):
        pointer_type = type_check_helper(node.pointer_expr, cx).check_read(node.pointer_expr, cx)
        assert types.is_pointer(pointer_type)
        privileges_requested = [
            types.Privilege(types.Privilege.READ, None, region, ())
            for region in pointer_type.regions]
        success, failed_request = types.check_privileges(privileges_requested, cx)
        if not success:
            raise types.TypeError(node, 'Invalid privilege %s requested in read' % failed_request)
        value_type = pointer_type.points_to_type
        return value_type
    if isinstance(node, ast.ExprWrite):
        pointer_type = type_check_helper(node.pointer_expr, cx).check_read(node.pointer_expr, cx)
        value_type = type_check_helper(node.value_expr, cx).check_read(node.value_expr, cx)
        assert types.is_pointer(pointer_type) and pointer_type.points_to_type == value_type
        privileges_requested = [
            types.Privilege(types.Privilege.WRITE, None, region, ())
            for region in pointer_type.regions]
        success, failed_request = types.check_privileges(privileges_requested, cx)
        if not success:
            raise types.TypeError(node, 'Invalid privilege %s requested in write' % failed_request)
        return types.Void()
    if isinstance(node, ast.ExprReduce):
        pointer_type = type_check_helper(node.pointer_expr, cx).check_read(node.pointer_expr, cx)
        value_type = type_check_helper(node.value_expr, cx).check_read(node.value_expr, cx)
        assert types.is_pointer(pointer_type)

        if pointer_type.points_to_type != value_type:
            raise types.TypeError(node, 'Type mismatch in operands to reduce: %s and %s' % (
                    pointer_type.points_to_type, value_type))
        if not reduce_operator_table[node.op](pointer_type.points_to_type):
            raise types.TypeError(node, 'Type mismatch in operand to reduce: %s' % (
                    pointer_type.points_to_type))
        if not reduce_operator_table[node.op](value_type):
            raise types.TypeError(node, 'Type mismatch in operand to reduce: %s' % (
                    value_type))

        privileges_requested = [
            types.Privilege(types.Privilege.REDUCE, node.op, region, ())
            for region in pointer_type.regions]
        success, failed_request = types.check_privileges(privileges_requested, cx)
        if not success:
            raise types.TypeError(node, 'Invalid privilege %s requested in reduce' % failed_request)
        return types.Void()
    if isinstance(node, ast.ExprDereference):
        pointer_type = type_check_helper(node.pointer_expr, cx).check_read(node.pointer_expr, cx)
        if not types.is_pointer(pointer_type):
            raise types.TypeError(node, 'Type mismatch in pointer dereference: expected a pointer but got %s' % (
                    pointer_type))
        reference_type = types.Reference(
            refers_to_type = pointer_type.points_to_type,
            regions = pointer_type.regions)
        return reference_type
    if isinstance(node, ast.ExprArrayAccess):
        array_type = type_check_helper(node.array_expr, cx).check_read(node.array_expr, cx)
        index_type = type_check_helper(node.index_expr, cx).check_read(node.index_expr, cx)

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
    if isinstance(node, ast.ExprFieldAccess):
        wrapper_type = type_check_helper(node.struct_expr, cx)

        struct_type = wrapper_type.as_read()
        if not types.is_struct(struct_type):
            raise types.TypeError(node, 'Type mismatch in struct field access: expected a struct but got %s' % (
                struct_type))
        if node.field_name not in struct_type.field_map:
            raise types.TypeError(node, 'Struct %s has no field named %s' % (
                struct_type, node.field_name))

        return wrapper_type.get_field(node.field_name)
    if isinstance(node, ast.ExprFieldDereference):
        pointer_type = type_check_helper(node.pointer_expr, cx).check_read(node.pointer_expr, cx)

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
    if isinstance(node, ast.ExprFieldValues):
        field_values = type_check_helper(node.field_values, cx)

        field_map = OrderedDict()
        for field_name, value_type in field_values:
            field_map[field_name] = value_type
        struct_type = types.Struct(None, [], [], set(), field_map)

        return struct_type
    if isinstance(node, ast.FieldValues):
        return [type_check_helper(field_value, cx)
                for field_value in node.field_values]
    if isinstance(node, ast.FieldValue):
        return (
            node.field_name,
            type_check_helper(node.value_expr, cx).check_read(node.value_expr, cx))
    if isinstance(node, ast.ExprFieldUpdates):
        struct_type = type_check_helper(node.struct_expr, cx).check_read(node.struct_expr, cx)
        field_updates = type_check_helper(node.field_updates, cx)
        assert types.is_struct(struct_type)

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
    if isinstance(node, ast.FieldUpdates):
        return [type_check_helper(field_update, cx)
                for field_update in node.field_updates]
    if isinstance(node, ast.FieldUpdate):
        return (
            node.field_name,
            type_check_helper(node.update_expr, cx).check_read(node.update_expr, cx))
    if isinstance(node, ast.ExprColoring):
        region_type = type_check_helper(node.region, cx).check_read(node.region, cx)
        if not (types.is_region(region_type) or types.is_ispace(region_type)):
            raise types.TypeError(node, 'Type mismatch in coloring: expected a region or ispace but got %s' % (
                region_type))
        return types.Coloring(region_type)
    if isinstance(node, ast.ColoringRegion):
        return cx.lookup(node, node.name)
    if isinstance(node, ast.ExprColor):
        coloring_type = type_check_helper(node.coloring_expr, cx).check_read(node.coloring_expr, cx)
        pointer_type = type_check_helper(node.pointer_expr, cx).check_read(node.pointer_expr, cx)
        color_type = type_check_helper(node.color_expr, cx).check_read(node.color_expr, cx)

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
    if isinstance(node, ast.ExprUpregion):
        region_types = type_check_helper(node.regions, cx)
        expr_type = type_check_helper(node.expr, cx).check_read(node.expr, cx)

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
    if isinstance(node, ast.UpregionRegions):
        return [type_check_helper(region, cx).check_read(region, cx)
                for region in node.regions]
    if isinstance(node, ast.UpregionRegion):
        return cx.lookup(node, node.name)
    if isinstance(node, ast.ExprDownregion):
        region_types = type_check_helper(node.regions, cx)
        expr_type = type_check_helper(node.expr, cx).check_read(node.expr, cx)

        for index, region_type in zip(xrange(len(region_types)), region_types):
            if not types.is_region(region_type):
                raise types.TypeError(node, 'Type mismatch for type argument %s in call to task %s: expected %s but got %s' % (
                    index, 'downregion', 'a region', region_type))
        if not types.is_pointer(expr_type):
            raise types.TypeError(node, 'Type mismatch for argument %s in call to task %s: expected %s but got %s' % (
                index, 'downregion', 'a pointer', expr_type))

        return types.Pointer(expr_type.points_to_type, region_types)
    if isinstance(node, ast.DownregionRegions):
        return [type_check_helper(region, cx).check_read(region, cx)
                for region in node.regions]
    if isinstance(node, ast.DownregionRegion):
        return cx.lookup(node, node.name)
    if isinstance(node, ast.ExprPack):
        declared_kind = type_check_helper(node.type, cx)
        assert types.is_kind(declared_kind)
        declared_type = declared_kind.type
        region_types = type_check_helper(node.regions, cx)
        actual_type = declared_type.instantiate_regions(dict(zip(declared_type.regions, region_types)))

        expr_type = type_check_helper(node.expr, cx).check_read(node.expr, cx)

        if expr_type != actual_type:
            raise types.TypeError(node, 'Type mismatch in pack: expected %s but got %s' % (
                    actual_type, expr_type))
        success, failed_request = types.check_constraints(actual_type.constraints, cx.constraints)
        if not success:
            raise types.TypeError(node, 'Invalid constraint %s requested in pack expression' % (
                    failed_request))

        return declared_type
    if isinstance(node, ast.PackRegions):
        return [type_check_helper(region, cx) for region in node.regions]
    if isinstance(node, ast.PackRegion):
        region_type = cx.lookup(node, node.name)
        assert types.is_region(region_type)
        return region_type
    if isinstance(node, ast.ExprCall):
        fn_type = type_check_helper(node.function, cx).check_read(node.function, cx)
        assert types.is_function(fn_type)
        function_name = node.function.name

        arg_types = type_check_helper(node.args, cx)

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
    if isinstance(node, ast.Args):
        return [type_check_helper(arg, cx).check_read(arg, cx)
                for arg in node.args]
    if isinstance(node, ast.ExprConstBool):
        return types.Bool()
    if isinstance(node, ast.ExprConstDouble):
        return types.Double()
    if isinstance(node, ast.ExprConstInt):
        return types.Int()
    if isinstance(node, ast.ExprConstUInt):
        return types.UInt()
    raise Exception('Type checking failed at %s' % node)

def type_check(node):
    cx = types.Context()
    type_check_helper(node, cx)
    return cx.type_map, cx.constraints
