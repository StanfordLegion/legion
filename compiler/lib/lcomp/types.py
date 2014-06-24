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
### Types
###

# Work around for Orderedict missing in Python 2.6.
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import copy
from . import symbol_table, union_find

def is_same_class(a, b):
    return type(a) is type(b) and a.__class__ is b.__class__

class Context:
    def __init__(self, opts):
        self.opts = opts
        self.type_env = None
        self.type_map = {}
        self.privileges = None
        self.constraints = set()
        self.region_forest = None
        self.return_type = None
        self.foreign_types = []
    def new_block_scope(self):
        cx = copy.copy(self)
        cx.type_env = cx.type_env.new_scope()
        cx.privileges = copy.copy(cx.privileges)
        return cx
    def new_function_scope(self):
        cx = copy.copy(self)
        cx.type_env = cx.type_env.new_scope()
        cx.privileges = set()
        return cx
    def new_struct_scope(self):
        cx = copy.copy(self)
        cx.type_env = cx.type_env.new_scope()
        return cx
    def new_global_scope(self):
        cx = copy.copy(self)
        cx.type_env = symbol_table.SymbolTable()
        cx.region_forest = union_find.UnionFind()
        return cx
    def with_return_type(self, return_type):
        cx = copy.copy(self)
        cx.return_type = return_type
        return cx
    def lookup(self, node, name):
        try:
            return self.type_env.lookup(name)
        except symbol_table.UndefinedSymbolException:
            raise TypeError(node, 'Name %s is undefined' % (name,))
    def insert(self, node, name, value, shadow = False):
        try:
            return self.type_env.insert(name, value, shadow)
        except symbol_table.RedefinedSymbolException:
            raise TypeError(node, 'Name %s is already defined' % (name,))

class TypeError (Exception):
    def __init__(self, node, message):
        Exception.__init__(self, '\n%s:\n%s' % (
                node.span, message))

# Special set wrapper for key-based equality.
def default_key(elt):
    return elt.key()

def wrap(elts, keyfn = default_key):
    return dict((keyfn(elt), elt) for elt in elts)

def unwrap(set_by_key):
    return set_by_key.values()

def unwrap_iter(set_by_key):
    return set_by_key.itervalues()

def add_key(set_by_key, elt, keyfn = default_key):
    set_by_key[keyfn(elt)] = elt

def contains_key(set_by_key, elt, keyfn = default_key):
    return keyfn(elt) in set_by_key

def find_key(set_by_key, elt, keyfn = default_key):
    return set_by_key[keyfn(elt)]

def union(sets):
    sets = iter(sets)
    result = copy.copy(next(sets, dict()))
    for s in sets:
        result.update(s)
    return result

def difference(a, b):
    result = copy.copy(a)
    for k in b:
        del result[k]
    return result

# Limit recursion depth for algorithms that might recurse infinitely
# (like computing hash values).
MAX_REPR_DEPTH = 5
MAX_HASH_DEPTH = 5

class Type:
    def __repr__(self):
        return self.pretty()
    def __eq__(self, other):
        return self is other
    def __ne__(self, other):
        return not self.__eq__(other)
    def __hash__(self):
        return hash(self.hash_helper(0))
    def hash_helper(self, depth):
        raise Exception('hash_helper not defined for Type subclass %s' % self.__class__)
    # Used for strict equality checking, to ensure that all types that
    # need translation make it through to code gen.
    def key(self):
        return self.__class__.__name__
    # Pretty printing.
    def pretty(self):
        return self.__class__.__name__.lower()
    # Pretty printing for types that include separate kinds (e.g. regions).
    def pretty_kind(self):
        return self.pretty()
    # Returns true only if the type contains no wildcards.
    def is_concrete(self):
        return True
    # Returns the type as an r-value.
    def as_read(self):
        return self
    # Returns the type as an l-value.
    def as_write(self):
        raise Exception('Type %s is not an lval' % self)
    # Checks for read permissions, then returns the type as an r-value.
    def check_read(self, node, cx):
        return self
    # Checks for write permissions, then returns the type as an l-value.
    def check_write(self, node, cx):
        raise TypeError(node, 'Type %s is not an lval' % self)
    # Checks for reduce permissions, then returns the type as an l-value.
    def check_reduce(self, node, op, cx):
        raise TypeError(node, 'Type %s is not an lval' % self)
    # Checks that the usage of regions within a type is internally
    # consistent. Used because region construction can potentially
    # create inconsistent types.
    def validate_regions(self):
        return self.validate_regions_helper(set())
    def validate_regions_helper(self, visited):
        return True
    # Substitutes the given regions into the type.
    def substitute_regions(self, region_map):
        return self
    # Returns a list of types that the type contains, for code gen.
    def component_types(self):
        return self.component_types_helper(wrap([]))
    def component_types_helper(self, visited):
        if contains_key(visited, self):
            return wrap([])
        add_key(visited, self)
        return wrap([self])

class SingletonType(Type):
    def __eq__(self, other):
        return is_same_class(self, other)
    def hash_helper(self, depth):
        return self.__class__.__name__
    @property
    def cname(self):
        return self.pretty()

class Void(SingletonType): pass
class Bool(SingletonType): pass
class Double(SingletonType): pass
class Float(SingletonType): pass

class MachineDependentInteger(Type):
    # variables to defined by derived classes
    name = None
    cname = None
    def __init__(self):
        assert self.name is not None
    def __eq__(self, other):
        return is_same_class(self, other)
    def hash_helper(self, depth):
        return self.__class__.__name__
    def pretty(self):
        return self.name

class Int(MachineDependentInteger):
    name = 'int'
    cname = 'intptr_t'
class UInt(MachineDependentInteger):
    name = 'uint'
    cname = 'uintptr_t'

class FixedSizeInteger(Type):
    # variables to defined by derived classes
    size = None
    signed = None
    def __init__(self):
        assert self.size is not None and self.signed is not None
    def __eq__(self, other):
        return is_same_class(self, other)
    def hash_helper(self, depth):
        return self.__class__.__name__
    def pretty(self):
        return '%sint%s' % ('' if self.signed else 'u', self.size)
    @property
    def cname(self):
        return '%sint%s_t' % ('' if self.signed else 'u', self.size)

class Int8(FixedSizeInteger):
    size = 8
    signed = True
class Int16(FixedSizeInteger):
    size = 16
    signed = True
class Int32(FixedSizeInteger):
    size = 32
    signed = True
class Int64(FixedSizeInteger):
    size = 64
    signed = True
class UInt8(FixedSizeInteger):
    size = 8
    signed = False
class UInt16(FixedSizeInteger):
    size = 16
    signed = False
class UInt32(FixedSizeInteger):
    size = 32
    signed = False
class UInt64(FixedSizeInteger):
    size = 64
    signed = False

class Ispace(Type):
    def __init__(self, name, kind):
        self.name = name
        self.kind = kind
    def hash_helper(self, depth):
        return id(self)
    def key(self):
        return ('ispace', self.name, id(self))
    def pretty(self):
        return self.name
    def pretty_kind(self):
        return '%s: %s' % (self.name, self.kind.pretty())
    def validate_regions_helper(self, visited):
        if self in visited:
            return True
        visited.add(self)
        return self.kind.validate_regions_helper(visited)

class IspaceKind(Type):
    def __init__(self, index_type):
        self.index_type = index_type
    def __eq__(self, other):
        return is_same_class(self, other) and self.index_type == other.index_type
    def hash_helper(self, depth):
        if depth > MAX_HASH_DEPTH:
            return 'ispace_kind'
        return ('ispace_kind', self.index_type.hash_helper(depth + 1))
    def key(self):
        return ('ispace_kind', self.index_type.key())
    def pretty(self):
        return 'ispace<%s>' % self.index_type.pretty()
    def validate_regions_helper(self, visited):
        if self in visited:
            return True
        visited.add(self)
        return self.index_type.validate_regions_helper(visited)
    def substitute_regions(self, region_map):
        return IspaceKind(self.index_type.substitute_regions(region_map))

class Region(Type):
    def __init__(self, name, kind):
        self.name = name
        self.kind = kind
    def __eq__(self, other):
        return id(self) == id(other) or is_region_wild(other) or is_foreign_region(other)
    def hash_helper(self, depth):
        return id(self)
    def key(self):
        return ('region', self.name)
    def pretty(self):
        return self.name
    def pretty_kind(self):
        return '%s: %s' % (self.name, self.kind.pretty())
    def validate_regions_helper(self, visited):
        if self in visited:
            return True
        visited.add(self)
        return self.kind.validate_regions_helper(visited)

class RegionWild(Type):
    def __eq__(self, other):
        return is_same_class(self, other) or is_region(other) or is_foreign_region(other)
    def hash_helper(self, depth):
        return 'region_wild'
    def key(self):
        return 'region'
    def pretty(self):
        return '?'
    def is_concrete(self):
        return False

class RegionKind(Type):
    def __init__(self, ispace, contains_type):
        self.ispace = ispace
        self.contains_type = contains_type
    def __eq__(self, other):
        return is_same_class(self, other) and self.ispace == other.ispace and self.contains_type == other.contains_type
    def hash_helper(self, depth):
        if depth > MAX_HASH_DEPTH:
            return 'region_kind'
        if self.contains_type is None:
            return 'region_kind'
        if self.ispace is None:
            return ('region_kind', self.contains_type.hash_helper(depth + 1))
        return ('region_kind', self.ispace.hash_helper(depth + 1), self.contains_type.hash_helper(depth + 1))
    def key(self):
        if self.contains_type is None:
            return 'region_kind'
        if self.ispace is None:
            return ('region_kind', self.contains_type.key())
        return ('region_kind', self.ispace.key(), self.contains_type.key())
    def pretty(self):
        if self.contains_type is None:
            return 'region'
        if self.ispace is None:
            return 'region<%s>' % self.contains_type.pretty()
        return 'array<%s, %s>' % (
            self.ispace.pretty(),
            self.contains_type.pretty())
    def validate_regions_helper(self, visited):
        if self in visited:
            return True
        visited.add(self)
        if self.contains_type is None:
            return False
        if self.ispace is not None and not self.ispace.validate_regions_helper(visited):
            return False
        return self.contains_type.validate_regions_helper(visited)
    def substitute_regions(self, region_map):
        if self.contains_type is None:
            return self
        if self.ispace is None:
            return RegionKind(self.ispace, self.contains_type.substitute_regions(region_map))
        if self.ispace not in region_map:
            return RegionKind(self.ispace, self.contains_type.substitute_regions(region_map))
        return RegionKind(
            region_map[self.ispace],
            self.contains_type.substitute_regions(region_map))

class Coloring(Type):
    def __init__(self, region):
        self.region = region
    def __eq__(self, other):
        return (is_same_class(self, other) and self.region == other.region) or is_foreign_coloring(other)
    def hash_helper(self, depth):
        if depth > MAX_HASH_DEPTH:
            return 'coloring'
        return ('coloring', self.region.hash_helper(depth + 1))
    def pretty(self):
        return 'coloring<%s>' % self.region.pretty()
    def is_concrete(self):
        return self.region.is_concrete()
    def validate_regions_helper(self, visited):
        if self in visited:
            return True
        visited.add(self)
        return self.region.validate_regions_helper(visited)
    def substitute_regions(self, region_map):
        if self.region not in region_map:
            return self
        return Coloring(region_map[self.region])
    def component_types_helper(self, visited):
        if contains_key(visited, self):
            return wrap([])
        add_key(visited, self)
        return union([wrap([self]), self.region.component_types_helper(visited)])

class Partition(Type):
    DISJOINT = 'disjoint'
    ALIASED = 'aliased'

    def __init__(self, name, kind):
        self.name = name
        self.kind = kind
        self.static_subregions = {}
        self.dynamic_subregions = {}
    def __eq__(self, other):
        return id(self) == id(other)
    def hash_helper(self, depth):
        return id(self)
    def key(self):
        return ('partition', self.name)
    def pretty(self):
        return self.name
    def pretty_kind(self):
        return '%s: %s' % (self.name, self.kind.pretty())
    def validate_regions_helper(self, visited):
        if self in visited:
            return True
        visited.add(self)
        return self.kind.validate_regions_helper(visited)
    def static_subregion(self, index, cx):
        assert isinstance(index, int)
        if index in self.static_subregions:
            return self.static_subregions[index]

        if is_region(self.kind.region):
            region = Region('%s[%s]' % (self.name, index), self.kind.region.kind)
            cx.region_forest.union(region, self.kind.region)
        elif is_ispace(self.kind.region):
            region = Ispace('%s[%s]' % (self.name, index), self.kind.region.kind)
        else:
            assert False

        cx.constraints.add(
            Constraint(lhs = region, op = Constraint.SUBREGION, rhs = self.kind.region))
        if self.kind.mode == Partition.DISJOINT:
            for sibling in self.static_subregions.itervalues():
                cx.constraints.add(
                    Constraint(lhs = region, op = Constraint.DISJOINT, rhs = sibling))
        self.static_subregions[index] = region
        return region
    def dynamic_subregion(self, index_expr, cx):
        if index_expr in self.dynamic_subregions:
            return self.dynamic_subregions[index_expr]

        if is_region(self.kind.region):
            region = Region('%s[...]' % (self.name), self.kind.region.kind)
            cx.region_forest.union(region, self.kind.region)
        elif is_ispace(self.kind.region):
            region = Ispace('%s[...]' % (self.name), self.kind.region.kind)
        else:
            assert False

        self.dynamic_subregions[index_expr] = region
        cx.constraints.add(
            Constraint(lhs = region, op = Constraint.SUBREGION, rhs = self.kind.region))
        return region

class PartitionKind(Type):
    def __init__(self, region, mode):
        self.region = region
        self.mode = mode
    def __eq__(self, other):
        return is_same_class(self, other) and self.region == other.region and self.mode == other.mode
    def hash_helper(self, depth):
        if depth > MAX_HASH_DEPTH:
            return 'partition_kind'
        return ('partition_kind', self.region.hash_helper(depth + 1), self.mode.hash_helper(depth + 1))
    def key(self):
        return ('partition_kind', self.region.key(), self.mode)
    def pretty(self):
        return 'partition<%s, %s>' % (
            self.region.pretty(),
            self.mode)
    def validate_regions_helper(self, visited):
        if self in visited:
            return True
        visited.add(self)
        return self.region.validate_regions_helper(visited)
    def substitute_regions(self, region_map):
        if self.region not in region_map:
            return self
        return PartitionKind(
            region_map[self.region],
            self.mode)

class Pointer(Type):
    def __init__(self, points_to_type, regions):
        for region in regions:
            assert is_region(region) or is_region_wild(region)
        self.points_to_type = points_to_type
        self.regions = regions
    def __eq__(self, other):
        return ((is_same_class(self, other) and
                 self.points_to_type == other.points_to_type and
                 self.regions == other.regions) or
                is_foreign_pointer(other))
    def hash_helper(self, depth):
        if depth > MAX_HASH_DEPTH:
            return 'pointer'
        return ('pointer',
                self.points_to_type.hash_helper(depth + 1),
                tuple(region.hash_helper(depth + 1) for region in self.regions))
    def key(self):
        return ('pointer', len(self.regions) > 1)
    def pretty(self):
        if len(self.regions) == 1:
            return '%s@%s' % (
                self.points_to_type.pretty(),
                self.regions[0].pretty())
        return '%s@(%s)' % (
            self.points_to_type.pretty(),
            ', '.join(region.pretty() for region in self.regions))
    def is_concrete(self):
        return self.points_to_type.is_concrete() and \
            all(region.is_concrete() for region in self.regions)
    def validate_regions_helper(self, visited):
        if self in visited:
            return True
        visited.add(self)
        if not self.points_to_type.validate_regions_helper(visited):
            return False
        for region in self.regions:
            if not region.validate_regions_helper(visited):
                return False
            if is_region(region) and not self.points_to_type == region.kind.contains_type:
                return False
        return True
    def substitute_regions(self, region_map):
        return Pointer(
            self.points_to_type.substitute_regions(region_map),
            [(region_map[region] if region in region_map else region)
             for region in self.regions])
    def component_types_helper(self, visited):
        if contains_key(visited, self):
            return wrap([])
        add_key(visited, self)
        return union(
            [wrap([self]),
             self.points_to_type.component_types_helper(visited)] +
            [region.component_types_helper(visited) for region in self.regions])

# A reference to a element or a field of an element living in a
# region, which can potentially be used as an lval.
class Reference(Type):
    def __init__(self, refers_to_type, regions, field_path = ()):
        assert isinstance(field_path, tuple)
        nested_type = refers_to_type
        for field_name in field_path:
            assert is_struct(nested_type) and field_name in nested_type.field_map
            nested_type = nested_type.field_map[field_name]
        for region in regions:
            assert is_region(region)

        self.refers_to_type = refers_to_type
        self.regions = regions
        self.field_path = field_path
    def hash_helper(self, depth):
        if depth > MAX_HASH_DEPTH:
            return 'reference'
        return ('reference',
                self.refers_to_type.hash_helper(depth + 1),
                tuple(region.hash_helper(depth + 1) for region in self.regions))
    def key(self):
        return 'reference'
    def pretty(self):
        return self.as_read().pretty()
    def is_concrete(self):
        return self.as_read().is_concrete()
    def as_read(self):
        nested_type = self.refers_to_type
        for field_name in self.field_path:
            nested_type = nested_type.field_map[field_name]
        return nested_type
    def as_write(self):
        return self.as_read()
    def check_read(self, node, cx):
        privileges_requested = [
            Privilege(Privilege.READ, None, region, self.field_path)
            for region in self.regions]
        success, failed_request = check_privileges(privileges_requested, cx)
        if not success:
            raise TypeError(node, 'Invalid privilege %s requested in pointer dereference' % failed_request)
        return self.as_read()
    def check_write(self, node, cx):
        privileges_requested = [
            Privilege(Privilege.WRITE, None, region, self.field_path)
            for region in self.regions]
        success, failed_request = check_privileges(privileges_requested, cx)
        if not success:
            raise TypeError(node, 'Invalid privilege %s requested in pointer dereference' % failed_request)
        return self.as_write()
    def check_reduce(self, node, op, cx):
        privileges_requested = [
            Privilege(Privilege.REDUCE, op, region, self.field_path)
            for region in self.regions]
        success, failed_request = check_privileges(privileges_requested, cx)
        if not success:
            raise TypeError(node, 'Invalid privilege %s requested in pointer dereference' % failed_request)
        return self.as_write()
    def substitute_regions(self, region_map):
        raise TypeError(node, 'unreachable')
    def component_types_helper(self, visited):
        if contains_key(visited, self):
            return wrap([])
        add_key(visited, self)
        return union(
            [wrap([self]),
             self.refers_to_type.component_types_helper(visited)] +
            [region.component_types_helper(visited) for region in self.regions])
    def get_field(self, field_name):
        return Reference(self.refers_to_type, self.regions, self.field_path + (field_name,))

# A reference to a variable on the stack, which can potentially be
# used as an lval.
class StackReference(Type):
    def __init__(self, refers_to_type, field_path = ()):
        assert isinstance(field_path, tuple)
        nested_type = refers_to_type
        for field_name in field_path:
            assert is_struct(nested_type) and field_name in nested_type.field_map
            nested_type = nested_type.field_map[field_name]

        self.refers_to_type = refers_to_type
        self.field_path = field_path
    def hash_helper(self, depth):
        if depth > MAX_HASH_DEPTH:
            return 'stack_reference'
        return ('stack_reference', self.refers_to_type.hash_helper(depth + 1))
    def key(self):
        return 'reference'
    def pretty(self):
        return self.as_read().pretty()
    def is_concrete(self):
        return self.as_read().is_concrete()
    def as_read(self):
        nested_type = self.refers_to_type
        for field_name in self.field_path:
            nested_type = nested_type.field_map[field_name]
        return nested_type
    def as_write(self):
        return self.as_read()
    def check_read(self, node, cx):
        # Reads are ok because variables are always readable.
        return self.as_read()
    def check_write(self, node, cx):
        # Writes are ok because variables are always writable.
        return self.as_write()
    def check_reduce(self, node, op, cx):
        # Reductions are ok because variables are always readable and writable.
        return self.as_write()
    def substitute_regions(self, region_map):
        raise Exception('unreachable')
    def component_types_helper(self, visited):
        if contains_key(visited, self):
            return wrap([])
        add_key(visited, self)
        return union(
            [wrap([self]),
             self.refers_to_type.component_types_helper(visited)])
    def get_field(self, field_name):
        return StackReference(self.refers_to_type, self.field_path + (field_name,))

def merge_region_map(old_map, new_map, allow_keys):
    old_set = set(old_map.itervalues())
    return dict(
        [(k, new_map[v]) if v in new_map else (k, v)
         for k, v in old_map.iteritems()] +
        [(k, v)
         for k, v in new_map.iteritems()
         if k not in old_set and k in allow_keys])

def struct_eq(self, other):
    if self is other:
        return True
    if not is_struct(other):
        return False
    if self.name is not None and self.name == other.name:
        if all(own_param == other_param for own_param, other_param in zip(self.params, other.params)) and \
                all(own_region == other_region for own_region, other_region in zip(self.regions, other.regions)):
            return True
        return False
    for (own_field_name, own_field_type), (other_field_name, other_field_type) \
            in zip(self.field_map.iteritems(), other.field_map.iteritems()):
        if not (own_field_name == other_field_name and own_field_type == other_field_type):
            return False
    return True

def struct_key(self):
    if is_struct_instance(self):
        return self.struct_type.key()
    return ('struct', tuple(t.key() for t in self.regions),
            tuple((k, v.key()) for k, v in self.field_map.iteritems()))

class Struct(Type):
    def __init__(self, name, params, regions, constraints, field_map):
        self.name = name
        self.params = params
        self.regions = regions
        self.constraints = constraints
        self.field_map = field_map
    def __eq__(self, other):
        return struct_eq(self, other)
    def hash_helper(self, depth):
        if depth > MAX_HASH_DEPTH:
            return 'struct'
        return ('struct', tuple((k, v.hash_helper(depth + 1)) for k, v in self.field_map.iteritems()))
    def key(self):
        return struct_key(self)
    def pretty(self):
        if self.name is None:
            return '{%s}' % (', '.join(
                    '%s: %s' % (field_name, field_type.pretty())
                    for field_name, field_type in self.field_map.iteritems()))
        return self.name
    def is_concrete(self):
        if self.name is None:
            return all(field_type.is_concrete() for field_type in self.field_map.itervalues())
        return all(param_type.is_concrete() for param_type in self.params + self.regions)
    def instantiate_params(self, region_map):
        assert set(self.params) == set(region_map.keys())
        return StructInstanceWithParams(self, region_map)
    def instantiate_regions(self, region_map):
        assert len(self.params) == 0 and set(self.regions) == set(region_map.keys())
        return StructInstanceWithParamsAndRegions(self, region_map)
    def substitute_regions(self, region_map):
        # Never substitute params.
        assert len((set(self.params) | set(self.regions)) & set(region_map.keys())) == 0
        # Ignore requests to substitute regions.
        return self
    def component_types_helper(self, visited):
        if contains_key(visited, self):
            return wrap([])
        add_key(visited, self)
        return union(
            [wrap([self])] +
            [t.component_types_helper(visited) for t in self.field_map.itervalues()])
    def get_field(self, field_name):
        assert field_name in self.field_map
        return self.field_map[field_name]

class StructInstanceWithParams(Type):
    def __init__(self, struct_type, region_map):
        assert struct_type.name is not None
        self.struct_type = struct_type
        self.region_map = region_map
    def __eq__(self, other):
        return struct_eq(self, other)
    def hash_helper(self, depth):
        if depth > MAX_HASH_DEPTH:
            return 'struct'
        return ('struct', tuple((k, v.hash_helper(depth + 1)) for k, v in self.field_map.iteritems()))
    def key(self):
        return struct_key(self)
    # Compute struct fields dynamically so that updates to the struct
    # type will be reflected here.
    @property
    def name(self):
        return self.struct_type.name
    @property
    def params(self):
        return [
            (self.region_map[param] if param in self.region_map else param)
            for param in self.struct_type.params]
    @property
    def regions(self):
        return [
            (self.region_map[region] if region in self.region_map else region)
            for region in self.struct_type.regions]
    @property
    def constraints(self):
        return [
            constraint.substitute_regions(self.region_map)
            for constraint in self.struct_type.constraints]
    @property
    def field_map(self):
        return OrderedDict([
                (field_name, field_type.substitute_regions(self.region_map))
                for field_name, field_type in self.struct_type.field_map.iteritems()])
    def pretty(self):
        return '%s<%s>' % (
            self.name,
            ', '.join(param.pretty() for param in self.params))
    def is_concrete(self):
        return all(param_type.is_concrete() for param_type in self.params + self.regions)
    def instantiate_params(self, region_map):
        assert False
    def instantiate_regions(self, region_map):
        assert set(self.regions) == set(region_map.keys())
        new_region_map = self.region_map.copy()
        new_region_map.update(region_map)
        return StructInstanceWithParamsAndRegions(self.struct_type, new_region_map)
    def substitute_regions(self, region_map):
        if len(set(self.params) & set(region_map.keys())) == 0:
            return self
        new_region_map = merge_region_map(self.region_map, region_map, set())
        return StructInstanceWithParams(self.struct_type, new_region_map)
    def component_types_helper(self, visited):
        return self.struct_type.component_types_helper(visited)
    def get_field(self, field_name):
        if field_name not in self.field_map:
            raise TypeError(node, 'Field %s of struct %s is undeclared' % (field_name, self.name))
        return self.field_map[field_name]

class StructInstanceWithParamsAndRegions(Type):
    def __init__(self, struct_type, region_map):
        assert struct_type.name is not None
        self.struct_type = struct_type
        self.region_map = region_map
    def __eq__(self, other):
        return struct_eq(self, other)
    def hash_helper(self, depth):
        if depth > MAX_HASH_DEPTH:
            return 'struct'
        return ('struct', tuple((k, v.hash_helper(depth + 1)) for k, v in self.field_map.iteritems()))
    def key(self):
        return struct_key(self)
    # Compute struct fields dynamically so that updates to the struct
    # type will be reflected here.
    @property
    def name(self):
        return self.struct_type.name
    @property
    def params(self):
        return [
            (self.region_map[param] if param in self.region_map else param)
            for param in self.struct_type.params]
    @property
    def regions(self):
        return [
            (self.region_map[region] if region in self.region_map else region)
            for region in self.struct_type.regions]
    @property
    def constraints(self):
        return [
            constraint.substitute_regions(self.region_map)
            for constraint in self.struct_type.constraints]
    @property
    def field_map(self):
        return OrderedDict([
                (field_name, field_type.substitute_regions(self.region_map))
                for field_name, field_type in self.struct_type.field_map.iteritems()])
    def pretty(self):
        return '%s%s%s%s[%s]' % (
            self.name,
            ('<' if len(self.params) > 0 else ''),
            ', '.join(param.pretty() for param in self.params),
            ('>' if len(self.params) > 0 else ''),
            ', '.join(region.pretty() for region in self.regions))
    def is_concrete(self):
        return all(param_type.is_concrete() for param_type in self.params + self.regions)
    def instantiate_params(self, region_map):
        assert False
    def instantiate_regions(self, region_map):
        assert False
    def substitute_regions(self, region_map):
        if len((set(self.params) | set(self.regions)) & set(region_map.keys())) == 0:
            return self
        new_region_map = merge_region_map(self.region_map, region_map, set())
        return StructInstanceWithParamsAndRegions(self.struct_type, new_region_map)
    def component_types_helper(self, visited):
        return self.struct_type.component_types_helper(visited)
    def get_field(self, field_name):
        if field_name not in self.field_map:
            raise TypeError(node, 'Field %s of struct %s is undeclared' % (field_name, self.name))
        return self.field_map[field_name]

# Kind is a wrapper type used when types themselves are referenced in
# the program.
class Kind(Type):
    def __init__(self, type):
        self.type = type
    def __eq__(self, other):
        return is_same_class(self, other) and self.type == other.type
    def hash_helper(self, depth):
        if depth > MAX_HASH_DEPTH:
            return 'kind'
        return ('kind', self.type.hash_helper(depth + 1))
    def key(self):
        return 'kind'
    def pretty(self):
        return 'kind<%s>' % self.type.pretty()
    def instantiate_params(self, region_map):
        if len(region_map) == 0:
            return self
        return Kind(self.type.instantiate_params(region_map))
    def substitute_regions(self, region_map):
        return Kind(self.type.substitute_regions(region_map))
    def component_types_helper(self, visited):
        if contains_key(visited, self):
            return wrap([])
        add_key(visited, self)
        return union(
            [wrap([self]),
             self.type.component_types_helper(visited)])

class Function(Type):
    def __init__(self, param_types, privileges, return_type):
        self.param_types = param_types
        self.privileges = privileges
        self.return_type = return_type
    def __eq__(self, other):
        return is_same_class(self, other) and \
            all(own_param == other_param for own_param, other_param in zip(self.param_types, other.param_types)) and \
            self.return_type == other.return_type
    def hash_helper(self, depth):
        if depth > MAX_HASH_DEPTH:
            return 'function'
        return ('function',
                tuple(t.hash_helper(depth + 1) for t in self.param_types),
                (self.return_type.hash_helper(depth + 1)))
    def key(self):
        return 'function'
    def pretty(self):
        return 'task%s%s%s' % (
            '(%s)' % (
                ', '.join(param_type.pretty_kind() for param_type in self.param_types),
                ),
            '%s%s' % (
                (': ' if not is_void(self.return_type) else ''),
                (self.return_type.pretty() if not is_void(self.return_type) else ''),
                ),
            '%s%s' % (
                (', ' if len(self.privileges) > 0 else ''),
                ', '.join(privilege.pretty() for privilege in self.privileges),
                ),
            )
    def component_types_helper(self, visited):
        if contains_key(visited, self):
            return wrap([])
        add_key(visited, self)
        return union(
            [wrap([self]),
             self.return_type.component_types_helper(visited)] +
            [t.component_types_helper(visited) for t in self.param_types])

###
### Foreign types are used in the FFI between Legion and
### C++. Generally, they exist implicitly in all Legion tasks, but are
### managed by the compiler and are not directly accessible to Legion
### code. However, in order to do useful work in C++, it is frequently
### necessary to pass them along to C++ code.
###

class ForeignColoring(Type):
    def __eq__(self, other):
        return is_same_class(self, other) or is_coloring(other)
class ForeignContext(Type): pass
class ForeignPointer(Type):
    def __eq__(self, other):
        return is_same_class(self, other) or is_pointer(other)
class ForeignRegion(Type):
    def __eq__(self, other):
        return is_same_class(self, other) or is_region(other) or is_region_wild(other)
class ForeignRuntime(Type): pass

class ForeignFunction(Function):
    def __init__(self, foreign_param_types, param_types, privileges, return_type):
        Function.__init__(self, param_types, privileges, return_type)

        assert len(foreign_param_types) >= len(param_types)
        self.foreign_param_types = foreign_param_types

class Module(Type):
    def __init__(self, def_types):
        self.def_types = def_types
    def __eq__(self, other):
        if not is_same_class(self, other):
            return False
        if len(self.def_types) != len(other.def_types):
            return False
        for (self_name, self_type), (other_name, other_type) in zip(self.def_types.iteritems(),
                                                                    other.def_types.iteritems()):
            if self_name != other_name or self_type != other_type:
                return False
    def hash_helper(self, depth):
        if depth > MAX_HASH_DEPTH:
            return 'program'
        return ('module', tuple((n, t.hash_helper(depth + 1)) for n, t in self.def_types.iteritems()))
    def key(self):
        return 'module'
    def pretty(self):
        return '\n\n'.join('%s: %s' % (n, t.pretty()) for n, t in self.def_types.iteritems())
    def component_types_helper(self, visited):
        if contains_key(visited, self):
            return wrap([])
        add_key(visited, self)
        return union(
            [wrap([self])] +
            [t.component_types_helper(visited) for t in self.def_types.itervalues()])

class Program(Type):
    def __init__(self, def_types):
        self.def_types = def_types
    def __eq__(self, other):
        if not is_same_class(self, other):
            return False
        if len(self.def_types) != len(other.def_types):
            return False
        for self_def, other_def in zip(self.def_types, other.def_types):
            if self_def != other_def:
                return False
    def hash_helper(self, depth):
        if depth > MAX_HASH_DEPTH:
            return 'program'
        return ('program', tuple(t.hash_helper(depth + 1) for t in self.def_types))
    def key(self):
        return 'program'
    def pretty(self):
        return '\n\n'.join(t.pretty() for t in self.def_types)
    def component_types_helper(self, visited):
        if contains_key(visited, self):
            return wrap([])
        add_key(visited, self)
        return union(
            [wrap([self])] +
            [t.component_types_helper(visited) for t in self.def_types])

def is_type(t):
    return isinstance(t, Type)

def is_void(t):
    return isinstance(t, Void)

def is_int(t):
    return isinstance(t, Int)

def is_integral(t):
    return isinstance(t, MachineDependentInteger) or isinstance(t, FixedSizeInteger)

def is_float(t):
    return isinstance(t, Float)

def is_double(t):
    return isinstance(t, Double)

def is_floating_point(t):
    return is_float(t) or is_double(t)

def is_numeric(t):
    return is_floating_point(t) or is_integral(t)

def is_bool(t):
    return isinstance(t, Bool)

def is_POD(t):
    return is_numeric(t) or is_bool(t)

def is_pointer(t):
    return isinstance(t, Pointer)

def is_reference(t):
    return isinstance(t, Reference) or isinstance(t, StackReference)

def is_struct(t):
    return isinstance(t, Struct) or isinstance(t, StructInstanceWithParams) or isinstance(t, StructInstanceWithParamsAndRegions)

def is_struct_instance(t):
    return isinstance(t, StructInstanceWithParams) or isinstance(t, StructInstanceWithParamsAndRegions)

def is_function(t):
    return isinstance(t, Function)

def is_foreign_coloring(t):
    return isinstance(t, ForeignColoring)

def is_foreign_context(t):
    return isinstance(t, ForeignContext)

def is_foreign_pointer(t):
    return isinstance(t, ForeignPointer)

def is_foreign_runtime(t):
    return isinstance(t, ForeignRuntime)

def is_foreign_region(t):
    return isinstance(t, ForeignRegion)

def is_foreign_function(t):
    return isinstance(t, ForeignFunction)

def is_ispace(t):
    return isinstance(t, Ispace)

def is_ispace_kind(t):
    return isinstance(t, IspaceKind)

def is_region(t):
    return isinstance(t, Region)

def is_region_wild(t):
    return isinstance(t, RegionWild)

def is_region_kind(t):
    return isinstance(t, RegionKind)

def is_coloring(t):
    return isinstance(t, Coloring)

def is_partition(t):
    return isinstance(t, Partition)

def is_partition_kind(t):
    return isinstance(t, PartitionKind)

def is_module(t):
    return isinstance(t, Module)

def is_program(t):
    return isinstance(t, Program)

def is_kind(t):
    return isinstance(t, Kind)

def is_concrete(t):
    return t.is_concrete()

def allows_var_binding(t):
    return not (is_region(t) or is_ispace(t))

def type_name(t):
    return t.cname

###
### Constraints
###

class Constraint:
    SUBREGION = '<='
    DISJOINT = '*'

    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs
    def __repr__(self):
        return '%s %s %s' % (self.lhs, self.op, self.rhs)
    def substitute_regions(self, region_map):
        if self.lhs in region_map and self.rhs in region_map:
            return Constraint(self.op, region_map[self.lhs], region_map[self.rhs])
        if self.lhs in region_map:
            return Constraint(self.op, region_map[self.lhs], self.rhs)
        if self.rhs in region_map:
            return Constraint(self.op, self.lhs, region_map[self.rhs])
        return self

def make_constraint_graph(constraints, op, symmetric = False, inverse = False):
    assert not (symmetric and inverse)
    forward_edges = [(constraint.lhs, constraint.rhs)
                     for constraint in constraints if constraint.op == op]
    backward_edges = [(constraint.rhs, constraint.lhs)
                      for constraint in constraints if constraint.op == op]
    edges = (forward_edges if not inverse else []) + \
        (backward_edges if inverse or symmetric else [])
    graph = dict()
    for edge in edges:
        if edge[0] not in graph:
            graph[edge[0]] = set()
        graph[edge[0]].add(edge[1])
    return graph

def search_constraints_for_constraint_helper_inner(request, constraint_graph,
                                                   auxiliary_graph, region, visited):
    if region in visited:
        return False
    visited.add(region)

    if region in constraint_graph and request in constraint_graph[region]:
        return True

    if region in constraint_graph:
        found = any(
            search_constraints_for_constraint_helper_inner(
                request, constraint_graph, auxiliary_graph, next_region, visited)
            for next_region in constraint_graph[region])
        if found:
            return True

    if auxiliary_graph is not None and region in auxiliary_graph:
        found = any(
            search_constraints_for_constraint_helper_inner(
                request, constraint_graph, auxiliary_graph, next_region, visited)
            for next_region in auxiliary_graph[region])
        if found:
            return True

    return False

def search_constraints_for_constraint_helper(request, constraint_graph,
                                             auxiliary_graph, region, visited):
    if request in visited:
        return False
    visited.add(request)

    if search_constraints_for_constraint_helper_inner(
        request, constraint_graph, auxiliary_graph, region, set()):
        return True

    if auxiliary_graph is not None and request in auxiliary_graph:
        found = any(
            search_constraints_for_constraint_helper(
                next_request, constraint_graph, auxiliary_graph, region, visited)
            for next_request in auxiliary_graph[request])
        if found:
            return True

    return False

def search_constraints_for_constraint(request, constraints):
    if request.op == Constraint.SUBREGION and request.rhs == request.lhs:
        return True

    constraint_graph = make_constraint_graph(constraints, request.op, symmetric = request.op == Constraint.DISJOINT)
    auxiliary_graph = None
    if request.op == Constraint.DISJOINT:
        auxiliary_graph = make_constraint_graph(constraints, Constraint.SUBREGION)

    return search_constraints_for_constraint_helper(
        request.rhs, constraint_graph, auxiliary_graph, request.lhs, set())

def check_constraints(constraints_requested, constraints_available):
    for request in constraints_requested:
        if not search_constraints_for_constraint(request, constraints_available):
            return (False, request)
    return (True, None)

###
### Privileges
###

class Privilege:
    READ = 'reads'
    WRITE = 'writes'
    REDUCE = 'reduce'

    def __init__(self, privilege, op, region, field_path, allow_ispace = False):
        assert (privilege == Privilege.REDUCE) == (op is not None)
        assert (is_region(region) or (allow_ispace == True and is_ispace(region))) and isinstance(field_path, tuple)
        self.privilege = privilege
        self.op = op
        self.region = region
        self.field_path = field_path
    def __repr__(self):
        return self.pretty()
    def __eq__(self, other):
        return is_same_class(self, other) and \
            self.privilege == other.privilege and \
            self.op == other.op and \
            self.region == other.region and self.field_path == other.field_path
    def __ne__(self, other):
        return not self.__eq__(other)
    def __hash__(self):
        return hash((self.privilege, self.op, self.region, self.field_path))
    def pretty(self):
        if self.op is not None:
            return '%s<%s>(%s)' % (self.privilege, self.op, '.'.join((self.region.name,) + self.field_path))
        return '%s(%s)' % (self.privilege, '.'.join((self.region.name,) + self.field_path))
    def substitute_regions(self, region_map):
        if self.region in region_map:
            return Privilege(self.privilege, self.op, region_map[self.region], self.field_path)
        return self
    def as_ispace(self):
        if self.region.kind.ispace is not None:
            return Privilege(self.privilege, self.op, self.region.kind.ispace, self.field_path, True)
        return self

def is_privilege(p):
    return isinstance(p, Privilege)

def search_constraints_for_privilege_helper(request, privileges_available,
                                             constraint_graph, region, visited):
    if region in visited:
        return False

    for prefix in reversed(xrange(len(request.field_path) + 1)):
        field_path = request.field_path[:prefix]
        tentative_privilege = Privilege(request.privilege, request.op, region, field_path, True)
        if tentative_privilege in privileges_available:
            return True
    if region in constraint_graph:
        visited.add(region)
        return any(
            search_constraints_for_privilege_helper(
                request, privileges_available, constraint_graph,
                next_region, visited)
            for next_region in constraint_graph[region])
    return False

def search_constraints_for_privilege(request, privileges_available, constraints):
    constraint_graph = make_constraint_graph(constraints, Constraint.SUBREGION)
    visited = set()
    return search_constraints_for_privilege_helper(
        request, privileges_available,
        constraint_graph, request.region, visited)

def check_privileges(privileges_requested, cx):
    for request in privileges_requested:
        original_request = request

        # For arrays, map everything into index spaces.

        # FIXME: Theoretically this should be safe everywhere, but
        # something's getting messed up when I turn this on
        # globally. For now as a stopgap just turn it on for arrays.
        request = request
        privileges_available = cx.privileges
        if request.region.kind.ispace is not None:
            request = request.as_ispace()
            privileges_available = set(
                privilege.as_ispace()
                for privilege in cx.privileges
                if cx.region_forest.find(privilege.region) == cx.region_forest.find(original_request.region))

        if not search_constraints_for_privilege(request, privileges_available, cx.constraints):
            if request.privilege != Privilege.REDUCE:
                return (False, original_request)

            # If the initial search fails on a reduce request, search
            # again for read-write.
            read_request = Privilege(Privilege.READ, None, request.region, request.field_path, True)
            write_request = Privilege(Privilege.WRITE, None, request.region, request.field_path, True)
            if not (search_constraints_for_privilege(read_request, privileges_available, cx.constraints) and
                    search_constraints_for_privilege(write_request, privileges_available, cx.constraints)):
                return (False, original_request)
    return (True, None)
