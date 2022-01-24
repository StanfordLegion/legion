#!/usr/bin/env python3

# Copyright 2022 Stanford University, NVIDIA Corporation
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys, os, shutil, gc
import string, re
from math import sqrt, log
from getopt import getopt

GC_REF_KIND = 0
VALID_REF_KIND = 1
RESOURCE_REF_KIND = 2
REMOTE_REF_KIND = 3

# Help for python 2/3 foolishness
long_type = int if sys.version_info > (3,) else eval("long")

def iteritems(obj):
    return obj.items() if sys.version_info > (3,) else obj.viewitems()

def iterkeys(obj):
    return obj.keys() if sys.version_info > (3,) else obj.viewkeys()

def itervalues(obj):
    return obj.values() if sys.version_info > (3,) else obj.viewvalues()

prefix = r'\[(?P<realmnode>[0-9]+) - (?P<thread>[0-9a-f]+)\](?:\s+[0-9]+\.[0-9]+)? \{\w+\}\{legion_gc\}: '
# References
add_base_ref_pat = re.compile(prefix + r'GC Add Base Ref (?P<kind>[0-9]+) (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<src>[0-9]+) (?P<cnt>[0-9]+)')
add_nested_ref_pat = re.compile(prefix + r'GC Add Nested Ref (?P<kind>[0-9]+) (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<src>[0-9]+) (?P<cnt>[0-9]+)')
remove_base_ref_pat = re.compile(prefix + r'GC Remove Base Ref (?P<kind>[0-9]+) (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<src>[0-9]+) (?P<cnt>[0-9]+)')
remove_nested_ref_pat = re.compile(prefix + r'GC Remove Nested Ref (?P<kind>[0-9]+) (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<src>[0-9]+) (?P<cnt>[0-9]+)')
# Instances
inst_manager_pat = re.compile(prefix + r'GC Instance Manager (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<iid>[a-f0-9]+) (?P<mem>[a-f0-9]+)')
collective_manager_pat = re.compile(prefix + r'GC Collective Manager (?P<did>[0-9]+) (?P<node>[0-9]+)')
virtual_manager_pat = re.compile(prefix + r'GC Virtual Manager (?P<did>[0-9]+) (?P<node>[0-9]+)')
# Views
materialize_pat = re.compile(prefix + r'GC Materialized View (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<inst>[0-9]+)')
fill_pat = re.compile(prefix + r'GC Fill View (?P<did>[0-9]+) (?P<node>[0-9]+)')
phi_pat = re.compile(prefix + r'GC Phi View (?P<did>[0-9]+) (?P<node>[0-9]+)')
reduction_pat = re.compile(prefix + r'GC Reduction View (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<inst>[0-9]+)')
# Equivalence Set 
equivalence_set_pat = re.compile(prefix + r'GC Equivalence Set (?P<did>[0-9]+) (?P<node>[0-9]+)')
# Future
future_pat = re.compile(prefix + r'GC Future (?P<did>[0-9]+) (?P<node>[0-9]+)')
# Future Map
future_map_pat = re.compile(prefix + r'GC Future Map (?P<did>[0-9]+) (?P<node>[0-9]+)')
# Constraints
constraints_pat = re.compile(prefix + r'GC Constraints (?P<did>[0-9]+) (?P<node>[0-9]+)')
# Region Tree
index_space_pat = re.compile(prefix + r'GC Index Space (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<handle>[0-9]+)')
index_part_pat  = re.compile(prefix + r'GC Index Partition (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<handle>[0-9]+)')
index_expr_pat  = re.compile(prefix + r'GC Index Expr (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<handle>[0-9]+)')
field_space_pat = re.compile(prefix + r'GC Field Space (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<handle>[0-9]+)')
region_pat      = re.compile(prefix + r'GC Region (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<is>[0-9]+) (?P<fs>[0-9]+) (?P<tid>[0-9]+)')
partition_pat   = re.compile(prefix + r'GC Partition (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<ip>[0-9]+) (?P<fs>[0-9]+) (?P<tid>[0-9]+)')
# Source Kinds
source_kind_pat = re.compile(prefix + r'GC Source Kind (?P<kind>[0-9]+) (?P<name>[0-9a-zA-Z_ ]+)')
# Deletion Pattern
deletion_pat = re.compile(prefix + r'GC Deletion (?P<did>[0-9]+) (?P<node>[0-9]+)')

class TracePrinter(object):
    def __init__(self):
        self.depth = 0

    def down(self):
        self.depth += 1

    def up(self):
        assert self.depth > 0
        self.depth -= 1

    def print_base(self, obj):
        self.println(repr(obj))

    def println(self, line):
        for idx in range(self.depth):
            line = '  '+line
        print(line)

class Base(object):
    def __init__(self, did, node):
        self.did = did
        self.node = node
        self.base_gc_refs = {}
        self.base_gc_adds = {}
        self.base_gc_rems = {}
        self.base_valid_refs = {}
        self.base_valid_adds = {}
        self.base_valid_rems = {}
        self.base_remote_refs = {}
        self.base_remote_adds = {}
        self.base_remote_rems = {}
        self.base_resource_refs = {}
        self.base_resource_adds = {}
        self.base_resource_rems = {}
        self.nested_gc_refs = {}
        self.nested_gc_adds = {}
        self.nested_gc_rems = {}
        self.nested_valid_refs = {}
        self.nested_valid_adds = {}
        self.nested_valid_rems = {}
        self.nested_remote_refs = {}
        self.nested_remote_adds = {}
        self.nested_remote_rems = {}
        self.nested_resource_refs = {}
        self.nested_resource_adds = {}
        self.nested_resource_rems = {}
        self.on_stack = False
        self.deleted = False
        self.checked = False

    def add_base_ref(self, kind, src, cnt):
        if kind is GC_REF_KIND:
            if src in self.base_gc_refs:
                self.base_gc_refs[src] += cnt
                self.base_gc_adds[src] += cnt
            else:
                self.base_gc_refs[src] = cnt
                self.base_gc_adds[src] = cnt
                self.base_gc_rems[src] = 0
        elif kind is VALID_REF_KIND:
            if src in self.base_valid_refs:
                self.base_valid_refs[src] += cnt
                self.base_valid_adds[src] += cnt
            else:
                self.base_valid_refs[src] = cnt
                self.base_valid_adds[src] = cnt
                self.base_valid_rems[src] = 0
        elif kind is REMOTE_REF_KIND:
            if src in self.base_remote_refs:
                self.base_remote_refs[src] += cnt
                self.base_remote_adds[src] += cnt
            else:
                self.base_remote_refs[src] = cnt
                self.base_remote_adds[src] = cnt
                self.base_remote_rems[src] = 0
        elif kind is RESOURCE_REF_KIND:
            if src in self.base_resource_refs:
                self.base_resource_refs[src] += cnt
                self.base_resource_adds[src] += cnt
            else:
                self.base_resource_refs[src] = cnt
                self.base_resource_adds[src] = cnt
                self.base_resource_rems[src] = 0
        else:
            print('BAD BASE REF '+str(kind))
            assert False

    def add_nested_ref(self, kind, src, cnt):
        if kind is GC_REF_KIND:
            if src in self.nested_gc_refs:
                self.nested_gc_refs[src] += cnt
                self.nested_gc_adds[src] += cnt
            else:
                self.nested_gc_refs[src] = cnt
                self.nested_gc_adds[src] = cnt
                self.nested_gc_rems[src] = 0
        elif kind is VALID_REF_KIND:
            if src in self.nested_valid_refs:
                self.nested_valid_refs[src] += cnt
                self.nested_valid_adds[src] += cnt
            else:
                self.nested_valid_refs[src] = cnt
                self.nested_valid_adds[src] = cnt
                self.nested_valid_rems[src] = 0
        elif kind is REMOTE_REF_KIND:
            if src in self.nested_remote_refs:
                self.nested_remote_refs[src] += cnt
                self.nested_remote_adds[src] += cnt
            else:
                self.nested_remote_refs[src] = cnt
                self.nested_remote_adds[src] = cnt
                self.nested_remote_rems[src] = 0
        elif kind is RESOURCE_REF_KIND:
            if src in self.nested_resource_refs:
                self.nested_resource_refs[src] += cnt
                self.nested_resource_adds[src] += cnt
            else:
                self.nested_resource_refs[src] = cnt
                self.nested_resource_adds[src] = cnt
                self.nested_resource_rems[src] = 0
        else:
            assert False

    def remove_base_ref(self, kind, src, cnt):
        if kind is GC_REF_KIND:
            if src in self.base_gc_refs:
                self.base_gc_refs[src] -= cnt
                self.base_gc_rems[src] += cnt
            else:
                self.base_gc_refs[src] = -cnt
                self.base_gc_adds[src] = 0
                self.base_gc_rems[src] = cnt
        elif kind is VALID_REF_KIND:
            if src in self.base_valid_refs:
                self.base_valid_refs[src] -= cnt
                self.base_valid_rems[src] += cnt
            else:
                self.base_valid_refs[src] = -cnt
                self.base_valid_adds[src] = 0
                self.base_valid_rems[src] = cnt
        elif kind is REMOTE_REF_KIND:
            if src in self.base_remote_refs:
                self.base_remote_refs[src] -= cnt
                self.base_remote_rems[src] += cnt
            else:
                self.base_remote_refs[src] = -cnt
                self.base_remote_adds[src] = 0
                self.base_remote_rems[src] = cnt
        elif kind is RESOURCE_REF_KIND:
            if src in self.base_resource_refs:
                self.base_resource_refs[src] -= cnt
                self.base_resource_rems[src] += cnt
            else:
                self.base_resource_refs[src] = -cnt
                self.base_resource_adds[src] = 0
                self.base_resource_rems[src] = cnt
        else:
            assert False

    def remove_nested_ref(self, kind, src, cnt):
        if kind is GC_REF_KIND:
            if src in self.nested_gc_refs:
                self.nested_gc_refs[src] -= cnt
                self.nested_gc_rems[src] += cnt
            else:
                self.nested_gc_refs[src] = -cnt
                self.nested_gc_adds[src] = 0
                self.nested_gc_rems[src] = cnt
        elif kind is VALID_REF_KIND:
            if src in self.nested_valid_refs:
                self.nested_valid_refs[src] -= cnt
                self.nested_valid_rems[src] += cnt
            else:
                self.nested_valid_refs[src] = -cnt
                self.nested_valid_adds[src] = 0
                self.nested_valid_rems[src] = cnt
        elif kind is REMOTE_REF_KIND:
            if src in self.nested_remote_refs:
                self.nested_remote_refs[src] -= cnt
                self.nested_remote_rems[src] += cnt
            else:
                self.nested_remote_refs[src] = -cnt
                self.nested_remote_adds[src] = 0
                self.nested_remote_rems[src] = cnt
        elif kind is RESOURCE_REF_KIND:
            if src in self.nested_resource_refs:
                self.nested_resource_refs[src] -= cnt
                self.nested_resource_rems[src] += cnt
            else:
                self.nested_resource_refs[src] = -cnt
                self.nested_resource_adds[src] = 0
                self.nested_resource_rems[src] = cnt
        else:
            assert False

    def clone(self, other):
        assert self.did == other.did
        self.base_gc_refs = other.base_gc_refs
        self.base_gc_adds = other.base_gc_adds
        self.base_gc_rems = other.base_gc_rems
        self.base_valid_refs = other.base_valid_refs
        self.base_valid_adds = other.base_valid_adds
        self.base_valid_rems = other.base_valid_rems
        self.base_remote_refs = other.base_remote_refs
        self.base_remote_adds = other.base_remote_adds
        self.base_remote_rems = other.base_remote_rems
        self.base_resource_refs = other.base_resource_refs
        self.base_resource_adds = other.base_resource_adds
        self.base_resource_rems = other.base_resource_rems
        self.nested_gc_refs = other.nested_gc_refs
        self.nested_gc_adds = other.nested_gc_adds
        self.nested_gc_rems = other.nested_gc_rems
        self.nested_valid_refs = other.nested_valid_refs
        self.nested_valid_adds = other.nested_valid_adds
        self.nested_valid_rems = other.nested_valid_rems
        self.nested_remote_refs = other.nested_remote_refs
        self.nested_remote_adds = other.nested_remote_adds
        self.nested_remote_rems = other.nested_remote_rems
        self.nested_resource_refs = other.nested_resource_refs
        self.nested_resource_adds = other.nested_resource_adds
        self.nested_resource_rems = other.nested_resource_rems
        self.deleted = other.deleted

    def update_nested_references(self, state):
        if self.nested_gc_refs:
            new_gc_refs = dict()
            new_gc_adds = dict()
            new_gc_rems = dict()
            for did,refs in iteritems(self.nested_gc_refs):
                src = state.get_obj(did, self.node)
                new_gc_refs[src] = refs
                new_gc_adds[src] = self.nested_gc_adds[did]
                new_gc_rems[src] = self.nested_gc_rems[did]
            self.nested_gc_refs = new_gc_refs
            self.nested_gc_adds = new_gc_adds
            self.nested_gc_rems = new_gc_rems
        if self.nested_valid_refs:
            new_valid_refs = dict()
            new_valid_adds = dict()
            new_valid_rems = dict()
            for did,refs in iteritems(self.nested_valid_refs):
                src = state.get_obj(did, self.node)
                new_valid_refs[src] = refs
                new_valid_adds[src] = self.nested_valid_adds[did]
                new_valid_rems[src] = self.nested_valid_rems[did]
            self.nested_valid_refs = new_valid_refs
            self.nested_valid_adds = new_valid_adds
            self.nested_valid_rems = new_valid_rems
        if self.nested_remote_refs:
            new_remote_refs = dict()
            new_remote_adds = dict()
            new_remote_rems = dict()
            for did,refs in iteritems(self.nested_remote_refs):
                src = state.get_obj(did, self.node)
                new_remote_refs[src] = refs
                new_remote_adds[src] = self.nested_remote_adds[did]
                new_remote_rems[src] = self.nested_remote_rems[did]
            self.nested_remote_refs = new_remote_refs
            self.nested_remote_adds = new_remote_adds
            self.nested_remote_rems = new_remote_rems
        if self.nested_resource_refs:
            new_resource_refs = dict()
            new_resource_adds = dict()
            new_resource_rems = dict()
            for did,refs in iteritems(self.nested_resource_refs):
                src = state.get_obj(did, self.node)
                new_resource_refs[src] = refs
                new_resource_adds[src] = self.nested_resource_adds[did]
                new_resource_rems[src] = self.nested_resource_rems[did]
            self.nested_resource_refs = new_resource_refs
            self.nested_resource_adds = new_resource_adds
            self.nested_resource_rems = new_resource_rems

    def check_for_leaks(self, assert_on_error, verbose):
        if self.deleted:
            if verbose:
                print("----------------------------------------------------------------")
                print(str(self)+' was properly deleted')
                printer = TracePrinter()
                self.report_references(printer, GC_REF_KIND, verbose)
                self.report_references(printer, VALID_REF_KIND, verbose)
                self.report_references(printer, REMOTE_REF_KIND, verbose)
                self.report_references(printer, RESOURCE_REF_KIND, verbose)
                print("----------------------------------------------------------------")
            else:
                print(str(self)+' was properly deleted')
            if isinstance(self,Manager):
                return (True,False)
            return True
        # Special case if this is an instance that the user pinned
        # then we don't need to report this as an error
        if isinstance(self,Manager):
            is_pinned = False 
            for kind in iterkeys(self.base_valid_refs):
                if kind == 'Never GC Reference':
                    is_pinned = True
                    break
            if is_pinned:
                if verbose:
                    print("----------------------------------------------------------------")
                    print("INFO: "+str(self)+' was not deleted because it was pinned by the user')
                    printer = TracePrinter()
                    self.report_references(printer, GC_REF_KIND, verbose)
                    self.report_references(printer, VALID_REF_KIND, verbose)
                    self.report_references(printer, REMOTE_REF_KIND, verbose)
                    self.report_references(printer, RESOURCE_REF_KIND, verbose)
                    print("----------------------------------------------------------------")
                else:
                    print("INFO: "+str(self)+' was not deleted because it was pinned by the user')
                return (False,True)
        print("----------------------------------------------------------------")
        print("ERROR: "+str(self)+" was not properly deleted")
        printer = TracePrinter()
        self.report_references(printer, GC_REF_KIND, verbose)
        self.report_references(printer, VALID_REF_KIND, verbose)
        self.report_references(printer, REMOTE_REF_KIND, verbose)
        self.report_references(printer, RESOURCE_REF_KIND, verbose)
        print("----------------------------------------------------------------")
        if isinstance(self,Manager):
            return (False,False)
        return False

    def report_references(self, printer, kind, verbose):
        printer.down()
        if kind == GC_REF_KIND and self.base_gc_refs:
            for src,refs in iteritems(self.base_gc_refs):
                if refs == 0:
                    if verbose:
                        printer.println('Empty (Refs=0): Base GC '+repr(src)+
                            ' (Adds='+str(self.base_gc_adds[src])+',Rems='+str(self.base_gc_rems[src])+')')
                else:
                    if verbose:
                        printer.println('NON-EMPTY (Refs='+str(refs)+'): Base GC '+repr(src)+
                            ' (Adds='+str(self.base_gc_adds[src])+',Rems='+str(self.base_gc_rems[src])+')')
                    else:
                        printer.println('Base GC '+repr(src)+' (Refs='+str(refs)+')')
        if kind == VALID_REF_KIND and self.base_valid_refs:
            for src,refs in iteritems(self.base_valid_refs):
                if refs == 0:
                    if verbose:
                        printer.println('Empty (Refs=0): Base Valid '+repr(src)+
                            ' (Adds='+str(self.base_valid_adds[src])+',Rems='+str(self.base_valid_rems[src])+')')
                else:
                    if verbose:
                        printer.println('NON-EMPTY (Refs='+str(refs)+'): Base Valid '+repr(src)+
                            ' (Adds='+str(self.base_valid_adds[src])+',Rems='+str(self.base_valid_rems[src])+')')
                    else:
                        printer.println('Base Valid '+repr(src)+' (Refs='+str(refs)+')')
        if kind == REMOTE_REF_KIND and self.base_remote_refs:
            for src,refs in iteritems(self.base_remote_refs):
                if refs == 0:
                    if verbose:
                        printer.println('Empty (Refs=0): Base Remote '+repr(src)+
                            ' (Adds='+str(self.base_remote_adds[src])+',Rems='+str(self.base_remote_rems[src])+')')
                else:
                    if verbose:
                        printer.println('NON-EMPTY (Refs='+str(refs)+'): Base Remote '+repr(src)+
                            ' (Adds='+str(self.base_remote_adds[src])+',Rems='+str(self.base_remote_rems[src])+')')
                    else:
                        printer.println('Base Remote '+repr(src)+' (Refs='+str(refs)+')')
        if kind == RESOURCE_REF_KIND and self.base_resource_refs:
            for src,refs in iteritems(self.base_resource_refs):
                if refs == 0:
                    if verbose:
                        printer.println('Empty (Refs=0): Base Resource '+repr(src)+
                            ' (Adds='+str(self.base_resource_adds[src])+',Rems='+str(self.base_resource_rems[src])+')')
                else:
                    if verbose:
                        printer.println('NON-EMPTY (Refs='+str(refs)+'): Base Resource '+repr(src)+
                            ' (Adds='+str(self.base_resource_adds[src])+',Rems='+str(self.base_resource_rems[src])+')')
                    else:
                        printer.println('Base Resource '+repr(src)+' (Refs='+str(refs)+')')
        if kind == GC_REF_KIND and self.nested_gc_refs:
            for src,refs in iteritems(self.nested_gc_refs):
                if refs == 0:
                    if verbose:
                        printer.println('Empty (Refs=0): Nested GC '+repr(src)+
                            ' (Adds='+str(self.nested_gc_adds[src])+',Rems='+str(self.nested_gc_rems[src])+')')
                    continue
                if verbose:
                    printer.println('NON-EMPTY (Refs='+str(refs)+'): Nested GC '+repr(src)+
                        ' (Adds='+str(self.nested_gc_adds[src])+',Rems='+str(self.nested_gc_rems[src])+')')    
                else:
                    printer.println('Nested GC '+repr(src)+' (Refs='+str(refs)+')')
                printer.down()
                src.report_references(printer, kind, verbose)
                printer.up()
        if kind == VALID_REF_KIND and self.nested_valid_refs:
            for src,refs in iteritems(self.nested_valid_refs):
                if refs == 0:
                    if verbose:
                        printer.println('Empty (Refs=0): Nested Valid '+repr(src)+
                            ' (Adds='+str(self.nested_valid_adds[src])+',Rems='+str(self.nested_valid_rems[src])+')')
                    continue
                if verbose:
                    printer.println('NON-EMPTY (Refs='+str(refs)+'): Nested Valid '+repr(src)+
                        ' (Adds='+str(self.nested_valid_adds[src])+',Rems='+str(self.nested_valid_rems[src])+')')
                else:
                    printer.println('Nested Valid '+repr(src)+' (Refs='+str(refs)+')')
                printer.down()
                src.report_references(printer, kind, verbose)
                printer.up()
        if kind == REMOTE_REF_KIND and self.nested_remote_refs:
            for src,refs in iteritems(self.nested_remote_refs):
                if refs == 0:
                    if verbose:
                        printer.println('Empty (Refs=0): Nested Remote '+repr(src)+
                            ' (Adds='+str(self.nested_remote_adds[src])+',Rems='+str(self.nested_remote_rems[src])+')')
                    continue
                if verbose:
                    printer.println('NON-EMPTY (Refs='+str(refs)+'): Nested Remote '+repr(src)+
                        ' (Adds='+str(self.nested_remote_adds[src])+',Rems='+str(self.nested_remote_rems[src])+')')
                else:
                    printer.println('Nested Remote '+repr(src)+' (Refs='+str(refs)+')')
                printer.down()
                src.report_references(printer, kind, verbose)
                printer.up()
        if kind == RESOURCE_REF_KIND and self.nested_resource_refs:
            for src,refs in iteritems(self.nested_resource_refs):
                if refs == 0:
                    if verbose:
                        printer.println('Empty (Refs=0): Nested Resource '+repr(src)+
                            ' (Adds='+str(self.nested_resource_adds[src])+',Rems='+str(self.nested_resource_rems[src])+')')
                    continue
                if verbose:
                    printer.println('NON-EMPTY (Refs='+str(refs)+'): Nested Resource '+repr(src)+
                        ' (Adds='+str(self.nested_resource_adds[src])+',Rems='+str(self.nested_resource_rems[src])+')')
                else:
                    printer.println('Nested Resource '+repr(src)+' (Refs='+str(refs)+')')
                printer.down()
                src.report_references(printer, kind, verbose)
                printer.up()
        printer.up()
        return False

    def check_for_cycles(self, assert_on_error):
        stack = list()
        self.check_for_cycles_by_kind(assert_on_error, stack, GC_REF_KIND)
        stack = list()
        self.check_for_cycles_by_kind(assert_on_error, stack, VALID_REF_KIND)
        stack = list()
        self.check_for_cycles_by_kind(assert_on_error, stack, REMOTE_REF_KIND)
        stack = list()
        self.check_for_cycles_by_kind(assert_on_error, stack, RESOURCE_REF_KIND)

    def check_for_cycles_by_kind(self, assert_on_error, stack, kind):
        if self.on_stack:
            print('CYCLE DETECTED!')
            for obj in stack:
                print('  '+repr(obj))
            if assert_on_error:
                assert False
            else:
                print('Exiting...')
                sys.exit(0)
        stack.append(self)
        self.on_stack = True
        if kind == GC_REF_KIND and self.nested_gc_refs:
            for src,refs in iteritems(self.nested_gc_refs):
                if refs == 0:
                    continue
                src.check_for_cycles_by_kind(assert_on_error, stack, kind)
        if kind == VALID_REF_KIND and self.nested_valid_refs:
            for src,refs in iteritems(self.nested_valid_refs):
                if refs == 0:
                    continue
                src.check_for_cycles_by_kind(assert_on_error, stack, kind)
        if kind == REMOTE_REF_KIND and self.nested_remote_refs:
            for src,refs in iteritems(self.nested_remote_refs):
                if refs == 0:
                    continue
                src.check_for_cycles_by_kind(assert_on_error, stack, kind)
        if kind == RESOURCE_REF_KIND and self.nested_resource_refs:
            for src,refs in iteritems(self.nested_resource_refs):
                if refs == 0:
                    continue
                src.check_for_cycles_by_kind(assert_on_error, stack, kind)
        stack.pop()
        self.on_stack = False

class Manager(Base):
    def __init__(self, did, node):
        super(Manager,self).__init__(did, node)
        self.instance = None

    def add_inst(self, inst):
        self.instance = inst

    def __repr__(self):
        result = 'Manager '+str(self.did)+' (Node='+str(self.node)+')'
        if self.instance is not None:
            result += ' '+repr(self.instance)
        return result

class View(Base):
    def __init__(self, did, node, kind):
        super(View,self).__init__(did, node)
        self.kind = kind
        self.manager = None

    def add_manager(self, manager):
        self.manager = manager

    def __repr__(self):
        result = self.kind +' View '+str(self.did)+' (Node='+str(self.node)+')'
        return result

class Future(Base):
    def __init__(self, did, node):
        super(Future,self).__init__(did, node)

    def __repr__(self):
        return 'Future '+str(self.did)+' (Node='+str(self.node)+')'

class FutureMap(Base):
    def __init__(self, did, node):
        super(FutureMap,self).__init__(did, node)

    def __repr__(self):
        return 'Future Map '+str(self.did)+' (Node='+str(self.node)+')'

class EquivalenceSet(Base):
    def __init__(self, did, node):
        super(EquivalenceSet,self).__init__(did, node)

    def __repr__(self):
        return 'Equivalence Set '+str(self.did)+' (Node='+str(self.node)+')'

class Constraints(Base):
    def __init__(self, did, node):
        super(Constraints,self).__init__(did, node)

    def __repr__(self):
        return 'Layout Constraints '+str(self.did)+' (Node='+str(self.node)+')'

class IndexSpace(Base):
    def __init__(self, did, node, handle):
        super(IndexSpace,self).__init__(did, node)
        self.handle = handle

    def __repr__(self):
        return 'Index Space '+str(self.did)+' (Node='+str(self.node)+') Handle '+str(self.handle)
        

class IndexPartition(Base):
    def __init__(self, did, node, handle):
        super(IndexPartition,self).__init__(did, node)
        self.handle = handle

    def __repr__(self):
        return 'Index Partition '+str(self.did)+' (Node='+str(self.node)+') Handle '+str(self.handle)

class IndexExpression(Base):
    def __init__(self, did, node, handle):
        super(IndexExpression,self).__init__(did, node)
        self.handle = handle

    def __repr__(self):
        return 'Index Expression '+str(self.did)+' (Node='+str(self.node)+') ExprID '+str(self.handle)

class FieldSpace(Base):
    def __init__(self, did, node, handle):
        super(FieldSpace,self).__init__(did, node)
        self.handle = handle

    def __repr__(self):
        return 'Field Space '+str(self.did)+' (Node='+str(self.node)+') Handle '+str(self.handle)

class Region(Base):
    def __init__(self, did, node, index_space, field_space, tid):
        super(Region,self).__init__(did, node)
        self.index_space = index_space
        self.field_space = field_space
        self.tree_id = tid

    def __repr__(self):
        return 'Region '+str(self.did)+' (Node='+str(self.node)+') ('+str(self.index_space)+','+str(self.field_space)+','+str(self.tree_id)+')'

class Partition(Base):
    def __init__(self, did, node, index_partition, field_space, tid):
        super(Partition,self).__init__(did, node)
        self.index_partition = index_partition
        self.field_space = field_space
        self.tree_id = tid

    def __repr__(self):
        return 'Partition '+str(self.did)+' (Node='+str(self.node)+') ('+str(self.index_partition)+','+str(self.field_space)+','+str(self.tree_id)+')'

class Instance(object):
    def __init__(self, iid, mem, kind):
        self.iid = iid
        self.mem = mem
        self.kind = kind

    def __repr__(self):
        return self.kind + ' Instance '+hex(self.iid)

class State(object):
    def __init__(self):
        self.managers = {}
        self.views = {}
        self.futures = {}
        self.future_maps = {}
        self.unknowns = {}
        self.instances = {}
        self.src_names = {}
        self.constraints = {}
        self.equivalence_sets = {}
        self.index_spaces = {}
        self.index_partitions = {}
        self.index_expressions = {}
        self.field_spaces = {}
        self.regions = {}
        self.partitions = {}

    def parse_log_file(self, file_name):
        with open(file_name, 'rb') as log:
            matches = 0
            for line in log:
                matches += 1
                # Python 2/3 foolishness
                if sys.version_info > (3,):
                    line = line.decode('utf-8')
                m = add_base_ref_pat.match(line)
                if m is not None:
                    self.log_add_base_ref(int(m.group('kind')),
                                          long_type(m.group('did')),
                                          long_type(m.group('node')),
                                          int(m.group('src')),
                                          int(m.group('cnt')))
                    continue
                m = add_nested_ref_pat.match(line)
                if m is not None:
                    self.log_add_nested_ref(int(m.group('kind')),
                                            long_type(m.group('did')),
                                            long_type(m.group('node')),
                                            long_type(m.group('src')),
                                            int(m.group('cnt')))
                    continue
                m = remove_base_ref_pat.match(line)
                if m is not None:
                    self.log_remove_base_ref(int(m.group('kind')),
                                             long_type(m.group('did')),
                                             long_type(m.group('node')),
                                             int(m.group('src')),
                                             int(m.group('cnt')))
                    continue
                m = remove_nested_ref_pat.match(line)
                if m is not None:
                    self.log_remove_nested_ref(int(m.group('kind')),
                                               long_type(m.group('did')),
                                               long_type(m.group('node')),
                                               long_type(m.group('src')),
                                               int(m.group('cnt')))
                    continue
                m = inst_manager_pat.match(line)
                if m is not None:
                    self.log_inst_manager(long_type(m.group('did')),
                                          long_type(m.group('node')), 
                                          long_type(m.group('iid'),16),
                                          long_type(m.group('mem'),16))
                    continue
                m = collective_manager_pat.match(line)
                if m is not None:
                    self.log_collective_manager(long_type(m.group('did')),
                                                long_type(m.group('node')))
                    continue
                m = virtual_manager_pat.match(line)
                if m is not None:
                    self.log_virtual_manager(long_type(m.group('did')),
                                             long_type(m.group('node')))
                    continue
                m = materialize_pat.match(line)
                if m is not None:
                    self.log_materialized_view(long_type(m.group('did')),
                                               long_type(m.group('node')),
                                               long_type(m.group('inst')))
                    continue
                m = fill_pat.match(line)
                if m is not None:
                    self.log_fill_view(long_type(m.group('did')),
                                       long_type(m.group('node')))
                    continue
                m = phi_pat.match(line)
                if m is not None:
                    self.log_phi_view(long_type(m.group('did')),
                                      long_type(m.group('node')))
                m = reduction_pat.match(line)
                if m is not None:
                    self.log_reduction_view(long_type(m.group('did')),
                                            long_type(m.group('node')),
                                            long_type(m.group('inst')))
                    continue
                m = equivalence_set_pat.match(line)
                if m is not None:
                    self.log_equivalence_set(long_type(m.group('did')),
                                             long_type(m.group('node')))
                    continue
                m = future_pat.match(line)
                if m is not None:
                    self.log_future(long_type(m.group('did')),
                                    long_type(m.group('node')))
                    continue
                m = future_map_pat.match(line)
                if m is not None:
                    self.log_future_map(long_type(m.group('did')),
                                        long_type(m.group('node')))
                    continue
                m = constraints_pat.match(line)
                if m is not None:
                    self.log_constraints(long_type(m.group('did')),
                                         long_type(m.group('node')))
                    continue
                m = index_space_pat.match(line)
                if m is not None:
                    self.log_index_space(long_type(m.group('did')),
                                         long_type(m.group('node')),
                                         long_type(m.group('handle')))
                    continue
                m = index_part_pat.match(line)
                if m is not None:
                    self.log_index_partition(long_type(m.group('did')),
                                             long_type(m.group('node')),
                                             long_type(m.group('handle')))
                    continue
                m = index_expr_pat.match(line)
                if m is not None:
                    self.log_index_expression(long_type(m.group('did')),
                                              long_type(m.group('node')),
                                              long_type(m.group('handle')))
                    continue
                m = field_space_pat.match(line)
                if m is not None:
                    self.log_field_space(long_type(m.group('did')),
                                         long_type(m.group('node')),
                                         long_type(m.group('handle')))
                    continue
                m = region_pat.match(line)
                if m is not None:
                    self.log_region(long_type(m.group('did')),
                                    long_type(m.group('node')),
                                    long_type(m.group('is')),
                                    long_type(m.group('fs')),
                                    long_type(m.group('tid')))
                    continue
                m = partition_pat.match(line)
                if m is not None:
                    self.log_partition(long_type(m.group('did')),
                                       long_type(m.group('node')),
                                       long_type(m.group('ip')),
                                       long_type(m.group('fs')),
                                       long_type(m.group('tid')))
                    continue
                m = source_kind_pat.match(line)
                if m is not None:
                    self.log_source_kind(int(m.group('kind')),
                                         m.group('name'))
                    continue
                m = deletion_pat.match(line)
                if m is not None:
                    self.log_deletion(long_type(m.group('did')),
                                      long_type(m.group('node')))
                    continue
                matches -= 1
                print('Skipping unmatched line: '+line)
        return matches

    def post_parse(self):
        # Delete the virtual instance it is special
        to_del = list()
        for key,val in iteritems(self.unknowns):
            if key[0] == 0:
                to_del.append(key)
        for key in to_del:
            del self.unknowns[key]
        if self.unknowns:
            print("WARNING: Found %d unknown objects!" % len(self.unknowns))
            for did in iterkeys(self.unknowns):
                print('  Unknown DID '+str(hex(did[0]))+' on node '+str(did[1]))
        # Now update all the pointers to references
        for man in itervalues(self.managers):
            man.update_nested_references(self)
        for view in itervalues(self.views):
            view.update_nested_references(self)
        for future in itervalues(self.futures):
            future.update_nested_references(self)
        for future_map in itervalues(self.future_maps):
            future_map.update_nested_references(self)
        for constraint in itervalues(self.constraints):
            constraint.update_nested_references(self)
        for state in itervalues(self.equivalence_sets):
            state.update_nested_references(self)
        for index_space in itervalues(self.index_spaces):
            index_space.update_nested_references(self)
        for index_part in itervalues(self.index_partitions):
            index_part.update_nested_references(self)
        for index_expr in itervalues(self.index_expressions):
            index_expr.update_nested_references(self)
        for field_space in itervalues(self.field_spaces):
            field_space.update_nested_references(self)
        for region in itervalues(self.regions):
            region.update_nested_references(self)
        for partition in itervalues(self.partitions):
            partition.update_nested_references(self)
        # Run the garbage collector
        gc.collect()

    def log_add_base_ref(self, kind, did, node, src, cnt):
        obj = self.get_obj(did, node) 
        assert src in self.src_names
        obj.add_base_ref(kind, self.src_names[src], cnt)

    def log_add_nested_ref(self, kind, did, node, src, cnt):
        obj = self.get_obj(did, node)
        obj.add_nested_ref(kind, src, cnt)

    def log_remove_base_ref(self, kind, did, node, src, cnt):
        obj = self.get_obj(did, node)
        assert src in self.src_names
        obj.remove_base_ref(kind, self.src_names[src], cnt)

    def log_remove_nested_ref(self, kind, did, node, src, cnt):
        obj = self.get_obj(did, node)
        obj.remove_nested_ref(kind, src, cnt)

    def log_inst_manager(self, did, node, iid, mem):
        inst = self.get_instance(iid, mem, 'Individual')
        manager = self.get_manager(did, node)
        manager.add_inst(inst)

    def log_collective_manager(self, did, node):
        inst = Instance(0, 0, 'Collective')
        manager = self.get_manager(did, node)
        manager.add_inst(inst)

    def log_virtual_manager(self, did, node):
        inst = Instance(0, 0, 'Virtual')
        manager = self.get_manager(did, node)
        manager.add_inst(inst)

    def log_materialized_view(self, did, node, inst):
        manager = self.get_manager(inst, node)
        view = self.get_view(did, node, 'Materialized')
        view.add_manager(manager)

    def log_fill_view(self, did, node):
        self.get_view(did, node, 'Fill')

    def log_phi_view(self, did, node):
        self.get_view(did, node, 'Phi')

    def log_reduction_view(self, did, node, inst):
        manager = self.get_manager(inst, node)
        view = self.get_view(did, node, 'Reduction')
        view.add_manager(manager)

    def log_equivalence_set(self, did, node):
        self.get_equivalence_set(did, node);

    def log_future(self, did, node):
        self.get_future(did, node)

    def log_future_map(self, did, node):
        self.get_future_map(did, node)

    def log_constraints(self, did, node):
        self.get_constraints(did, node)

    def log_index_space(self, did, node, handle):
        self.get_index_space(did, node, handle)

    def log_index_partition(self, did, node, handle):
        self.get_index_partition(did, node, handle)

    def log_index_expression(self, did, node, handle):
        self.get_index_expression(did, node, handle)

    def log_field_space(self, did, node, handle):
        self.get_field_space(did, node, handle)

    def log_region(self, did, node, index_space, field_space, tid):
        self.get_region(did, node, index_space, field_space, tid)

    def log_partition(self, did, node, index_partition, field_space, tid):
        self.get_partition(did, node, index_partition, field_space, tid)

    def log_source_kind(self, kind, name):
        if kind not in self.src_names:
            self.src_names[kind] = name

    def log_deletion(self, did, node):
        obj = self.get_obj(did, node)
        obj.deleted = True

    def get_instance(self, iid, mem, kind):
        if iid not in self.instances:
            self.instances[iid] = Instance(iid, mem, kind)
        return self.instances[iid]

    def get_manager(self, did, node):
        key = (did,node)
        if key not in self.managers:
            self.managers[key] = Manager(did, node)
            if key in self.unknowns:
                self.managers[key].clone(self.unknowns[key])
                del self.unknowns[key]
        return self.managers[key]

    def get_view(self, did, node, kind):
        key = (did,node)
        if key not in self.views:
            self.views[key] = View(did, node, kind)
            if key in self.unknowns:
                self.views[key].clone(self.unknowns[key])
                del self.unknowns[key]
        return self.views[key]

    def get_equivalence_set(self, did, node):
        key = (did,node)
        if key not in self.equivalence_sets:
            self.equivalence_sets[key] = EquivalenceSet(did, node)
            if key in self.unknowns:
                self.equivalence_sets[key].clone(self.unknowns[key])
                del self.unknowns[key]
        return self.equivalence_sets[key]

    def get_future(self, did, node):
        key = (did,node)
        if key not in self.futures:
            self.futures[key] = Future(did, node)
            if key in self.unknowns:
                self.futures[key].clone(self.unknowns[key])
                del self.unknowns[key]
        return self.futures[key]

    def get_future_map(self, did, node):
        key = (did,node)
        if key not in self.future_maps:
            self.future_maps[key] = FutureMap(did, node)
            if key in self.unknowns:
                self.future_maps[key].clone(self.unknowns[key])
                del self.unknowns[key]
        return self.future_maps[key]

    def get_constraints(self, did, node):
        key = (did,node)
        if key not in self.constraints:
            self.constraints[key] = Constraints(did, node)
            if key in self.unknowns:
                self.constraints[key].clone(self.unknowns[key])
                del self.unknowns[key]
        return self.constraints[key]

    def get_index_space(self, did, node, handle):
        key = (did,node)
        if key not in self.index_spaces:
            self.index_spaces[key] = IndexSpace(did, node, handle)
            if key in self.unknowns:
                self.index_spaces[key].clone(self.unknowns[key])
                del self.unknowns[key]
        return self.index_spaces[key]

    def get_index_partition(self, did, node, handle):
        key = (did,node)
        if key not in self.index_partitions:
            self.index_partitions[key] = IndexPartition(did, node, handle)
            if key in self.unknowns:
                self.index_partitions[key].clone(self.unknowns[key])
                del self.unknowns[key]
        return self.index_partitions[key]

    def get_index_expression(self, did, node, handle):
        key = (did,node)
        if key not in self.index_expressions:
            self.index_expressions[key] = IndexExpression(did, node, handle)
            if key in self.unknowns:
                self.index_expressions[key].clone(self.unknowns[key])
                del self.unknowns[key]
        return self.index_expressions[key]

    def get_field_space(self, did, node, handle):
        key = (did,node)
        if key not in self.field_spaces:
            self.field_spaces[key] = FieldSpace(did, node, handle)
            if key in self.unknowns:
                self.field_spaces[key].clone(self.unknowns[key])
                del self.unknowns[key]
        return self.field_spaces[key]

    def get_region(self, did, node, index_space, field_space, tid):
        key = (did,node)
        if key not in self.regions:
            self.regions[key] = Region(did, node, index_space, field_space, tid)
            if key in self.unknowns:
                self.regions[key].clone(self.unknowns[key])
                del self.unknowns[key]
        return self.regions[key]

    def get_partition(self, did, node, index_partition, field_space, tid):
        key = (did,node)
        if key not in self.partitions:
            self.partitions[key] = Partition(did, node, index_partition, field_space, tid)
            if key in self.unknowns:
                self.partitions[key].clone(self.unknowns[key])
                del self.unknowns[key]
        return self.partitions[key]

    def get_obj(self, did, node):
        key = (did,node)
        if key in self.views:
            return self.views[key]
        if key in self.managers:
            return self.managers[key]
        if key in self.futures:
            return self.futures[key]
        if key in self.future_maps:
            return self.future_maps[key]
        if key in self.equivalence_sets:
            return self.equivalence_sets[key]
        if key in self.constraints:
            return self.constraints[key]
        if key in self.index_spaces:
            return self.index_spaces[key]
        if key in self.index_partitions:
            return self.index_partitions[key]
        if key in self.index_expressions:
            return self.index_expressions[key]
        if key in self.field_spaces:
            return self.field_spaces[key]
        if key in self.regions:
            return self.regions[key]
        if key in self.partitions:
            return self.partitions[key]
        if key in self.unknowns:
            return self.unknowns[key]
        self.unknowns[key] = Base(did, node)
        return self.unknowns[key]

    def check_for_cycles(self, assert_on_error):
        for did,manager in iteritems(self.managers):
            print("Checking for cycles in "+repr(manager))
            manager.check_for_cycles(assert_on_error)
        for did,view in iteritems(self.views):
            print("Checking for cycles in "+repr(view))
            view.check_for_cycles(assert_on_error)
        for did,future in iteritems(self.futures):
            print("Checking for cycles in "+repr(future))
            future.check_for_cycles(assert_on_error)
        for did,future_map in iteritems(self.future_maps):
            print("Checking for cycles in "+repr(future_map))
            future_map.check_for_cycles(assert_on_error)
        for did,eq in iteritems(self.equivalence_sets):
            print("Checking for cycles in "+repr(eq))
            eq.check_for_cycles(assert_on_error)
        for did,constraint in iteritems(self.constraints):
            print("Checking for cycles in "+repr(constraint))
            constraint.check_for_cycles(assert_on_error)
        for did,index_space in iteritems(self.index_spaces):
            print("Checking for cycles in "+repr(index_space))
            index_space.check_for_cycles(assert_on_error)
        for did,index_part in iteritems(self.index_partitions):
            print("Checking for cycles in "+repr(index_part))
            index_part.check_for_cycles(assert_on_error)
        for did,index_expr in iteritems(self.index_expressions):
            print("Checking for cycles in "+repr(index_expr))
            index_expr.check_for_cycles(assert_on_error)
        for did,field_space in iteritems(self.field_spaces):
            print("Checking for cycles in "+repr(field_space))
            field_space.check_for_cycles(assert_on_error)
        for did,region in iteritems(self.regions):
            print("Checking for cycles in "+repr(region))
            region.check_for_cycles(assert_on_error)
        for did,partition in iteritems(self.partitions):
            print("Checking for cycles in "+repr(partition))
        print("NO CYCLES")

    def check_for_leaks(self, assert_on_error, verbose): 
        leaked_futures = 0
        leaked_future_maps = 0
        leaked_constraints = 0
        leaked_managers = 0
        pinned_managers = 0
        leaked_views = 0
        leaked_eq_sets = 0
        leaked_index_spaces = 0
        leaked_index_partitions = 0
        leaked_index_expressions = 0
        leaked_field_spaces = 0
        leaked_regions = 0
        leaked_partitions = 0
        for future in itervalues(self.futures):
            if not future.check_for_leaks(assert_on_error, verbose):
                leaked_futures += 1
        for future_map in itervalues(self.future_maps):
            if not future_map.check_for_leaks(assert_on_error, verbose):
                leaked_future_maps += 1
        for constraint in itervalues(self.constraints):
            if not constraint.check_for_leaks(assert_on_error, verbose): 
                leaked_constraints += 1
        for manager in itervalues(self.managers):
            deleted,pinned = manager.check_for_leaks(assert_on_error, verbose)
            if not deleted:
                if pinned:
                    pinned_managers += 1
                else:
                    leaked_managers += 1
        for view in itervalues(self.views):
            if not view.check_for_leaks(assert_on_error, verbose):
                leaked_views += 1
        for eq in itervalues(self.equivalence_sets):
            if not eq.check_for_leaks(assert_on_error, verbose):
                leaked_eq_sets += 1
        for index_space in itervalues(self.index_spaces):
            if not index_space.check_for_leaks(assert_on_error, verbose):
                leaked_index_spaces += 1
        for index_part in itervalues(self.index_partitions):
            if not index_part.check_for_leaks(assert_on_error, verbose):
                leaked_index_partitions += 1
        for index_expr in itervalues(self.index_expressions):
            if not index_expr.check_for_leaks(assert_on_error, verbose):
                leaked_index_expressions += 1
        for field_space in itervalues(self.field_spaces):
            if not field_space.check_for_leaks(assert_on_error, verbose):
                leaked_field_spaces += 1
        for region in itervalues(self.regions):
            if not region.check_for_leaks(assert_on_error, verbose):
                leaked_regions += 1
        for partition in itervalues(self.partitions):
            if not partition.check_for_leaks(assert_on_error, verbose):
                leaked_partitions += 1
        print("LEAK SUMMARY")
        if leaked_futures > 0:
            print("  LEAKED FUTURES: "+str(leaked_futures))
            if assert_on_error: assert False
        else:
            print("  Leaked Futures: "+str(leaked_futures))
        if leaked_future_maps > 0:
            print("  LEAKD FUTURE MAPS: "+str(leaked_future_maps))
            if assert_on_error: assert False
        else:
            print("  Leaked Future Maps: "+str(leaked_future_maps))
        if leaked_constraints > 0:
            print("  LEAKED CONSTRAINTS: "+str(leaked_constraints))
            if assert_on_error: assert False
        else:
            print("  Leaked Constraints: "+str(leaked_constraints))
        if leaked_managers > 0:
            print("  LEAKED MANAGERS: "+str(leaked_managers))
            if assert_on_error: assert False
        else:
            print("  Leaked Managers: "+str(leaked_managers))
        if pinned_managers > 0:
            print("  PINNED MANAGERS: "+str(pinned_managers))
            if assert_on_error: assert False
        else:
            print("  Pinned Managers: "+str(pinned_managers))
        if leaked_views > 0:
            print("  LEAKED VIEWS: "+str(leaked_views))
            if assert_on_error: assert False
        else:
            print("  Leaked Views: "+str(leaked_views))
        if leaked_eq_sets > 0:
            print("  LEAKED EQUIVALENCE SETS: "+str(leaked_eq_sets))
            if assert_on_error: assert False
        else:
            print("  Leaked Equivalence Sets: "+str(leaked_eq_sets))
        if leaked_index_spaces > 0:
            print("  LEAKED INDEX SPACES: "+str(leaked_index_spaces))
            if assert_on_error: assert False
        else:
            print("  Leaked Index Spaces: "+str(leaked_index_spaces))
        if leaked_index_partitions > 0:
            print("  LEAKED INDEX PARTITIONS: "+str(leaked_index_partitions))
            if assert_on_error: assert False
        else:
            print("  Leaked Index Partitions: "+str(leaked_index_partitions))
        if leaked_index_expressions > 0:
            print("  LEAKED INDEX EXPRESSIONS: "+str(leaked_index_expressions))
            if assert_on_error: assert False
        else:
            print("  Leaked Index Expressions: "+str(leaked_index_expressions))
        if leaked_field_spaces > 0:
            print("  LEAKED FIELD SPACES: "+str(leaked_field_spaces))
            if assert_on_error: assert False
        else:
            print("  Leaked Field Spaces: "+str(leaked_field_spaces))
        if leaked_regions > 0:
            print("  LEAKED REGIONS: "+str(leaked_regions))
            if assert_on_error: assert False
        else:
            print("  Leaked Regions: "+str(leaked_regions))
        if leaked_partitions > 0:
            print("  LEAKED PARTITIONS: "+str(leaked_partitions))
            if assert_on_error: assert False
        else:
            print("  Leaked Partitions: "+str(leaked_partitions))

def main():
    parser = argparse.ArgumentParser(
        description='Legion GC analysis and verification')
    parser.add_argument(
        '-l', '--leaks', action='store_true',
        help='check for leaks')
    parser.add_argument(
        '-c', '--cycles', action='store_true',
        help='check for cycles')
    parser.add_argument(
        '-a', '--assert', dest='assert_on_error', action='store_true',
        help='assert on errors')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='verbose output')
    parser.add_argument(
        dest='filenames', nargs='+',
        help='input log filenames')
    args = parser.parse_args()

    check_cycles = args.cycles
    check_leaks = args.leaks
    assert_on_error = args.assert_on_error
    verbose = args.verbose

    file_names = args.filenames

    state = State()
    has_matches = False
    for file_name in file_names:
        print('Reading log file %s...' % file_name)
        total_matches = state.parse_log_file(file_name)
        print('Matched %s lines' % total_matches)
        if total_matches > 0:
            has_matches = True
    if not has_matches:
        print('No matches found! Exiting...')
        return

    state.post_parse()
  
    if check_cycles:
        state.check_for_cycles(assert_on_error)
    if check_leaks:
        state.check_for_leaks(assert_on_error, verbose)


if __name__ == '__main__':
    main()

