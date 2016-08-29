#!/usr/bin/env python

# Copyright 2016 Stanford University, NVIDIA Corporation
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

import sys, os, shutil, gc
import string, re
from math import sqrt, log
from getopt import getopt

GC_REF_KIND = 0
VALID_REF_KIND = 1
RESOURCE_REF_KIND = 2
REMOTE_REF_KIND = 3

prefix = r'\[(?P<realmnode>[0-9]+) - (?P<thread>[0-9a-f]+)\] \{\w+\}\{legion_gc\}: '
# References
add_base_ref_pat = re.compile(prefix + r'GC Add Base Ref (?P<kind>[0-9]+) (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<src>[0-9]+) (?P<cnt>[0-9]+)')
add_nested_ref_pat = re.compile(prefix + r'GC Add Nested Ref (?P<kind>[0-9]+) (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<src>[0-9]+) (?P<cnt>[0-9]+)')
remove_base_ref_pat = re.compile(prefix + r'GC Remove Base Ref (?P<kind>[0-9]+) (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<src>[0-9]+) (?P<cnt>[0-9]+)')
remove_nested_ref_pat = re.compile(prefix + r'GC Remove Nested Ref (?P<kind>[0-9]+) (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<src>[0-9]+) (?P<cnt>[0-9]+)')
# Instances
inst_manager_pat = re.compile(prefix + r'GC Instance Manager (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<iid>[a-f0-9]+) (?P<mem>[a-f0-9]+)')
list_manager_pat = re.compile(prefix + r'GC List Reduction Manager (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<iid>[a-f0-9]+) (?P<mem>[a-f0-9]+)')
fold_manager_pat = re.compile(prefix + r'GC Fold Reduction Manager (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<iid>[a-f0-9]+) (?P<mem>[a-f0-9]+)')
# Views
materialize_pat = re.compile(prefix + r'GC Materialized View (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<inst>[0-9]+)')
composite_pat = re.compile(prefix + r'GC Composite View (?P<did>[0-9]+) (?P<node>[0-9]+)')
fill_pat = re.compile(prefix + r'GC Fill View (?P<did>[0-9]+) (?P<node>[0-9]+)')
reduction_pat = re.compile(prefix + r'GC Reduction View (?P<did>[0-9]+) (?P<node>[0-9]+) (?P<inst>[0-9]+)')
# Version State
version_state_pat = re.compile(prefix + r'GC Version State (?P<did>[0-9]+) (?P<node>[0-9]+)')
# Future
future_pat = re.compile(prefix + r'GC Future (?P<did>[0-9]+) (?P<node>[0-9]+)')
# Constraints
constraints_pat = re.compile(prefix + r'GC Constraints (?P<did>[0-9]+) (?P<node>[0-9]+)')
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
        print line

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
            print 'BAD BASE REF '+str(kind)
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
            for did,refs in self.nested_gc_refs.iteritems():
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
            for did,refs in self.nested_valid_refs.iteritems():
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
            for did,refs in self.nested_remote_refs.iteritems():
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
            for did,refs in self.nested_resource_refs.iteritems():
                src = state.get_obj(did, self.node)
                new_resource_refs[src] = refs
                new_resource_adds[src] = self.nested_resource_adds[did]
                new_resource_rems[src] = self.nested_resource_rems[did]
            self.nested_resource_refs = new_resource_refs
            self.nested_resource_adds = new_resource_adds
            self.nested_resource_rems = new_resource_rems

    def check_for_leaks(self, verbose):
        if self.deleted:
            if verbose:
                print "----------------------------------------------------------------"
                print str(self)+' was properly deleted'
                printer = TracePrinter()
                self.report_references(printer, GC_REF_KIND, verbose)
                self.report_references(printer, VALID_REF_KIND, verbose)
                self.report_references(printer, REMOTE_REF_KIND, verbose)
                self.report_references(printer, RESOURCE_REF_KIND, verbose)
                print "----------------------------------------------------------------"
            else:
                print str(self)+' was properly deleted'
            if isinstance(self,Manager):
                return (True,False)
            return True
        # Special case if this is an instance that the user pinned
        # then we don't need to report this as an error
        if isinstance(self,Manager):
            is_pinned = False 
            for kind in self.base_valid_refs.iterkeys():
                if kind == 'Never GC Reference':
                    is_pinned = True
                    break
            if is_pinned:
                if verbose:
                    print "----------------------------------------------------------------"
                    print "INFO: "+str(self)+' was not deleted because it was pinned by the user'
                    printer = TracePrinter()
                    self.report_references(printer, GC_REF_KIND, verbose)
                    self.report_references(printer, VALID_REF_KIND, verbose)
                    self.report_references(printer, REMOTE_REF_KIND, verbose)
                    self.report_references(printer, RESOURCE_REF_KIND, verbose)
                    print "----------------------------------------------------------------"
                else:
                    print "INFO: "+str(self)+' was not deleted because it was pinned by the user'
                return (False,True)
        print "----------------------------------------------------------------"
        print "ERROR: "+str(self)+" was not properly deleted"
        printer = TracePrinter()
        self.report_references(printer, GC_REF_KIND, verbose)
        self.report_references(printer, VALID_REF_KIND, verbose)
        self.report_references(printer, REMOTE_REF_KIND, verbose)
        self.report_references(printer, RESOURCE_REF_KIND, verbose)
        print "----------------------------------------------------------------"
        if isinstance(self,Manager):
            return (False,False)
        return False

    def report_references(self, printer, kind, verbose):
        printer.down()
        if kind == GC_REF_KIND and self.base_gc_refs:
            for kind,refs in self.base_gc_refs.iteritems():
                if refs == 0:
                    if verbose:
                        printer.println('Empty (Refs=0): Base GC '+kind+
                            ' (Adds='+str(self.base_gc_adds[kind])+',Rems='+str(self.base_gc_rems[kind])+')')
                else:
                    if verbose:
                        printer.println('NON-EMPTY (Refs='+str(refs)+'): Base GC '+kind+
                            ' (Adds='+str(self.base_gc_adds[kind])+',Rems='+str(self.base_gc_rems[kind])+')')
                    else:
                        printer.println('Base GC '+kind+' (Refs='+str(refs)+')')
        if kind == VALID_REF_KIND and self.base_valid_refs:
            for kind,refs in self.base_valid_refs.iteritems():
                if refs == 0:
                    if verbose:
                        printer.println('Empty (Refs=0): Base Valid '+kind+
                            ' (Adds='+str(self.base_valid_adds[kind])+',Rems='+str(self.base_valid_rems[kind])+')')
                else:
                    if verbose:
                        printer.println('NON-EMPTY (Refs='+str(refs)+'): Base Valid '+kind+
                            ' (Adds='+str(self.base_valid_adds[kind])+',Rems='+str(self.base_valid_rems[kind])+')')
                    else:
                        printer.println('Base Valid '+kind+' (Refs='+str(refs)+')')
        if kind == REMOTE_REF_KIND and self.base_remote_refs:
            for kind,refs in self.base_remote_refs.iteritems():
                if refs == 0:
                    if verbose:
                        printer.println('Empty (Refs=0): Base Remote '+kind+
                            ' (Adds='+str(self.base_remote_adds[kind])+',Rems='+str(self.base_remote_rems[kind])+')')
                else:
                    if verbose:
                        printer.println('NON-EMPTY (Refs='+str(refs)+'): Base Remote '+kind+
                            ' (Adds='+str(self.base_remote_adds[kind])+',Rems='+str(self.base_remote_rems[kind])+')')
                    else:
                        printer.println('Base Remote '+kind+' (Refs='+str(refs)+')')
        if kind == RESOURCE_REF_KIND and self.base_resource_refs:
            for kind,refs in self.base_resource_refs.iteritems():
                if refs == 0:
                    if verbose:
                        printer.println('Empty (Refs=0): Base Resource '+kind+
                            ' (Adds='+str(self.base_resource_adds[kind])+',Rems='+str(self.base_resource_rems[kind])+')')
                else:
                    if verbose:
                        printer.println('NON-EMPTY (Refs='+str(refs)+'): Base Resource '+kind+
                            ' (Adds='+str(self.base_resource_adds[kind])+',Rems='+str(self.base_resource_rems[kind])+')')
                    else:
                        printer.println('Base Resource '+kind+' (Refs='+str(refs)+')')
        if kind == GC_REF_KIND and self.nested_gc_refs:
            for src,refs in self.nested_gc_refs.iteritems():
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
            for src,refs in self.nested_valid_refs.iteritems():
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
            for src,refs in self.nested_remote_refs.iteritems():
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
            for src,refs in self.nested_resource_refs.iteritems():
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

    def check_for_cycles(self):
        stack = list()
        self.check_for_cycles_by_kind(stack, GC_REF_KIND)
        stack = list()
        self.check_for_cycles_by_kind(stack, VALID_REF_KIND)
        stack = list()
        self.check_for_cycles_by_kind(stack, REMOTE_REF_KIND)
        stack = list()
        self.check_for_cycles_by_kind(stack, RESOURCE_REF_KIND)

    def check_for_cycles_by_kind(self, stack, kind):
        if self.on_stack:
            print 'CYCLE DETECTED!'
            for obj in stack:
                print '  '+repr(obj)
            print 'Exiting...'
            sys.exit(0)
        stack.append(self)
        self.on_stack = True
        if kind == GC_REF_KIND and self.nested_gc_refs:
            for src,refs in self.nested_gc_refs.iteritems():
                if refs is 0:
                    continue
                src.check_for_cycles_by_kind(stack, kind)
        if kind == VALID_REF_KIND and self.nested_valid_refs:
            for src,refs in self.nested_valid_refs.iteritems():
                if refs is 0:
                    continue
                src.check_for_cycles_by_kind(stack, kind)
        if kind == REMOTE_REF_KIND and self.nested_remote_refs:
            for src,refs in self.nested_remote_refs.iteritems():
                if refs is 0:
                    continue
                src.check_for_cycles_by_kind(stack, kind)
        if kind == RESOURCE_REF_KIND and self.nested_resource_refs:
            for src,refs in self.nested_resource_refs.iteritems():
                if refs is 0:
                    continue
                src.check_for_cycles_by_kind(stack, kind)
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

class VersionState(Base):
    def __init__(self, did, node):
        super(VersionState,self).__init__(did, node)

    def __repr__(self):
        return 'Version State '+str(self.did)+' (Node='+str(self.node)+')'

class Constraints(Base):
    def __init__(self, did, node):
        super(Constraints,self).__init__(did, node)

    def __repr__(self):
        return 'Layout Constraints '+str(self.did)+' (Node='+str(self.node)+')'

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
        self.unknowns = {}
        self.instances = {}
        self.src_names = {}
        self.constraints = {}
        self.version_states = {}

    def parse_log_file(self, file_name):
        with open(file_name, 'rb') as log:
            matches = 0
            for line in log:
                matches += 1
                m = add_base_ref_pat.match(line)
                if m is not None:
                    self.log_add_base_ref(int(m.group('kind')),
                                          long(m.group('did')),
                                          long(m.group('node')),
                                          int(m.group('src')),
                                          int(m.group('cnt')))
                    continue
                m = add_nested_ref_pat.match(line)
                if m is not None:
                    self.log_add_nested_ref(int(m.group('kind')),
                                            long(m.group('did')),
                                            long(m.group('node')),
                                            long(m.group('src')),
                                            int(m.group('cnt')))
                    continue
                m = remove_base_ref_pat.match(line)
                if m is not None:
                    self.log_remove_base_ref(int(m.group('kind')),
                                             long(m.group('did')),
                                             long(m.group('node')),
                                             int(m.group('src')),
                                             int(m.group('cnt')))
                    continue
                m = remove_nested_ref_pat.match(line)
                if m is not None:
                    self.log_remove_nested_ref(int(m.group('kind')),
                                               long(m.group('did')),
                                               long(m.group('node')),
                                               long(m.group('src')),
                                               int(m.group('cnt')))
                    continue
                m = inst_manager_pat.match(line)
                if m is not None:
                    self.log_inst_manager(long(m.group('did')),
                                          long(m.group('node')), 
                                          long(m.group('iid'),16),
                                          long(m.group('mem'),16))
                    continue
                m = list_manager_pat.match(line)
                if m is not None:
                    self.log_list_manager(long(m.group('did')),
                                          long(m.gropu('node')),
                                          long(m.group('iid'),16),
                                          long(m.group('mem'),16))
                    continue
                m = fold_manager_pat.match(line)
                if m is not None:
                    self.log_fold_manager(long(m.group('did')),
                                          long(m.group('node')),
                                          long(m.group('iid'),16),
                                          long(m.group('mem'),16))
                    continue
                m = materialize_pat.match(line)
                if m is not None:
                    self.log_materialized_view(long(m.group('did')),
                                               long(m.group('node')),
                                               long(m.group('inst')))
                    continue
                m = composite_pat.match(line)
                if m is not None:
                    self.log_composite_view(long(m.group('did')),
                                            long(m.group('node')))
                    continue
                m = fill_pat.match(line)
                if m is not None:
                    self.log_fill_view(long(m.group('did')),
                                       long(m.group('node')))
                    continue
                m = reduction_pat.match(line)
                if m is not None:
                    self.log_reduction_view(long(m.group('did')),
                                            long(m.group('node')),
                                            long(m.group('inst')))
                    continue
                m = version_state_pat.match(line)
                if m is not None:
                    self.log_version_state(long(m.group('did')),
                                           long(m.group('node')))
                    continue
                m = future_pat.match(line)
                if m is not None:
                    self.log_future(long(m.group('did')),
                                    long(m.group('node')))
                    continue
                m = constraints_pat.match(line)
                if m is not None:
                    self.log_constraints(long(m.group('did')),
                                         long(m.group('node')))
                    continue
                m = source_kind_pat.match(line)
                if m is not None:
                    self.log_source_kind(int(m.group('kind')),
                                         m.group('name'))
                    continue
                m = deletion_pat.match(line)
                if m is not None:
                    self.log_deletion(long(m.group('did')),
                                      long(m.group('node')))
                    continue
                matches -= 1
                print 'Skipping unmatched line: '+line
        return matches

    def post_parse(self):
        # Delete the virtual instance it is special
        to_del = list()
        for key,val in self.unknowns.iteritems():
            if key[0] == 0:
                to_del.append(key)
        for key in to_del:
            del self.unknowns[key]
        if self.unknowns:
            print "WARNING: Found %d unknown objects!" % len(self.unknowns)
            for did in self.unknowns.iterkeys():
                print '  Unknown DID '+str(hex(did[0]))+' on node '+str(did[1])
        # Now update all the pointers to references
        for man in self.managers.itervalues():
            man.update_nested_references(self)
        for view in self.views.itervalues():
            view.update_nested_references(self)
        for future in self.futures.itervalues():
            future.update_nested_references(self)
        for constraint in self.constraints.itervalues():
            constraint.update_nested_references(self)
        for state in self.version_states.itervalues():
            state.update_nested_references(self)
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
        inst = self.get_instance(iid, mem, 'Physical')
        manager = self.get_manager(did, node)
        manager.add_inst(inst)

    def log_list_manager(self, did, node, iid, mem):
        inst = self.get_instance(iid, mem, 'List Reduction')
        manager = self.get_manager(did, node)
        manager.add_inst(inst)

    def log_fold_manager(self, did, node, iid, mem):
        inst = self.get_instance(iid, mem, 'Fold Reduction')
        manager = self.get_manager(did, node)
        manager.add_inst(inst)

    def log_materialized_view(self, did, node, inst):
        manager = self.get_manager(inst, node)
        view = self.get_view(did, node, 'Materialized')
        view.add_manager(manager)

    def log_composite_view(self, did, node):
        self.get_view(did, node, 'Composite')

    def log_fill_view(self, did, node):
        self.get_view(did, node, 'Fill')

    def log_reduction_view(self, did, node, inst):
        manager = self.get_manager(inst, node)
        view = self.get_view(did, node, 'Reduction')
        view.add_manager(manager)

    def log_version_state(self, did, node):
        self.get_version_state(did, node);

    def log_future(self, did, node):
        self.get_future(did, node)

    def log_constraints(self, did, node):
        self.get_constraints(did, node)

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

    def get_version_state(self, did, node):
        key = (did,node)
        if key not in self.version_states:
            self.version_states[key] = VersionState(did, node)
            if key in self.unknowns:
                self.version_states[key].clone(self.unknowns[key])
                del self.unknowns[key]
        return self.version_states[key]

    def get_future(self, did, node):
        key = (did,node)
        if key not in self.futures:
            self.futures[key] = Future(did, node)
            if key in self.unknowns:
                self.futures[key].clone(self.unknowns[key])
                del self.unknowns[key]
        return self.futures[key]

    def get_constraints(self, did, node):
        key = (did,node)
        if key not in self.constraints:
            self.constraints[key] = Constraints(did, node)
            if key in self.unknowns:
                self.constraints[key].clone(self.unknowns[key])
                del self.unknowns[key]
        return self.constraints[key]

    def get_obj(self, did, node):
        key = (did,node)
        if key in self.views:
            return self.views[key]
        if key in self.managers:
            return self.managers[key]
        if key in self.futures:
            return self.futures[key]
        if key in self.version_states:
            return self.version_states[key]
        if key in self.constraints:
            return self.constraints[key]
        if key in self.unknowns:
            return self.unknowns[key]
        self.unknowns[key] = Base(did, node)
        return self.unknowns[key]

    def check_for_cycles(self):
        for did,manager in self.managers.iteritems():
            print "Checking for cycles in "+repr(manager)
            manager.check_for_cycles()
        for did,view in self.views.iteritems():
            print "Checking for cycles in "+repr(view)
            view.check_for_cycles()
        for did,future in self.futures.iteritems():
            print "Checking for cycles in "+repr(future)
            future.check_for_cycles()
        for did,version in self.version_states.iteritems():
            print "Checking for cycles in "+repr(version)
            version.check_for_cycles()
        for did,constraint in self.constraints.iteritems():
            print "Checking for cycles in "+repr(constraint)
            constraint.check_for_cycles()
        print "NO CYCLES"

    def check_for_leaks(self, verbose): 
        leaked_futures = 0
        leaked_constraints = 0
        leaked_managers = 0
        pinned_managers = 0
        leaked_views = 0
        leaked_states = 0
        for future in self.futures.itervalues():
            if not future.check_for_leaks(verbose):
                leaked_futures += 1
        for constraint in self.constraints.itervalues():
            if not constraint.check_for_leaks(verbose): 
                leaked_constraints += 1
        for manager in self.managers.itervalues():
            deleted,pinned = manager.check_for_leaks(verbose)
            if not deleted:
                if pinned:
                    pinned_managers += 1
                else:
                    leaked_managers += 1
        for view in self.views.itervalues():
            if not view.check_for_leaks(verbose):
                leaked_views += 1
        for state in self.version_states.itervalues():
            if not state.check_for_leaks(verbose):
                leaked_states += 1
        print "LEAK SUMMARY"
        if leaked_futures > 0:
            print "  LEAKED_FUTURES: "+str(leaked_futures)
        else:
            print "  Leaked Futures: "+str(leaked_futures)
        if leaked_constraints > 0:
            print "  LEAKED CONSTRAINTS: "+str(leaked_constraints)
        else:
            print "  Leaked Constraints: "+str(leaked_constraints)
        if leaked_managers > 0:
            print "  LEAKED MANAGERS: "+str(leaked_managers)
        else:
            print "  Leaked Managers: "+str(leaked_managers)
        if pinned_managers > 0:
            print "  PINNED MANAGERS: "+str(pinned_managers)
        else:
            print "  Pinned Managers: "+str(pinned_managers)
        if leaked_views > 0:
            print "  LEAKED VIEWS: "+str(leaked_views)
        else:
            print "  Leaked Views: "+str(leaked_views)
        if leaked_states > 0:
            print "  LEAKED VERSION STATES: "+str(leaked_states)
        else:
            print "  Leaked Version States: "+str(leaked_states)

def usage():
    print 'Usage: '+sys.argv[0]+' [-c -l -v] <file_names>+'
    print '  -c : check for cycles'
    print '  -l : check for leaks'
    print '  -v : verbose'
    sys.exit(1)

def main():
    opts, args = getopt(sys.argv[1:],'clv')
    opts = dict(opts)
    if len(args) == 0:
        usage()
    check_cycles = False
    check_leaks = False
    verbose = False
    if '-c' in opts:
        check_cycles = True
    if '-l' in opts:
        check_leaks = True
    if '-v' in opts:
        verbose = True

    file_names = args
    
    state = State()
    has_matches = False
    for file_name in file_names:
        print 'Reading log file %s...' % file_name
        total_matches = state.parse_log_file(file_name)
        print 'Matched %s lines' % total_matches
        if total_matches > 0:
            has_matches = True
    if not has_matches:
        print 'No matches found! Exiting...'
        return

    state.post_parse()
  
    if check_cycles:
        state.check_for_cycles()
    if check_leaks:
        state.check_for_leaks(verbose)


if __name__ == '__main__':
    main()

