#!/usr/bin/env python

# Copyright 2015 Stanford University, NVIDIA Corporation
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

import sys, os, shutil
import string, re
from math import sqrt, log
from getopt import getopt

GC_REF_KIND = 0
VALID_REF_KIND = 1
REMOTE_REF_KIND = 2
RESOURCE_REF_KIND = 3

prefix = r'\[(?P<node>[0-9]+) - (?P<thread>[0-9a-f]+)\] \{\w+\}\{legion_gc\}: '
add_base_ref_pat = re.compile(prefix + r'GC Add Base Ref (?P<kind>[0-9]+) (?P<did>[0-9]+) (?P<src>[0-9]+) (?P<cnt>[0-9]+)')
add_nested_ref_pat = re.compile(prefix + r'GC Add Nested Ref (?P<kind>[0-9]+) (?P<did>[0-9]+) (?P<src>[0-9]+) (?P<cnt>[0-9]+)')
remove_base_ref_pat = re.compile(prefix + r'GC Remove Base Ref (?P<kind>[0-9]+) (?P<did>[0-9]+) (?P<src>[0-9]+) (?P<cnt>[0-9]+)')
remove_nested_ref_pat = re.compile(prefix + r'GC Remove Nested Ref (?P<kind>[0-9]+) (?P<did>[0-9]+) (?P<src>[0-9]+) (?P<cnt>[0-9]+)')
inst_manager_pat = re.compile(prefix + r'GC Instance Manager (?P<did>[0-9]+) (?P<iid>[a-f0-9]+) (?P<mem>[a-f0-9]+)')
list_manager_pat = re.compile(prefix + r'GC List Reduction Manager (?P<did>[0-9]+) (?P<iid>[a-f0-9]+) (?P<mem>[a-f0-9]+)')
fold_manager_pat = re.compile(prefix + r'GC Fold Reduction Manager (?P<did>[0-9]+) (?P<iid>[a-f0-9]+) (?P<mem>[a-f0-9]+)')
materialize_pat = re.compile(prefix + r'GC Materialized View (?P<did>[0-9]+) (?P<inst>[0-9]+)')
composite_pat = re.compile(prefix + r'GC Composite View (?P<did>[0-9]+)')
fill_pat = re.compile(prefix + r'GC Fill View (?P<did>[0-9]+)')
reduction_pat = re.compile(prefix + r'GC Reduction View (?P<did>[0-9]+) (?P<inst>[0-9]+)')
future_pat = re.compile(prefix + r'GC Future (?P<did>[0-9]+)')
source_kind_pat = re.compile(prefix + r'GC Source Kind (?P<kind>[0-9]+) (?P<name>[0-9a-zA-Z_]+)')

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
    def __init__(self, did):
        self.did = did
        self.base_gc_refs = {}
        self.base_valid_refs = {}
        self.base_remote_refs = {}
        self.base_resource_refs = {}
        self.nested_gc_refs = {}
        self.nested_valid_refs = {}
        self.nested_remote_refs = {}
        self.nested_resource_refs = {}
        self.on_stack = False

    def add_base_ref(self, kind, src, cnt):
        if kind is GC_REF_KIND:
            if src in self.base_gc_refs:
                self.base_gc_refs[src] += cnt
            else:
                self.base_gc_refs[src] = cnt
        elif kind is VALID_REF_KIND:
            if src in self.base_valid_refs:
                self.base_valid_refs[src] += cnt
            else:
                self.base_valid_refs[src] = cnt
        elif kind is REMOTE_REF_KIND:
            if src in self.base_remote_refs:
                self.base_remote_refs[src] += cnt
            else:
                self.base_remote_refs[src] = cnt
        elif kind is RESOURCE_REF_KIND:
            if src in self.base_resource_refs:
                self.base_resource_refs[src] += cnt
            else:
                self.base_resource_refs[src] = cnt
        else:
            print 'BAD BASE REF '+str(kind)
            assert False

    def add_nested_ref(self, kind, src, cnt):
        if kind is GC_REF_KIND:
            if src in self.base_gc_refs:
                self.nested_gc_refs[src] += cnt
            else:
                self.nested_gc_refs[src] = cnt
        elif kind is VALID_REF_KIND:
            if src in self.base_valid_refs:
                self.nested_valid_refs[src] += cnt
            else:
                self.nested_valid_refs[src] = cnt
        elif kind is REMOTE_REF_KIND:
            if src in self.base_remote_refs:
                self.nested_remote_refs[src] += cnt
            else:
                self.nested_remote_refs[src] = cnt
        elif kind is RESOURCE_REF_KIND:
            if src in self.base_resource_refs:
                self.nested_resource_refs[src] += cnt
            else:
                self.nested_resource_refs[src] = cnt
        else:
            assert False

    def remove_base_ref(self, kind, src, cnt):
        if kind is GC_REF_KIND:
            if src in self.base_gc_refs:
                self.base_gc_refs[src] -= cnt
            else:
                self.base_gc_refs[src] = -cnt
        elif kind is VALID_REF_KIND:
            if src in self.base_valid_refs:
                self.base_valid_refs[src] -= -cnt
            else:
                self.base_valid_refs[src] = -cnt
        elif kind is REMOTE_REF_KIND:
            if src in self.base_remote_refs:
                self.base_remote_refs[src] -= cnt
            else:
                self.base_remote_refs[src] = -cnt
        elif kind is RESOURCE_REF_KIND:
            if src in self.base_resource_refs:
                self.base_resource_refs[src] -= cnt
            else:
                self.base_resource_refs[src] = -cnt
        else:
            assert False

    def remove_nested_ref(self, kind, src, cnt):
        if kind is GC_REF_KIND:
            if src in self.base_gc_refs:
                self.nested_gc_refs[src] -= cnt
            else:
                self.nested_gc_refs[src] = -cnt
        elif kind is VALID_REF_KIND:
            if src in self.base_valid_refs:
                self.nested_valid_refs[src] -= cnt
            else:
                self.nested_valid_refs[src] = -cnt
        elif kind is REMOTE_REF_KIND:
            if src in self.base_remote_refs:
                self.nested_remote_refs[src] -= cnt
            else:
                self.nested_remote_refs[src] = -cnt
        elif kind is RESOURCE_REF_KIND:
            if src in self.base_resource_refs:
                self.nested_resource_refs[src] -= cnt
            else:
                self.nested_resource_refs[src] = -cnt
        else:
            assert False

    def clone(self, other):
        self.base_gc_refs = other.base_gc_refs
        self.base_valid_refs = other.base_valid_refs
        self.base_remote_refs = other.base_remote_refs
        self.base_resource_refs = other.base_resource_refs
        self.nested_gc_refs = other.nested_gc_refs
        self.nested_valid_refs = other.nested_valid_refs
        self.nested_remote_refs = other.nested_remote_refs
        self.nested_resource_refs = other.nested_resource_refs

    def check_references(self):
        has_references = False
        if not has_references and len(self.base_gc_refs) > 0:
            for kind,refs in self.base_gc_refs.iteritems():
                if refs is not 0:
                    has_references = True
                    break
        if not has_references and len(self.base_valid_refs) > 0:
            for kind,refs in self.base_valid_refs.iteritems():
                if refs is not 0:
                    has_references = True
                    break
        if not has_references and len(self.base_remote_refs) > 0:
            for kind,refs in self.base_remote_refs.iteritems():
                if refs is not 0:
                    has_references = True
                    break
        if not has_references and len(self.nested_gc_refs) > 0:
            for kind,refs in self.nested_gc_refs.iteritems():
                if refs is not 0:
                    has_references = True
                    break
        if not has_references and len(self.nested_valid_refs) > 0:
            for kind,refs in self.nested_valid_refs.iteritems():
                if refs is not 0:
                    has_references = True
                    break
        if not has_references and len(self.nested_remote_refs) > 0:
            for kind,refs in self.nested_remote_refs.iteritems():
                if refs is not 0:
                    has_references = True
                    break
        if has_references:
            printer = TracePrinter()
            self.report_references(printer)
        return has_references

    def report_references(self, printer):
        printer.print_base(self)
        printer.down()
        if len(self.base_gc_refs) > 0:
            for kind,refs in self.base_gc_refs.iteritems():
                if refs is 0:
                    continue
                printer.println('Base GC '+kind+' '+str(refs))
        if len(self.base_valid_refs) > 0:
            for kind,refs in self.base_valid_refs.iteritems():
                if refs is 0:
                    continue
                printer.println('Base Valid '+kind+' '+str(refs))
        if len(self.base_remote_refs) > 0:
            for kind,refs in self.base_remote_refs.iteritems():
                if refs is 0:
                    continue
                printer.println('Base Remote '+kind+' '+str(refs))
        if len(self.nested_gc_refs) > 0:
            for src,refs in self.nested_gc_refs.iteritems():
                if refs is 0:
                    continue
                printer.println('Nested GC '+repr(src))
                printer.down()
                src.report_references(printer)
                printer.up()
        if len(self.nested_valid_refs) > 0:
            for src,refs in self.nested_valid_refs.iteritems():
                if refs is 0:
                    continue
                printer.println('Nested Valid '+repr(src))
                printer.down()
                src.report_references(printer)
                printer.up()
        if len(self.nested_remote_refs) > 0:
            for src,refs in self.nested_remote_refs.iteritems():
                if refs is 0:
                    continue
                printer.println('Nested Remote '+repr(src))
                printer.down()
                src.report_references(printer)
                printer.up()
        printer.up()

    def check_for_cycles(self, stack):
        if self.on_stack:
            print 'CYCLE DETECTED!'
            for obj in stack:
                print '  '+repr(obj)
            print 'Exiting...'
            sys.exit(0)
        stack.append(self)
        self.on_stack = True
        if len(self.nested_gc_refs) > 0:
            for src,refs in self.nested_gc_refs.iteritems():
                if refs is 0:
                    continue
                src.check_for_cycles(stack)
        if len(self.nested_valid_refs) > 0:
            for src,refs in self.nested_valid_refs.iteritems():
                if refs is 0:
                    continue
                src.check_for_cycles(stack)
        if len(self.nested_remote_refs) > 0:
            for src,refs in self.nested_remote_refs.iteritems():
                if refs is 0:
                    continue
                src.check_for_cycles(stack)
        stack.pop()
        self.on_stack = False

class Manager(Base):
    def __init__(self, did):
        super(Manager,self).__init__(did)
        self.instance = None

    def add_inst(self, inst):
        self.instance = inst

    def __repr__(self):
        result = 'Manager '+str(self.did)
        if self.instance is not None:
            result += ' '+repr(self.instance)
        return result

class View(Base):
    def __init__(self, did, kind):
        super(View,self).__init__(did)
        self.kind = kind
        self.manager = None

    def add_manager(self, manager):
        self.manager = manager

    def __repr__(self):
        result = self.kind +' View '+str(self.did)
        if self.manager is not None:
            result += ' of ' + repr(self.manager)
        return result

class Future(Base):
    def __init__(self, did):
        super(Future,self).__init__(did)

    def __repr__(self):
        return 'Future '+str(self.did)

class Instance(object):
    def __init__(self, iid, mem, kind):
        self.iid = iid
        self.mem = mem
        self.kind = kind

    def __repr__(self):
        return self.kind + ' Instance '+hex(self.iid)+' in Memory '+hex(self.mem)

class State(object):
    def __init__(self):
        self.managers = {}
        self.views = {}
        self.futures = {}
        self.unknowns = {}
        self.instances = {}
        self.src_names = {}

    def parse_log_file(self, file_name):
        with open(file_name, 'rb') as log:
            matches = 0
            for line in log:
                matches += 1
                m = add_base_ref_pat.match(line)
                if m is not None:
                    self.log_add_base_ref(int(m.group('kind')),
                                          long(m.group('did')),
                                          int(m.group('src')),
                                          int(m.group('cnt')))
                    continue
                m = add_nested_ref_pat.match(line)
                if m is not None:
                    self.log_add_nested_ref(int(m.group('kind')),
                                            long(m.group('did')),
                                            long(m.group('src')),
                                            int(m.group('cnt')))
                    continue
                m = remove_base_ref_pat.match(line)
                if m is not None:
                    self.log_remove_base_ref(int(m.group('kind')),
                                             long(m.group('did')),
                                             int(m.group('src')),
                                             int(m.group('cnt')))
                    continue
                m = remove_nested_ref_pat.match(line)
                if m is not None:
                    self.log_remove_nested_ref(int(m.group('kind')),
                                               long(m.group('did')),
                                               long(m.group('src')),
                                               int(m.group('cnt')))
                    continue
                m = inst_manager_pat.match(line)
                if m is not None:
                    self.log_inst_manager(long(m.group('did')),
                                          long(m.group('iid'),16),
                                          long(m.group('mem'),16))
                    continue
                m = list_manager_pat.match(line)
                if m is not None:
                    self.log_list_manager(long(m.group('did')),
                                          long(m.group('iid'),16),
                                          long(m.group('mem'),16))
                    continue
                m = fold_manager_pat.match(line)
                if m is not None:
                    self.log_fold_manager(long(m.group('did')),
                                          long(m.group('iid'),16),
                                          long(m.group('mem'),16))
                    continue
                m = materialize_pat.match(line)
                if m is not None:
                    self.log_materialized_view(long(m.group('did')),
                                               long(m.group('inst')))
                    continue
                m = composite_pat.match(line)
                if m is not None:
                    self.log_composite_view(long(m.group('did')))
                    continue
                m = fill_pat.match(line)
                if m is not None:
                    self.log_fill_view(long(m.group('did')))
                    continue
                m = reduction_pat.match(line)
                if m is not None:
                    self.log_reduction_view(long(m.group('did')),
                                            long(m.group('inst')))
                    continue
                m = future_pat.match(line)
                if m is not None:
                    self.log_future(long(m.group('did')))
                    continue
                m = source_kind_pat.match(line)
                if m is not None:
                    self.log_source_kind(int(m.group('kind')),
                                         m.group('name'))
                    continue
                matches -= 1
                print 'Skipping unmatched line: '+line
        return matches

    def log_add_base_ref(self, kind, did, src, cnt):
        obj = self.get_obj(did) 
        assert src in self.src_names
        obj.add_base_ref(kind, self.src_names[src], cnt)

    def log_add_nested_ref(self, kind, did, src, cnt):
        obj = self.get_obj(did)
        src_obj = self.get_obj(src)
        obj.add_nested_ref(kind, src_obj, cnt)

    def log_remove_base_ref(self, kind, did, src, cnt):
        obj = self.get_obj(did)
        assert src in self.src_names
        obj.remove_base_ref(kind, self.src_names[src], cnt)

    def log_remove_nested_ref(self, kind, did, src, cnt):
        obj = self.get_obj(did)
        src_obj = self.get_obj(src)
        obj.remove_nested_ref(kind, src_obj, cnt)

    def log_inst_manager(self, did, iid, mem):
        inst = self.get_instance(iid, mem, 'Instance')
        manager = self.get_manager(did)
        manager.add_inst(inst)

    def log_list_manager(self, did, iid, mem):
        inst = self.get_instance(iid, mem, 'List')
        manager = self.get_manager(did)
        manager.add_inst(inst)

    def log_fold_manager(self, did, iid, mem):
        inst = self.get_instance(iid, mem, 'Fold')
        manager = self.get_manager(did)
        manager.add_inst(inst)

    def log_materialized_view(self, did, inst):
        manager = self.get_manager(inst)
        view = self.get_view(did, 'Materialized')
        view.add_manager(manager)

    def log_composite_view(self, did):
        self.get_view(did, 'Composite')

    def log_fill_view(self, did):
        self.get_view(did, 'Fill')

    def log_reduction_view(self, did, inst):
        manager = self.get_manager(inst)
        view = self.get_view(did, 'Reduction')
        view.add_manager(manager)

    def log_future(self, did):
        self.get_future(did)

    def log_source_kind(self, kind, name):
        if kind not in self.src_names:
            self.src_names[kind] = name

    def get_instance(self, iid, mem, kind):
        if iid not in self.instances:
            self.instances[iid] = Instance(iid, mem, kind)
        return self.instances[iid]

    def get_manager(self, did):
        if did not in self.managers:
            self.managers[did] = Manager(did)
            if did in self.unknowns:
                self.managers[did].clone(self.unknowns[did])
                del self.unknowns[did]
        return self.managers[did]

    def get_view(self, did, kind):
        if did not in self.views:
            self.views[did] = View(did, kind)
            if did in self.unknowns:
                self.views[did].clone(self.unknowns[did])
                del self.unknowns[did]
        return self.views[did]

    def get_future(self, did):
        if did not in self.futures:
            self.futures[did] = Future(did)
            if did in self.unknowns:
                self.futures[did].clone(self.unknowns[did])
                del self.unknowns[did]
        return self.futures[did]

    def get_obj(self, did):
        if did in self.views:
            return self.views[did]
        if did in self.managers:
            return self.managers[did]
        if did in self.futures:
            return self.futures[did]
        if did in self.unknowns:
            return self.unknowns[did]
        self.unknowns[did] = Base(did)
        return self.unknowns[did]

    def find_cycles(self):
        stack = list()
        for did,manager in self.managers.iteritems():
            manager.check_for_cycles(stack)
        for did,view in self.views.iteritems():
            view.check_for_cycles(stack)
        for did,future in self.futures.iteritems():
            future.check_for_cycles(stack)

    def check_futures(self):
        for did,future in self.futures.iteritems():
            if future.check_references():
                break

    def check_managers(self):
        for did,manager in self.managers.iteritems():
            if manager.check_references():
                break

    def check_views(self):
        for did,views in self.views.iteritems():
            if view.check_references():
                break

def usage():
    print 'Usage: '+sys.argv[0]+' <file_names>+'
    sys.exit(1)

def main():
    opts, args = getopt(sys.argv[1:],'fmv')
    opts = dict(opts)
    if len(args) == 0:
        usage()
    check_all = True
    check_futures = False
    check_managers = False
    check_views = False
    if '-f' in opts:
        check_futures = True
        check_all = False
    if '-m' in opts:
        check_managers = True
        check_all = False
    if '-v' in opts:
        check_managers = True
        check_all = False
    if check_all:
        check_futures = True
        check_managers = True
        check_view = True

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

    state.find_cycles()

    if check_futures:
        state.check_futures()

    if check_managers:
        state.check_managers()

    if check_views:
        state.check_views()

if __name__ == '__main__':
    main()

