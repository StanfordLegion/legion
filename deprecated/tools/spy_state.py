#!/usr/bin/env python3

# Copyright 2023 Stanford University
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

import subprocess
import string
import sys
#from spy_timeline import TimeLine

# These are imported from legion_types.h
NO_DEPENDENCE = 0
TRUE_DEPENDENCE = 1
ANTI_DEPENDENCE = 2
ATOMIC_DEPENDENCE = 3
SIMULTANEOUS_DEPENDENCE = 4

NO_ACCESS  = 0x00000000
READ_ONLY  = 0x00000001
READ_WRITE = 0x00000111
WRITE_ONLY = 0x00000010
REDUCE     = 0x00000100

EXCLUSIVE = 0
ATOMICE = 1
SIMULTANEOUS = 2
RELAXED = 3

LOC_PROC = 0
TOC_PROC = 1
UTIL_PROC = 2

# Some helper methods
def is_mapping_dependence(dtype):
    if dtype == NO_DEPENDENCE:
        return False
    # All other types of dependences are mapping dependences
    return True

def check_for_anti_dependence(req1, req2, actual):
    if req1.is_read_only():
        assert req2.has_write()
        return ANTI_DEPENDENCE
    else:
        if req2.is_write_only():
            return ANTI_DEPENDENCE
        else:
            return actual

def compute_dependence_type(req1, req2):
    if req1.is_read_only() and req2.is_read_only():
        return NO_DEPENDENCE
    elif req1.is_reduce() and req2.is_reduce():
        if req1.redop == req2.redop:
            return NO_DEPENDENCE
        else:
            return TRUE_DEPENDENCE
    else:
        assert req1.has_write() or req2.has_write() 
        if req1.is_exclusive() or req2.is_exclusive():
            return check_for_anti_dependence(req1,req2,TRUE_DEPENDENCE)
        elif req1.is_atomic() or req2.is_atomic():
            if req1.is_atomic() and req2.is_atomic():
                return check_for_anti_dependence(req1,req1,ATOMIC_DEPENDENCE)
            elif ((not req1.is_atomic()) and req1.is_read_only()) or ((not req2.is_atomic()) and req2.is_read_only()):
                return NO_DEPENDENCE
            else:
                return check_for_anti_dependence(req1,req2,TRUE_DEPENDENCE)
        elif req1.is_simult() or req2.is_simult():
            return check_for_anti_dependence(req1,req2,SIMULTANEOUSE_DEPENDENCE)
        elif req1.is_relaxed() and req2.is_relaxed():
            return check_for_anti_dependence(req1,req2,SIMULTANEOUS_DEPENDENCE)
        # Should never get here
        assert False
        return NO_DEPENDENCE


class Processor(object):
    def __init__(self, state, uid, util, kind):
        self.state = state
        self.uid = uid
        self.utility = util
        assert kind <> UTIL_PROC
        self.kind = kind
        self.mem_latency = dict()
        self.mem_bandwidth = dict()
        self.executed_tasks = list()
        util.add_constituent(self)

    def add_memory(self, mem, bandwidth, latency):
        assert mem not in self.mem_latency
        assert mem not in self.mem_bandwidth
        self.mem_latency[mem] = latency
        self.mem_bandwidth[mem] = bandwidth

    def append_task_instance(self, task):
        self.executed_tasks.append(task)

#    def print_timeline(self):
 #       if len(self.executed_tasks) == 0:
 #           return
 #       # Find the first and last points in time 
 #       first = self.executed_tasks[0].start_time 
 #       assert first <> None
 #       last = self.executed_tasks[0].end_time
 #       assert last <> None
 #       for t in self.executed_tasks:
 #           if t.start_time < first:
 #               first = t.start_time
 #           if t.end_time > last:
 #               last = t.end_time
 #       assert first < last
 #       name = "processor_"+str(self.uid)+"_timeline" 
 #       timeline = TimeLine(name, first, last) 
 #       # Map from enclosing contexts to lines in the processor printout
 #       contexts = dict()
 #       for t in self.executed_tasks:
 #           # Find the enclosing context, first check if it is the top level task
 #           enclosing = None
 #           assert t.enclosing <> None
 #          if t.enclosing.enclosing <> None:
 #               enclosing = t.enclosing.enclosing
 #               assert enclosing <> None
 #           if enclosing not in contexts:
 #               # We need to make the line
 #               if enclosing == None:
 #                   # Handle the case where the task is the top level task
 #                   contexts[enclosing] = timeline.add_line("Top Level Task")
 #               else:
 #                   contexts[enclosing] = timeline.add_line(enclosing.get_name())
 #           assert enclosing in contexts
 #           line = contexts[enclosing]
 #           timeline.add_instance(line, t.get_name(), t.start_time, t.end_time, "black")
 #       timeline.write_pdf(1000)
            

class UtilityProcessor(object):
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.kind = UTIL_PROC 
        self.constituents = set()

    def add_constituent(self, proc):
        assert proc not in self.constituents
        self.constituents.add(proc)


class Memory(object):
    def __init__(self, state, uid, capacity):
        self.state = state
        self.uid = uid
        self.capacity = capacity
        self.proc_bandwidth = dict()
        self.proc_latency = dict()
        self.mem_bandwidth = dict()
        self.mem_latency = dict()
        self.physical_instances = dict()
        self.reduction_instances = dict()

    def add_processor(self, proc, bandwidth, latency):
        assert proc not in self.proc_bandwidth
        assert proc not in self.proc_latency
        self.proc_bandwidth[proc] = bandwidth
        self.proc_bandwidth[proc] = latency

    def add_memory(self, mem, bandwidth, latency):
        assert mem <> self
        assert mem not in self.mem_bandwidth
        assert mem not in self.mem_latency
        self.mem_bandwidth[mem] = bandwidth
        self.mem_latency[mem] = latency

    def add_physical_instance(self, inst):
        assert inst.uid not in self.physical_instances
        self.physical_instances[inst.uid] = inst

    def add_reduction_instance(self, inst):
        assert inst.uid not in self.reduction_instances
        self.reduction_instances[inst.uid] = inst

    def print_timeline(self):
        name = "memory_"+str(self.uid)+"_timeline"
        return


class IndexSpaceNode(object):
    def __init__(self, state, uid, color, parent):
        self.state = state
        self.uid = uid
        self.parent = parent
        self.instances = dict()
        self.children = dict()
        if parent <> None:
            parent.add_child(color, self)
            self.depth = parent.depth + 1
            # Also check to see if the parent has any instances for which
            # we need to instantiate this tree
            for tid,pinst in parent.instances.iteritems():
                self.instantiate(pinst,pinst.field_node,tid)
        else:
            self.depth = 0

    def instantiate(self, parent_inst, field_node, tid):
        assert tid not in self.instances
        region_node = RegionNode(self.state, self, field_node, tid, parent_inst)
        self.instances[tid] = region_node
        for color,child in self.children.iteritems():
            child.instantiate(region_node, field_node, tid)

    def add_child(self, color, child):
        assert color not in self.children
        self.children[color] = child 

    def get_instance(self, tid):
        assert tid in self.instances
        return self.instances[tid]

    def is_region(self):
        return True

class IndexPartNode(object):
    def __init__(self, state, uid, disjoint, color, parent):
        self.state = state
        self.uid = uid
        self.disjoint = disjoint
        self.parent = parent
        self.instances = dict()
        self.children = dict()
        assert parent <> None
        parent.add_child(color, self)
        self.depth = parent.depth + 1
        # Also need to instaitate any instances from the parent
        for tid,pinst in parent.instances.iteritems():
            self.instantiate(pinst, pinst.field_node, tid)

    def instantiate(self, parent_inst, field_node, tid):
        assert tid not in self.instances
        part_node = PartitionNode(self.state, self, field_node, tid, parent_inst)
        self.instances[tid] = part_node
        for color,child in self.children.iteritems():
            child.instantiate(part_node, field_node, tid)

    def add_child(self, color, child):
        assert color not in self.children
        self.children[color] = child

    def is_region(self):
        return False

class FieldSpaceNode(object):
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.fields = set()

    def add_field(self, fid):
        assert fid not in self.fields
        self.fields.add(fid)

class RegionNode(object):
    def __init__(self, state, index_node, field_node, tid, parent):
        self.state = state
        self.index_node = index_node
        self.field_node = field_node
        self.tid = tid
        self.parent = parent
        self.children = set()
        if parent <> None:
            parent.add_child(self)

    def add_child(self, child):
        assert child not in self.children
        self.children.add(child)

class PartitionNode(object):
    def __init__(self, state, index_node, field_node, tid, parent):
        self.state = state
        self.index_node = index_node
        self.field_node = field_node
        self.tid = tid
        self.parent = parent
        self.children = set()
        if parent <> None:
            parent.add_child(self)

    def add_child(self, child):
        assert child not in self.children
        self.children.add(child)

# A way to uniquely identify task instances is the quad 
# consisting of the task ID, the context, the context
# generation, and the high-level runtime utility processor
class TaskHandle(object):
    def __init__(self, uid, ctx, hid, gen):
        self.uid = uid
        self.ctx = ctx
        self.hid = hid
        self.gen = gen

    def __hash__(self):
        return hash((self.uid,self.ctx,self.hid,self.gen))

    def __eq__(self,other):
        return (self.uid,self.ctx,self.hid,self.gen) == (other.uid,other.ctx,other.hid,other.gen)


class TaskInstance(object):
    def __init__(self, state, handle, enclosing, index_space):
        self.state = state
        self.handle = handle
        # The enclosing parent task
        self.enclosing = enclosing
        self.index_space = index_space
        self.physical_marked = False
        self.ops = list()
        self.mdeps = list() # The runtime computed mapping dependences
        self.adeps = list() # The mapping dependences computed here
        self.prev_event_deps = set()
        self.managers = dict() # Managers for the mapped regions
        self.requirements = dict()
        self.executing_processor = None
        self.start_time = None
        self.end_time = None

    def print_instance(self):
        if self.enclosing.index_space:
            print "Point "+self.point.to_string()+" of task "+str(self.enclosing.name)+" UID "+str(self.enclosing.uid)
        else:
            print "Single task "+str(self.enclosing.name)+" UID "+str(self.enclosing.uid)
        for idx,req in self.requirements.iteritems():
            print "  Region Requirement "+str(idx)
            req.print_requirement()

    def update_info(self, point, start_event, term_event):
        self.point = point
        self.start_event = start_event
        self.term_event = term_event
        self.node_name = 'task_node_'+str(self.handle.uid)+'_'+self.point.to_simple_string()
        if self.index_space:
            self.name = str(self.enclosing.name)+" "+self.point.to_string()+" UID:"+str(self.enclosing.uid)
        else:
            self.name = str(self.enclosing.name)+" UID:"+str(self.enclosing.uid)

    def uses_memory(self, mem):
        for idx,man in self.managers.iteritems():
            if man.inst.memory == mem:
                return True
        return False

    def append_op(self, op):
        self.ops.append(op)

    def add_mdep(self, op1, op2, idx1, idx2, dtype):
        assert op1 in self.ops
        assert op2 in self.ops
        self.mdeps.append(MappingDependence(self, op1, op2, idx1, idx2, dtype))

    def add_adep(self, op1, op2, idx1, idx2, dtype):
        self.adeps.append(MappingDependence(self, op1, op2, idx1, idx2, dtype))

    def update_requirement(self, idx, index):
        assert self.index_space
        assert idx not in self.requirements
        req = self.enclosing.get_requirement(idx)
        new_req = Requirement(idx, True, index, req.fspace, req.tid, req.priv, req.coher, req.redop)
        for f in req.fields:
            new_req.add_field(f)
        self.requirements[idx] = new_req

    def set_processor(self, proc):
        assert self.executing_processor == None
        self.executing_processor = proc
        proc.append_task_instance(self)

    def set_start_time(self, start):
        assert self.start_time == None
        self.start_time = start

    def set_end_time(self, end):
        assert self.end_time == None
        self.end_time = end

    def compute_dependence_diff(self, verbose):
        if (len(self.ops) == 0) or (len(self.ops) == 1):
            return
        print "Checking mapping dependences for task "+str(self.name)
        # First compute the list of adeps between each pair of operations
        for idx in range(1,len(self.ops)):
            for prev in range(idx):
                self.ops[prev].find_dependences(self.ops[idx])
        print "    Found "+str(len(self.adeps))+" dependences in all-pairs test for task "+str(self.enclosing.name)
        if verbose:
            print "        Computed Dependences:"
            for dep in self.mdeps:
                print "          index "+str(dep.idx1)+" of "+dep.op1.get_name()+" (ID "+str(dep.op1.uid)+") and index "+\
                                         str(dep.idx2)+" of "+dep.op2.get_name()+" (ID "+str(dep.op2.uid)+")"
            print "        Actual Dependences:"
            for dep in self.adeps:
                print "          index "+str(dep.idx1)+" of "+dep.op1.get_name()+" (ID "+str(dep.op1.uid)+") and index "+\
                                         str(dep.idx2)+" of "+dep.op2.get_name()+" (ID "+str(dep.op2.uid)+")"

        # For all the actual dependences make sure we have a dependence path between the
        # two different operations based on the dependences computed by the runtime
        count = 0
        errors = 0
        for adep in self.adeps:
            sys.stdout.write("    Checking dependence: %d \r" % (count))
            if not adep.op2.has_logical_path(adep.op1):
                print "    ERROR: Failed to compute mapping dependence between index "+str(adep.idx1)+ \
                  " of "+adep.op1.get_name()+" (ID "+str(adep.op1.uid)+") and index "+str(adep.idx2)+ \
                  " of "+adep.op2.get_name()+" (ID "+str(adep.op2.uid)+")"
                errors = errors + 1
            for op in self.ops:
                op.unmark_logical()
            count = count + 1

        # Now go through all the mdeps and see if there were any in there that were not
        # computed by the adep computation, which indicates that we computed an unnecessary dependence
        warnings = 0
        for mdep in self.mdeps:
            found = False
            for adep in self.adeps:
                if adep == mdep:
                    found = True
                    break
            if not found:
                print "    WARNING: Computed extra mapping dependence between index "+str(mdep.idx1)+ \
                  " of "+mdep.op1.get_name()+" (ID "+str(mdep.op1.uid)+") and index "+str(mdep.idx2)+ \
                  " of "+mdep.op2.get_name()+" (ID "+str(mdep.op2.uid)+") in context of task "+str(self.enclosing.name)
                warnings = warnings + 1

        print "    Mapping Dependence Errors: "+str(errors)
        print "    Mapping Dependence Warnings: "+str(warnings)

    def physical_traverse(self, component):
        if self.physical_marked:
            return
        self.physical_marked = True
        component.add_task(self)
        self.start_event.physical_traverse(component)
        self.term_event.physical_traverse(component)

    def physical_unmark(self):
        self.physical_marked = False

    def print_event_dependences(self, printer):
        self.start_event.print_prev_event_dependences(printer, self.node_name)

    def print_prev_event_dependences(self, printer, later_name):
        if later_name not in self.prev_event_deps:
            printer.println(self.node_name+' -> '+later_name+' [style=solid,color=black,penwidth=2];')
            self.prev_event_deps.add(later_name)

    def clear_prev_event_dependences(self):
        self.prev_event_deps = set()

    def print_filtered_dependences(self, printer, printed_nodes):
        assert self in printed_nodes
        self.start_event.print_prev_filtered_dependences(printer, self.node_name, printed_nodes) 

    def print_prev_filtered_dependences(self, printer, later_name, printed_nodes):
        if self not in printed_nodes:
            self.print_physical_node(printer)
            printed_nodes.add(self)
        if later_name not in self.prev_event_deps:
            printer.println(self.node_name+' -> '+later_name+' [style=solid,color=black,penwidth=2];')
            self.prev_event_deps.add(later_name)

    def print_physical_node(self, printer):
        index_string = ""
        if self.index_space:
            index_string = '\\nIndex\ Space\ Point\ '+self.point.to_string()
        label_string = str(self.enclosing.name)+index_string+'\\nUnique\ ID\ '+str(self.handle.uid)
        for idx,man in self.managers.iteritems():
            assert idx in self.requirements
            req = self.requirements[idx] 
            label_string = label_string+'\\nInst\ '+hex(man.inst.uid)+'\ '+req.dot_requirement() 
        printer.println(self.node_name+' [style=filled,label="'+label_string+ 
            '",fillcolor=lightskyblue,fontsize=14,fontcolor=black,shape=record,penwidth=2];')

    def set_manager(self, idx, manager):
        assert idx not in self.managers
        self.managers[idx] = manager

    def get_manager(self, idx):
        assert idx in self.managers
        return self.managers[idx]

    def event_graph_traverse(self, traverser):
        traverser.visit_task(self)

    def get_node_name(self):
        return self.node_name

    def get_name(self):
        return self.name

    def get_requirement(self, idx):
        if self.index_space:
            assert idx in self.requirements
            return self.requirements[idx]
        else:
            # Single task, just use the enclosing version
            return self.enclosing.get_requirement(idx)

    def check_data_flow(self):
        print "Performing data flow check for task "+self.get_name()
        # Check that all the true mapping dependences
        # have been satisfied
        for dep in self.mdeps:
            if dep.dtype == TRUE_DEPENDENCE:
                # Handle cross product of instances
                # in case we have multiple instances for an op
                for handle,inst1 in dep.op1.get_instances().iteritems():
                    for handle,inst2 in dep.op2.get_instances().iteritems():
                        # Check to see if they are still aliased
                        req1 = inst1.get_requirement(dep.idx1)
                        req2 = inst2.get_requirement(dep.idx2)
                        # If the second requirement is a reduction, there is no need to
                        # have a dataflow dependence since we can make a separate reduction instance
                        if req2.priv == REDUCE:
                            continue
                        assert req1.tid == req2.tid
                        index1 = self.state.get_index_node(True, req1.ispace)
                        index2 = self.state.get_index_node(True, req2.ispace)
                        if self.state.is_aliased(index1, index2):
                            def traverse_event(node, traverser):
                                if traverser.found:
                                    return False
                                return True
                            def traverse_task(node, traverser):
                                if node == traverser.target:
                                    traverser.found = True
                                if traverser.found:
                                    return False
                                return True
                            def traverse_map(node, traverser):
                                if node == traverser.target:
                                    traverser.found = True
                                if traverser.found:
                                    return False
                                return True
                            def traverse_copy(node, traverser):
                                if traverser.found:
                                    return False
                                # Check to see if we can have the matching
                                # manager on the stack
                                last_idx = len(traverser.manager_stack) - 1
                                assert last_idx >= 0 
                                last_man = traverser.manager_stack[last_idx]
                                if (last_man == node.dst_manager) and (traverser.field in node.fields):
                                    traverser.manager_stack.append(node.src_manager)
                                    return True
                                return False
                            def traverse_reduce(node, traverser):
                                if traverser.found:
                                    return False
                                last_idx = len(traverser.manager_stack) - 1
                                assert last_idx >= 0
                                last_man = traverser.manager_stack[last_idx]
                                if (last_man == node.dst_manager) and (traverser.field in node.fields):
                                    traverser.manager_stack.append(node.dst_manager)
                                    return True
                                return False
                            def post_traverse_event(node, traverser):
                                pass
                            def post_traverse_task(node, traverser):
                                pass
                            def post_traverse_map(node, traverser):
                                pass
                            def post_traverse_copy(node, traverser):
                                traverser.manager_stack.pop()
                            def post_traverse_reduce(node, traverser):
                                traverser.manager_stack.pop()
                            # Do the traversal for each field of the shared fields
                            fields = req1.fields & req2.fields
                            for f in fields:
                                traverser = EventGraphTraverser(False, traverse_event, traverse_task, traverse_map, traverse_copy, traverse_reduce, post_traverse_event, post_traverse_task, post_traverse_map, post_traverse_copy, post_traverse_reduce)
                                traverser.found = False
                                traverser.target = inst1
                                traverser.field = f
                                traverser.manager_stack = list()
                                # TODO: support virtual mappings here where an instance manager doesn't necessarily exist
                                traverser.manager_stack.append(inst2.get_manager(dep.idx2))
                                # Traverse and see if we can find inst1
                                inst2.event_graph_traverse(traverser)
                                # Sanity check, only one thing on the stack
                                assert len(traverser.manager_stack) == 1
                                if not traverser.found:
                                    print "   ERROR: Unable to find data flow path between requirement "+str(dep.idx1)+" of "+inst1.get_name()+" and requirement "+str(dep.idx2)+" of "+inst2.get_name()+" for field "+str(f)
            


class TaskOp(object):
    def __init__(self, state, uid, tid, enclosing, index_space, name):
        self.state = state
        self.uid = uid
        self.tid = tid
        # The enclosing task instance
        self.enclosing = enclosing
        self.index_space = index_space
        self.reqs = dict()
        self.instances = dict()
        self.term_event = None
        self.logical_incoming = set()
        self.logical_outgoing = set()
        self.logical_marked = False
        if name <> None:
            self.name = name
        else:
            self.name = str(tid)

    def add_requirement(self, index, is_reg, ispace, fspace, tid, priv, coher, redop):
        assert index not in self.reqs
        self.reqs[index] = Requirement(index, is_reg, ispace, fspace, tid, priv, coher, redop)

    def add_instance(self, point, inst):
        assert point not in self.instances
        self.instances[point] = inst
        # Anything that is not an index space should only have one instance
        if not self.index_space:
            assert len(self.instances) == 1
        elif self.term_event <> None: # Otherwise if we've already set up the term event add event dependences
            inst.term_event.add_physical_outgoing(self.term_event)
            self.term_event.add_physical_incoming(inst.term_event)

    def add_term_event(self, term_event):
        assert self.index_space
        self.term_event = term_event
        for point,inst in self.instances.iteritems():
            inst.term_event.add_physical_outgoing(self.term_event)
            self.term_event.add_physical_incoming(inst.term_event)

    def add_req_field(self, index, fid):
        assert index in self.reqs
        self.reqs[index].add_field(fid)

    def find_dependences(self, op):
        for idx,req in self.reqs.iteritems():
            op.find_individual_dependences(self, req)

    def find_individual_dependences(self, other_op, other_req):
      for idx,req in self.reqs.iteritems():
          dtype = self.state.compute_dependence(other_req, req)
          if is_mapping_dependence(dtype):
              self.enclosing.add_adep(other_op, self, other_req.index, req.index, dtype)

    def add_logical_incoming(self, op):
        if op == self:
            return
        assert op <> self
        self.logical_incoming.add(op)

    def add_logical_outgoing(self, op):
        if op == self:
            return
        assert op <> self
        self.logical_outgoing.add(op)

    def has_logical_path(self, target):
        if target == self:
            return True
        if self.logical_marked:
            return False
        # Otherwise check all the outgoing edges
        for op in self.logical_outgoing:
            if op.has_logical_path(target):
                return True
        self.logical_marked = True
        return False

    def unmark_logical(self):
        self.logical_marked = False

    def get_name(self):
        return self.name

    def get_requirement(self, idx):
        assert idx in self.reqs
        return self.reqs[idx]

    def get_instances(self):
        return self.instances


class Mapping(object):
    def __init__(self, state, uid, enclosing):
        self.state = state
        self.uid = uid
        self.enclosing = enclosing
        self.req = None
        self.logical_incoming = set()
        self.logical_outgoing = set()
        self.logical_marked = False
        self.physical_marked = False
        self.name = "Mapping "+str(uid)
        self.node_name = 'mapping_node_'+str(uid)
        self.prev_event_deps = set()
        self.manager = None

    def print_instance(self):
        print "Inline Mapping UID "+str(self.uid)
        print "  Requirement 0"
        self.req.print_requirement()

    def add_requirement(self, index, is_reg, ispace, fspace, tid, priv, coher, redop):
        assert index == 0
        self.req = Requirement(index, is_reg, ispace, fspace, tid, priv, coher, redop)

    def add_instance(self, start_event, term_event):
        self.start_event = start_event
        self.term_event = term_event

    def add_req_field(self, index, fid):
        assert index == 0
        self.req.add_field(fid)

    def add_logical_incoming(self,op):
        assert op <> self
        self.logical_incoming.add(op)

    def add_logical_outgoing(self,op):
        assert op <> self
        self.logical_outgoing.add(op)

    def uses_memory(self, mem):
        if self.manager.inst.memory == mem:
            return True
        return False

    def find_dependences(self, op):
        op.find_individual_dependences(self, self.req)

    def find_individual_dependences(self, other_op, other_req):
        dtype = self.state.compute_dependence(other_req, self.req)
        if is_mapping_dependence(dtype):
            self.enclosing.add_adep(other_op, self, other_req.index, self.req.index, dtype)

    def has_logical_path(self, target):
        if target == self:
            return True
        if self.logical_marked:
            return False
        # Otherwise check all the outgoing edges
        for op in self.logical_outgoing:
            if op.has_logical_path(target):
                return True
        self.logical_marked = True
        return False

    def unmark_logical(self):
        self.logical_marked = False

    def get_name(self):
        return self.name

    def physical_traverse(self, component):
        if self.physical_marked:
            return
        self.physical_marked = True
        component.add_map(self)
        self.start_event.physical_traverse(component)
        self.term_event.physical_traverse(component)

    def physical_unmark(self):
        self.physical_marked = False

    def print_event_dependences(self, printer):
        self.start_event.print_prev_event_dependences(printer, self.node_name)

    def print_prev_event_dependences(self, printer, later_name):
        if later_name not in self.prev_event_deps:
            printer.println(self.node_name+' -> '+later_name+' [style=solid,color=black,penwidth=2];')
            self.prev_event_deps.add(later_name)

    def clear_prev_event_dependences(self):
        self.prev_event_deps = set()

    def print_filtered_dependences(self, printer, printed_nodes):
        assert self in printed_nodes
        self.start_event.print_prev_filtered_dependences(printer, self.node_name, printed_nodes)

    def print_prev_filtered_dependences(self, printer, later_name, printed_nodes):
        if self not in printed_nodes:
            self.print_physical_nodes(printer)
            printed_nodes.add(self)
        if later_name not in self.prev_event_deps:
            printer.println(self.node_name+' -> '+later_name+' [style=solid,color=black,penwidth=2];')
            self.prev_event_deps.add(later_name)

    def print_physical_node(self, printer):
        printer.println(self.node_name+' [style=filled,label="Inline\ Mapping\ '+str(self.uid)+
            '",fillcolor=mediumseagreen,fontsize=14,fontcolor=black,shape=record,penwidth=2];')

    def set_manager(self, manager):
        assert self.manager == None
        self.manager = manager

    def get_manager(self, idx):
        assert idx == 0
        assert self.manager <> None
        return self.manager

    def event_graph_traverse(self, traverser):
        traverser.visit_mapping(self)

    def get_node_name(self):
        return self.node_name

    def get_requirement(self, idx):
        assert idx == 0
        assert self.req <> None
        return self.req

    def get_instances(self):
        result = dict()
        result[0] = self
        return result


class Deletion(object):
    def __init__(self, state, uid, enclosing):
        self.state = state
        self.uid = uid
        self.enclosing = enclosing
        self.logical_outgoing = set()
        self.name = "Deletion "+str(uid)

    def find_dependences(self, op):
        # No need to do anything
        pass

    def find_individual_dependences(self, other_op, other_req):
        # TODO: implement this for deletion
        pass

    def add_logical_incoming(self, op):
        # Should never happen
        assert False

    def add_logical_outgoing(self, op): 
        assert op <> self
        self.logical_outgoing.add(op)

    def has_path(self, target):
        if target == self:
            return True
        return False

    def unmark_logical(self):
        # No need to do anything
        pass

    def get_name(self):
        return self.name

    def get_instances(self):
        # No instances for deletions since
        # they don't have any true data dependences
        result = dict()
        return result

    def get_manager(self, idx):
        # Should never be called
        assert False
        return None


class EventHandle(object):
    def __init__(self, uid, gen):
        self.uid = uid
        self.gen = gen

    def __hash__(self):
        return hash((self.uid, self.gen))

    def __eq__(self, other):
        return (self.uid,self.gen) == (other.uid,other.gen)


class Requirement(object):
    def __init__(self, index, is_reg, ispace, fspace, tid, priv, coher, redop):
        self.index = index
        self.is_reg = is_reg
        self.ispace = ispace
        self.fspace = fspace
        self.tid = tid
        self.priv = priv
        self.coher = coher
        self.redop = redop
        self.fields = set()

    def print_requirement(self):
        if self.is_reg:
            print "    Logical Region Requirement ("+str(hex(self.ispace))+","+str(self.fspace)+","+str(self.tid)+")"
        else:
            print "    Logical Partition Requirement ("+str(self.ispace)+","+str(self.fsapce)+","+str(self.tid)+")"
        field_str = "    Fields: "
        first = True
        for f in self.fields:
            if first:
                field_str = field_str+str(f)
                first = False
            else:
                field_str = field_str+", "+str(f)
        print field_str
        print "    Privilege: "+self.get_privilege()
        print "    Coherence: "+self.get_coherence()

    def add_field(self, fid):
        assert fid not in self.fields
        self.fields.add(fid)

    def is_read_only(self):
        return (self.priv == NO_ACCESS) or (self.priv == READ_ONLY)

    def has_write(self):
        return (self.priv == READ_WRITE) or (self.priv == REDUCE) or (self.priv == WRITE_ONLY)

    def is_write(self):
        return (self.priv == READ_WRITE) or (self.priv == WRITE_ONLY)

    def is_write_only(self):
        return self.priv == WRITE_ONLY

    def is_reduce(self):
        return self.priv == REDUCE

    def is_exclusive(self):
        return self.coher == EXCLUSIVE

    def is_atomic(self):
        return self.coher == ATOMIC

    def is_simult(self):
        return self.coher == SIMULTANEOUS

    def is_relaxed(self):
        return self.coher == RELAXED

    def to_string(self):
        if self.is_reg:
            print "Region Requirement for ("+str(hex(self.ispace))+","+str(self.fspace)+","+str(self.tid)+")"
        else:
            print "Partition Requirement for ("+str(self.ispace)+","+str(self.fspace)+","+str(self.tid)+")"
        print "    Privilege: "+self.get_privilege()
        print "    Coherence: "+self.get_coherence()
        print "    Fields: "+self.get_fields()

    def get_privilege(self):
        if self.priv == NO_ACCESS:
            return "NO ACCESS"
        elif self.priv == READ_ONLY:
            return "READ-ONLY"
        elif self.priv == READ_WRITE:
            return "READ-WRITE"
        elif self.priv == WRITE_ONLY:
            return "WRITE-ONLY"
        else:
            assert self.priv == REDUCE
            return "REDUCE with Reduction Op "+str(self.redop)

    def get_coherence(self):
        if self.coher == EXCLUSIVE:
            return "EXCLUSIVE"
        elif self.coher == ATOMIC:
            return "ATOMIC"
        elif self.coher == SIMULTANEOUS:
            return "SIMULTANEOUS"
        else:
            assert self.coher == RELAXED
            return "RELAXED"

    def get_fields(self):
        result = ""
        for f in self.fields:
            result = result + str(f) + " "
        return result

    def dot_requirement(self):
        result = ''
        if self.priv == NO_ACCESS:
            result = result + "NA"
        elif self.priv == READ_ONLY:
            result = result + "RO"
        elif self.priv == READ_WRITE:
            result = result + "RW"
        elif self.priv == WRITE_ONLY:
            result = result + "WO"
        else:
            assert self.priv == REDUCE
            result = result + "Red"+str(self.redop)+'-'
        if self.coher == EXCLUSIVE:
            result = result + "E"
        elif self.coher == ATOMIC:
            result = result + "A"
        elif self.coher == SIMULTANEOUS:
            result = result + "S"
        else:
            assert self.coher == RELAXED
            result = result + "R"
        result = result + '\ Fields:'
        for f in self.fields:
            result = result + str(f) + ','
        return result


class MappingDependence(object):
    def __init__(self, ctx, op1, op2, idx1, idx2, dtype):
        self.ctx = ctx
        self.op1 = op1
        self.op2 = op2
        self.idx1 = idx1
        self.idx2 = idx2
        self.dtype = dtype

    def __eq__(self,other):
        return (self.ctx == other.ctx) and (self.op1 is other.op1) and (self.op2 is other.op2) and (self.idx1 == other.idx1) and (self.idx2 == other.idx2) and (self.dtype == other.dtype)


class Event(object):
    def __init__(self, state, handle):
        self.state = state
        self.handle = handle
        self.physical_incoming = set()
        self.physical_outgoing = set()
        self.physical_marked = False

    def add_physical_incoming(self, event):
        assert self <> event
        self.physical_incoming.add(event)

    def add_physical_outgoing(self, event):
        assert self <> event
        self.physical_outgoing.add(event)

    def physical_traverse(self, component):
        if self.physical_marked:
            return
        self.physical_marked = True
        component.add_event(self)
        for n in self.physical_incoming:
            n.physical_traverse(component)
        for n in self.physical_outgoing:
            n.physical_traverse(component)

    def physical_unmark(self):
        self.physical_marked = False

    def print_prev_event_dependences(self, printer, name):
        for n in self.physical_incoming:
            n.print_prev_event_dependences(printer, name)

    def print_prev_filtered_dependences(self, printer, name, printed_nodes):
        for n in self.physical_incoming:
            n.print_prev_filtered_dependences(printer, name, printed_nodes)

    def event_graph_traverse(self, traverser):
        traverser.visit_event(self)


class Point(object):
    def __init__(self, point_id, dim):
        self.point_id = point_id
        self.dim = dim
        self.values = list()

    def add_value(self, val):
        self.values.append(val)

    def to_string(self):
        result = '('
        first = True
        for val in self.values:
            if not(first):
                result = result + ','
            result = result + str(val)
            first = False
        result = result + ')'
        return result

    def to_simple_string(self):
        result = ''
        first = True
        for val in self.values:
            if not(first):
                result = result + '_'
            result = result + str(val)
            first = False
        return result


def parse_mask(mask):
    result = set()
    fields = string.split(mask,',')
    for f in fields:
        result.add(int(f))
    return result

class CopyInstance(object):
    def __init__(self, state, uid, src_manager, dst_manager, index, field, tree, start, term, mask):
        self.state = state
        self.uid = uid
        self.src_manager = src_manager
        self.dst_manager = dst_manager
        self.index_space = index
        self.field_space = field
        self.tree_id = tree
        self.start_event = start
        self.term_event = term
        self.mask = mask
        self.physical_marked = False
        self.name = 'copy_node_'+str(uid)
        self.prev_event_deps = set()
        self.fields = parse_mask(mask)
        assert len(self.fields) > 0

    def physical_traverse(self, component):
        if self.physical_marked:
            return
        self.physical_marked = True
        component.add_copy(self)
        self.start_event.physical_traverse(component)
        self.term_event.physical_traverse(component)

    def physical_unmark(self):
        self.physical_marked = False

    def uses_memory(self, mem):
        if self.src_manager.inst.memory == mem:
            return True
        if self.dst_manager.inst_memory == mem:
            return True
        return False

    def print_event_dependences(self, printer):
        self.start_event.print_prev_event_dependences(printer, self.name)

    def print_prev_event_dependences(self, printer, later_name):
        if later_name not in self.prev_event_deps:
            printer.println(self.name+' -> '+later_name+' [style=solid,color=black,penwidth=2];')
            self.prev_event_deps.add(later_name)

    def clear_prev_event_dependences(self):
        self.prev_event_deps = set()

    def print_filtered_dependences(self, printer, printed_nodes):
        assert self in printed_nodes
        self.start_event.print_prev_filtered_dependences(printer, self.name, printed_nodes)

    def print_prev_filtered_dependences(self, printer, later_name, printed_nodes):
        if self not in printed_nodes:
            self.print_physical_node(printer)
            printed_nodes.add(self)
        if later_name not in self.prev_event_deps:
            printer.println(self.name+' -> '+later_name+' [style=solid,color=black,penwidth=2];')
            self.prev_event_deps.add(later_name)

    def print_physical_node(self, printer):
        printer.println(self.name+' [style=filled,label="Src\ Inst:\ '+str(hex(self.src_manager.inst.uid))+'\ Src\ Loc:\ '+
            str(hex(self.src_manager.inst.memory.uid))+
            '\\nDst\ Inst:\ '+str(hex(self.dst_manager.inst.uid))+'\ Dst\ Loc:\ '+str(hex(self.dst_manager.inst.memory.uid))+
            '\\nLogical\ Region:\ (index:'+str(hex(self.index_space))+',field:'+str(self.field_space)+',tree:'+str(self.tree_id)+')'+
            '\\nCopy\ Fields:\ '+self.mask+
            '\\nCopy\ ID:\ '+str(self.uid)+
            '",fillcolor=darkgoldenrod1,fontsize=14,fontcolor=black,shape=record,penwidth=2];')

    def event_graph_traverse(self, traverser):
        traverser.visit_copy(self)

    def get_node_name(self):
        return self.name

    def get_redop(self):
        # Always the invalid reduction
        return 0


class ReduceInstance(object):
    def __init__(self, state, uid, src_manager, dst_manager, index, field, tree, start, term, redop, mask):
        self.state = state
        self.uid = uid
        self.src_manager = src_manager
        self.dst_manager = dst_manager
        self.index_space = index
        self.field_space = field
        self.tree_id = tree
        self.start_event = start
        self.term_event = term
        self.redop = redop
        self.mask = mask
        self.physical_marked = False
        self.name = 'reduce_node_'+str(uid)
        self.prev_event_deps = set()
        self.fields = parse_mask(mask)
        assert len(self.fields) > 0

    def physical_traverse(self, component):
        if self.physical_marked:
            return
        self.physical_marked = True
        component.add_reduction(self)
        self.start_event.physical_traverse(component)
        self.term_event.physical_traverse(component)

    def physical_unmark(self):
        self.physical_marked = False

    def uses_memory(self, mem):
        if self.src_manager.inst.memory == mem:
            return True
        if self.dst_manager.inst.memory == mem:
            return True
        return False

    def print_event_dependences(self, printer):
        self.start_event.print_prev_event_dependences(printer, self.name)

    def print_prev_event_dependences(self, printer, later_name):
        if later_name not in self.prev_event_deps:
            printer.println(self.name+' -> '+later_name+' [style=solid,color=black,penwidth=2];')
            self.prev_event_deps.add(later_name)

    def clear_prev_event_dependences(self):
        self.prev_event_deps = set()

    def print_filtered_dependences(self, printer, printed_nodes):
        assert self in printed_nodes
        self.start_event.print_prev_filtered_dependences(printer, self.node_name, printed_nodes)

    def print_prev_filtered_dependences(self, printer, later_name, printed_nodes):
        if self not in printed_nodes:
            self.print_physical_node(printer)
            printed_nodes.add(self)
        if later_name not in self.prev_event_deps:
            printer.println(self.node_name+' -> '+later_name+' [style=solid,color=black,penwidth=2];')
            self.prev_event_deps.add(later_name)

    def print_physical_node(self, printer):
        printer.println(self.name+' [style=filled,label="Src\ Inst:\ '+str(hex(self.src_manager.inst.uid))+'\ Src\ Loc:\ '+
            str(hex(self.src_manager.inst.memory.uid))+
            '\\nDst\ Inst:\ '+str(hex(self.dst_manager.inst.uid))+'\ Dst\ Loc:\ '+str(hex(self.dst_manager.inst.memory.uid))+
            '\\nLogical\ Region:\ (index:'+str(hex(self.index_space))+',field:'+str(self.field_space)+',tree:'+str(self.tree_id)+')'+
            '\\nReduce\ Fields:\ '+self.mask+
            '\\nReduction\ Op:\ '+str(self.redop)+
            '\\nReduction\ ID:\ '+str(self.uid)+
            '",fillcolor=crimson,fontsize=14,fontcolor=black,shape=record,penwidth=2];')

    def event_graph_traverse(self, traverser):
        traverser.visit_reduce(self)

    def get_node_name(self):
        return self.name

    def get_redop(self):
        return self.redop


class PhysicalInstance(object):
    def __init__(self, state, uid, memory, region):
        self.state = state
        self.uid = uid
        self.memory = memory
        self.region = region
        self.managers = dict()
        # Add ourselves to the list of instances
        memory.add_physical_instance(self)

    def add_manager(self, manager):
        assert manager.uid not in self.managers
        assert manager.inst == self
        self.managers[manager.uid] = manager


class ReductionInstance(object):
    def __init__(self, state, uid, memory, region, foldable, indirect):
        assert foldable or (indirect > 0)
        self.state = state
        self.uid = uid
        self.memory = memory
        self.region = region
        self.foldable = foldable
        self.indirect = indirect
        self.managers = dict()
        memory.add_reduction_instance(self)

    def add_manager(self, manager):
        assert manager.uid not in self.managers
        assert manager.inst == self
        self.managers[manager.uid] = manager


class TaskUser(object):
    def __init__(self, task_inst, idx):
        self.task_inst = task_inst
        self.indexes = set()
        self.indexes.add(idx)

    def add_index(self, idx):
        assert idx not in self.indexes
        self.indexes.add(idx)

class CopyUser(object):
    def __init__(self, copy_op, writing):
        self.op = copy_op
        self.writing = writing

class ReduceUser(object):
    def __init__(self, reduce_op, writing):
        self.op = reduce_op
        self.writing = writing


class InstanceManager(object):
    def __init__(self, state, uid, inst):
        self.state = state
        self.uid = uid
        self.inst = inst
        self.task_users = dict()
        self.map_users = dict()
        self.copy_users = dict()
        self.reduce_users = dict()

    def add_task_user(self, handle, task_inst, idx):
        #if handle not in self.task_users:
        #    self.task_users[handle] = TaskUser(task_inst, idx)
        #else:
        #    self.task_users[handle].add_index(idx)
        task_inst.set_manager(idx, self)

    def add_map_user(self, uid, mapping):
        assert uid not in self.map_users
        self.map_users[uid] = TaskUser(mapping, 0)
        mapping.set_manager(self)

    def add_copy_user(self, copy_op, writing):
        assert copy_op.uid not in self.copy_users
        self.copy_users[copy_op.uid] = CopyUser(copy_op, writing)

    def add_reduce_user(self, reduce_op, writing):
        assert writing # Better be reducing to this instance
        assert reduce_op.uid not in self.reduce_users
        self.reduce_users[reduce_op.uid] = ReduceUser(reduce_op, writing)

    def check_instance_dependences(self):
        if len(self.task_users) == 0 and len(self.copy_users) == 0 and len(self.reduce_users) == 0:
            return
        self.dependences = set()
        for h,u in self.task_users.iteritems():
            self.find_transitive_dependences(u.task_inst)
        for h,u in self.map_users.iteritems():
            self.find_transitive_dependences(u.task_inst)
        for h,u in self.copy_users.iteritems():
            self.find_transitive_dependences(u.op)
        for h,u in self.reduce_users.iteritems():
            self.find_transitive_dependences(u.op) 
        print "Performing data-race detection for manager "+str(self.uid)+" of instance "+str(hex(self.inst.uid))+" in memory "+str(hex(self.inst.memory.uid))
        # Now do all of the dependence checks between all pairs
        for h1,u1 in self.task_users.iteritems():
            for h2,u2 in self.task_users.iteritems():
                if u1 == u2:
                    continue
                if (u1.task_inst,u2.task_inst) in self.dependences:
                    continue
                if (u2.task_inst,u1.task_inst) in self.dependences:
                    continue
                self.task_task_analysis(u1,u2)
            for h2,m2 in self.map_users.iteritems():
                if (u1.task_inst,m2.task_inst) in self.dependences:
                    continue
                if (m2.task_inst,u1.task_inst) in self.dependences:
                    continue
                self.task_map_analysis(u1,m2)
            for h2,c2 in self.copy_users.iteritems():
                if (u1.task_inst,c2.op) in self.dependences:
                    continue
                if (c2.op,u1.task_inst) in self.dependences:
                    continue
                self.task_copy_analysis(u1,c2)
            for h2,r2 in self.reduce_users.iteritems():
                if (u1.task_inst,r2.op) in self.dependences:
                    continue
                if (r2.op,u1.task_inst) in self.dependences:
                    continue
                self.task_reduce_analysis(u1,r2)
        for h1,m1 in self.map_users.iteritems():
            for h2,m2 in self.map_users.iteritems():
                if m1 == m2:
                    continue
                if (m1.task_inst,m2.task_inst) in self.dependences:
                    continue
                if (m2.task_inst,m1.task_inst) in self.dependences:
                    continue
                self.map_map_analysis(m1,m2)
            for h2,c2 in self.copy_users.iteritems():
                if (m1.task_inst,c2.op) in self.dependences:
                    continue
                if (c2.op,m1.task_inst) in self.dependences:
                    continue
                self.map_copy_analysis(m1,c2)
            for h2,r2 in self.reduce_users.iteritems():
                if (m1.task_inst,r2.op) in self.dependences:
                    continue
                if (r2.op,m1.task_inst) in self.dependences:
                    continue
                self.map_reduce_analysis(m1,r2)
        for h1,c1 in self.copy_users.iteritems():
            for h2,c2 in self.copy_users.iteritems():
                if c1 == c2:
                    continue
                if (c1.op,c2.op) in self.dependences:
                    continue
                if (c2.op,c1.op) in self.dependences:
                    continue
                self.copy_copy_analysis(c1, c2)
            for h2,r2 in self.reduce_users.iteritems():
                if (c1.op,r2.op) in self.dependences:
                    continue
                if (r2.op,c1.op) in self.dependences:
                    continue
                self.copy_reduce_analysis(c1,r2)
        for h1,r1 in self.reduce_users.iteritems():
            for h2,r2 in self.reduce_users.iteritems():
                if r1 == r2:
                    continue
                if (r1.op,r2.op) in self.dependences:
                    continue
                if (r2.op,r1.op) in self.dependences:
                    continue
                self.reduce_reduce_analysis(r1,r2)
                  
    def task_task_analysis(self, t1, t2):
        # Get the region requirements and figure out the dependence type
        for idx1 in t1.indexes:
            for idx2 in t2.indexes:
                req1 = t1.task_inst.get_requirement(idx1)
                req2 = t2.task_inst.get_requirement(idx2)
                self.req_req_analysis(req1,req2, "index "+str(idx1)+" of task instance "+str(t1.task_inst.name),
                                                  "index "+str(idx2)+" of task instance "+str(t2.task_inst.name))
                

    def task_map_analysis(self, t, m):
        for idx in t.indexes:
            req1 = t.task_inst.get_requirement(idx)
            req2 = m.task_inst.req
            self.req_req_analysis(req1,req2,"index "+str(idx)+" of task instance "+str(t.task_inst.name),
                                            "mapping "+str(m.task_inst.uid))
        
    def task_copy_analysis(self, t, c):
        for idx in t.indexes:
            req = t.task_inst.get_requirement(idx)
            self.req_op_analysis(req, c, "index "+str(idx)+" of task instance "+str(t.task_inst.name),
                                         "copy operation "+str(c.op.uid))

    def task_reduce_analysis(self, t, r):
        for idx in t.indexes:
            req = t.task_inst.get_requirement(idx)
            self.req_op_analysis(req, r, "index "+str(idx)+" of task instance "+str(t.task_inst.name),
                                         "reduce operation "+str(r.op.uid))

    def map_map_analysis(self, m1, m2):
        req1 = m1.task_inst.req
        req2 = m2.task_inst.req
        self.req_req_analysis(req1,req2,"mapping "+str(m1.task_inst.uid), "mapping "+str(m2.task_inst.uid))

    def map_copy_analysis(self, m, c):
        self.req_op_analysis(m.task_inst.req, c, "mapping "+str(m.task_inst.uid), "copy operation "+str(c.op.uid))

    def map_reduce_analysis(self, m, r):
        self.req_op_analysis(m.task_inst.req, r, "mapping "+str(m.task_inst.uid), "reduce operation "+str(r.op.uid))

    def copy_copy_analysis(self, c1, c2):
        self.op_op_analysis(c1, c2, "copy operation "+str(c1.op.uid), "copy_operation "+str(c2.op.uid))

    def copy_reduce_analysis(self, c, r):
        self.op_op_analysis(c, r, "copy operation "+str(c.op.uid), "reduce operation "+str(r.op.uid))

    def reduce_reduce_analysis(self, r1, r2):
        self.op_op_analysis(r1, r2, "reduce operation "+str(r1.op.uid), "reduce operation "+str(r2.op.uid))

    def req_req_analysis(self, req1, req2, name1, name2):
        dep = self.state.compute_dependence(req1,req2)
        if (dep == TRUE_DEPENDENCE) or (dep == ANTI_DEPENDENCE):
            self.report_error(name1,name2)

    def req_op_analysis(self, req, op, name1, name2):
        # Do all the disjointness tests 
        if len(req.fields & op.op.fields) == 0:
            return # Independent fields
        assert req.tid == op.op.tree_id # Sanity check: should always be the same tree
        assert req.is_reg
        req_index = self.state.get_index_node(True, req.ispace)
        op_index = self.state.get_index_node(True, op.op.index_space)
        if not self.state.is_aliased(req_index, op_index):
            return # Independent parts of the region tree 
        if ((req.priv == READ_ONLY) or (req.priv == NO_ACCESS)) and not op.writing:
            return # Two readers
        # Same kind of reduction is ok too
        if (req.redop > 0) and (req.redop == op.op.get_redop()):
            return # Same reduction operation
        # Otherwise we've got a problem
        self.report_error(name1, name2)

    def op_op_analysis(self, op1, op2, name1, name2):
        # Do all the disjointness tests
        if len(op1.op.fields & op2.op.fields) == 0:
            return # Independent fields
        assert op1.op.tree_id == op2.op.tree_id # Sanity check: should always be the same tree
        op1_index = self.state.get_index_node(True, op1.op.index_space)
        op2_index = self.state.get_index_node(True, op2.op.index_space)
        if not self.state.is_aliased(op1_index, op2_index):
            return # Independent parts of the region tree
        if not op1.writing and not op2.writing:
            return # Both readers, doesn't matter
        if (op1.op.get_redop() > 0) and (op1.op.get_redop() == op2.op.get_redop()):
            return # Same reduction is also ok
        # Otherwise we have a problem
        self.report_error(name1, name2)


    def report_error(self, one, two):
        print "    ERROR: Missing event dependence between "+one+" and "+two

    def find_transitive_dependences(self, user):
        def traverse_event(node, traverser):
            if node in traverser.visited_events:
                return False
            traverser.visited_events.add(node)
            return True
        def traverse_task(node, traverser):
            if node in traverser.visited_tasks:
                return False
            traverser.visited_tasks.add(node)
            return True
        def traverse_map(node, traverser):
            if node in traverser.visited_maps:
                return False
            traverser.visited_maps.add(node)
            return True
        def traverse_copy(node, traverser):
            if node in traverser.visited_copies:
                return False
            traverser.visited_copies.add(node)
            return True
        def traverse_reduce(node, traverser):
            if node in traverser.visited_reduces:
                return False
            traverser.visited_reduces.add(node)
            return True
        traverser = EventGraphTraverser(False, traverse_event, traverse_task, traverse_map, traverse_copy, traverse_reduce)
        traverser.visited_events = set()
        traverser.visited_tasks = set()
        traverser.visited_maps = set()
        traverser.visited_copies = set()
        traverser.visited_reduces = set()
        user.event_graph_traverse(traverser)
        for t in traverser.visited_tasks:
            if t == user:
                continue
            if t.handle in self.task_users:
                self.dependences.add((t, user))
        for m in traverser.visited_maps:
            if m == user:
                continue
            if m.uid in self.map_users:
                self.dependences.add((m, user))
        for c in traverser.visited_copies:
            if c == user:
                continue
            if c.uid in self.copy_users:
                self.dependences.add((c, user))
        for r in traverser.visited_reduces:
            if r == user:
                continue
            if r.uid in self.reduce_users:
                self.dependences.add((r, user))

    def print_instance_graphs(self, path):
        if len(self.task_users) == 0 and len(self.copy_users) == 0 and len(self.reduce_users) == 0:
            return
        self.dependences = set()
        for h,u in self.task_users.iteritems():
            self.find_dependences(u.task_inst)
        for h,u in self.map_users.iteritems():
            self.find_dependences(u.task_inst)
        for h,u in self.copy_users.iteritems():
            self.find_dependences(u.op)
        for h,u in self.reduce_users.iteritems():
            self.find_dependences(u.op)
        name = 'users_of_manager_'+str(self.uid)+'_of_instance_'+str(self.inst.uid)+'_in_memory_'+str(self.inst.memory.uid)
        printer = GraphPrinter(path,name)
        for h,u in self.task_users.iteritems():
            u.task_inst.print_physical_node(printer)
        for h,u in self.map_users.iteritems():
            u.task_inst.print_physical_node(printer)
        for h,u in self.copy_users.iteritems():
            u.op.print_physical_node(printer)
        for h,u in self.reduce_users.iteritems():
            u.op.print_physical_node(printer)
        self.print_dependences(printer)
        dot_file = printer.close()
        pdf_file = name+'.pdf'
        try:
            subprocess.check_call(['dot', '-Tpdf', '-o', pdf_file, dot_file])
        except:
            print "WARNING: DOT failure, image for "+pdf_file+" not generated"
            subprocess.call(['rm', '-f', 'core', pdf_file])

    def find_dependences(self, user):
        def traverse_event(node, traverser):
            if node in traverser.visited_events:
                return False
            traverser.visited_events.add(node)
            return True
        def traverse_task(node, traverser):
            if node in traverser.visited_tasks:
                return False
            traverser.visited_tasks.add(node)
            if node == traverser.first:
                return True
            if node.handle in traverser.manager.task_users:
                return False
            return True
        def traverse_map(node, traverser):
            if node in traverser.visited_maps:
                return False
            traverser.visited_maps.add(node)
            if node == traverser.first:
                return True
            if node.uid in traverser.manager.map_users:
                return False
            return True
        def traverse_copy(node, traverser):
            if node in traverser.visited_copies:
                return False
            traverser.visited_copies.add(node)
            if node == traverser.first:
                return True
            if node.uid in traverser.manager.copy_users:
                return False
            return True
        def traverse_reduce(node, traverser):
            if node in traverser.visited_reduces:
                return False
            traverser.visited_reduces.add(node)
            if node == traverser.first:
                return True
            if node.uid in traverser.manager.reduce_users:
                return False
            return True
        traverser = EventGraphTraverser(False, traverse_event, traverse_task, traverse_map, traverse_copy, traverse_reduce)
        # It's unclear to me if python functions are thunks or whether they are stateless, so we'll just be safe
        traverser.manager = self
        traverser.first = user
        traverser.visited_events = set()
        traverser.visited_tasks = set()
        traverser.visited_maps = set()
        traverser.visited_copies = set()
        traverser.visited_reduces = set()
        user.event_graph_traverse(traverser)
        for t in traverser.visited_tasks:
            if t == user:
                continue
            if t.handle in self.task_users:
                self.dependences.add((t, user))
        for m in traverser.visited_maps:
            if m == user:
                continue
            if m.uid in self.map_users:
                self.dependences.add((m, user))
        for c in traverser.visited_copies:
            if c == user:
                continue
            if c.uid in self.copy_users:
                self.dependences.add((c, user))
        for r in traverser.visited_reduces:
            if r == user:
                continue
            if r.uid in self.reduce_users:
                self.dependences.add((r, user)) 

    def print_dependences(self, printer):
        for dep in self.dependences:
            printer.println(dep[0].get_node_name()+' -> '+dep[1].get_node_name()+' [style=solid,color=black,penwidth=2];')


class ReductionManager(object):
    def __init__(self, state, uid, inst):
        self.state = state
        self.uid = uid
        self.inst = inst
        self.task_users = dict()
        self.reduce_users = set()

    def add_task_user(self, handle, task_inst, idx):
        assert handle not in self.task_users
        #self.task_users[handle] = TaskUser(task_inst, idx)
        task_inst.set_manager(idx, self)

    def add_map_user(self, uid, mapping):
        mapping.set_manager(self)

    def add_reduce_user(self, reduce_op, writing):
        self.reduce_users.add(ReduceUser(reduce_op, writing))


class EventGraphTraverser(object):
    def __init__(self, forwards, event_fn = None, task_fn = None, map_fn = None, copy_fn = None, reduce_fn = None, post_event_fn = None, post_task_fn = None, post_map_fn = None, post_copy_fn = None, post_reduce_fn = None):
        self.forwards = forwards
        self.event_fn = event_fn
        self.task_fn = task_fn
        self.map_fn = map_fn
        self.copy_fn = copy_fn
        self.reduce_fn = reduce_fn
        self.post_event_fn = post_event_fn
        self.post_task_fn = post_task_fn
        self.post_map_fn = post_map_fn
        self.post_copy_fn = post_copy_fn
        self.post_reduce_fn = post_reduce_fn

    def visit_event(self, node):
        do_next = True
        if self.event_fn <> None:
            do_next = self.event_fn(node, self)
        if not do_next:
            return
        if self.forwards:
            for n in node.physical_outgoing:
                n.event_graph_traverse(self)
        else:
            for n in node.physical_incoming:
                n.event_graph_traverse(self)
        if self.post_event_fn <> None:
            self.post_event_fn(node, self)

    def visit_task(self, node):
        do_next = True
        if self.task_fn <> None:
            do_next = self.task_fn(node, self)
        if not do_next:
            return
        if self.forwards:
            node.term_event.event_graph_traverse(self)
        else:
            node.start_event.event_graph_traverse(self)
        if self.post_task_fn <> None:
            self.post_task_fn(node, self)

    def visit_mapping(self, node):
        do_next = True
        if self.map_fn <> None:
            do_next = self.map_fn(node, self)
        if not do_next:
            return
        if self.forwards:
            node.term_event.event_graph_traverse(self)
        else:
            node.start_event.event_graph_traverse(self)
        if self.post_map_fn <> None:
            self.post_map_fn(node, self)

    def visit_copy(self, node):
        do_next = True
        if self.copy_fn <> None:
            do_next = self.copy_fn(node, self)
        if not do_next:
            return
        if self.forwards:
            node.term_event.event_graph_traverse(self)
        else:
            node.start_event.event_graph_traverse(self)
        if self.post_copy_fn <> None:
            self.post_copy_fn(node, self)

    def visit_reduce(self, node):
        do_next = True
        if self.reduce_fn <> None:
            do_next = self.reduce_fn(node, self)
        if not do_next:
            return
        if self.forwards:
            node.term_event.event_graph_traverse(self)
        else:
            node.start_event.event_graph_traverse(self)
        if self.post_reduce_fn <> None:
            self.post_reduce_fn(node, self)


class ConnectedComponent(object):
    def __init__(self, state):
        self.state = state
        self.events = set()
        self.tasks = set()
        self.maps = set()
        self.copies = set()
        self.reductions = set()

    def add_event(self, event):
        assert event not in self.events
        self.events.add(event)

    def add_task(self, task):
        assert task not in self.tasks
        self.tasks.add(task)

    def add_map(self, mapp):
        assert mapp not in self.maps
        self.maps.add(mapp)

    def add_copy(self, copy):
        assert copy not in self.copies
        self.copies.add(copy)

    def add_reduction(self, reduction):
        assert reduction not in self.reductions
        self.reductions.add(reduction)

    def empty(self):
        if ((len(self.tasks) == 0) or (len(self.tasks) == 1)) and len(self.maps) == 0 and len(self.copies) == 0 and len(self.reductions) == 0:
            return True
        return False

    def generate_graph(self, idx, path):
        name = 'event_graph_'+str(idx)
        printer = GraphPrinter(path,name)
        # Print the nodes
        for t in self.tasks:
            t.print_physical_node(printer)
        for m in self.maps:
            m.print_physical_node(printer)
        for c in self.copies:
            c.print_physical_node(printer)
        for r in self.reductions:
            r.print_physical_node(printer)
        # Now print the dependences
        for t in self.tasks:
            t.print_event_dependences(printer)
        for m in self.maps:
            m.print_event_dependences(printer)
        for c in self.copies:
            c.print_event_dependences(printer) 
        for r in self.reductions:
            r.print_event_dependences(printer)
        dot_file = printer.close()
        pdf_file = name+'.pdf'
        try:
            subprocess.check_call(['dot', '-Tpdf', '-o', pdf_file, dot_file])
        except:
            print "WARNING: DOT failure, image for event graph "+str(idx)+" not generated"
            subprocess.call(['rm', '-f', 'core', pdf_file])
        self.unmark_all()

    def unmark_all(self):
        for t in self.tasks:
            t.physical_unmark()
        for m in self.maps:
            m.physical_unmark()
        for c in self.copies:
            c.physical_unmark()
        for r in self.reductions:
            r.physical_unmark()
        for e in self.events:
            e.physical_unmark()

class GraphPrinter(object):
    def __init__(self,path,name):
        self.filename = path+name+'.dot'
        self.out = open(self.filename,'w')
        self.depth = 0
        self.println('digraph '+name)
        self.println('{')
        self.down()
        #self.println('aspect = ".00001,100";')
        #self.println('ratio = 1;')
        #self.println('size = "10,10";')
        self.println('compound = true;')
        self.println('rankdir="LR";')
        self.println('size = "36,36";')

    def close(self):
        self.up()
        self.println('}')
        self.out.close()
        return self.filename

    def up(self):
        assert self.depth > 0
        self.depth = self.depth-1

    def down(self):
        self.depth = self.depth+1

    def println(self,string):
        for i in range(self.depth):
            self.out.write('  ')
        self.out.write(string)
        self.out.write('\n')


class State(object):
    def __init__(self, verbose):
        self.processors = dict()
        self.utilities = dict()
        self.memories = dict()
        self.index_space_nodes = dict()
        self.index_part_nodes = dict()
        self.field_space_nodes = dict()
        self.region_trees = dict()
        self.task_instances = dict()
        self.task_ops = dict()
        self.slice_index = dict()
        self.slice_slice = dict()
        self.point_slice = dict()
        self.point_point = dict()
        self.mappings = dict()
        self.deletions = dict()
        self.events = dict()
        self.copies = set()
        self.reduces = set()
        self.physical_instances = dict()
        self.reduction_instances = dict()
        self.physical_managers = dict()
        self.reduction_managers = dict()
        self.points = dict()
        self.next_copy = 1
        self.next_reduce = 1
        self.verbose = verbose 

    def add_utility(self, pid):
        assert pid not in self.utilities
        self.utilities[pid] = UtilityProcessor(self, pid)

    def add_processor(self, pid, util, kind):
        assert pid not in self.processors
        if util not in self.utilities:
            return False
        self.processors[pid] = Processor(self, pid, self.utilities[util], kind)
        return True

    def add_memory(self, mid, capacity):
        assert mid not in self.memories
        self.memories[mid] = Memory(self, mid, capacity)

    def set_proc_mem(self, pid, mid, bandwidth, latency):
        if pid not in self.processors:
            return False
        if mid not in self.memories:
            return False
        self.processors[pid].add_memory(self.memories[mid], bandwidth, latency)
        self.memories[mid].add_processor(self.processors[pid], bandwidth, latency)
        return True

    def set_mem_mem(self, mem1, mem2, bandwidth, latency):
        if mem1 not in self.memories:
            return False
        if mem2 not in self.memories:
            return False
        self.memories[mem1].add_memory(self.memories[mem2], bandwidth, latency)
        self.memories[mem2].add_memory(self.memories[mem1], bandwidth, latency)
        return True

    def add_point(self, node, pid, dim, val1=0, val2=0, val3=0):
        if node not in self.points:
            self.points[node] = dict()
        assert pid not in self.points[node]
        self.points[node][pid] = Point(pid, dim)
        if dim > 0:
          self.points[node][pid].add_value(val1)
        if dim > 1:
          self.points[node][pid].add_value(val2)
        if dim > 2:
          self.points[node][pid].add_value(val3)

    def get_point(self, node, pid):
        assert node in self.points
        assert pid in self.points[node]
        return self.points[node][pid]

    def add_index_space(self, uid):
        assert uid not in self.index_space_nodes
        self.index_space_nodes[uid] = IndexSpaceNode(self, uid, 0, None)

    def add_index_partition(self, pid, uid, disjoint, color):
        assert uid not in self.index_part_nodes
        if pid not in self.index_space_nodes:
            return False
        self.index_part_nodes[uid] = IndexPartNode(self, uid, disjoint, color, self.index_space_nodes[pid])
        return True

    def add_index_subspace(self, pid, uid, color):
        assert uid not in self.index_space_nodes
        if pid not in self.index_part_nodes:
            return False
        self.index_space_nodes[uid] = IndexSpaceNode(self, uid, color, self.index_part_nodes[pid])
        return True

    def add_field_space(self, uid):
        assert uid not in self.field_space_nodes
        self.field_space_nodes[uid] = FieldSpaceNode(self, uid)

    def add_field(self, uid, fid):
        if uid not in self.field_space_nodes:
            return False
        self.field_space_nodes[uid].add_field(fid)
        return True

    def add_region(self, iid, fid, tid):
        assert tid not in self.region_trees
        if iid not in self.index_space_nodes:
            return False
        if fid not in self.field_space_nodes:
            return False
        self.region_trees[tid] = self.index_space_nodes[iid].instantiate(None, self.field_space_nodes[fid], tid)
        return True

    def add_top_task(self, hid, gen, uid, ctx, tid):
        assert uid not in self.task_ops
        self.task_ops[uid] = TaskOp(self, uid, tid, None, False, None)

    def add_task(self, uid, tid, pid, ctx, hid, gen, index_space):
        parent_handle = TaskHandle(pid, ctx, hid, gen)
        if parent_handle not in self.task_instances:
            return False
        task_op = TaskOp(self, uid, tid, self.task_instances[parent_handle], index_space, None)
        assert uid not in self.task_ops
        self.task_ops[uid] = task_op
        # Tell the parent task instance about the operation
        self.task_instances[parent_handle].append_op(task_op)
        return True

    def add_mapping(self, uid, pid, ctx, hid, gen, implicit):
        parent_handle = TaskHandle(pid, ctx, hid, gen)
        if parent_handle not in self.task_instances:
            return False
        mapping = Mapping(self, uid, self.task_instances[parent_handle])
        assert uid not in self.mappings
        self.mappings[uid] = mapping
        if not implicit:
            self.task_instances[parent_handle].append_op(mapping)
        return True

    def add_deletion(self, uid, pid, ctx, hid, gen):
        parent_handle = TaskHandle(pid, ctx, hid, gen)
        if parent_handle not in self.task_instances:
            return False
        deletion = Deletion(self, uid, self.task_instances[parent_handle])
        assert uid not in self.deletions
        self.deletions[uid] = deletion
        self.task_instances[parent_handle].append_op(deletion)
        return True

    def add_name(self, uid, name):
        # Who cares if we miss some names
        if uid not in self.task_ops:
            return True 
        self.task_ops[uid].name = name
        return True
    
    def add_index_slice(self, index_id, slice_id):
        if index_id not in self.task_ops:
            return False
        self.slice_index[slice_id] = index_id
        return True

    def add_slice_slice(self, slice_id1, slice_id2):
        if slice_id1 not in self.slice_index and slice_id1 not in self.slice_slice:
            return False
        assert slice_id1 <> slice_id2
        self.slice_slice[slice_id2] = slice_id1
        return True

    def add_slice_point(self, slice_id, point_id):
        if slice_id not in self.slice_slice and slice_id not in self.slice_index:
            return False
        self.point_slice[point_id] = slice_id
        return True

    def add_point_point(self, point_id1, point_id2):
        if point_id1 <> point_id2 and point_id1 not in self.point_point:
            return False
        self.point_point[point_id2] = point_id1
        return True

    def add_requirement(self, uid, index, is_reg, ispace, fspace, tid, priv, coher, redop):
        if uid in self.mappings:
            self.mappings[uid].add_requirement(index, is_reg, ispace, fspace, tid, priv, coher, redop)
        else:
            if uid not in self.task_ops:
                return False
            self.task_ops[uid].add_requirement(index, is_reg, ispace, fspace, tid, priv, coher, redop)
        return True

    def add_req_field(self, uid, index, fid):
        if uid in self.mappings:
            self.mappings[uid].add_req_field(index, fid)
        else:
            if uid not in self.task_ops:
                return False
            self.task_ops[uid].add_req_field(index, fid)
        return True

    def add_mapping_dependence(self, pid, ctx, hid, gen, prev_id, pidx, next_id, nidx, dtype):
        if not self.has_op(prev_id):
            return False
        if not self.has_op(next_id):
            return False
        prev_op = self.get_op(prev_id)
        next_op = self.get_op(next_id)
        handle = TaskHandle(pid, ctx, hid, gen)
        if not handle in self.task_instances:
            return False
        self.task_instances[handle].add_mdep(prev_op, next_op, pidx, nidx, dtype)
        prev_op.add_logical_incoming(next_op)
        next_op.add_logical_outgoing(prev_op)
        return True

    def add_instance_requirement(self, uid, ctx, gen, hid, idx, index):
        handle = TaskHandle(uid, ctx, hid, gen)
        if handle not in self.task_instances:
            if not uid in self.task_ops:
                return False
            # Has to be an index space if we are doing this
            self.task_instances[handle] = TaskInstance(self, handle, self.task_ops[uid], True)
        self.task_instances[handle].update_requirement(idx, index)
        return True

    def add_task_instance(self, uid, ctx, hid, gen, index_space, point, startid, startgen, termid, termgen):
        if uid not in self.point_slice and uid not in self.point_point:
            return False
        if uid in self.point_point:
            enclosing_uid = self.point_point[uid]
        else:
            enclosing_uid = self.find_enclosing_point_uid(uid)
        assert enclosing_uid in self.task_ops
        handle = TaskHandle(uid, ctx, hid, gen) 
        if handle not in self.task_instances:
            self.task_instances[handle] = TaskInstance(self, handle, self.task_ops[enclosing_uid], index_space)
        start_event = self.get_event(EventHandle(startid,startgen))
        term_event = self.get_event(EventHandle(termid,termgen))
        task_inst = self.task_instances[handle]
        task_inst.update_info(point, start_event, term_event)
        self.task_ops[enclosing_uid].add_instance(point,task_inst)
        start_event.add_physical_outgoing(task_inst)
        term_event.add_physical_incoming(task_inst)
        return True

    def add_event_dependence(self, id1, gen1, id2, gen2):
        e1 = self.get_event(EventHandle(id1,gen1))
        e2 = self.get_event(EventHandle(id2,gen2))
        e1.add_physical_outgoing(e2)
        e2.add_physical_incoming(e1)

    def add_index_term(self, uid, termid, termgen):
        if uid not in self.task_ops:
            return False
        term_event = self.get_event(EventHandle(termid,termgen))
        self.task_ops[uid].add_term_event(term_event)
        return True

    def add_map_instance(self, uid, startid, startgen, termid, termgen):
        if uid not in self.mappings:
            return False
        start_event = self.get_event(EventHandle(startid,startgen))
        term_event = self.get_event(EventHandle(termid,termgen))
        self.mappings[uid].add_instance(start_event,term_event)
        start_event.add_physical_outgoing(self.mappings[uid])
        term_event.add_physical_incoming(self.mappings[uid])
        return True

    def add_copy_instance(self, srcman, dstman, index, field, tree, startid, startgen, termid, termgen, mask):
        if srcman not in self.physical_managers:
            return False
        if dstman not in self.physical_managers:
            return False
        src_manager = self.physical_managers[srcman]
        dst_manager = self.physical_managers[dstman]
        start_event = self.get_event(EventHandle(startid,startgen))
        term_event = self.get_event(EventHandle(termid,termgen))
        copy_op = CopyInstance(self, self.next_copy, src_manager, dst_manager, index, field, tree, start_event, term_event, mask)
        self.copies.add(copy_op)
        self.next_copy = self.next_copy + 1
        start_event.add_physical_outgoing(copy_op)
        term_event.add_physical_incoming(copy_op)
        src_manager.add_copy_user(copy_op, False)
        dst_manager.add_copy_user(copy_op, True)
        return True

    def add_reduce_instance(self, srcman, dstman, index, field, tree, startid, startgen, termid, termgen, redop, mask):
        if srcman not in self.reduction_managers:
            return False
        src_manager = self.reduction_managers[srcman]
        dst_manager = None
        if dstman in self.reduction_managers:
            dst_manager = self.reduction_managers[dstman]
        else:
            if dstman not in self.physical_managers:
                return False
            dst_manager = self.physical_managers[dstman]
        start_event = self.get_event(EventHandle(startid,startgen))
        term_event = self.get_event(EventHandle(termid,termgen))
        reduce_op = ReduceInstance(self, self.next_reduce, src_manager, dst_manager, index, field, tree, start_event, term_event, redop, mask)
        self.reduces.add(reduce_op)
        self.next_reduce = self.next_reduce + 1
        start_event.add_physical_outgoing(reduce_op)
        term_event.add_physical_incoming(reduce_op)
        src_manager.add_reduce_user(reduce_op, False)
        dst_manager.add_reduce_user(reduce_op, True)
        return True

    def add_physical_instance(self, iid, memory, index, field, tid):
        assert iid not in self.physical_instances
        if memory not in self.memories:
            return False
        if index not in self.index_space_nodes:
            return False
        if field not in self.field_space_nodes:
            return False
        if tid not in self.region_trees:
            return False
        index_node = self.get_index_node(True, index)
        self.physical_instances[iid] = PhysicalInstance(self, iid, self.memories[memory], index_node.get_instance(tid))
        return True

    def add_reduction_instance(self, iid, memory, index, field, tid, foldable, indirect):
        assert iid not in self.reduction_instances
        if memory not in self.memories:
            return False
        if index not in self.index_space_nodes:
            return False
        if field not in self.field_space_nodes:
            return False
        if tid not in self.region_trees:
            return False
        index_node = self.get_index_node(True, index)
        self.reduction_instances[iid] = ReductionInstance(self, iid, self.memories[memory], index_node.get_instance(tid), foldable, indirect)
        return True

    def add_instance_manager(self, iid, manager_id):
        if iid not in self.physical_instances:
            return False
        assert manager_id not in self.physical_managers
        manager = InstanceManager(self, manager_id, self.physical_instances[iid])
        self.physical_managers[manager_id] = manager
        self.physical_instances[iid].add_manager(manager)
        return True

    def add_reduction_manager(self, iid, manager_id):
        if iid not in self.reduction_instances:
            return False
        assert manager_id not in self.reduction_managers
        manager = ReductionManager(self, manager_id, self.reduction_instances[iid])
        self.reduction_managers[manager_id] = manager
        self.reduction_instances[iid].add_manager(manager)
        return True

    def add_task_user(self, uid, ctx, gen, hid, idx, manager_id):
        handle = TaskHandle(uid, ctx, hid, gen) 
        if handle not in self.task_instances:
            return False
        task_inst = self.task_instances[handle]
        manager = None
        if manager_id in self.physical_managers:
            manager = self.physical_managers[manager_id]
        else:
            if manager_id not in self.reduction_managers:
                return False
            manager = self.reduction_managers[manager_id]
        manager.add_task_user(handle, task_inst, idx)
        return True

    def add_mapping_user(self, uid, manager_id):
        if uid not in self.mappings:
            return False
        mapping = self.mappings[uid]
        manager = None
        if manager_id in self.physical_managers:
            manager = self.physical_managers[manager_id]
        else:
            if manager_id not in self.reduction_managers:
                return False
            manager = self.reduction_managers[manager_id]
        manager.add_map_user(uid, mapping)
        return True

    def set_exec_info(self, uid, ctx, gen, hid, proc):
        handle = TaskHandle(uid, ctx, hid, gen)
        if not handle in self.task_instances:
            return False
        if not proc in self.processors:
            return False
        self.task_instances[handle].set_processor(self.processors[proc])
        return True

    def set_task_start(self, uid, ctx, gen, hid, start):
        handle = TaskHandle(uid, ctx, hid, gen)
        if handle not in self.task_instances:
            return False
        self.task_instances[handle].set_start_time(start)
        return True

    def set_task_end(self, uid, ctx, gen, hid, end):
        handle = TaskHandle(uid, ctx, hid, gen)
        if handle not in self.task_instances:
            return False
        self.task_instances[handle].set_end_time(end)
        return True

    def has_op(self, uid):
        if uid in self.mappings:
            return True
        elif uid in self.deletions:
            return True
        elif uid in self.task_ops:
            return True
        return False

    def get_op(self, uid):
        if uid in self.mappings:
            return self.mappings[uid]
        elif uid in self.deletions:
            return self.deletions[uid]
        else:
            assert uid in self.task_ops
            return self.task_ops[uid]

    def find_enclosing_point_uid(self, uid):
        assert uid in self.point_slice
        return self.find_enclosing_slice_uid(self.point_slice[uid])

    def find_enclosing_slice_uid(self, uid):
        assert uid in self.slice_slice or uid in self.slice_index
        if uid in self.slice_index:
            return self.slice_index[uid];
        return find_enclosing_slice_uid(self.slice_slice[uid])

    def get_event(self, handle):
        if handle not in self.events:
            self.events[handle] = Event(self, handle)
        return self.events[handle]

    def get_index_node(self, is_reg, iid):
        if is_reg:
            assert iid in self.index_space_nodes
            return self.index_space_nodes[iid]
        else:
            assert iid in self.index_part_nodes
            return self.index_part_nodes[iid]

    def check_logical(self):
        for handle,ctx in self.task_instances.iteritems():
            ctx.compute_dependence_diff(self.verbose)

    def compute_dependence(self, req1, req2):
        # Check to see if there is any overlap in fields or regions
        if len(req1.fields & req2.fields) == 0:
            return NO_DEPENDENCE
        # Check to see if they are in different region trees, in which case
        # there can be no aliasing
        if req1.tid != req2.tid:
            return NO_DEPENDENCE
        node1 = self.get_index_node(req1.is_reg, req1.ispace)
        node2 = self.get_index_node(req2.is_reg, req2.ispace) 
        if not self.is_aliased(node1, node2):
            return NO_DEPENDENCE
        # Otherwise check the coherence and the privilege
        return compute_dependence_type(req1, req2)

    def is_aliased(self, inode1, inode2):
        orig1 = inode1
        orig2 = inode2
        # We need to find their common ancestor 
        if inode1.depth <> inode2.depth:
            if inode1.depth > inode2.depth:
                while inode1.depth > inode2.depth:
                    inode1 = inode1.parent
            else:
                while inode2.depth > inode1.depth:
                    inode2 = inode2.parent
        assert inode1.depth == inode2.depth
        # Handle the case where one is a subset of the other
        if (inode1 is orig2) or (inode2 is orig1):
            return True
        # Now walk backwards up the tree in sync until either we either
        # find a common ancestor or we run out of parents in which case
        # they are in different trees and are therefore disjoint
        while inode1 is not inode2:
            if inode1.parent == None:
                return False
            if inode2.parent == None:
                return False
            inode1 = inode1.parent
            inode2 = inode2.parent
        assert inode1 is inode2
        # Least common ancestor is a region, so they came from different
        # partitions and are therefore not disjoint
        # TODO: handle when partitions are computed to be disjoint
        if inode1.is_region():
            return True
        return not inode1.disjoint

    def print_pictures(self, path):
        # First compute the connected components of the graph 
        components = list()
        for h,e in self.events.iteritems():
            comp = ConnectedComponent(self)
            e.physical_traverse(comp)
            if not comp.empty():
                components.append(comp)
            else:
                comp.unmark_all()
        print "Found "+str(len(components))+" event graphs"
        for idx in range(len(components)):
            components[idx].generate_graph(idx,path)
      
    def check_instance_dependences(self):
        for uid,m in self.physical_managers.iteritems():
            m.check_instance_dependences() 

    def print_instance_graphs(self, path):
        for uid,m in self.physical_managers.iteritems():
            m.print_instance_graphs(path)

    def check_data_flow(self):
        for handle,task_inst in self.task_instances.iteritems():  
            task_inst.check_data_flow()

    def print_instances(self, path):
        for handle,inst in self.task_instances.iteritems():
            inst.print_instance()
        for handle,mapping in self.mappings.iteritems():
            mapping.print_instance()

    def print_processor_graphs(self, path):
        for pid,proc in self.processors.iteritems():
            self.clear_prev_event_dependences()
            name = 'processor_graph_'+str(proc.uid)
            printer = GraphPrinter(path,name)
            orig_nodes = set()
            printed_nodes = set()
            # First find the set of nodes which use this processor and print them
            for handle,task in self.task_instances.iteritems():
                if task.executing_processor == proc:
                    orig_nodes.add(task)
                    printed_nodes.add(task)
                    task.print_physical_node(printer)
            for uid,mapping in self.mappings.iteritems():
                if mapping.enclosing.executing_processor == proc:
                    orig_nodes.add(mapping)
                    printed_nodes.add(mapping)
                    mapping.print_physical_node(printer)
            # Then print all the dependences and nodes which they might depend on
            for node in orig_nodes:
                node.print_filtered_dependences(printer, printed_nodes)
            dot_file = printer.close()
            pdf_file = name+'.pdf'
            try:
                subprocess.check_call(['dot', '-Tpdf', '-o', pdf_file, dot_file])
            except:
                print "WARNING: DOT failure, image for "+pdf_file+" not generated"
                subprocess.call(['rm', '-f', 'core', pdf_file])

    def print_memory_graphs(self,path):
        for mid,mem in self.memories.iteritems():
            self.clear_prev_event_dependences()
            name = 'memory_graph_'+str(mem.uid)
            printer = GraphPrinter(path,name)
            orig_nodes = set()
            printed_nodes = set()
            for handle,task in self.task_instances.iteritems():
                if task.uses_memory(mem):
                    orig_nodes.add(task)
                    printed_nodes.add(task)
                    task.print_physical_node(printer)
            for uid,mapping in self.mappings.iteritems():
                if mapping.uses_memory(mem):
                    orig_nodes.add(mapping)
                    printed_nodes.add(mapping)
                    mapping.print_physical_node(printer)
            for copy in self.copies:
                if copy.uses_memory(mem):
                    orig_nodes.add(copy)
                    printed_nodes.add(copy)
                    copy.print_physical_node(printer)
            for reduce_op in self.reduces:
                if reduce_op.uses_memory(mem):
                    orig_nodes.add(reduce_op)
                    printed_nodes.add(reduce_op)
                    reduce_op.print_physical_node(printer)
            # Now print the depenendences
            for node in orig_nodes:
                node.print_filtered_dependences(printer, printed_nodes)
            dot_file = printer.close()
            pdf_file = name+'.pdf'
            try:
                subprocess.check_call(['dot', '-Tpdf', '-o', pdf_file, dot_file])
            except:
                print "WARNING: DOT failure, image for "+pdf_file+" not generated"
                subprocess.call(['rm', '-f', 'core', pdf_file])

    def clear_prev_event_dependences(self):
        for handle,task in self.task_instances.iteritems():
            task.clear_prev_event_dependences()
        for uid,mapping in self.mappings.iteritems():
            mapping.clear_prev_event_dependences()
        for copy in self.copies:
            copy.clear_prev_event_dependences()
        for reduce_op in self.reduces:
            reduce_op.clear_prev_event_dependences()

    # Legacy functions.  Have been usurped by LegionProf
    def print_processor_timelines(self):
        for uid,p in self.processors.iteritems():
            p.print_timeline()

    def print_memory_timelines(self):
        for uid,m in self.memories.iteritems():
            m.print_timeline()


# EOF

