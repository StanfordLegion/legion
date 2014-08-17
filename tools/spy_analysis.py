#!/usr/bin/env python

# Copyright 2014 Stanford University
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
ATOMIC = 1
SIMULTANEOUS = 2
RELAXED = 3

LOC_PROC = 0
TOC_PROC = 1
UTIL_PROC = 2

# Operation Kinds
SINGLE_OP = 0
INDEX_OP = 1
MAPPING_OP = 2
CLOSE_OP = 3
DELETION_OP = 4
COPY_OP = 5
FENCE_OP = 6

# Instance Kinds
TASK_INST = 0
MAPPING_INST = 1
CLOSE_INST = 2
COPY_INST = 3


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
            elif (((not req1.is_atomic()) and req1.is_read_only()) or 
                  ((not req2.is_atomic()) and req2.is_read_only())):
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

#########################################################
#   Operations
#########################################################

class SingleTask(object):
    def __init__(self, state, uid, tid, ctx, name):
        self.state = state
        self.uid = uid
        self.ctx = ctx
        self.tid = tid
        self.name = name
        self.mdeps = list()
        self.adeps = list()
        self.ops = list()
        self.reqs = dict()
        self.logical_incoming = set()
        self.logical_outgoing = set()
        self.logical_marked = False
        self.enclosing = None
        self.point = None
        self.start_event = None
        self.term_event = None
        self.op_instances = set()
        self.op_instances.add(self)
        self.instances = dict()
        self.physical_marked = False
        self.node_name = 'task_node_'+str(self.uid)
        self.prev_event_deps = set()
        self.generation = 0

    def get_name(self):
        return self.name

    def get_op_kind(self):
        return SINGLE_OP

    def get_inst_kind(self):
        return TASK_INST

    def set_enclosing(self, enc, point):
        self.enclosing = enc
        self.point = point

    def add_operation(self, op):
        self.ops.append(op)

    def add_requirement(self, idx, req):
        # Should only happen to individual tasks
        assert self.enclosing == None
        assert idx not in self.reqs
        self.reqs[idx] = req

    def get_requirement(self, idx):
        assert idx in self.reqs
        return self.reqs[idx]

    def add_req_field(self, idx, fid):
        assert idx in self.reqs
        self.reqs[idx].add_field(fid)

    def add_logical_incoming(self, op):
        if op <> self:
            self.logical_incoming.add(op)

    def add_logical_outgoing(self, op):
        if op <> self:
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

    def add_mdep(self, op1, op2, idx1, idx2, dtype):
        assert op1 in self.ops
        assert op2 in self.ops
        self.mdeps.append(MappingDependence(self, op1, op2, idx1, idx2, dtype))

    def add_adep(self, op1, op2, idx1, idx2, dtype):
        assert op1 in self.ops
        assert op2 in self.ops
        self.adeps.append(MappingDependence(self, op1, op2, idx1, idx2, dtype))

    def add_instance_requirement(self, idx, index):
        if idx in self.reqs:
            assert self.enclosing == None
            assert self.reqs[idx].ispace == index
            return 
        self.reqs[idx] = self.enclosing.get_child_req(idx, index)

    def add_events(self, start, term):
        assert self.start_event == None
        assert self.term_event == None
        self.start_event = start
        self.term_event = term

    def event_graph_traverse(self, traverser):
        traverser.visit_task(self)

    def add_instance(self, idx, inst):
        assert idx not in self.instances
        self.instances[idx] = inst

    def get_instance(self, idx):
        assert idx in self.instances
        return self.instances[idx]

    def physical_traverse(self, component):
        if self.physical_marked:
            return
        self.physical_marked = True
        component.add_task(self)
        self.start_event.physical_traverse(component)
        self.term_event.physical_traverse(component)

    def physical_unmark(self):
        self.physical_marked = False

    def print_physical_node(self, printer):
        inst_string = ''
        if self.state.verbose:
            for idx,inst in self.instances.iteritems():
                assert idx in self.reqs
                req = self.reqs[idx]
                inst_string = inst_string+'\\nInst\ '+hex(inst.iid)+'\ '+req.dot_requirement() 
        if self.enclosing <> None:
            # Index Space Point
            index_string = '\\nPoint\ '+self.point.to_string()
            printer.println(self.node_name+' [style=filled,label="'+\
                str(self.name)+index_string+'\\nUnique\ ID\ '+str(self.uid)+inst_string+\
                '",fillcolor=mediumslateblue,fontsize=14,fontcolor=black,'+\
                'shape=record,penwidth=2];')
        else:
            # Individual Task 
            printer.println(self.node_name+' [style=filled,label="'+\
                str(self.name)+'\\nUnique\ ID\ '+str(self.uid)+inst_string+'",'+\
                'fillcolor=lightskyblue,fontsize=14,fontcolor=black,'+\
                'shape=record,penwidth=2];')

    def print_event_dependences(self, printer):
        self.start_event.print_prev_event_dependences(printer, self.node_name)

    def print_prev_event_dependences(self, printer, later_name):
        if later_name not in self.prev_event_deps:
            printer.println(self.node_name+' -> '+later_name+
                ' [style=solid,color=black,penwidth=2];')
            self.prev_event_deps.add(later_name)

    def compute_dependence_diff(self, verbose):
        print "Checking mapping dependences for task context "+str(self.name)+ \
              " (UID "+str(self.uid)+")"
        if (len(self.ops) == 0) or (len(self.ops) == 1):
            return
        # First compute the list of actual dependences betwee
        for idx in range(1,len(self.ops)):
            for prev in range(idx):
                self.ops[prev].find_dependences(self.ops[idx])
        print "    Found "+str(len(self.adeps))+" dependences in all-pairs "+ \
              "test for task "+str(self.name)
        if verbose:
            print "        Computed Dependences:"
            for dep in self.mdeps:
                print "          index "+str(dep.idx1)+" of "+dep.op1.get_name()+\
                      " (ID "+str(dep.op1.uid)+") and index "+\
                      str(dep.idx2)+" of "+dep.op2.get_name()+" (ID "+\
                      str(dep.op2.uid)+")"
            print "        Actual Dependences:"
            for dep in self.adeps:
                print "          index "+str(dep.idx1)+" of "+dep.op1.get_name()+\
                      " (ID "+str(dep.op1.uid)+") and index "+\
                      str(dep.idx2)+" of "+dep.op2.get_name()+\
                      " (ID "+str(dep.op2.uid)+")"
        # For all the actual dependences make sure we have a dependence path between the
        # two different operations based on the dependences computed by the runtime
        count = 0
        errors = 0
        for adep in self.adeps:
            sys.stdout.write("    Checking dependence: %d \r" % (count))
            if not adep.op2.has_logical_path(adep.op1):
                print "    ERROR: Failed to compute mapping dependence between "+\
                      "index "+str(adep.idx1)+" of "+adep.op1.get_name()+\
                      " (ID "+str(adep.op1.uid)+") and index "+str(adep.idx2)+ \
                      " of "+adep.op2.get_name()+" (ID "+str(adep.op2.uid)+")"
                if adep.op1.get_op_kind() == FENCE_OP:
                    print "      FENCE OPERATION"
                else:
                    print "      First Requirement:"
                    adep.op1.get_requirement(adep.idx1).print_requirement()
                if adep.op2.get_op_kind() == FENCE_OP:
                    print "      FENCE OPERATION:"
                else:
                    print "      Second Requirement:"
                    adep.op2.get_requirement(adep.idx2).print_requirement()
                errors = errors + 1
            for op in self.ops:
                op.unmark_logical()
            count = count + 1

        # Now go through all the mdeps and see if there were any in there that were not
        # computed by the adep computation, which indicates that we 
        # computed an unnecessary dependence
        warnings = 0
        for mdep in self.mdeps:
            found = False
            for adep in self.adeps:
                if adep == mdep:
                    found = True
                    break
            if not found:
                # Skip any deletion op dependences since we haven't taught
                # LegionSpy how to properly compute them yet
                if mdep.op2.get_op_kind() == DELETION_OP:
                    continue
                print "    WARNING: Computed extra mapping dependence "+\
                      "between index "+str(mdep.idx1)+" of "+\
                      mdep.op1.get_name()+" (ID "+str(mdep.op1.uid)+\
                      ") and index "+str(mdep.idx2)+ " of "+mdep.op2.get_name()+\
                      " (ID "+str(mdep.op2.uid)+") in context of task "+\
                      str(self.name)
                warnings = warnings + 1

        if (self.mdeps > 0) or (errors > 0) or (warnings > 0):
            print "    Mapping Dependence Errors: "+str(errors)
            print "    Mapping Dependence Warnings: "+str(warnings)

    def find_dependences(self, op):
        for idx,req in self.reqs.iteritems():
            op.find_individual_dependences(self, req)

    def find_individual_dependences(self, other_op, other_req):
        for idx,req in self.reqs.iteritems():
            dtype = self.state.compute_dependence(other_req, req)
            if is_mapping_dependence(dtype):
                self.ctx.add_adep(other_op, self, other_req.index, req.index, dtype)

    def check_data_flow(self):
        print "Performing data flow check for task "+self.name+" (UID "+str(self.uid)+")"
        for dep in self.mdeps:
            if dep.dtype == TRUE_DEPENDENCE:
                # Handle cross product of instances
                # in case we have multiple instances for an op
                for inst1 in dep.op1.op_instances:
                    if inst1.get_op_kind() == FENCE_OP:
                        continue
                    for inst2 in dep.op2.op_instances:
                        if inst2.get_op_kind() == FENCE_OP:
                            continue
                        # Check to see if they are still aliased
                        req1 = inst1.get_requirement(dep.idx1)
                        req2 = inst2.get_requirement(dep.idx2)
                        # If the second requirement is a reduction, there is no need to
                        # have a dataflow dependence since we can make a separate 
                        # reduction instance
                        if req2.priv == REDUCE:
                            continue
                        assert req1.tid == req2.tid
                        index1 = self.state.get_index_node(True, req1.ispace)
                        index2 = self.state.get_index_node(True, req2.ispace)
                        if self.state.is_aliased(index1, index2):
                            def record_visit(node, traverser):
                                # Get the last instance of the traverser
                                last_idx = len(traverser.instance_stack) - 1
                                last_inst = traverser.instance_stack[last_idx]
                                # See if we have visited this node in this context before  
                                if node in traverser.visited:
                                    contexts = traverser.visited[node]
                                    if last_inst in contexts:
                                        return True # Already been here
                                    else:
                                        contexts.add(last_inst)
                                        return False
                                else:
                                    contexts = set()
                                    contexts.add(last_inst)
                                    traverser.visited[node] = contexts
                                    return False
                            def traverse_event(node, traverser):
                                if record_visit(node, traverser):
                                    return False
                                if traverser.found:
                                    return False
                                return True
                            def traverse_task(node, traverser):
                                if record_visit(node, traverser):
                                    return False
                                if node == traverser.target:
                                    traverser.found = True
                                if traverser.found:
                                    return False
                                return True
                            def traverse_map(node, traverser):
                                if record_visit(node, traverser):
                                    return False
                                if node == traverser.target:
                                    traverser.found = True
                                if traverser.found:
                                    return False
                                return True
                            def traverse_close(node, traverser):
                                if record_visit(node, traverser):
                                    return False
                                if node == traverser.target:
                                    traverser.found = True
                                if traverser.found:
                                    return False
                                return True
                            def traverse_copy(node, traverser):
                                if record_visit(node, traverser):
                                    return False
                                if traverser.found:
                                    return False
                                # Check to see if we have the matching
                                # manager on the stack
                                last_idx = len(traverser.instance_stack) - 1
                                assert last_idx >= 0
                                last_inst = traverser.instance_stack[last_idx]
                                if ((last_inst == node.dst_inst) and 
                                    (traverser.field in node.fields)):
                                    traverser.instance_stack.append(node.src_inst)
                                    return True
                                return False
                            def post_traverse_event(node, traverser):
                                pass
                            def post_traverse_task(node, traverser):
                                pass
                            def post_traverse_map(node, traverser):
                                pass
                            def post_traverse_close(node, traverser):
                                pass
                            def post_traverse_copy(node, traverser):
                                traverser.instance_stack.pop()
                            # Do the traversal for each overlapping field
                            fields = req1.fields & req2.fields
                            for f in fields:
                                traverser = EventGraphTraverser(False, False, False, 
                                    self.state.get_next_traverser_gen(), traverse_event,
                                    traverse_task, traverse_map, traverse_close, 
                                    traverse_copy, post_traverse_event, 
                                    post_traverse_task, post_traverse_map,
                                    post_traverse_close, post_traverse_copy)
                                traverser.found = False
                                traverser.target = inst1
                                traverser.field = f
                                traverser.instance_stack = list()
                                # Maintain a list of nodes that we have visited and
                                # context in which we have visted them.  We may need
                                # to visit some nodes in more than one context
                                traverser.visited = dict()
                                # TODO: support virtual mappings
                                traverser.instance_stack.append(inst2.get_instance(dep.idx2))
                                # Traverse and see if we find inst1
                                inst2.event_graph_traverse(traverser)
                                # Sanity check, only one thing on the stack
                                assert len(traverser.instance_stack) == 1
                                if not traverser.found:
                                    print "   ERROR: Unable to find data flow path "+\
                                          "between requirement "+str(dep.idx1)+" of "+\
                                          inst1.get_name()+" (UID "+str(inst1.uid)+") "+\
                                          "and requirement "+str(dep.idx2)+" of "+inst2.get_name()+\
                                          " (UID "+str(inst2.uid)+") for field "+str(f)
                                    if self.state.verbose:
                                        print "      First Requirement:"
                                        req1.print_requirement()
                                        print "      Second Requirement:"
                                        req2.print_requirement()
    

class IndexTask(object):
    def __init__(self, state, uid, tid, ctx, name):
        assert ctx <> None
        self.state = state
        self.uid = uid
        self.ctx = ctx
        self.tid = tid
        self.name = name
        self.points = set()
        self.logical_incoming = set()
        self.logical_outgoing = set()
        self.logical_marked = False
        self.reqs = dict()
        self.op_instances = set()

    def get_name(self):
        return self.name

    def get_op_kind(self):
        return INDEX_OP

    def add_point(self, uid, point):
        point_task = SingleTask(self.state, uid, self.tid, None, self.name)
        point_task.set_enclosing(self, point)
        self.points.add(point_task)
        self.op_instances.add(point_task)
        return point_task

    def add_requirement(self, idx, req):
        assert idx not in self.reqs
        self.reqs[idx] = req

    def get_requirement(self, idx):
        assert idx in self.reqs
        return self.reqs[idx]

    def add_req_field(self, idx, fid):
        assert idx in self.reqs
        self.reqs[idx].add_field(fid)

    def get_child_req(self, idx, index):
        assert idx in self.reqs
        r = self.reqs[idx]
        result = Requirement(idx, True, index, r.fspace, r.tid, r.priv, r.coher, r.redop)
        for f in r.fields:
            result.add_field(f)
        return result

    def add_logical_incoming(self, op):
        assert op <> self
        self.logical_incoming.add(op)

    def add_logical_outgoing(self, op):
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

    def compute_dependence_diff(self, verbose):
        # No need to do anything
        pass

    def find_dependences(self, op):
        for idx,req in self.reqs.iteritems():
            op.find_individual_dependences(self, req)

    def find_individual_dependences(self, other_op, other_req):
        for idx,req in self.reqs.iteritems():
            dtype = self.state.compute_dependence(other_req, req)
            if is_mapping_dependence(dtype):
                self.ctx.add_adep(other_op, self, other_req.index, req.index, dtype)

    def check_data_flow(self):
        # No need to do anything
        pass

class Mapping(object):
    def __init__(self, state, uid, ctx):
        assert ctx <> None
        self.state = state
        self.uid = uid
        self.ctx = ctx
        self.requirement = None
        self.logical_incoming = set()
        self.logical_outgoing = set()
        self.logical_marked = False
        self.start_event = None
        self.term_event = None
        self.op_instances = set()
        self.op_instances.add(self)
        self.instance = None
        self.physical_marked = False
        self.node_name = 'mapping_node_'+str(uid)
        self.prev_event_deps = set()
        self.generation = 0

    def get_name(self):
        return "Mapping "+str(self.uid)

    def get_op_kind(self):
        return MAPPING_OP

    def get_inst_kind(self):
        return MAPPING_INST

    def add_requirement(self, idx, req):
        assert idx == 0
        assert self.requirement == None
        self.requirement = req

    def get_requirement(self, idx):
        assert idx == 0
        assert self.requirement <> None
        return self.requirement

    def add_req_field(self, idx, fid):
        assert idx == 0
        assert self.requirement <> None
        self.requirement.add_field(fid)

    def add_logical_incoming(self, op):
        assert self <> op
        self.logical_incoming.add(op)

    def add_logical_outgoing(self, op):
        assert self <> op
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

    def add_events(self, start, term):
        assert self.start_event == None
        assert self.term_event == None
        self.start_event = start
        self.term_event = term

    def event_graph_traverse(self, traverser):
        traverser.visit_mapping(self)

    def add_instance(self, idx, inst):
        assert idx == 0
        assert self.instance == None
        self.instance = inst

    def get_instance(self, idx):
        assert idx == 0
        assert self.instance <> None
        return self.instance

    def physical_traverse(self, component):
        if self.physical_marked:
            return
        self.physical_marked = True
        component.add_map(self)
        self.start_event.physical_traverse(component)
        self.term_event.physical_traverse(component)

    def physical_unmark(self):
        self.physical_marked = False

    def print_physical_node(self, printer):
        inst_string = ''
        if self.state.verbose:
            inst_string = '\\nInst\ '+hex(self.instance.iid)+'\ '+self.requirement.dot_requirement()
        printer.println(self.node_name+' [style=filled,label="'+\
            'Mapping\ '+str(self.uid)+'\ in\ '+self.ctx.name+inst_string+'",'+\
            'fillcolor=mediumseagreen,fontsize=14,fontcolor=black,'+\
            'shape=record,penwidth=2];')

    def print_event_dependences(self, printer):
        self.start_event.print_prev_event_dependences(printer, self.node_name)

    def print_prev_event_dependences(self, printer, later_name):
        if later_name not in self.prev_event_deps:
            printer.println(self.node_name+' -> '+later_name+
                ' [style=solid,color=black,penwidth=2];')
            self.prev_event_deps.add(later_name)

    def compute_dependence_diff(self, verbose):
        # No need to do anything
        pass

    def find_dependences(self, op):
        op.find_individual_dependences(self, self.requirement)

    def find_individual_dependences(self, other_op, other_req):
        dtype = self.state.compute_dependence(other_req, self.requirement)
        if is_mapping_dependence(dtype):
            self.ctx.add_adep(other_op, self, other_req.index, self.requirement.index, dtype)

    def check_data_flow(self):
        # No need to do anything
        pass

class Deletion(object):
    def __init__(self, state, uid, ctx):
        assert ctx <> None
        self.state = state
        self.uid = uid
        self.ctx = ctx
        self.logical_outgoing = set()
        self.logical_marked = False
        self.op_instances = set()

    def get_name(self):
        return "Deletion "+str(self.uid)

    def get_op_kind(self):
        return DELETION_OP

    def add_logical_incoming(self, op):
        # Should never happen
        assert False

    def add_logical_outgoing(self, op):
        assert self <> op
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

    def compute_dependence_diff(self, verbose):
        # No need to do anything
        pass

    def find_dependences(self, op):
        # No need to do anything
        pass

    def find_individual_dependences(self, other_op, other_req):
        # TODO: implement this for deletion
        pass

    def check_data_flow(self):
        # No need to do anything
        pass

class CopyOp(object):
    def __init__(self, state, uid, ctx):
        assert ctx is not None
        self.state = state
        self.uid = uid
        self.ctx = ctx
        self.reqs = dict()
        self.logical_incoming = set()
        self.logical_outgoing = set()
        self.logical_marked = False
        self.op_instances = set()
        self.op_instances.add(self)
        self.instances = dict()
        self.generation = 0
        self.start_event = None
        self.term_event = None
        self.physical_marked = False
        self.node_name = 'copy_across_'+str(self.uid)
        self.prev_event_deps = set()

    def get_name(self):
        return "Copy Op "+str(self.uid)

    def get_op_kind(self):
        return COPY_OP

    def add_requirement(self, idx, req):
        assert idx not in self.reqs
        self.reqs[idx] = req

    def get_requirement(self, idx):
        assert idx in self.reqs
        return self.reqs[idx]

    def add_req_field(self, idx, fid):
        assert idx in self.reqs
        self.reqs[idx].add_field(fid)

    def add_logical_incoming(self, op):
        assert self <> op
        self.logical_incoming.add(op)

    def add_logical_outgoing(self, op):
        assert self <> op
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

    def compute_dependence_diff(self, verbose):
        # No need to do anything
        pass

    def find_dependences(self, op):
        for idx,req in self.reqs.iteritems():
            op.find_individual_dependences(self, req)

    def find_individual_dependences(self, other_op, other_req):
        for idx,req in self.reqs.iteritems():
            dtype = self.state.compute_dependence(other_req, req)
            if is_mapping_dependence(dtype):
                self.ctx.add_adep(other_op, self, other_req.index, req.index, dtype)
    
    def add_instance(self, idx, inst):
        assert idx not in self.instances
        self.instances[idx] = inst

    def get_instance(self, idx):
        assert idx in self.instances
        return self.instances[idx]

    def event_graph_traverse(self, traverser):
        traverser.visit_task(self)
        
    def add_events(self, start, term):
        assert self.start_event == None
        assert self.term_event == None
        self.start_event = start
        self.term_event = term

    def check_data_flow(self):
        # No need to do anything
        pass

    def physical_traverse(self, component):
        if self.physical_marked:
            return
        self.physical_marked = True
        component.add_copy(self)
        self.start_event.physical_traverse(component)
        self.term_event.physical_traverse(component)

    def physical_unmark(self):
        self.physical_marked = False

    def print_physical_node(self, printer):
        printer.println(self.node_name+' [style=filled,label="Copy Across'+\
                '",fillcolor=darkgoldenrod1,fontsize=14,fontcolor=black,'+\
                'shape=record,penwidth=2];')

    def print_event_dependences(self, printer):
        self.start_event.print_prev_event_dependences(printer, self.node_name)
        pass

    def print_prev_event_dependences(self, printer, later_name):
        if later_name not in self.prev_event_deps:
            printer.println(self.node_name+' -> '+later_name+
                ' [style=solid,color=black,penwidth=2];')
            self.prev_event_deps.add(later_name)
        self.start_event.print_prev_event_dependences(printer, later_name)


class Close(object):
    def __init__(self, state, uid, ctx):
        assert ctx is not None
        self.state = state
        self.uid = uid
        self.ctx = ctx
        self.requirement = None
        self.logical_incoming = set()
        self.logical_outgoing = set()
        self.logical_marked = False
        self.start_event = None
        self.term_event = None
        self.op_instances = set()
        self.op_instances.add(self)
        self.instance = None
        self.physical_marked = False
        self.node_name = 'close_op_'+str(self.uid)
        self.prev_event_deps = set()
        self.generation = 0

    def get_name(self):
        return "Close "+str(self.uid)

    def get_op_kind(self):
        return CLOSE_OP

    def get_inst_kind(self):
        return CLOSE_INST

    def add_requirement(self, idx, req):
        assert idx == 0
        assert self.requirement == None
        self.requirement = req

    def get_requirement(self, idx):
        assert idx == 0
        assert self.requirement <> None
        return self.requirement

    def add_req_field(self, idx, fid):
        assert idx == 0
        assert self.requirement <> None
        self.requirement.add_field(fid)

    def add_logical_incoming(self, op):
        assert self <> op
        self.logical_incoming.add(op)

    def add_logical_outgoing(self, op):
        assert self <> op
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

    def print_physical_node(self, printer):
        # Let's not print these for right now
        #printer.println(self.node_name+' [style=filled,label="'+\
        #    'Close\ '+str(self.uid)+'\ in\ '+self.ctx.name+'",'+\
        #    'fillcolor=orangered,fontsize=14,fontcolor=black,'+\
        #    'shape=record,penwidth=2];')
        pass

    def print_event_dependences(self, printer):
        #self.start_event.print_prev_event_dependences(printer, self.node_name)
        pass

    def print_prev_event_dependences(self, printer, later_name):
        #if later_name not in self.prev_event_deps:
        #    printer.println(self.node_name+' -> '+later_name+
        #        ' [style=solid,color=black,penwidth=2];')
        #    self.prev_event_deps.add(later_name)
        self.start_event.print_prev_event_dependences(printer, later_name)

    def add_events(self, start, term):
        assert self.start_event == None
        assert self.term_event == None
        self.start_event = start
        self.term_event = term

    def event_graph_traverse(self, traverser):
        traverser.visit_close(self)

    def add_instance(self, idx, inst):
        assert idx == 0
        assert self.instance == None
        self.instance = inst

    def get_instance(self, idx):
        assert idx == 0
        assert self.instance <> None
        return self.instance

    def physical_traverse(self, component):
        if self.physical_marked:
            return
        self.physical_marked = True
        component.add_close(self)
        self.start_event.physical_traverse(component)
        self.term_event.physical_traverse(component)

    def physical_unmark(self):
        self.physical_marked = False

    def compute_dependence_diff(self, verbose):
        # No need to do anything
        pass

    def find_dependences(self, op):
        op.find_individual_dependences(self, self.requirement) 

    def find_individual_dependences(self, other_op, other_req):
        dtype = self.state.compute_dependence(other_req, self.requirement)
        if is_mapping_dependence(dtype):
            self.ctx.add_adep(other_op, self, other_req.index, self.requirement.index, dtype)

    def check_data_flow(self):
        # No need to do anything
        pass

class Fence(object):
    def __init__(self, state, uid, ctx):
        self.state = state
        self.uid = uid
        self.ctx = ctx
        self.logical_incoming = set()
        self.logical_outgoing = set()
        self.logical_marked = False
        self.op_instances = set()
        self.op_instances.add(self)

    def get_name(self):
        return "Fence "+str(self.uid)

    def get_op_kind(self):
        return FENCE_OP

    def add_logical_incoming(self, op):
        assert self <> op
        self.logical_incoming.add(op)

    def add_logical_outgoing(self, op):
        assert self <> op
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

    def find_dependences(self, op):
        # Everybody who comes after us has a dependence on us 
        #self.ctx.add_adep(self, op, 0, 0, TRUE_DEPENDENCE)
        pass

    def find_individual_dependences(self, other_op, other_req):
        # Everybody who comes after us has a dependence on us
        #self.ctx.add_adep(other_op, self, other_req.index, 0, TRUE_DEPENDENCE)
        pass

    def compute_dependence_diff(self, verbose):
        # No need to do anything
        pass

    def check_data_flow(self):
        pass

class Copy(object):
    def __init__(self, state, srcman, dstman, start_event, term_event, region, redop, mask, uid):
        self.state = state
        self.uid = uid
        self.src_inst = srcman
        self.dst_inst = dstman
        self.start_event = start_event
        self.term_event = term_event
        self.region = region
        self.redop = redop
        def parse_mask(mask):
            result = set()
            fields = string.split(mask,',')
            for f in fields:
                result.add(int(f))
            return result
        self.fields = parse_mask(mask)
        self.physical_marked = False
        self.mask = mask
        self.node_name = 'copy_node_'+str(self.uid)
        self.prev_event_deps = set()
        self.generation = 0

    def get_inst_kind(self):
        return COPY_INST
      
    def event_graph_traverse(self, traverser):
        traverser.visit_copy(self)

    def physical_traverse(self, component):
        if self.physical_marked:
            return
        self.physical_marked = True
        component.add_copy(self)
        self.start_event.physical_traverse(component)
        self.term_event.physical_traverse(component)

    def physical_unmark(self):
        self.physical_marked = False

    def print_physical_node(self, printer):
        if self.redop <> 0:
            # Reduction copy
            printer.println(self.node_name+' [style=filled,label="'+\
                'Src\ Inst:\ '+str(hex(self.src_inst.iid))+'\ Src\ Loc:\ '+\
                str(hex(self.src_inst.memory.uid))+'\\n'+\
                'Dst\ Inst:\ '+str(hex(self.dst_inst.iid))+'\ Dst\ Loc:\ '+\
                str(hex(self.dst_inst.memory.uid))+'\\n'+\
                'Logical\ Region:\ (index:'+str(hex(self.region.index_node.uid))+\
                ',field:'+str(hex(self.region.field_node.uid))+',tree:'+\
                str(self.region.tid)+')\\nCopy\ Fields:\ '+self.mask+\
                '\\nReduction\ Op:\ '+str(self.redop)+\
                '",fillcolor=crimson,fontsize=14,fontcolor=black,'+\
                'shape=record,penwidth=2];')
        else:
            # Normal copy
            printer.println(self.node_name+' [style=filled,label="'+\
                'Src\ Inst:\ '+str(hex(self.src_inst.iid))+'\ Src\ Loc:\ '+\
                str(hex(self.src_inst.memory.uid))+'\\n'+\
                'Dst\ Inst:\ '+str(hex(self.dst_inst.iid))+'\ Dst\ Loc:\ '+\
                str(hex(self.dst_inst.memory.uid))+'\\n'+\
                'Logical\ Region:\ (index:'+str(hex(self.region.index_node.uid))+\
                ',field:'+str(hex(self.region.field_node.uid))+',tree:'+\
                str(self.region.tid)+')\\nCopy\ Fields:\ '+self.mask+\
                '",fillcolor=darkgoldenrod1,fontsize=14,fontcolor=black,'+\
                'shape=record,penwidth=2];')

    def print_event_dependences(self, printer):
        self.start_event.print_prev_event_dependences(printer, self.node_name)

    def print_prev_event_dependences(self, printer, later_name):
        if later_name not in self.prev_event_deps:
            printer.println(self.node_name+' -> '+later_name+
                ' [style=solid,color=black,penwidth=2];')
            self.prev_event_deps.add(later_name)


class PhysicalInstance(object):
    def __init__(self, state, iid, memory, region):
        self.state = state
        self.iid = iid
        self.memory = memory
        self.region = region
        self.op_users = dict()

    def add_op_user(self, op, idx):
        req = op.get_requirement(idx)
        for field in req.fields:
            if field not in self.op_users:
                self.op_users[field] = dict()
            if op not in self.op_users[field]:
                self.op_users[field][op] = list()
            self.op_users[field][op].append(req)


class ReductionInstance(object):
    def __init__(self, state, iid, memory, region, fold, indirect):
        self.state = state
        self.iid = iid
        self.memory = memory
        self.region = region
        self.fold = fold
        self.indirect = indirect
        self.op_users = dict()

    def add_op_user(self, op, idx):
        req = op.get_requirement(idx)
        for field in req.fields:
            if field not in self.op_users:
                self.op_users[field] = dict()
            if op not in self.op_users[field]:
                self.op_users[field][op] = list()
            self.op_users[field][op].append(req)


class EventHandle(object):
    def __init__(self, uid, gen):
        self.uid = uid
        self.gen = gen

    def __hash__(self):
        return hash((self.uid, self.gen))

    def __eq__(self, other):
        return (self.uid,self.gen) == (other.uid,other.gen)

    def exists(self):
        return (self.uid <> 0)


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
            print "        Logical Region Requirement ("+str(hex(self.ispace))+","+\
                  str(self.fspace)+","+str(self.tid)+")"
        else:
            print "        Logical Partition Requirement ("+str(self.ispace)+","+\
                  str(self.fspace)+","+str(self.tid)+")"
        field_str = "          Fields: "
        first = True
        for f in self.fields:
            if first:
                field_str = field_str+str(f)
                first = False
            else:
                field_str = field_str+", "+str(f)
        print field_str
        print "        Privilege: "+self.get_privilege()
        print "        Coherence: "+self.get_coherence()

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
            print "Region Requirement for ("+str(hex(self.ispace))+","+\
                  str(self.fspace)+","+str(self.tid)+")"
        else:
            print "Partition Requirement for ("+str(self.ispace)+","+\
                  str(self.fspace)+","+str(self.tid)+")"
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
        self.implicit_incoming = set()
        self.implicit_outgoing = set()
        self.physical_marked = False
        self.generation = 0

    def add_physical_incoming(self, event):
        assert self <> event
        self.physical_incoming.add(event)

    def add_physical_outgoing(self, event):
        assert self <> event
        self.physical_outgoing.add(event)

    def add_implicit_incoming(self, event):
        assert self <> event
        self.implicit_incoming.add(event)

    def add_implicit_outgoing(self, event):
        assert self <> event
        self.implicit_outgoing.add(event)

    def physical_traverse(self, component):
        if self.physical_marked or not self.handle.exists():
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


class EventGraphTraverser(object):
    def __init__(self, forwards, implicit, use_gen, generation, 
                 event_fn = None, task_fn = None, 
                 map_fn = None, close_fn = None, copy_fn = None,
                 post_event_fn = None, post_task_fn = None,
                 post_map_fn = None, post_close_fn = None,
                 post_copy_fn = None):
        self.forwards = forwards
        self.implicit = implicit
        self.use_gen = use_gen
        self.generation = generation
        self.event_fn = event_fn
        self.task_fn = task_fn
        self.map_fn = map_fn
        self.close_fn = close_fn
        self.copy_fn = copy_fn
        self.post_event_fn = post_event_fn
        self.post_task_fn = post_task_fn
        self.post_map_fn = post_map_fn
        self.post_close_fn = post_close_fn
        self.post_copy_fn = post_copy_fn

    def visit_event(self, node):
        if self.use_gen:
            if node.generation == self.generation:
                return
            else:
                node.generation = self.generation
        do_next = True
        if self.event_fn <> None:
            do_next = self.event_fn(node, self)
        if not do_next:
            return
        if self.forwards:
            for n in node.physical_outgoing:
                n.event_graph_traverse(self)
            if self.implicit:
                for n in node.implicit_outgoing:
                    n.event_graph_traverse(self)
        else:
            for n in node.physical_incoming:
                n.event_graph_traverse(self)
            if self.implicit:
                for n in node.implicit_incoming:
                    n.event_graph_traverse(self)
        if self.post_event_fn <> None:
            self.post_event_fn(node, self)

    def visit_task(self, node):
        if self.use_gen:
            if node.generation == self.generation:
                return
            else:
                node.generation = self.generation
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
        if self.use_gen:
            if node.generation == self.generation:
                return
            else:
                node.generation = self.generation
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

    def visit_close(self, node):
        if self.use_gen:
            if node.generation == self.generation:
                return
            else:
                node.generation = self.generation
        do_next = True
        if self.close_fn <> None:
            do_next = self.close_fn(node, self)
        if not do_next:
            return
        if self.forwards:
            node.term_event.event_graph_traverse(self)
        else:
            node.start_event.event_graph_traverse(self)
        if self.post_close_fn <> None:
            self.post_close_fn(node, self)

    def visit_copy(self, node):
        if self.use_gen:
            if node.generation == self.generation:
                return
            else:
                node.generation = self.generation
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


class ConnectedComponent(object):
    def __init__(self, state):
        self.state = state
        self.events = set()
        self.tasks = set()
        self.maps = set()
        self.closes = set()
        self.copies = set()

    def add_event(self, event):
        assert event not in self.events
        self.events.add(event)

    def add_task(self, task):
        assert task not in self.tasks
        self.tasks.add(task)

    def add_map(self, mapp):
        assert mapp not in self.maps
        self.maps.add(mapp)

    def add_close(self, close):
        assert close not in self.closes
        self.closes.add(close)

    def add_copy(self, copy):
        assert copy not in self.copies
        self.copies.add(copy)

    def empty(self):
        total = len(self.tasks)+len(self.maps)+len(self.closes)+len(self.copies)
        return (total < 2)

    def unmark_all(self):
        for t in self.tasks:
            t.physical_unmark()
        for m in self.maps:
            m.physical_unmark()
        for c in self.copies:
            c.physical_unmark()
        for c in self.closes:
            c.physical_unmark()
        for e in self.events:
            e.physical_unmark()

    def generate_graph(self, idx, path):
        name = 'event_graph_'+str(idx)
        printer = GraphPrinter(path,name)
        # Print the nodes
        for t in self.tasks:
            t.print_physical_node(printer)
        for m in self.maps:
            m.print_physical_node(printer)
        for c in self.closes:
            c.print_physical_node(printer)
        for c in self.copies:
            c.print_physical_node(printer)
        # Now print the dependences
        for t in self.tasks:
            t.print_event_dependences(printer)
        for m in self.maps:
            m.print_event_dependences(printer)
        for c in self.closes:
            c.print_event_dependences(printer)
        for c in self.copies:
            c.print_event_dependences(printer) 
        dot_file = printer.close()
        pdf_file = name+'.pdf'
        try:
            subprocess.check_call(['dot -Tpdf -o '+pdf_file+' '+dot_file],shell=True)
        except:
            print "WARNING: DOT failure, image for event graph "+str(idx)+" not generated"
            subprocess.call(['rm -f core '+pdf_file],shell=True)


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
        self.ops = dict()
        self.copies = set()
        self.slice_index = dict()
        self.slice_slice = dict()
        self.point_slice = dict()
        self.instances = dict()
        self.events = dict()
        self.verbose = verbose
        self.top_level_uid = None
        self.traverser_gen = 1
        self.copy_uid = 1

    def get_next_traverser_gen(self):
        result = self.traverser_gen
        self.traverser_gen = result + 1
        return result

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

    def add_index_space(self, uid):
        assert uid not in self.index_space_nodes
        self.index_space_nodes[uid] = IndexSpaceNode(self, uid, 0, None)

    def add_index_partition(self, pid, uid, disjoint, color):
        assert uid not in self.index_part_nodes
        if pid not in self.index_space_nodes:
            return False
        self.index_part_nodes[uid] = IndexPartNode(self, uid, disjoint, color, 
                                                    self.index_space_nodes[pid])
        return True

    def add_index_subspace(self, pid, uid, color):
        assert uid not in self.index_space_nodes
        if pid not in self.index_part_nodes:
            return False
        self.index_space_nodes[uid] = IndexSpaceNode(self, uid, color, 
                                              self.index_part_nodes[pid])
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
        self.region_trees[tid] = self.index_space_nodes[iid].instantiate(
                                    None, self.field_space_nodes[fid], tid)
        return True

    def add_top_task(self, tid, uid, name):
        assert uid not in self.ops
        self.ops[uid] = SingleTask(self, uid, tid, None, name)
        self.top_level_uid = uid
      
    def add_single_task(self, ctx, tid, uid, name):
        if uid == self.top_level_uid:
            return True
        assert uid not in self.ops
        if ctx not in self.ops:
            return False
        self.ops[uid] = SingleTask(self, uid, tid, self.ops[ctx], name)
        self.ops[ctx].add_operation(self.ops[uid])
        return True

    def add_index_task(self, ctx, tid, uid, name):
        assert uid not in self.ops
        if ctx not in self.ops:
            return False
        self.ops[uid] = IndexTask(self, uid, tid, self.ops[ctx], name)
        self.ops[ctx].add_operation(self.ops[uid])
        return True
    
    def add_mapping(self, ctx, uid):
        assert uid not in self.ops
        if ctx not in self.ops:
            return False
        self.ops[uid] = Mapping(self, uid, self.ops[ctx])
        self.ops[ctx].add_operation(self.ops[uid])
        return True

    def add_close(self, ctx, uid):
        assert uid not in self.ops
        if ctx not in self.ops:
            return False
        self.ops[uid] = Close(self, uid, self.ops[ctx])
        self.ops[ctx].add_operation(self.ops[uid])
        return True

    def add_fence(self, ctx, uid):
        assert uid not in self.ops
        if ctx not in self.ops:
            return False
        self.ops[uid] = Fence(self, uid, self.ops[ctx])
        self.ops[ctx].add_operation(self.ops[uid])
        return True

    def add_copy_op(self, ctx, uid):
        assert uid not in self.ops
        if ctx not in self.ops:
            return False
        self.ops[uid] = CopyOp(self, uid, self.ops[ctx])
        self.ops[ctx].add_operation(self.ops[uid])
        return True

    def add_deletion(self, ctx, uid):
        assert uid not in self.ops
        if ctx not in self.ops:
            return False
        self.ops[uid] = Deletion(self, uid, self.ops[ctx])
        self.ops[ctx].add_operation(self.ops[uid])
        return True

    def add_index_slice(self, index_id, slice_id):
        if index_id not in self.ops:
            return False
        assert slice_id not in self.slice_index
        self.slice_index[slice_id] = index_id
        return True

    def add_slice_slice(self, slice1, slice2):
        if slice1 not in self.slice_slice and slice1 not in self.slice_index:
            return False
        assert slice2 not in self.slice_slice
        self.slice_slice[slice2] = slice1
        return True
    
    def add_slice_point(self, slice_id, point_id, dim, v1, v2, v3):
        if slice_id not in self.slice_slice and slice_id not in self.slice_index:
            return False
        assert point_id not in self.point_slice
        self.point_slice[point_id] = slice_id
        point = Point(0, dim)
        point.add_value(v1)
        if dim > 1:
            point.add_value(v2)
            if dim > 2:
                point.add_value(v3)
        index_id = self.find_enclosing_index_id(slice_id) 
        assert index_id in self.ops
        child_task = self.ops[index_id].add_point(point_id, point)
        self.ops[point_id] = child_task
        return True

    def add_point_point(self, point1, point2):
        if point1 not in self.ops:
            return False
        assert point2 not in self.ops
        self.ops[point2] = self.ops[point1]
        return True

    def add_requirement(self, uid, index, is_reg, ispace, fspace, tid, priv, coher, redop):
        if uid not in self.ops:
            return False
        self.ops[uid].add_requirement(index, Requirement(index, is_reg, ispace, 
                                                  fspace, tid, priv, coher, redop))
        return True

    def add_req_field(self, uid, idx, fid):
        if uid not in self.ops:
            return False
        self.ops[uid].add_req_field(idx, fid)
        return True

    def add_mapping_dependence(self, ctx, prev_id, pidx, next_id, nidx, dtype):
        if ctx not in self.ops:
            return False
        if prev_id not in self.ops:
            return False
        if next_id not in self.ops:
            return False
        context = self.ops[ctx]
        prev_op = self.ops[prev_id]
        next_op = self.ops[next_id]
        context.add_mdep(prev_op, next_op, pidx, nidx, dtype)
        prev_op.add_logical_incoming(next_op)
        next_op.add_logical_outgoing(prev_op)
        return True

    def add_instance_requirement(self, uid, idx, index):
        if uid not in self.ops:
            return False
        self.ops[uid].add_instance_requirement(idx, index)
        return True

    def add_op_events(self, uid, startid, startgen, termid, termgen):
        if uid not in self.ops:
            return False
        e1 = self.get_event(EventHandle(startid,startgen))
        e2 = self.get_event(EventHandle(termid,termgen))
        e1.add_physical_outgoing(self.ops[uid])
        e2.add_physical_incoming(self.ops[uid])
        self.ops[uid].add_events(e1, e2)
        return True

    def add_copy_events(self, srcman, dstman, index, field, tree, 
                        startid, startgen, termid, termgen, redop, mask):
        if srcman not in self.instances:
            return False
        if dstman not in self.instances:
            return False
        if index not in self.index_space_nodes:
            return False
        if field not in self.field_space_nodes:
            return False
        if tree not in self.region_trees:
            return False
        e1 = self.get_event(EventHandle(startid,startgen))
        e2 = self.get_event(EventHandle(termid,termgen))
        region = self.get_index_node(True, index).get_instance(tree)
        copy = Copy(self, self.instances[srcman], self.instances[dstman], 
                             e1, e2, region, redop, mask, self.copy_uid)
        self.copy_uid = self.copy_uid + 1
        self.copies.add(copy)
        e1.add_physical_outgoing(copy)
        e2.add_physical_incoming(copy)
        return True

    def add_physical_instance(self, iid, mem, index, field, tree):
        if iid in self.instances:
            return True
        if mem not in self.memories:
            return False
        if index not in self.index_space_nodes:
            return False
        if field not in self.field_space_nodes:
            return False
        if tree not in self.region_trees:
            return False
        region = self.get_index_node(True, index).get_instance(tree)
        self.instances[iid] = PhysicalInstance(self, iid, self.memories[mem], region)
        return True

    def add_reduction_instance(self, iid, mem, index, field, tree, fold, indirect):
        assert iid not in self.instances
        if mem not in self.memories:
            return False
        if index not in self.index_space_nodes:
            return False
        if field not in self.field_space_nodes:
            return False
        if tree not in self.region_trees:
            return False
        region = self.get_index_node(True, index).get_instance(tree)
        self.instances[iid] = ReductionInstance(self, iid, self.memories[mem],
                                                region, fold, indirect)
        return True

    def add_op_user(self, uid, idx, iid):
        if uid not in self.ops:
            return False
        if iid not in self.instances:
            return False
        self.instances[iid].add_op_user(self.ops[uid], idx)
        self.ops[uid].add_instance(idx, self.instances[iid])
        return True

    def add_event_dependence(self, id1, gen1, id2, gen2):
        e1 = self.get_event(EventHandle(id1,gen1))
        e2 = self.get_event(EventHandle(id2,gen2))
        e1.add_physical_outgoing(e2)
        e2.add_physical_incoming(e1)

    def add_implicit_dependence(self, id1, gen1, id2, gen2):
        e1 = self.get_event(EventHandle(id1,gen1))
        e2 = self.get_event(EventHandle(id2,gen2))
        e1.add_implicit_outgoing(e2)
        e2.add_implicit_incoming(e1)

    def find_enclosing_index_id(self, slice_id):
        if slice_id in self.slice_index:
            return self.slice_index[slice_id]
        assert slice_id in self.slice_slice
        return self.find_enclosing_index_id(self.slice_slice[slice_id])

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
        for uid,op in self.ops.iteritems():
            op.compute_dependence_diff(self.verbose)

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

    def check_data_flow(self):
        for uid,op in self.ops.iteritems():
            op.check_data_flow()

    def check_instance_dependences(self):
        for instance in self.instances.itervalues():
            print "Checking physical instance "+str(instance.iid)+"..."
            for field, op_users in instance.op_users.iteritems():
                for op1, reqs1 in op_users.iteritems():
                    for req1 in reqs1:
                        for op2, reqs2 in op_users.iteritems():
                            for req2 in reqs2:
                                if op1 != op2 and self.compute_dependence(req1, req2) in (TRUE_DEPENDENCE, ANTI_DEPENDENCE):
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
                                    def traverse_close(node, traverser):
                                        if node == traverser.target:
                                            traverser.found = True
                                        if traverser.found:
                                            return False
                                        return True
                                    def traverse_copy(node, traverser):
                                        if node == traverser.target:
                                            traverser.found = True
                                        if traverser.found:
                                            return False
                                        return True
                                    def post_traverse_event(node, traverser):
                                        pass
                                    def post_traverse_task(node, traverser):
                                        pass
                                    def post_traverse_map(node, traverser):
                                        pass
                                    def post_traverse_close(node, traverser):
                                        pass
                                    def post_traverse_copy(node, traverser):
                                        pass
                                    traverser = EventGraphTraverser(False, True, True,
                                        self.get_next_traverser_gen(), traverse_event,
                                        traverse_task, traverse_map, traverse_close,
                                        traverse_copy, post_traverse_event,
                                        post_traverse_task, post_traverse_map,
                                        post_traverse_close, post_traverse_copy)
                                    traverser.found = False
                                    traverser.target = op1
                                    # Traverse and see if we find op1 from op2
                                    op2.event_graph_traverse(traverser)
                                    if not traverser.found:
                                        traverser.found = False
                                        traverser.target = op2
                                        traverser.generation = self.get_next_traverser_gen()
                                        # Otherwise flip around and try to find op2 from op1
                                        op1.event_graph_traverse(traverser)
                                        if not traverser.found:
                                            print "   ERROR: Potential data race between "+\
                                                "requirement "+str(req1.index)+" of "+\
                                                op1.get_name()+" (UID "+str(op1.uid)+") "+\
                                                "and requirement "+str(req2.index)+" of "+\
                                                op2.get_name()+" (UID "+str(op2.uid)+") "+\
                                                "for field "+str(field)
                                            if self.verbose:
                                                print "      First Requirement:"
                                                req1.print_requirement()
                                                print "      Second Requirement:"
                                                req2.print_requirement()

    def print_pictures(self, path):
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
            components[idx].unmark_all()

