#!/usr/bin/python

# Copyright 2012 Stanford University
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

#!/usr/bin/python

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
ATOMICE = 1
SIMULTANEOUS = 2
RELAXED = 3


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
            print "Region Requirement for region ("+str(self.ispace)+","+str(self.fspace)+","+str(self.tid)+")"
        else:
            print "Region Requirement for partition ("+str(self.ispace)+","+str(self.fspace)+","+str(self.tid)+")"
        print "    Privilege: "+self.get_privilege()
        print "    Coherence: "+self.get_coherence()
        print "    Fields: "+self.get_fields()

    def get_privilege(self):
        if self.priv == NO_ACCESS:
            return "NO ACCESS"
        elif self.priv == READ_ONLY:
            return "READ ONLY"
        elif self.priv == READ_WRITE:
            return "READ-WRITE"
        elif self.priv == WRITE_ONLY:
            return "WRITE ONLY"
        else:
            return "REDUCE with Reduction Op "+str(self.redop)

    def get_coherence(self):
        if self.coher == EXCLUSIVE:
            return "EXCLUSIVE"
        elif self.coher == ATOMIC:
            return "ATOMIC"
        elif self.coher == SIMULTANEOUS:
            return "SIMULTANEOUS"
        else:
            return "RELAXED"

    def get_fields(self):
        result = ""
        for f in self.fields:
            result = result + str(f) + " "
        return result


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
        else:
            self.depth = 0

    def instantiate(self, parent_inst, field_node, tid):
        assert tid not in self.instances
        region_node = RegionNode(self.state, self, field_node, tid, parent_inst)
        self.instances[tid] = region_node
        for color,child in self.children:
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

    def instantiate(self, parent_inst, field_node, tid):
        assert tid not in self.instances
        part_node = PartitionNode(self.state, self, field_node, tid, parent_inst)
        self.instances[tid] = part_node
        for color,child in self.children:
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
        assert child not in children
        self.children.add(child)


class TreeState(object):
    def __init__(self):
        self.index_space_nodes = dict()
        self.index_part_nodes = dict()
        self.field_space_nodes = dict()
        self.region_trees = dict()

    def add_index_space(self, uid):
        assert uid not in self.index_space_nodes
        self.index_space_nodes[uid] = IndexSpaceNode(self, uid, 0, None)

    def add_index_partition(self, pid, uid, disjoint, color):
        assert uid not in self.index_part_nodes
        assert pid in self.index_space_nodes
        self.index_part_nodes[uid] = IndexPartNode(self, uid, disjoint, color, self.index_space_nodes[pid])

    def add_index_subspace(self, pid, uid, color):
        assert uid not in self.index_space_nodes
        assert pid in self.index_part_nodes
        self.index_space_nodes[uid] = IndexSpaceNode(self, uid, color, self.index_part_nodes[pid])

    def add_field_space(self, uid):
        assert uid not in self.field_space_nodes
        self.field_space_nodes[uid] = FieldSpaceNode(self, uid)

    def add_field(self, uid, fid):
        assert uid in self.field_space_nodes
        self.field_space_nodes[uid].add_field(fid)

    def add_region(self, iid, fid, tid):
        assert tid not in self.region_trees
        assert iid in self.index_space_nodes
        assert fid in self.field_space_nodes
        self.region_trees[tid] = self.index_space_nodes[iid].instantiate(None, self.field_space_nodes[fid], tid)

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
        # Handle the easy cases
        # If one was a subregion of the other, they are definitely aliased
        if (inode1 is orig1) or (inode1 is orig2):
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
            inode2 = indoe2.parent
        assert inode1 is inode2
        # Least common ancestor is a region, so they came from different
        # partitions and are therefore disjoint
        # TODO: handle when partitions are computed to be disjoint
        if inode1.is_region():
            return True
        return not inode1.disjoint 

    def get_index_node(self, is_reg, iid):
        if is_reg:
            if iid not in self.index_space_nodes:
                print "MISSING iid "+str(iid)
            assert iid in self.index_space_nodes
            return self.index_space_nodes[iid]
        else:
            assert iid in self.index_part_nodes
            return self.index_part_nodes[iid]


class ContextHandle(object):
    def __init__(self, uid, ctx):
        self.uid = uid
        self.ctx = ctx

    def __hash__(self):
        return hash((self.uid,self.ctx))

    def __eq__(self,other):
        return (self.uid,self.ctx) == (other.uid,other.ctx)


class Context(object):
    def __init__(self,handle,state):
        self.handle = handle
        self.op_state = state
        self.ops = list()
        self.mdeps = list() # The runtime computed mapping dependences
        self.adeps = list() # The mapping dependences computed here

    def append_op(self,op):
        self.ops.append(op)

    def add_mdep(self,mdep):
        self.mdeps.append(mdep)

    def add_adep(self,adep):
        self.adeps.append(adep)

    def compute_dependence_diff(self,tree_state):
        if (len(self.ops) == 0) or (len(self.ops) == 1):
            return
        parent_task = self.op_state.get_op(self.handle.uid)
        print "Checking mapping dependences for task "+str(parent_task.name)
        # First compute the list of adeps between each pair of operations
        for idx in range(1,len(self.ops)):
            for prev in range(idx):
                self.ops[prev].find_dependences(self.ops[idx], self.handle, self, tree_state)
        print "    Found "+str(len(self.adeps))+" dependences in all-pairs test for task "+str(parent_task.name)

        # For all the actual dependences make sure we have a dependence path between the
        # two different operations based on the dependences computed by the runtime
        count = 0
        errors = 0
        for adep in self.adeps:
            sys.stdout.write("    Checking dependence: %d \r" % (count))
            if not adep.op2.has_path(adep.op1):
                print "    ERROR: Failed to compute mapping dependence between index "+str(adep.idx1)+ \
                  " of "+adep.op1.get_name()+" (ID "+str(adep.op1.uid)+") and index "+str(adep.idx2)+ \
                  " of "+adep.op2.get_name()+" (ID "+str(adep.op2.uid)+")"
                errors = errors + 1
            for op in self.ops:
                op.unmark()
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
                  " of "+mdep.op2.get_name()+" (ID "+str(mdep.op2.uid)+") in context of task "+parent_task.get_name()
                warnings = warnings + 1

        print "    Mapping Dependence Errors: "+str(errors)
        print "    Mapping Dependence Warnings: "+str(warnings)
                

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


class TaskOp(object):
    def __init__(self, uid, tid, name):
        self.uid = uid
        self.tid = tid
        if name <> None:
            self.name = name
        else:
            self.name = str(tid)
        self.reqs = dict()
        self.incoming = set()
        self.outgoing = set()
        self.marked = False

    def add_requirement(self, index, is_reg, ispace, fspace, tid, priv, coher, redop):
        assert index not in self.reqs
        self.reqs[index] = Requirement(index, is_reg, ispace, fspace, tid, priv, coher, redop)

    def print_req(self,idx):
        assert idx in self.reqs
        self.reqs[idx].to_string()

    def add_req_field(self, index, fid):
        assert index in self.reqs
        self.reqs[index].add_field(fid)

    def get_name(self):
        return "Task "+self.name

    def find_dependences(self, op, handle, context, tree_state):
        for idx,req in self.reqs.items():
            op.find_individual_dependences(self, req, handle, context, tree_state)

    def find_individual_dependences(self, other_op, other_req, handle, context, tree_state):
        for idx,req in self.reqs.items():
            dtype = tree_state.compute_dependence(other_req, req)
            if is_mapping_dependence(dtype):
                context.add_adep(MappingDependence(handle, other_op, self, other_req.index, req.index, dtype))

    def add_incoming(self,op):
        assert op <> self
        self.incoming.add(op)

    def add_outgoing(self,op):
        assert op <> self
        self.outgoing.add(op)

    def has_path(self,target):
        if target == self:
            return True
        if self.marked:
            return False
        # Otherwise check all the outgoing edges
        for op in self.outgoing:
            if op.has_path(target):
                return True
        self.marked = True
        return False

    def unmark(self):
        self.marked = False


class MapOp(object):
    def __init__(self, uid):
        self.uid = uid
        self.req = None
        self.incoming = set()
        self.outgoing = set()
        self.marked = False

    def add_requirement(self, index, is_reg, ispace, fspace, tid, priv, coher, redop):
        assert index == 0
        self.req = Requirement(index, is_reg, ispace, fspace, tid, priv, coher, redop)

    def print_req(self,idx):
        assert idx == 0
        self.req.to_string()

    def add_req_field(self, index, fid):
        assert self.req <> None
        self.req.add_field(fid)

    def get_name(self):
        return "Mapping "+str(self.uid)

    def find_dependences(self, op, handle, context, tree_state):
        op.find_individual_dependences(self, self.req, handle, context, tree_state)

    def find_individual_dependences(self, other_op, other_req, handle, context, tree_state):
        dtype = tree_state.compute_dependence(other_req, self.req)
        if is_mapping_dependence(dtype):
            context.add_adep(MappingDependence(handle, other_op, self, other_req.index, self.req.index, dtype))

    def add_incoming(self,op):
        assert op <> self
        self.incoming.add(op)

    def add_outgoing(self,op):
        assert op <> self
        self.outgoing.add(op)

    def has_path(self,target):
        if target == self:
            return True
        if self.marked:
            return False
        # Otherwise check all the outgoing edges
        for op in self.outgoing:
            if op.has_path(target):
                return True
        self.marked = True
        return False

    def unmark(self):
        self.marked = False


class DeletionOp(object):
    def __init__(self, uid):
        self.uid = uid
        self.incoming = set()
        self.marked = False

    def get_name(self):
        return "Deletion "+str(self.uid)

    def find_dependences(self, op, handle, context, tree_state):
        # No need to do anything
        pass

    def find_individual_dependences(self, other_op, other_req, handle, context, tree_state):
        # TODO: implement this for deletion
        pass

    def add_incoming(self,op):
        assert op <> self
        self.incoming.add(op)

    def add_outgoin(self,op):
        # Should never happen
        assert False

    def has_path(self,target):
        if target == self:
            return True
        if self.marked:
            return False
        # Otherwise check all the outgoing edges
        for op in self.outgoing:
            if op.has_path(target):
                return True
        self.marked = True
        return False

    def unmark(self):
        self.marked = False


class OpState(object):
    def __init__(self):
        self.tasks = dict()
        self.maps = dict()
        self.deletions = dict()
        self.contexts = dict()

    def add_top_task(self, uid, tid):
        assert uid not in self.tasks
        self.tasks[uid] = TaskOp(uid, tid, None)

    def add_task(self, uid, tid, pid, ctx):
        # Index space tasks can be duplicated
        if uid not in self.tasks:
            self.tasks[uid] = TaskOp(uid, tid, None)
        context = self.get_context(pid, ctx)
        context.append_op(self.tasks[uid])

    def add_mapping(self, uid, pid, ctx):
        assert uid not in self.maps
        mapping = MapOp(uid)
        self.maps[uid] = mapping
        context = self.get_context(pid, ctx)
        context.append_op(mapping)

    def add_deletion(self, uid, pid, ctx):
        assert uid not in self.deletions
        deletion = DeletionOp(uid)
        self.deletions[uid] = deletion
        context = self.get_context(pid, ctx)
        context.append_op(deletion)

    def add_name(self, uid, name):
        assert uid in self.tasks
        self.tasks[uid].name = name

    def get_context(self, pid, ctx):
        ctx = ContextHandle(pid, ctx)
        if ctx not in self.contexts:
            self.contexts[ctx] = Context(ctx,self)
        return self.contexts[ctx]

    def get_op(self, uid):
        assert (uid in self.tasks) or (uid in self.maps) or (uid in self.deletions)
        if uid in self.tasks:
            return self.tasks[uid]
        elif uid in self.maps:
            return self.maps[uid]
        return self.deletions[uid]

    def get_name(self, uid):
        assert uid in self.tasks
        return self.tasks[uid].get_name()

    def add_requirement(self, uid, index, is_reg, ispace, fspace, tid, priv, coher, redop):
        self.get_op(uid).add_requirement(index, is_reg, ispace, fspace, tid, priv, coher, redop)

    def add_req_field(self, uid, index, fid):
        self.get_op(uid).add_req_field(index, fid)

    def add_mapping_dependence(self, pid, ctx, prev_id, pidx, next_id, nidx, dtype):
        context = self.get_context(pid, ctx)
        op1 = self.get_op(prev_id)
        op2 = self.get_op(next_id)
        context.add_mdep(MappingDependence(context.handle, op1, op2, pidx, nidx, dtype))
        # Graph edges point backwards in time from second task to first task
        op1.add_incoming(op2)
        op2.add_outgoing(op1)

    def check_logical(self, tree_state):
        for handle,ctx in self.contexts.iteritems():
            ctx.compute_dependence_diff(tree_state)


class EventHandle(object):
    def __init__(self, uid, gen):
        self.uid = uid
        self.gen = gen

    def __hash__(self):
        return hash((self.uid,self.gen))

    def __eq__(self,other):
        return (self.uid,self.gen) == (other.uid,other.gen)

class TaskHandle(object):
    def __init__(self, uid, point):
        self.uid = uid
        self.point = point 

    def __hash__(self):
        return hash((self.uid,self.point))

    def __eq__(self,other):
        return (self.uid,self.point) == (other.uid,other.point)

class Event(object):
    def __init__(self, handle):
        self.handle = handle
        self.incoming = set()
        self.outgoing = set()
        self.marked = False

    def add_incoming(self, event):
        assert self <> event
        self.incoming.add(event)

    def add_outgoing(self, event):
        if self == event:
            print str(self.handle.uid)+" "+str(self.handle.gen)
        assert self <> event
        self.outgoing.add(event)

    def traverse(self, component):
        if self.marked:
            return
        self.marked = True
        component.add_event(self)
        for n in self.incoming:
            n.traverse(component)
        for n in self.outgoing:
            n.traverse(component)

    def print_prev_dependences(self, printer, name):
        for n in self.incoming:
            n.print_prev_dependences(printer, name)

class TaskInstance(object):
    def __init__(self, handle, start, term, is_index):
        self.handle = handle
        self.start_event = start
        self.term_event = term
        self.is_index_space = is_index
        self.marked = False
        self.name = 'task_node_'+str(handle.uid)+'_'+str(handle.point)
        self.previous_dependences = set()

    def traverse(self, component):
        if self.marked:
            return
        self.marked = True
        component.add_task(self)
        self.start_event.traverse(component)
        self.term_event.traverse(component)

    def print_node(self, printer, ops):
        index_string = ""
        if self.is_index_space:
            index_string = '\\nIndex\ Space\ Point\ '+str(self.handle.point)
        printer.println(self.name+' [style=filled,label="'+ops.get_name(self.handle.uid)+index_string+ 
            '\\nUnique\ ID\ '+str(self.handle.uid)+'",fillcolor=lightskyblue,fontsize=14,fontcolor=black,shape=record,penwidth=2];')

    def print_dependences(self, printer):
        self.start_event.print_prev_dependences(printer, self.name)

    def print_prev_dependences(self, printer, later_name):
        # Check to see if we already printed this dependence
        if later_name not in self.previous_dependences:
            # Print the dependence, don't traverse back any farther
            printer.println(self.name+' -> '+later_name+' [style=solid,color=black,penwidth=2];') 
            self.previous_dependences.add(later_name)

class IndexInstance(object):
    def __init__(self, uid, term):
        self.uid = uid
        self.term_event = term
        self.points = dict()

    def add_point(self, handle, point):
        assert handle not in self.points
        self.points[handle] = point

class CopyInstance(object):
    def __init__(self, uid, srcid, dstid, srcloc, dstloc, index, field, tree, start, term, mask):
        self.srcid = srcid
        self.dstid = dstid
        self.srcloc = srcloc
        self.dstloc = dstloc
        self.index_space = index
        self.field_space = field
        self.tree_id = tree
        self.start_event = start
        self.term_event = term
        self.mask = mask
        self.marked = False
        self.name = 'copy_node_'+str(uid)
        self.previous_dependences = set()

    def traverse(self, component):
        if self.marked:
            return
        self.marked = True
        component.add_copy(self)
        self.start_event.traverse(component)
        self.term_event.traverse(component)

    def print_node(self, printer):
        printer.println(self.name+' [style=filled,label="Src\ Inst:\ '+str(self.srcid)+'\ Src\ Loc:\ '+str(self.srcloc)+
            '\\nDst\ Inst:\ '+str(self.dstid)+'\ Dst\ Loc:\ '+str(self.dstloc)+
            '\\nLogical\ Region:\ (index:'+str(self.index_space)+',field:'+str(self.field_space)+',tree:'+str(self.tree_id)+')'+
            '\\nCopy\ Mask:\ '+self.mask+
            '",fillcolor=darkgoldenrod1,fontsize=14,fontcolor=black,shape=record,penwidth=2];')

    def print_dependences(self, printer):
        self.start_event.print_prev_dependences(printer, self.name)

    def print_prev_dependences(self, printer, later_name):
        # Check to make sure we haven't printed this dependence before
        if later_name not in self.previous_dependences:
            printer.println(self.name+' -> '+later_name+' [style=solid,color=black,penwidth=2];')
            self.previous_dependences.add(later_name)

class ReduceInstance(object):
    def __init__(self, uid, srcid, dstid, srcloc, dstloc, index, field, tree, start, term, mask):
        self.srcid = srcid
        self.dstid = dstid
        self.srcloc = srcloc
        self.dstloc = dstloc
        self.index_space = index
        self.field_space = field
        self.tree_id = tree
        self.start_event = start
        self.term_event = term
        self.mask = mask
        self.marked = False
        self.name = 'reduce_node_'+str(uid)
        self.previous_dependences = set()

    def traverse(self, component):
        if self.marked:
            return
        self.marked = True
        component.add_reduction(self)
        self.start_event.traverse(component)
        self.term_event.traverse(component)

    def print_node(self, printer):
        printer.println(self.name+' [style=filled,label="Src\ Inst:\ '+str(self.srcid)+'\ Src\ Loc:\ '+str(self.srcloc)+
            '\\nDst\ Inst:\ '+str(self.dstid)+'\ Dst\ Loc:\ '+str(self.dstloc)+
            '\\nLogical\ Region:\ (index:'+str(self.index_space)+',field:'+str(self.field_space)+',tree:'+str(self.tree_id)+')'+
            '\\nReduce\ Mask:\ '+self.mask+
            '",fillcolor=crimson,fontsize=14,fontcolor=black,shape=record,penwidth=2];')

    def print_dependences(self, printer):
        self.start_event.print_prev_dependences(printer, self.name)

    def print_prev_dependences(self, printer, later_name):
        # Make sure we haven't printed this dependence before
        if later_name not in self.previous_dependences:
            printer.println(self.name+' -> '+later_name+' [style=solid,color=black,penwidth=2];')
            self.previous_dependences.add(later_name)


class MapInstance(object):
    def __init__(self, uid, start, term):
        self.uid = uid
        self.start_event = start
        self.term_event = term
        self.name = 'mapping_node_'+str(uid)
        self.marked = False
        self.previous_dependences = set()

    def traverse(self, component):
        if self.marked:
            return
        self.marked = True
        component.add_map(self)
        self.start_event.traverse(component)
        self.term_event.traverse(component)

    def print_node(self, printer):
        printer.println(self.name+' [style=filled,label="Inline\ Mapping\ '+str(self.uid)+
            '",fillcolor=mediumseagreen,fontsize=14,fontcolor=black,shape=record,penwidth=2];')

    def print_dependences(self, printer):
        self.start_event.print_prev_dependences(printer, self.name)

    def print_prev_dependences(self, printer, later_name):
        if later_name not in self.previous_dependences:
            printer.println(self.name+' -> '+later_name+' [style=solid,color=black,penwidth=2];')
            self.previous_dependences.add(later_name)

class EventGraphPrinter(object):
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


class ConnectedComponent(object):
    def __init__(self):
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
        if len(self.tasks) == 0 and len(self.maps) == 0 and len(self.copies) == 0 and len(self.reductions) == 0:
            return True
        return False

    def generate_graph(self, idx, ops, path):
        name = 'event_graph_'+str(idx)
        printer = EventGraphPrinter(path,name)
        # Print the nodes
        for t in self.tasks:
            t.print_node(printer,ops)
        for m in self.maps:
            m.print_node(printer)
        for c in self.copies:
            c.print_node(printer)
        for r in self.reductions:
            r.print_node(printer)
        # Now print the dependences
        for t in self.tasks:
            t.print_dependences(printer)
        for m in self.maps:
            m.print_dependences(printer)
        for c in self.copies:
            c.print_dependences(printer) 
        for r in self.reductions:
            r.print_dependences(printer)
        dot_file = printer.close()
        pdf_file = name+'.pdf'
        try:
            subprocess.check_call(['dot -Tpdf -o '+pdf_file+' '+dot_file],shell=True)
        except:
            print "WARNING: DOT failure, image for event graph "+str(idx)+" not generated"
            subprocess.call(['rm -f core '+pdf_file],shell=True)


class EventGraph(object):
    def __init__(self):
        self.events = dict()
        self.tasks = dict()
        self.index_tasks = dict()
        self.maps = dict()
        self.copies = set()
        self.next_copy = 1
        self.next_reduce = 1

    def get_event(self, handle):
        if handle not in self.events:
            self.events[handle] = Event(handle)
        return self.events[handle]

    def add_event_dependence(self, id1, gen1, id2, gen2):
        e1 = self.get_event(EventHandle(id1,gen1))
        e2 = self.get_event(EventHandle(id2,gen2))
        e1.add_outgoing(e2)
        e2.add_incoming(e1)

    def add_task_instance(self, uid, index_space, point, startid, startgen, termid, termgen):
        handle = TaskHandle(uid, point)
        assert handle not in self.tasks
        start_event = self.get_event(EventHandle(startid,startgen))
        term_event = self.get_event(EventHandle(termid,termgen))
        task = TaskInstance(handle,start_event,term_event,index_space)
        self.tasks[handle] = task
        if uid in self.index_tasks:
            self.index_tasks[uid].add_point(handle, task)
            global_term = self.index_tasks[uid].term_event
            term_event.add_outgoing(global_term)
            global_term.add_incoming(term_event)
        start_event.add_outgoing(task)
        term_event.add_incoming(task)

    def add_index_term(self, uid, termid, termgen):
        assert uid not in self.index_tasks
        term_event = self.get_event(EventHandle(termid, termgen))
        self.index_tasks[uid] = IndexInstance(uid, term_event)

    def add_copy_instance(self, srcid, dstid, srcloc, dstloc, index, field, tree, startid, startgen, termid, termgen, mask):
        start_event = self.get_event(EventHandle(startid,startgen))
        term_event = self.get_event(EventHandle(termid,termgen))
        copy_op = CopyInstance(self.next_copy, srcid, dstid, srcloc, dstloc, index, field, tree, start_event, term_event, mask)
        self.next_copy = self.next_copy + 1
        start_event.add_outgoing(copy_op)
        term_event.add_incoming(copy_op)

    def add_reduce_instance(self, srcid, dstid, srcloc, dstloc, index, field, tree, startid, startgen, termid, termgen, mask):
        start_event = self.get_event(EventHandle(startid,startgen))
        term_event = self.get_event(EventHandle(termid,termgen))
        reduce_op = ReduceInstance(self.next_reduce, srcid, dstid, srcloc, dstloc, index, field, tree, start_event, term_event, mask)
        self.next_reduce = self.next_reduce + 1
        start_event.add_outgoing(reduce_op)
        term_event.add_incoming(reduce_op)

    def add_map_instance(self, uid, startid, startgen, termid, termgen):
        assert uid not in self.maps
        start_event = self.get_event(EventHandle(startid,startgen))
        term_event = self.get_event(EventHandle(termid,termgen))
        map_inst = MapInstance(uid, start_event, term_event)
        self.maps[uid] = map_inst
        start_event.add_outgoing(map_inst)
        term_event.add_incoming(map_inst)

    def make_pictures(self, ops, path):
        # First compute the connected components of the graph 
        components = list()
        # Go through all the events and find the components
        for h,e in self.events.iteritems():
            comp = ConnectedComponent()
            e.traverse(comp)
            if not comp.empty():
                components.append(comp)

        print "Found "+str(len(components))+" event graphs"

        for idx in range(len(components)):
            components[idx].generate_graph(idx,ops,path)

