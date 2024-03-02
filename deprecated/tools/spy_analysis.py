#!/usr/bin/env python3

# Copyright 2024 Stanford University
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
import time

# These are imported from legion_types.h
NO_DEPENDENCE = 0
TRUE_DEPENDENCE = 1
ANTI_DEPENDENCE = 2
ATOMIC_DEPENDENCE = 3
SIMULTANEOUS_DEPENDENCE = 4
DEPENDENCE_TYPES = [
"no",
"true",
"anti",
"atomic",
"simultaneous",
]

NO_ACCESS  = 0x00000000
READ_ONLY  = 0x00000001
READ_WRITE = 0x00000007
WRITE_ONLY = 0x00000002
REDUCE     = 0x00000004

EXCLUSIVE = 0
ATOMIC = 1
SIMULTANEOUS = 2
RELAXED = 3

LOC_PROC = 0
TOC_PROC = 1
UTIL_PROC = 2

MEMORY_TYPES = [
"GASNET",
"SYSTEM",
"REGDMA",
"SOCKET",
"Z_COPY",
"GPU_FB",
"DISK",
"HDF",
"FILE",
"LEVEL3_CACHE",
"LEVEL2_CACHE",
"LEVEL1_CACHE",
]

# Operation Kinds
SINGLE_OP = 0
INDEX_OP = 1
MAPPING_OP = 2
CLOSE_OP = 3
DELETION_OP = 4
COPY_OP = 5
FENCE_OP = 6
ACQUIRE_OP = 7
RELEASE_OP = 8
DEPENDENT_PARTITION_OP = 9
PENDING_PARTITION_OP = 10
FILL_OP = 11

# Realm copies are not Legion operations, but we will treat them in the same way
COPY_EVENTS = 100

# Instance Kinds
TASK_INST = 0
MAPPING_INST = 1
CLOSE_INST = 2
COPY_INST = 3
ACQUIRE_INST = 4
RELEASE_INST = 5
DEPENDENT_PARTITION_INST = 6
PENDING_PARTITION_INST = 7

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

def wrap_with_trtd(labels):
    line = "<tr>"
    for label in labels:
        if isinstance(label, str):
            l = label
            rowspan = 1
            colspan = 1
        else:
            l = label["label"]
            rowspan = label["rowspan"] if "rowspan" in label else 1
            colspan = label["colspan"] if "colspan" in label else 1
        line = line + ('<td colspan="%d" rowspan="%d">%s</td>' % (colspan, rowspan, l))
    line = line + "</tr>"
    return line

def generate_html_op_label(title, requirements, instances, color, verbose):
    lines = list()
    lines.append([{"label" : title, "colspan" : 2}])

    for i in range(0, len(requirements)):
        req = requirements[i]
        region_name = req.region_node.get_name()
        priv = req.get_privilege_and_coherence()
        if req.is_reg:
            prefix = "Region"
        else:
            prefix = "Partition"

        if region_name <> None:
            line = [prefix + " " + str(i), region_name+" (priv: "+priv+")"]
        else:
            line = [{"label" : prefix+" "+str(i)+" (priv:"+priv+")", "colspan" : 2}]
        lines.append(line)
        req_summary = 'index:%s,field:%s,tree:%s' % \
                (hex(req.region_node.index_node.uid),
                        hex(req.region_node.field_node.uid),
                        str(req.region_node.tid))
        lines.append(["Requirement", req_summary])

        if verbose:
            if instances <> None and i in instances:
                lines.append(["Memory", instances[i].memory.dot_memory()])
                lines.append(["Instance", instances[i].dot_instance()])

            field_names = req.get_field_names()
            first_field = True
            for f in field_names:
                line = []
                if first_field:
                    line.append({"label" : "Fields", "rowspan" : len(field_names)})
                line.append(f)
                lines.append(line)
                first_field = False

    return '<table border="0" cellborder="1" cellpadding="3" cellspacing="0" bgcolor="%s">' % color + \
            "".join([wrap_with_trtd(line) for line in lines]) +\
            '</table>'

def generate_html_instance_label(title, instance, color, verbose):
    colspan = 2 if len(instance.fields) > 1 and verbose else 1

    lines = list()
    if title <> None:
        lines.append([{"label" : title, "colspan" : colspan}])
    lines.append([{"label" : "Memory: " + instance.memory.dot_memory(),
        "colspan" : colspan}])
    lines.append([{"label" : "Instance: " + instance.dot_instance(),
        "colspan" : colspan}])

    if verbose:
        lines.append([{"label" : "", "colspan" : colspan}])
        lines.append([{"label" : "Fields", "colspan" : colspan}])
        fields = instance.region.field_node.get_field_names(instance.fields)
        if len(fields) % 2 == 1:
            fields.append("")
        for i in range(0, len(fields) / 2):
            lines.append([fields[2 * i], fields[2 * i + 1]])

    return '<table border="0" cellborder="0" cellpadding="2" cellspacing="0" bgcolor="%s">' % color + \
            "".join([wrap_with_trtd(line) for line in lines]) +\
            '</table>'

def compute_dependence_type(req1, req2):
    if req1.is_no_access() or req2.is_no_access():
        return NO_DEPENDENCE
    elif req1.is_read_only() and req2.is_read_only():
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
            return SIMULTANEOUS_DEPENDENCE
        elif req1.is_relaxed() and req2.is_relaxed():
            return req1,req2,SIMULTANEOUS_DEPENDENCE
        # Should never get here
        assert False
        return NO_DEPENDENCE

def list_to_ranges(int_list):
    sorted_list = sorted(int_list)

    def add_range(ranges, range):
        if range[0] == range[1]:
            ranges.append(str(range[0]))
        else:
            ranges.append(str(range[0])+"~"+str(range[1]))

    range = [None, None]
    ranges = list()
    for val in sorted_list:
        if range[1] == None or val != range[1] + 1:
            if range[1] != None:
                add_range(ranges, range)
            range[0] = val
        range[1] = val
    add_range(ranges, range)
    return ranges

class Processor(object):
    def __init__(self, state, uid, kind):
        self.state = state
        self.uid = uid
        self.kind = kind
        self.mem_latency = dict()
        self.mem_bandwidth = dict()
        self.executed_ops = list()

    def add_memory(self, mem, bandwidth, latency):
        assert mem not in self.mem_latency
        assert mem not in self.mem_bandwidth
        self.mem_latency[mem] = latency
        self.mem_bandwidth[mem] = bandwidth

    def append_op_instance(self, op):
        self.executed_ops.append(op)


class UtilityProcessor(object):
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.kind = UTIL_PROC
        self.constituents = set()
        self.mem_latency = dict()
        self.mem_bandwidth = dict()
        self.executed_ops = list()

    def add_memory(self, mem, bandwidth, latency):
        assert mem not in self.mem_latency
        assert mem not in self.mem_bandwidth
        self.mem_latency[mem] = latency
        self.mem_bandwidth[mem] = bandwidth

    def add_constituent(self, proc):
        assert proc not in self.constituents
        self.constituents.add(proc)

    def append_op_instance(self, op):
        self.executed_ops.append(op)

class Memory(object):
    def __init__(self, state, uid, capacity, kind):
        self.state = state
        self.uid = uid
        self.capacity = capacity
        self.kind = kind
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
        if inst.iid not in self.physical_instances:
            self.physical_instances[inst.iid] = [inst]
        else:
            self.physical_instances[inst.iid].append(inst)

    def add_reduction_instance(self, inst):
        if inst.iid not in self.reduction_instances:
            self.reduction_instances[inst.iid] = [inst]
        else:
            self.reduction_instances[inst.iid].append(inst)

    def print_timeline(self):
        name = "memory_"+str(self.uid)+"_timeline"
        return

    def dot_memory(self):
        return hex(self.uid)+' ('+MEMORY_TYPES[self.kind]+')'


class IndexSpaceNode(object):
    def __init__(self, state, uid, point, parent):
        self.state = state
        self.uid = uid
        self.parent = parent
        self.instances = dict()
        self.children = dict()
        if parent <> None:
            parent.add_child(point, self)
            self.depth = parent.depth + 1
            # Also check to see if the parent has any instances for which
            # we need to instantiate this tree
            for tid,pinst in parent.instances.iteritems():
                self.instantiate(pinst,pinst.field_node,tid)
        else:
            self.depth = 0
        self.name = None
        self.node_name = "index_space_node_" + str(uid)
        self.independent_pairs = set()

    def instantiate(self, parent_inst, field_node, tid):
        assert tid not in self.instances
        region_node = RegionNode(self.state, self, field_node, tid, parent_inst)
        self.instances[tid] = region_node
        for color,child in self.children.iteritems():
            child.instantiate(region_node, field_node, tid)
        return region_node

    def add_child(self, point, child):
        assert point not in self.children
        self.children[point] = child 

    def get_instance(self, tid):
        assert tid in self.instances
        return self.instances[tid]

    def is_region(self):
        return True

    def get_name(self):
        if self.name <> None:
            return self.name
        else:
            return "index space " + hex(self.uid)

    def set_name(self, name):
        self.name = name

    def find_child(self, uid):
        for point,child in self.children.iteritems():
            if child.uid == uid:
                return child
        return None

    def find_child_color(self, child):
        for c, child_ in self.children.iteritems():
            if child_ == child:
                return c
        return None

    def mark_independent(self, uid1, uid2):
        assert self.find_child(uid1) <> None
        assert self.find_child(uid2) <> None
        if uid1 < uid2:
            self.independent_pairs.add((uid1, uid2))
        else:
            self.independent_pairs.add((uid2, uid1))

    def is_independent(self, uid1, uid2):
        assert self.find_child(uid1) <> None
        assert self.find_child(uid2) <> None
        if uid1 < uid2:
            pair = (uid1, uid2)
        else:
            pair = (uid2, uid1)
        return pair in self.independent_pairs

    def print_link_to_parent(self, printer, parent):
        printer.println(parent+' -> '+ self.node_name+\
                " [style=solid,color=black,penwidth=2];")

    def print_graph(self, printer):
        # print node itself
        if self.name != None:
            label = self.name+' (ID: '+hex(self.uid)+')'
        else:
            if self.parent == None:
                label = 'index space '+hex(self.uid)
            else:
                color = None
                for c, child in self.parent.children.iteritems():
                    if child == self:
                        color = c
                        break
                assert color != None
                label = 'subspace '+hex(self.uid)+\
                        ' (color: ' + color.to_simple_string() +')'

        printer.println(self.node_name+' [label="'+label+\
                '",shape=plaintext,fontsize=14,'+\
                'fontcolor=black,fontname="Helvetica"];')
        # print links to children
        for child in self.children.itervalues():
            child.print_link_to_parent(printer, self.node_name)
        for child in self.children.itervalues():
            child.print_graph(printer)

class IndexPartNode(object):
    def __init__(self, state, uid, disjoint, point, parent):
        self.state = state
        self.uid = uid
        self.disjoint = disjoint
        self.parent = parent
        self.instances = dict()
        self.children = dict()
        assert parent <> None
        parent.add_child(point, self)
        self.depth = parent.depth + 1
        # Also need to instaitate any instances from the parent
        for tid,pinst in parent.instances.iteritems():
            self.instantiate(pinst, pinst.field_node, tid)
        self.name = None
        self.node_name = "index_part_node_" + str(uid)
        self.independent_pairs = set()

    def instantiate(self, parent_inst, field_node, tid):
        assert tid not in self.instances
        part_node = PartitionNode(self.state, self, field_node, tid, parent_inst)
        self.instances[tid] = part_node
        for color,child in self.children.iteritems():
            child.instantiate(part_node, field_node, tid)
        return part_node

    def add_child(self, point, child):
        assert point not in self.children
        self.children[point] = child

    def is_region(self):
        return False

    def get_name(self):
        if self.name <> None:
            return self.name
        else:
            return "index partition " + hex(self.uid)

    def set_name(self, name):
        self.name = name

    def find_child(self, uid):
        for point,child in self.children.iteritems():
            if child.uid == uid:
                return child
        return None

    def find_child_color(self, child):
        for c, child_ in self.children.iteritems():
            if child_ == child:
                return c
        return None

    def mark_independent(self, uid1, uid2):
        assert self.find_child(uid1) <> None
        assert self.find_child(uid2) <> None
        if uid1 < uid2:
            self.independent_pairs.add((uid1, uid2))
        else:
            self.independent_pairs.add((uid2, uid1))

    def is_independent(self, uid1, uid2):
        assert self.find_child(uid1) <> None
        assert self.find_child(uid2) <> None
        if uid1 < uid2:
            pair = (uid1, uid2)
        else:
            pair = (uid2, uid1)
        return pair in self.independent_pairs

    def print_link_to_parent(self, printer, parent):
        if self.disjoint:
            edge_label = '*'
        else:
            edge_label = ''
        printer.println(parent+' -> '+ self.node_name+\
                ' [label="'+edge_label+'",style=dotted,color=black,penwidth=2];')

    def print_graph(self, printer):
        # print node itself
        if self.name != None:
            label = self.name+' (ID: '+hex(self.uid)+')'
        else:
            label = 'index partition '+hex(self.uid)
        printer.println(self.node_name+' [label="'+label+\
                '",shape=plaintext,fontsize=14,'+\
                'fontcolor=black,fontname="times italic"];')
        # print links to children
        for child in self.children.itervalues():
            child.print_link_to_parent(printer, self.node_name)
        for child in self.children.itervalues():
            child.print_graph(printer)

class FieldSpaceNode(object):
    def __init__(self, state, uid):
        self.state = state
        self.uid = uid
        self.fields = set()
        self.name = None
        self.field_names = dict()
        self.node_name = "field_space_node_"+str(uid)

    def add_field(self, fid):
        assert fid not in self.fields
        self.fields.add(fid)

    def set_name(self, name):
        self.name = name

    def set_field_name(self, fid, name):
        self.field_names[fid] = name

    def get_field_name(self, fid):
        if fid in self.field_names:
            return self.field_names[fid]
        else:
            return None

    def get_field_names(self, fields):
        field_names = list()
        for f in fields:
            field_name = self.get_field_name(f)
            if field_name != None:
                field_names.append(field_name)
            else:
                field_names.append(str(f))
        return field_names

    def field_mask_string(self, fields):
        return ','.join(self.get_field_names(fields))

    def print_graph(self, printer):
        if self.name != None:
            label = self.name+' (ID: '+str(self.uid)+')'
        else:
            label = 'field space '+str(self.uid)
        printer.println(self.node_name+' [label="'+label+\
                '",shape=plaintext,fontsize=14,'+\
                'fontcolor=black,fontname="Helvetica"];')

        for field in self.fields:
            field_id = "field_node_"+str(self.uid)+"_"+str(field)
            if field in self.field_names:
                field_name = self.field_names[field]+' (FID: '+str(field)+')'
            else:
                field_name = 'FID: ' + str(field)
            printer.println(field_id+' [label="'+field_name+\
                    '",shape=plaintext,fontsize=14,'+\
                    'fontcolor=black,fontname="Helvetica"]')
            printer.println(self.node_name+' -> '+ field_id+\
                    " [style=dotted,color=black,penwidth=2];")

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
        self.name = None
        self.node_name = "region_node_"+str(index_node.uid)+"_"+\
                str(field_node.uid)+"_"+str(tid)
        self.has_named_children = False

    def add_child(self, child):
        assert child not in self.children
        self.children.add(child)

    def set_name(self, name, index_node, field_node):
        if self.index_node == index_node and \
                self.field_node == field_node:
                    self.name = name
        else:
            for child in self.children:
                child.set_name(name, index_node, field_node)

    def get_name(self):
        if self.name <> None:
            return self.name
        elif self.parent <> None:
            parent_name = self.parent.get_name()
            color = self.parent.index_node.find_child_color(self.index_node)
            assert(color <> None)
            if parent_name <> None:
                return parent_name + "[" + color.to_dim_string() + "]"
            else:
                return None
        return None

    def find_node(self, index_node, field_node):
        if self.index_node == index_node and \
                self.field_node == field_node:
                    return self
        else:
            for child in self.children:
                node = child.find_node(index_node, field_node)
                if node != None:
                    return node
        return None

    def mark_named_children(self):
        if self.name != None:
            self.has_named_children = True
        for child in self.children:
            has_named_children = child.mark_named_children()
            self.has_named_children = self.has_named_children or \
                    has_named_children
        return self.has_named_children

    def print_link_to_parent(self, printer, parent):
        printer.println(parent+' -> '+ self.node_name+\
                " [style=solid,color=black,penwidth=2];")

    def gen_id(self):
        return 'index: '+hex(self.index_node.uid)+','+\
                'field: '+str(self.field_node.uid)+','+\
                'tree: '+str(self.tid)

    def print_node(self, printer):
        if self.name != None:
            label = self.name+' ('+self.gen_id()+')'
        else:
            if self.parent == None:
                label = 'region ('+self.gen_id()+')'
            else:
                label = 'subregion ('+self.gen_id()+')'

        printer.println(self.node_name+' [label="'+label+\
                '",shape=plaintext,fontsize=14,'+\
                'fontcolor=black,fontname="Helvetica"];')

    def print_graph(self, printer, simplify_graph):
        if simplify_graph and not self.has_named_children:
            return
        # print node itself
        self.print_node(printer)
        # print links to children
        for child in self.children:
            if not simplify_graph or child.has_named_children:
                child.print_link_to_parent(printer, self.node_name)
        for child in self.children:
            child.print_graph(printer, simplify_graph)

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
        self.name = None
        self.node_name = "part_node_"+str(index_node.uid)+"_"+\
                str(field_node.uid)+"_"+str(tid)
        self.has_named_children = False

    def add_child(self, child):
        assert child not in self.children
        self.children.add(child)

    def set_name(self, name, index_node, field_node):
        if self.index_node == index_node and \
                self.field_node == field_node:
                    self.name = name
        else:
            for child in self.children:
                child.set_name(name, index_node, field_node)

    def get_name(self):
        if self.name <> None:
            return self.name
        elif self.parent <> None:
            parent_name = self.parent.get_name()
            if parent_name <> None:
                return parent_name + ".partition"
        return None

    def find_node(self, index_node, field_node):
        if self.index_node == index_node and \
                self.field_node == field_node:
                    return self
        else:
            for child in self.children:
                node = child.find_node(index_node, field_node)
                if node != None:
                    return node
        return None

    def mark_named_children(self):
        if self.name != None:
            self.has_named_children = True
        for child in self.children:
            has_named_children = child.mark_named_children()
            self.has_named_children = self.has_named_children or \
                    has_named_children
        return self.has_named_children


    def print_link_to_parent(self, printer, parent):
        if self.index_node.disjoint:
            edge_label = '*'
        else:
            edge_label = ''
        printer.println(parent+' -> '+ self.node_name+\
                ' [label="'+edge_label+'",style=dotted,color=black,penwidth=2];')

    def gen_id(self):
        return 'part: '+hex(self.index_node.uid)+','+\
                'field: '+str(self.field_node.uid)+','+\
                'tree: '+str(self.tid)

    def print_node(self, printer):
        if self.name != None:
            label = self.name+' ('+self.gen_id() +')'
        else:
            label = 'partition ('+self.gen_id() +')'
        printer.println(self.node_name+' [label="'+label+\
                '",shape=plaintext,fontsize=14,'+\
                'fontcolor=black,fontname="times italic"];')

    def print_graph(self, printer, simplify_graph):
        if simplify_graph and not self.has_named_children:
            return
        # print node itself
        self.print_node(printer)
        # print links to children
        for child in self.children:
            child.print_link_to_parent(printer, self.node_name)
            if simplify_graph and not child.has_named_children:
                child.print_node(printer)
        for child in self.children:
            child.print_graph(printer, simplify_graph)

#########################################################
#   Operations
#########################################################

class Operation(object):
    def __init__(self, state, uid, ctx, kind, name, node_name, color):
        self.state = state
        self.uid = uid
        self.ctx = ctx
        self.kind = kind
        self.reqs = dict()
        self.logical_incoming = set()
        self.logical_outgoing = set()
        self.logical_mark = dict()
        self.start_event = None
        self.term_event = None
        self.op_instances = set()
        self.op_instances.add(self)
        self.instances = dict()
        self.physical_marked = False
        self.name = name
        self.node_name = node_name + str(self.uid)
        self.color = color
        self.prev_event_deps = set()
        self.generation = 0

    def get_name(self):
        return self.name + " (ID: " + str(self.uid) + ")"

    def get_op_kind(self):
        return self.kind

    def add_requirement(self, idx, req):
        assert idx not in self.reqs
        self.reqs[idx] = req

    def get_requirement(self, idx):
        assert idx in self.reqs
        return self.reqs[idx]

    def get_fields(self, idx):
        return self.get_requirement(idx).fields

    def get_field_name(self, idx, field):
        return self.get_requirement(idx).get_field_name(field)

    def get_num_requirements(self):
        return len(self.reqs)

    def add_req_field(self, idx, fid):
        if not idx in self.reqs:
            return False
        self.reqs[idx].add_field(fid)
        return True

    def add_logical_incoming(self, op):
        #assert op <> self
        self.logical_incoming.add(op)

    def add_logical_outgoing(self, op):
        #assert op <> self
        self.logical_outgoing.add(op)

    def get_logical_mark(self, idx):
        if idx not in self.logical_mark:
            self.logical_mark[idx] = 0
        return self.logical_mark[idx]

    def has_logical_path(self, idx, target, target_idx, mark):
        if target == self and idx == target_idx:
            return True
        if self.get_logical_mark(idx) == mark:
            return False
        # Otherwise check all the outgoing edges
        sources = self.ctx.find_logically_dependent_sources(self, idx)
        for (src_op, src_idx) in sources:
            if src_op.has_logical_path(src_idx, target, target_idx, mark):
                return True
        self.logical_mark[idx] = mark
        return False

    def get_reachable(self, reachable, forward):
        if self in reachable:
            return
        reachable.add(self)
        if forward:
            for op in self.logical_outgoing:
                op.get_reachable(reachable, True)
        else:
            for op in self.logical_incoming:
                op.get_reachable(reachable, False)

    def add_events(self, start, term):
        assert self.start_event == None
        assert self.term_event == None
        self.start_event = start
        self.term_event = term

    def add_instance(self, idx, inst):
        assert idx not in self.instances
        self.instances[idx] = inst

    def get_instance(self, idx, field=None):
        assert idx in self.instances
        return self.instances[idx]

    def get_all_instances(self):
        return set(self.instances.values())

    def physical_traverse(self, component):
        if self.physical_marked:
            return
        self.physical_marked = True
        self.add_to_component(component)
        self.start_event.physical_traverse(component)
        self.term_event.physical_traverse(component)

    def physical_unmark(self):
        self.physical_marked = False

    def compute_dependence_diff(self, verbose):
        pass

    def find_dependences(self, op):
        for idx, req in self.reqs.iteritems():
            op.find_individual_dependences(self, req)

    def find_individual_dependences(self, other_op, other_req):
        for idx, req in self.reqs.iteritems():
            dtype = self.state.compute_dependence(other_req, req)
            if is_mapping_dependence(dtype):
                self.ctx.add_adep(other_op, self, other_req.index, req.index, dtype)

    def check_data_flow(self):
        pass

    def print_dataflow(self, path, simplify):
        return 0

    def print_logical_node(self, printer):
        self.print_base_node(printer, True)

    def print_physical_node(self, printer):
        self.print_base_node(printer, False)

    def print_base_node(self, printer, logical):
        title = self.get_name()
        if self.ctx <> None:
            title += ' in ' + self.ctx.get_name()
        label = generate_html_op_label(title, self.reqs,
                                       self.instances if not logical else None,
                                       self.color, self.state.verbose)
        printer.println(self.node_name+' [label=<'+label+'>,fontsize=14,'+\
                'fontcolor=black,shape=record,penwidth=0];')

    def print_igraph_node(self, printer):
        self.print_base_node(printer, False)

    def print_igraph_edges(self, printer):
        for idx, inst in self.instances.iteritems():
            req = self.reqs[idx]
            label = req.get_privilege_and_coherence()
            if req.is_read_only():
                inst.print_outgoing_edge(printer, self.node_name, edge_label=label)
            elif req.is_write_only():
                inst.print_incoming_edge(printer, self.node_name, edge_label=label)
            elif req.is_read_write():
                inst.print_incoming_edge(printer, self.node_name, edge_label=label, dir='both')
            elif req.is_reduce():
                inst.print_incoming_edge(printer, self.node_name, edge_label=label)
            else:
                assert False

    def print_event_dependences(self, printer):
        self.start_event.print_prev_event_dependences(printer, self.node_name)

    def print_prev_event_dependences(self, printer, later_name):
        if later_name not in self.prev_event_deps:
            printer.println(self.node_name+' -> '+later_name+
                ' [style=solid,color=black,penwidth=2];')
            self.prev_event_deps.add(later_name)

class SingleTask(Operation):
    def __init__(self, state, uid, tid, ctx, name):
        Operation.__init__(self, state, uid, ctx, SINGLE_OP, name,
                "task_node_", "lightskyblue")
        self.tid = tid
        self.mdeps = list()
        self.adeps = list()
        self.ops = list()
        self.enclosing = None
        self.point = None

    def get_name(self):
        name = ""
        if self.name <> None:
            name = self.name
        else:
            name = "task"
        if self.enclosing <> None:
            point = "["+self.point.to_dim_string()+"]"
            name += point
        name += " (ID: " + str(self.uid) + ")"
        return name

    def set_enclosing(self, enc, point):
        self.enclosing = enc
        self.color = "mediumslateblue"
        self.point = point

    def add_operation(self, op):
        self.ops.append(op)

    def add_requirement(self, idx, req):
        # Should only happen to individual tasks
        assert self.enclosing == None
        Operation.add_requirement(self, idx, req)

    def add_mdep(self, op1, op2, idx1, idx2, dtype):
        assert isinstance(op1, Fence) or op1 in self.ops
        assert isinstance(op2, Fence) or op2 in self.ops
        self.mdeps.append(MappingDependence(self, op1, op2, idx1, idx2, dtype))

    def find_logically_dependent_sources(self, op, idx):
        result = set()
        for mdep in self.mdeps:
            if mdep.op2 == op and mdep.idx2 == idx:
                result.add((mdep.op1, mdep.idx1))
        return result

    def add_adep(self, op1, op2, idx1, idx2, dtype):
        assert isinstance(op1, Fence) or op1 in self.ops
        assert isinstance(op2, Fence) or op2 in self.ops
        self.adeps.append(MappingDependence(self, op1, op2, idx1, idx2, dtype))

    def add_instance_requirement(self, idx, index):
        if idx in self.reqs:
            assert self.enclosing == None
            assert self.reqs[idx].ispace == index
            return True
        if self.enclosing == None:
            return False
        self.reqs[idx] = self.enclosing.get_child_req(idx, index)
        return True

    def event_graph_traverse(self, traverser):
        traverser.visit_task(self)

    def add_to_component(self, component):
        component.add_task(self)

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
                print "          " + str(dep)
            print "        Actual Dependences:"
            for dep in self.adeps:
                print "          " + str(dep)
        # For all the actual dependences make sure we have a dependence path between the
        # two different operations based on the dependences computed by the runtime
        count = 0
        errors = 0
        for adep in self.adeps:
            sys.stdout.write("    Checking dependence: %d \r" % (count))
            check = adep.op2.has_logical_path(adep.idx2, adep.op1, adep.idx1,
                    self.state.get_next_logical_mark())

            if not check:
                print "    ERROR: Failed to compute mapping dependence between "+str(adep)
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
                # Skip fences too
                if mdep.op1.get_op_kind() == FENCE_OP or mdep.op2.get_op_kind() == FENCE_OP:
                    continue
                # TODO: need to track mapping dependencies from futures
                if not mdep.idx1 in mdep.op1.reqs:
                    continue
                if not mdep.idx2 in mdep.op2.reqs:
                    continue
                req1 = mdep.op1.get_requirement(mdep.idx1)
                req2 = mdep.op2.get_requirement(mdep.idx2)
                if req1.tid <> req2.tid:
                    continue
                print "    WARNING: Computed extra mapping dependence between "+str(mdep)+\
                      " in context of task "+str(self.name)
                warnings = warnings + 1

        if (self.mdeps > 0) or (errors > 0) or (warnings > 0):
            print "    Mapping Dependence Errors: "+str(errors)
            print "    Mapping Dependence Warnings: "+str(warnings)

    def print_dataflow(self, path, simplify_graphs):
        if len(self.ops) < 2:
            return 0
        filename = 'dataflow_'+self.name.replace(' ', '_')+'_'+str(self.uid)
        printer = GraphPrinter(path,filename)
        # First emit the nodes
        for op in self.ops:
            op.print_logical_node(printer) 
        # Simplify our graph if necessary
        if simplify_graphs:
            count = 0
            for src in reversed(self.ops):
                actual_out = src.logical_outgoing.copy()
                #print 'Handling node %d of %d with %d edges' % (count, len(self.ops), len(actual_out))
                diff = False
                for next_vert in src.logical_outgoing:
                    if not next_vert in actual_out:
                        continue
                    #start = time.clock()
                    reachable = set()
                    next_vert.get_reachable(reachable, True)
                    # See which edges we can remove
                    to_remove = list()
                    for other in actual_out:
                        if other == next_vert:
                            continue
                        if other in reachable:
                            to_remove.append(other)
                    del reachable
                    if len(to_remove) > 0:
                        diff = True
                        for rem in to_remove:
                            actual_out.remove(rem)
                            rem.logical_incoming.remove(src)
                    del to_remove
                    #stop = time.clock()
                    #print '  Elapsed %g us, %d' % ((stop-start)*1e6, len(actual_out))
                if diff:
                    src.logical_outgoing = actual_out
                for dst in actual_out:
                    printer.println(dst.node_name+' -> '+src.node_name+
                                    ' [style=solid,color=black,penwidth=2];')
                count+=1
        else:
            previous_pairs = set()
            for dep in self.mdeps:
                dep.print_dataflow_edge(printer, previous_pairs)
        printer.print_pdf_after_close(False)
        # We printed our datflow graph
        return 1

    def check_data_flow(self):
        print "Performing data flow check for task "+self.name+" (UID "+str(self.uid)+")"
        for dep in self.mdeps:
            if dep.dtype == TRUE_DEPENDENCE:
                # Handle cross product of instances
                # in case we have multiple instances for an op
                for inst1 in dep.op1.op_instances:
                    if inst1.get_op_kind() == FENCE_OP or \
                            inst1.get_op_kind() == FILL_OP or \
                            inst1.get_op_kind() == COPY_OP:
                        continue
                    if inst1.get_num_requirements() == 0:
                        continue
                    for inst2 in dep.op2.op_instances:
                        if inst2.get_op_kind() == FENCE_OP or \
                                inst2.get_op_kind() == FILL_OP or \
                                inst2.get_op_kind() == COPY_OP:
                            continue
                        if inst2.get_num_requirements() == 0:
                            continue
                        # Check to see if they are still aliased
                        req1 = inst1.get_requirement(dep.idx1)
                        req2 = inst2.get_requirement(dep.idx2)
                        def is_empty_inter_close_op(op):
                            return isinstance(op, Close) and \
                                    op.is_inter_close_op and \
                                    len(op.get_all_instances()) == 0
                        if is_empty_inter_close_op(inst1) or \
                                is_empty_inter_close_op(inst2):
                            # this is an InterCloseOp that uses no physical instance
                            continue
                        # If the second requirement is a reduction, there is no need to
                        # have a dataflow dependence since we can make a separate 
                        # reduction instance
                        if req2.priv == REDUCE or req2.priv == WRITE_ONLY:
                            continue
                        # A mapping dependence between different region trees
                        # comes from futures, which does not involve any data flow
                        # between physical instances.
                        if req1.tid <> req2.tid:
                            continue
                        index1 = self.state.get_index_node(True, req1.ispace)
                        index2 = self.state.get_index_node(True, req2.ispace)
                        if self.state.is_aliased(index1, index2):
                            def record_visit(node, traverser):
                                # Get the last instance of the traverser
                                last_inst = traverser.instance_stack[-1]
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
                                return not traverser.found
                            def traverse_task(node, traverser):
                                if record_visit(node, traverser):
                                    return False
                                traverser.found = traverser.found or \
                                        (node == traverser.target)
                                return not traverser.found
                            def traverse_map(node, traverser):
                                if record_visit(node, traverser):
                                    return False
                                traverser.found = traverser.found or \
                                        (node == traverser.target)
                                return not traverser.found
                            def traverse_close(node, traverser):
                                if record_visit(node, traverser):
                                    return False
                                traverser.found = traverser.found or \
                                        (node == traverser.target)
                                return not traverser.found
                            def traverse_copy(node, traverser):
                                if record_visit(node, traverser):
                                    return False
                                if traverser.found:
                                    return False
                                # Check to see if we have the matching
                                # manager on the stack
                                assert len(traverser.instance_stack) >= 1
                                last_inst = traverser.instance_stack[-1]
                                assert len(traverser.field_stack) >= 1
                                last_field = traverser.field_stack[-1]

                                dst_req = node.get_requirement(1)
                                if last_field not in dst_req.fields:
                                    return False
                                dst_inst = node.get_instance(1)
                                if last_inst <> dst_inst:
                                    return False

                                src_inst = node.get_instance(0)
                                src_req = node.get_requirement(0)
                                fidx = dst_req.fields.index(last_field)
                                src_field = src_req.fields[fidx]

                                traverser.found = node == traverser.target
                                if traverser.found:
                                    return False
                                else:
                                    traverser.instance_stack.append(src_inst)
                                    traverser.field_stack.append(src_field)
                                    return True
                            def traverse_acquire(node, traverser):
                                if record_visit(node, traverser):
                                    return False
                                traverser.found = traverser.found or \
                                        (node == traverser.target)
                                last_inst = traverser.instance_stack[-1]
                                if (last_inst != node.instances[0]):
                                    return False
                                return not traverser.found
                            def traverse_release(node, traverser):
                                if record_visit(node, traverser):
                                    return False
                                traverser.found = traverser.found or \
                                        (node == traverser.target)
                                return not traverser.found
                            def post_traverse_event(node, traverser):
                                pass
                            def post_traverse_task(node, traverser):
                                pass
                            def post_traverse_map(node, traverser):
                                pass
                            def post_traverse_close(node, traverser):
                                pass
                            def post_traverse_copy(node, traverser):
                                traverser.field_stack.pop()
                                traverser.instance_stack.pop()
                            def post_traverse_acquire(node, traverser):
                                pass
                            def post_traverse_release(node, traverser):
                                pass
                            # Do the traversal for each overlapping field
                            fields = set(req1.fields) & set(req2.fields)
                            for f in fields:
                                traverser = EventGraphTraverser(False, False, False, 
                                    self.state.get_next_traverser_gen(), traverse_event,
                                    traverse_task, traverse_map, traverse_close, 
                                    traverse_copy, traverse_acquire, traverse_release,
                                    post_traverse_event, post_traverse_task, post_traverse_map,
                                    post_traverse_close, post_traverse_copy, post_traverse_acquire,
                                    post_traverse_release)
                                traverser.found = False
                                traverser.target = inst1
                                traverser.instance_stack = list()
                                traverser.field_stack = list()
                                # Maintain a list of nodes that we have visited and
                                # context in which we have visted them.  We may need
                                # to visit some nodes in more than one context
                                traverser.visited = dict()
                                # TODO: support virtual mappings
                                traverser.instance_stack.append(
                                        inst2.get_instance(dep.idx2, f))
                                traverser.field_stack.append(f)
                                # Traverse and see if we find inst1
                                inst2.event_graph_traverse(traverser)
                                # Sanity check, only one thing on the stack
                                assert len(traverser.instance_stack) == 1
                                if not traverser.found:
                                    print "   ERROR: Unable to find data flow path "+\
                                          "between requirement "+str(dep.idx1)+" of "+\
                                          inst1.get_name()+" and requirement "+\
                                          str(dep.idx2)+" of "+inst2.get_name()+\
                                          " for field "+str(f)
                                    if self.state.verbose:
                                        print "      First Requirement:"
                                        req1.print_requirement()
                                        print "      Second Requirement:"
                                        req2.print_requirement()

class IndexTask(Operation):
    def __init__(self, state, uid, tid, ctx, name):
        assert ctx <> None
        Operation.__init__(self, state, uid, ctx, INDEX_OP, name,
                "task_node_", "mediumslateblue")
        self.tid = tid
        self.points = set()
        self.op_instances.remove(self)

    def get_name(self):
        name = ""
        if self.name <> None:
            name = self.name
        else:
            name = "index task"
        name += " (ID: " + str(self.uid) + ")"
        return name

    def add_point(self, uid, point):
        point_task = SingleTask(self.state, uid, self.tid, self.ctx, self.name)
        point_task.set_enclosing(self, point)
        self.points.add(point_task)
        self.op_instances.add(point_task)
        return point_task

    def get_child_req(self, idx, index):
        assert idx in self.reqs
        r = self.reqs[idx]
        result = Requirement(idx, True, index, r.fspace, r.tid, r.region_node,
                r.priv, r.coher, r.redop, self.state)
        for f in r.fields:
            result.add_field(f)
        return result

    def event_graph_traverse(self, traverser):
        # Indexspace tasks only appear in dataflow graphs
        assert False

    def add_to_component(self, component):
        # impossible
        assert False

class Mapping(Operation):
    def __init__(self, state, uid, ctx):
        assert ctx <> None
        Operation.__init__(self, state, uid, ctx, MAPPING_OP, "Mapping",
                "mapping_node_", "mediumseagreen")

    def add_requirement(self, idx, req):
        assert idx == 0
        Operation.add_requirement(self, idx, req)

    def get_requirement(self, idx):
        assert idx == 0
        return Operation.get_requirement(self, idx)

    def add_instance(self, idx, inst):
        assert idx == 0
        Operation.add_instance(self, idx, inst)

    def get_instance(self, idx, field = None):
        assert idx == 0
        return Operation.get_instance(self, idx, field)

    def event_graph_traverse(self, traverser):
        traverser.visit_mapping(self)

    def add_to_component(self, component):
        component.add_map(self)

class Deletion(Operation):
    def __init__(self, state, uid, ctx):
        assert ctx <> None
        Operation.__init__(self, state, uid, ctx, DELETION_OP, "Deletion",
                "deletion_node_", "dodgerblue3")

    def add_logical_incoming(self, op):
        # Should never happen
        assert False

    def find_dependences(self, op):
        # No need to do anything
        pass

    def find_individual_dependences(self, other_op, other_req):
        # TODO: implement this for deletion
        pass

class CopyOp(Operation):
    def __init__(self, state, uid, ctx):
        assert ctx is not None
        Operation.__init__(self, state, uid, ctx, COPY_OP, "Copy Across",
                "copy_across_", "darkgoldenrod3")
        self.prev_event_deps = set()

    def get_num_copies(self):
        return self.get_num_requirements() / 2

    def get_index_pair(self, cidx):
        return (cidx, cidx + self.get_num_copies())

    def print_base_node(self, printer, logical):
        size = 14
        title = self.get_name() + ' in '+self.ctx.get_name()
        color = 'darkgoldenrod3'

        num_copies = len(self.reqs) / 2
        for cidx in range(0, num_copies):
            (src_idx, dst_idx) = self.get_index_pair(cidx)
            dst_req = self.get_requirement(dst_idx)
            if dst_req.redop <> 0:
                title = 'Reduction ' + title
                color = 'tomato'
                break

        lines = [[{ "label" : title, "colspan" : 3 }]]

        for cidx in range(0, num_copies):
            (src_idx, dst_idx) = self.get_index_pair(cidx)
            src_req = self.get_requirement(src_idx)
            dst_req = self.get_requirement(dst_idx)

            src_region_name = src_req.region_node.get_name()
            dst_region_name = dst_req.region_node.get_name()

            if src_region_name <> None or dst_region_name <> None:
                lines.append(["Region Pair " + str(cidx),
                    src_region_name if src_region_name <> None else '',
                    dst_region_name if dst_region_name <> None else ''])
            else:
                lines.append([{"label" : "Region Pair " + str(cidx), "colspan" : 3}])

            if self.state.verbose:
                lines.append(["Requirement",
                    src_req.to_summary_string(),
                    dst_req.to_summary_string()])

                num_fields = len(src_req.fields)
                assert num_fields == len(dst_req.fields)

                first_field = True
                for fidx in range(0, num_fields):
                    line = []
                    src_field = src_req.fields[fidx]
                    dst_field = dst_req.fields[fidx]
                    if src_field == dst_field:
                        field_name = self.get_field_name(src_idx, src_field)
                        line.append({"label" : field_name, "colspan" : 2})
                    else:
                        line.append(self.get_field_name(src_idx, src_field))
                        line.append(self.get_field_name(dst_idx, dst_field))

                    if first_field:
                        line.insert(0, {"label" : "Fields",
                                        "rowspan" : num_fields})
                        first_field = False
                    lines.append(line)

            if dst_req.redop <> 0:
                lines.append(["Reduction Op",
                              {"colspan" : 2, "label" : str(dst_req.redop)}])

        label = '<table border="0" cellborder="1" cellspacing="0" cellpadding="3" bgcolor="%s">' % color + \
                "".join([wrap_with_trtd(line) for line in lines]) +\
                '</table>'

        printer.println(self.node_name+' [label=<'+label+'>,fontsize='+str(size)+\
                ',fontcolor=black,shape=record,penwidth=0];')

    def event_graph_traverse(self, traverser):
        # Copy operations only appear in dataflow graphs
        assert False

    def add_to_component(self, component):
        component.add_copy(self)

    def print_igraph_edges(self, printer):
        fields = self.reqs[0].field_mask_string()+\
                ' \-\> '+\
                self.reqs[1].field_mask_string()
        self.instances[1].print_incoming_edge(printer,
                self.instances[0].node_name,
                edge_type='dashed',
                edge_label='copy across '+hex(self.uid)+' (fields: '+fields+')')

class FillOp(Operation):
    def __init__(self, state, uid, ctx):
        Operation.__init__(self, state, uid, ctx, FILL_OP, "Fill",
                "fill_node_", "darkorange1")

    def add_requirement(self, idx, req):
        assert idx == 0
        Operation.add_requirement(self, idx, req)

    def get_requirement(self, idx):
        assert idx == 0
        return Operation.get_requirement(self, idx)

    def add_req_field(self, idx, fid):
        assert idx == 0
        return Operation.add_req_field(self, idx, fid)

    def print_igraph_edges(self, printer):
        self.instance.print_incoming_edge(printer, self.node_name)

class AcquireOp(Operation):
    def __init__(self, state, uid, ctx):
        assert ctx is not None
        Operation.__init__(self, state, uid, ctx, ACQUIRE_OP, "Acquire",
                "acquire_node_", "darkolivegreen")

    def event_graph_traverse(self, traverser):
        traverser.visit_acquire(self)

    def add_to_component(self, component):
        component.add_acquire(self)

class ReleaseOp(Operation):
    def __init__(self, state, uid, ctx):
        assert ctx is not None
        Operation.__init__(self, state, uid, ctx, RELEASE_OP, "Release",
                "release_node_", "darksalmon")

    def event_graph_traverse(self, traverser):
        traverser.visit_release(self)

    def add_to_component(self, component):
        component.add_release(self)

class DependentPartitionOp(Operation):
    def __init__(self, state, uid, ctx, part, kind):
        assert ctx is not None
        Operation.__init__(self, state, uid, ctx, DEPENDENT_PARTITION_OP,
                "Dependent Partition", "dep_partition_node_", "steelblue")
        self.partition_kind = kind
        self.index_partition_node = part

    def print_base_node(self, printer, logical):
        field_names = []
        fspace = self.state.field_space_nodes[self.reqs[0].fspace]
        for f in self.reqs[0].fields:
            field_name = fspace.get_field_name(f)
            if field_name == None:
                field_name = str(f)
            field_names.append(field_name)

        if self.partition_kind == 0:
            subtitle = 'By Field ' + ','.join(field_names)
        elif self.partition_kind == 1:
            subtitle = 'By Image of Field ' + ','.join(field_names)
        elif self.partition_kind == 2:
            subtitle = 'By Pre-Image of Field ' + ','.join(field_names)
        else:
            assert(false)

        src = self.index_partition_node.parent.get_name()
        dst = self.index_partition_node.get_name()

        printer.println(self.node_name+\
                ' [style=filled,label="Dependent Partition '+str(self.uid)+'\\nReq:'+\
                self.reqs[0].to_summary_string()+'\\n'+\
                ('' if self.reqs[0].region_node.name == None
                    else '('+self.reqs[0].region_node.name+')\\n')+\
                subtitle+'\\n'+src+' \=\=\> '+dst+\
                '",fillcolor=steelblue,fontsize=14,fontcolor=black,'+\
                'shape=record,penwidth=2];')

    def event_graph_traverse(self, traverser):
        pass
        #traverser.visit_partition_op(self)

class PendingPartitionOp(Operation):
    def __init__(self, state, uid, ctx):
        assert ctx is not None
        Operation.__init__(self, state, uid, ctx, PENDING_PARTITION_OP,
                "Pending Partition", "pending_partition_node_", "honeydew")
        self.index_partition_node = None
        self.kind = None

    def set_index_partition_node(self, part):
        self.index_partition_node = part

    def set_pending_partition_kind(self, kind):
        self.kind = kind

    def print_base_node(self, printer):
        kind_str = None
        if self.kind == 0:
            kind_str = "Equal Partition"
        elif self.kind == 1:
            kind_str = "Weighted Partition"
        elif self.kind == 2:
            kind_str = "Union Partition"
        elif self.kind == 3:
            kind_str = "Intersection Partition"
        elif self.kind == 4:
            kind_str = "Difference Partition"

        if self.index_partition_node <> None:
            dst = self.index_partition_node.get_name()
        else:
            dst = ""

        printer.println(self.node_name+\
                ' [style=filled,label="Pending Partition '+str(self.uid)+\
                ("" if kind_str == None else "\\nType: " + kind_str + "\\n")+\
                dst+'",fillcolor=honeydew,fontsize=14,fontcolor=black,'+\
                'shape=record,penwidth=2];')

class Close(Operation):
    def __init__(self, state, uid, ctx, is_inter_close_op):
        assert ctx is not None
        color = "orangered"
        if is_inter_close_op:
            color = "red"
        Operation.__init__(self, state, uid, ctx, CLOSE_OP, "Close",
                "close_node_", color)
        self.is_inter_close_op = is_inter_close_op
        self.creator = None
        self.creator_req_idx = None

    def add_requirement(self, idx, req):
        assert idx == 0
        Operation.add_requirement(self, idx, req)

    def get_requirement(self, idx):
        assert idx == 0
        return Operation.get_requirement(self, idx)

    def add_req_field(self, idx, fid):
        assert idx == 0
        return Operation.add_req_field(self, idx, fid)

    def event_graph_traverse(self, traverser):
        traverser.visit_close(self)

    def add_to_component(self, component):
        component.add_close(self)

    def add_instance(self, idx, inst):
        assert idx == 0
        Operation.add_instance(self, idx, inst)

    def get_instance(self, idx, field = None):
        assert idx == 0
        return Operation.get_instance(self, idx, field)

    def find_individual_dependences(self, other_op, other_req):
        req = self.get_requirement(0)
        dtype = self.state.compute_dependence(other_req, req)
        if not is_mapping_dependence(dtype):
            return
        if self.creator == None:
            self.ctx.add_adep(other_op, self, other_req.index, req.index, dtype)
            return

        if self.creator == other_op:
            other_op.find_individual_dependences(self, req)
        else:
            req = self.creator.get_requirement(self.creator_req_idx)
            dtype = self.state.compute_dependence(other_req, req)
            if dtype == TRUE_DEPENDENCE:
                self.ctx.add_adep(other_op, self, other_req.index, 0, dtype)
            elif dtype == ANTI_DEPENDENCE:
                inode1 = self.state.get_index_node(other_req.is_reg, other_req.ispace)
                inode2 = self.state.get_index_node(req.is_reg, req.ispace)
                if self.state.is_aliased(inode1, inode2) and \
                        not self.state.is_subtree(inode1, inode2):
                            self.ctx.add_adep(other_op, self, other_req.index, 0, dtype)

class Fence(Operation):
    def __init__(self, state, uid, ctx):
        Operation.__init__(self, state, uid, ctx, FENCE_OP,
                "Fence", "fence_node_", "darkorchid2")

    def find_dependences(self, op):
        # Everybody who comes after us has a dependence on us
        #self.ctx.add_adep(self, op, 0, 0, TRUE_DEPENDENCE)
        pass

    def find_individual_dependences(self, other_op, other_req):
        # Everybody who comes after us has a dependence on us
        #self.ctx.add_adep(other_op, self, other_req.index, 0, TRUE_DEPENDENCE)
        pass

    def print_logical_node(self, printer):
        printer.println(self.node_name+' [style=filled,label="'+self.get_name()+
                '",fillcolor=darkorchid2,fontsize=14,fontcolor=black,'+\
                'shape=record,penwidth=2];')

class RealmCopy(Operation):
    def __init__(self, state, uid, ctx):
        Operation.__init__(self, state, uid, ctx, COPY_EVENTS, "Copy",
                "copy_node_", "darkgoldenrod1")

    def get_name(self):
        name = "Copy (ID: "+str(self.uid)+")"
        if self.get_requirement(1).redop <> 0:
            name = 'Reduction ' + name
        return name

    def event_graph_traverse(self, traverser):
        traverser.visit_copy(self)

    def add_to_component(self, component):
        component.add_copy_event(self)

    def field_mask_string(self):
        return self.region.field_node.field_mask_string(self.fields)

    def print_physical_node(self, printer):
        title = self.get_name()
        if self.ctx <> None:
            title += ' triggered by ' + self.ctx.get_name()
        lines = [[{ "label" : title, "colspan" : 3 }]]

        src_req = self.get_requirement(0)
        dst_req = self.get_requirement(1)

        src_region_name = src_req.region_node.get_name()
        dst_region_name = dst_req.region_node.get_name()

        if src_region_name <> None or dst_region_name <> None:
            lines.append(["Region Pair",
                src_region_name if src_region_name <> None else '',
                dst_region_name if dst_region_name <> None else ''])

        if self.state.verbose:
            lines.append(["Requirement",
                src_req.to_summary_string(),
                dst_req.to_summary_string()])

        src_inst = self.get_instance(0)
        dst_inst = self.get_instance(1)
        lines.append(
                ["Memory",
                    src_inst.memory.dot_memory(),
                    dst_inst.memory.dot_memory()])
        lines.append(
                ["Instance",
                    src_inst.dot_instance(),
                    dst_inst.dot_instance()])

        if self.state.verbose:
            num_fields = len(src_req.fields)
            assert num_fields == len(dst_req.fields)

            first_field = True
            for fidx in range(0, num_fields):
                line = []
                src_field = src_req.fields[fidx]
                dst_field = dst_req.fields[fidx]
                if src_field == dst_field:
                    field_name = self.get_field_name(0, src_field)
                    line.append({"label" : field_name, "colspan" : 2})
                else:
                    line.append(self.get_field_name(0, src_field))
                    line.append(self.get_field_name(1, dst_field))

                if first_field:
                    line.insert(0, {"label" : "Fields",
                                    "rowspan" : num_fields})
                    first_field = False
                lines.append(line)

        color = 'darkgoldenrod1'
        size = 14
        if dst_req.redop <> 0:
            lines.append(["Reduction Op", {"colspan" : 2, "label" : str(dst_req.redop)}])
            color = 'tomato'
        else:
            max_blocking = max(src_inst.blocking, dst_inst.blocking)
            min_blocking = min(src_inst.blocking, dst_inst.blocking)
            if max_blocking > 1 and min_blocking == 1:
                color = 'red'
                size = 16

        label = '<table border="0" cellborder="1" cellspacing="0" cellpadding="3" bgcolor="%s">' % color + \
                "".join([wrap_with_trtd(line) for line in lines]) +\
                '</table>'

        printer.println(self.node_name+' [label=<'+label+'>,fontsize='+str(size)+\
                ',fontcolor=black,shape=record,penwidth=0];')

    def print_igraph_edges(self, printer):
        self.mask = self.get_requirement(0).field_mask_string()
        self.dst_inst.print_incoming_edge(printer,
                self.src_inst.node_name,
                edge_type='dashed',
                edge_label='copy '+hex(self.uid)+' (fields: '+self.mask+')')

class PhysicalInstance(object):
    def __init__(self, state, iid, ver, memory, region, blocking):
        self.state = state
        self.iid = iid
        self.memory = memory
        self.region = region
        self.blocking = blocking
        self.op_users = dict()
        memory.add_physical_instance(self)
        self.node_name = "physical_inst_"+str(iid)+"_"+str(ver)
        self.igraph_outgoing_deps = set()
        self.igraph_incoming_deps = set()
        self.fields = list()
        self.simultaneous = False

    def get_field_name(self, field):
        field_name = self.region.field_node.get_field_name(field)
        if field_name <> None:
            return field_name
        else:
            return str(field)

    def mark_simultaneous(self):
        self.simultaneous = True

    def is_simultaneous(self):
        return self.simultaneous

    def add_op_user(self, op, idx):
        req = op.get_requirement(idx)
        if req == None:
            return False
        for field in req.fields:
            if field not in self.op_users:
                self.op_users[field] = dict()
            if op not in self.op_users[field]:
                self.op_users[field][op] = list()
            self.op_users[field][op].append(req)
        if req.is_simult():
            self.mark_simultaneous()
        return True

    def add_field(self, fid):
        self.fields.append(fid)

    def dot_instance(self):
        return hex(self.iid)+' ('+str(self.blocking)+')'

    def print_igraph_node(self, printer):
        label = generate_html_instance_label(self.region.get_name(),
                self, "dodgerblue4", self.state.verbose)
        printer.println(self.node_name+' [style=filled,label=<'+label+\
                '>,fillcolor=dodgerblue4,fontsize=11,fontcolor=white,'+\
                'shape=Mrecord,penwidth=0];')

    def print_incoming_edge(self, printer, prev, edge_type='solid', edge_label='', dir='forward'):
        if prev not in self.igraph_incoming_deps:
            printer.println(prev+' -> '+self.node_name+\
                ' [label="'+edge_label+'",style='+edge_type+',color=black,penwidth=2,dir="'+dir+'"];')
            self.igraph_incoming_deps.add(prev)

    def print_outgoing_edge(self, printer, succ, edge_type='solid', edge_label='', dir='forward'):
        if succ not in self.igraph_outgoing_deps:
            printer.println(self.node_name+' -> '+succ+\
                ' [label="'+edge_label+'",style='+edge_type+',color=black,penwidth=2,dir="'+dir+'"];')
            self.igraph_outgoing_deps.add(succ)


class ReductionInstance(object):
    def __init__(self, state, iid, ver, memory, region, fold, indirect):
        self.state = state
        self.iid = iid
        self.memory = memory
        self.region = region
        self.fold = fold
        self.indirect = indirect
        self.op_users = dict()
        memory.add_reduction_instance(self)
        self.node_name = "reduction_inst_"+str(iid)+"_"+str(ver)
        self.igraph_outgoing_deps = set()
        self.igraph_incoming_deps = set()
        self.fields = list()
        self.simultaneous = False

    def get_field_name(self, field):
        field_name = self.region.field_node.get_field_name(field)
        if field_name <> None:
            return field_name
        else:
            return str(field)

    def mark_simultaneous(self):
        self.simultaneous = True

    def is_simultaneous(self):
        return self.simultaneous

    def add_op_user(self, op, idx):
        req = op.get_requirement(idx)
        if req == None:
            return False
        for field in req.fields:
            if field not in self.op_users:
                self.op_users[field] = dict()
            if op not in self.op_users[field]:
                self.op_users[field][op] = list()
            self.op_users[field][op].append(req)
        if req.is_simult():
            self.mark_simultaneous()
        return True

    def add_field(self, fid):
        self.fields.append(fid)

    def dot_instance(self):
        return hex(self.iid)+" (reduction)"

    def print_igraph_node(self, printer):
        label = generate_html_instance_label(self.region.get_name(),
                self, "deeppink3", self.state.verbose)
        printer.println(self.node_name+' [style=filled,label=<'+label+\
                '>,fillcolor=deeppink3,fontsize=11,fontcolor=white,'+\
                'shape=Mrecord,penwidth=0];')

    def print_incoming_edge(self, printer, prev, edge_type='solid', edge_label='', dir='forward'):
        if prev not in self.igraph_incoming_deps:
            printer.println(prev+' -> '+self.node_name+\
                ' [label="'+edge_label+'",style='+edge_type+',color=black,penwidth=2,dir="'+dir+'"];')
            self.igraph_incoming_deps.add(prev)

    def print_outgoing_edge(self, printer, succ, edge_type='solid', edge_label='', dir='forward'):
        if succ not in self.igraph_outgoing_deps:
            printer.println(self.node_name+' -> '+succ+\
                ' [label="'+edge_label+'",style='+edge_type+',color=black,penwidth=2,dir="'+dir+'"];')
            self.igraph_outgoing_deps.add(succ)

class EventHandle(object):
    def __init__(self, uid, gen):
        self.uid = uid
        self.gen = gen

    def __hash__(self):
        return hash((self.uid, self.gen))

    def __eq__(self, other):
        return (self.uid,self.gen) == (other.uid,other.gen)

    def __repr__(self):
        return "ev(" + hex(self.uid) + "," + str(self.gen) + ")"

    def exists(self):
        return (self.uid <> 0)

class PhaseBarrier(object):
    def __init__(self, uid, gen):
        self.uid = uid
        self.gen = gen
        self.node_name = 'phase_barrier_'+str(self.uid)+'_'+str(gen)
        self.prev_event_deps = set()

    def __hash__(self):
        return hash((self.uid, self.gen))

    def __eq__(self, other):
        return (self.uid,self.gen) == (other.uid,other.gen)

    def __repr__(self):
        return "pb(" + hex(self.uid) + "," + str(self.gen) + ")"

    def exists(self):
        return (self.uid <> 0)

    def print_physical_node(self, printer):
        printer.println(self.node_name+\
                ' [style=filled,label="PB '+hex(self.uid)+'\\n '+\
                str(self.gen)+"->"+str(self.gen+1)+\
                '",fillcolor=deeppink3,fontsize=12,fontcolor=white,'+\
                'shape=circle,penwidth=0,margin=0];')

    def print_prev_event_dependences(self, printer, later_name):
        if later_name not in self.prev_event_deps:
            printer.println(self.node_name+' -> '+later_name+
                ' [style=solid,color=black,penwidth=2];')
            self.prev_event_deps.add(later_name)

class Requirement(object):
    def __init__(self, index, is_reg, ispace, fspace, tid, node, priv, coher, redop, state):
        self.index = index
        self.is_reg = is_reg
        self.ispace = ispace
        self.fspace = fspace
        self.tid = tid
        self.region_node = node
        self.priv = priv
        self.coher = coher
        self.redop = redop
        self.fields = list()
        self.state = state

    def find_field_index(self, field):
        return self.fields.index(field)

    def print_requirement(self):
        if self.is_reg:
            print "        Logical Region Requirement ("+hex(self.ispace)+","+\
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
        self.fields.append(fid)

    def is_no_access(self):
        return self.priv == NO_ACCESS

    def is_read_only(self):
        return self.priv == READ_ONLY

    def has_write(self):
        return (self.priv == READ_WRITE) or (self.priv == REDUCE) or (self.priv == WRITE_ONLY)

    def is_write(self):
        return (self.priv == READ_WRITE) or (self.priv == WRITE_ONLY)

    def is_read_write(self):
        return self.priv == READ_WRITE

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
            print "Region Requirement for ("+hex(self.ispace)+","+\
                  str(self.fspace)+","+str(self.tid)+")"
        else:
            print "Partition Requirement for ("+str(self.ispace)+","+\
                  str(self.fspace)+","+str(self.tid)+")"
        print "    Privilege: "+self.get_privilege()
        print "    Coherence: "+self.get_coherence()
        print "    Fields: "+" ".join(self.get_field_names())

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

    def get_field_names(self):
        return self.state.field_space_nodes[self.fspace].get_field_names(self.fields)

    def get_privilege_and_coherence(self):
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
        return result

    def dot_requirement(self):
        result = self.get_privilege_and_coherence()
        result = result + '\ Fields:' + self.field_mask_string()
        return result

    def to_summary_string(self):
        return 'index:%s,field:%s,tree:%s' % \
                (hex(self.ispace), hex(self.fspace), str(self.tid))

    def get_field_name(self, field):
        name = self.state.field_space_nodes[self.fspace].get_field_name(field)
        if name <> None:
            return name
        else:
            return str(field)

    def field_mask_string(self):
        return self.state.field_space_nodes[self.fspace].field_mask_string(self.fields)


class MappingDependence(object):
    def __init__(self, ctx, op1, op2, idx1, idx2, dtype):
        self.ctx = ctx
        self.op1 = op1
        self.op2 = op2
        self.idx1 = idx1
        self.idx2 = idx2
        self.dtype = dtype
        self.is_reversed = False

    def __eq__(self,other):
        return (self.ctx == other.ctx) and \
                (self.op1 is other.op1) and \
                (self.op2 is other.op2) and \
                (self.idx1 == other.idx1) and \
                (self.idx2 == other.idx2) and \
                (self.dtype == other.dtype)

    def __repr__(self):
        return "index %d of %s and index %d of %s (type: %s)" % \
                (self.idx1, self.op1.get_name(),
                 self.idx2, self.op2.get_name(),
                 DEPENDENCE_TYPES[self.dtype])


    def print_dataflow_edge(self, printer, previous_pairs):
        pair = (self.op1,self.op2)
        if pair not in previous_pairs:
            printer.println(self.op1.node_name+' -> '+self.op2.node_name+
                            ' [style=solid,color=black,penwidth=2];')
            previous_pairs.add(pair)

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
        self.prev_event_deps = set()

    def get_name(self):
        return repr(self.handle)

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
        if self.is_phase_barrier() and \
                len(self.physical_outgoing) == 0 and \
                len(self.physical_incoming) == 1 and \
                list(self.physical_incoming)[0].is_phase_barrier():
            return
        component.add_event(self)
        if self.is_phase_barrier():
            component.add_phase_barrier(self.handle)
        for n in self.physical_incoming:
            n.physical_traverse(component)
        for n in self.physical_outgoing:
            n.physical_traverse(component)

    def physical_unmark(self):
        self.physical_marked = False

    def is_phase_barrier(self):
        return isinstance(self.handle, PhaseBarrier)

    def print_event_dependences(self, printer):
        if isinstance(self.handle, PhaseBarrier):
            for n in self.physical_incoming:
                n.print_prev_event_dependences(printer, self.handle.node_name)

    def print_prev_event_dependences(self, printer, name):
        if name not in self.prev_event_deps:
            #self.prev_event_deps.add(name)
            if self.is_phase_barrier():
                self.handle.print_prev_event_dependences(printer, name)
                for n in self.physical_incoming:
                    n.print_prev_event_dependences(printer, self.handle.node_name)
            else:
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

    def mk_string(self, start, delim, end):
        result = start
        first = True
        for val in self.values:
            if not first:
                result = result + delim
            result = result + str(val)
            first = False
        result = result + end
        return result

    def to_string(self):
        return self.mk_string("(", ",", ")")

    def to_dim_string(self):
        return self.mk_string("", "][", "")

    def to_simple_string(self):
        return self.mk_string("", "_", "")

    def __hash__(self):
        return hash(self.to_simple_string())

class EventGraphTraverser(object):
    def __init__(self, forwards, implicit, use_gen, generation, 
                 event_fn = None, task_fn = None, 
                 map_fn = None, close_fn = None, copy_fn = None,
                 acquire_fn = None, release_fn = None,
                 post_event_fn = None, post_task_fn = None,
                 post_map_fn = None, post_close_fn = None,
                 post_copy_fn = None, post_acquire_fn = None,
                 post_release_fn = None):
        self.forwards = forwards
        self.implicit = implicit
        self.use_gen = use_gen
        self.generation = generation
        self.event_fn = event_fn
        self.task_fn = task_fn
        self.map_fn = map_fn
        self.close_fn = close_fn
        self.copy_fn = copy_fn
        self.acquire_fn = acquire_fn
        self.release_fn = release_fn
        self.post_event_fn = post_event_fn
        self.post_task_fn = post_task_fn
        self.post_map_fn = post_map_fn
        self.post_close_fn = post_close_fn
        self.post_copy_fn = post_copy_fn
        self.post_acquire_fn = post_acquire_fn
        self.post_release_fn = post_release_fn

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
            if node.term_event <> None:
                node.term_event.event_graph_traverse(self)
        else:
            if node.start_event <> None:
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

    def visit_acquire(self, node):
        if self.use_gen:
            if node.generation == self.generation:
                return
            else:
                node.generation = self.generation
        do_next = True
        if self.acquire_fn <> None:
            do_next = self.acquire_fn(node, self)
        if not do_next:
            return
        if self.forwards:
            node.term_event.event_graph_traverse(self)
        else:
            node.start_event.event_graph_traverse(self)
        if self.post_acquire_fn <> None:
            self.post_acquire_fn(node, self)

    def visit_release(self, node):
        if self.use_gen:
            if node.generation == self.generation:
                return
            else:
                node.generation = self.generation
        do_next = True
        if self.release_fn <> None:
            do_next = self.release_fn(node, self)
        if not do_next:
            return
        if self.forwards:
            node.term_event.event_graph_traverse(self)
        else:
            node.start_event.event_graph_traverse(self)
        if self.post_release_fn <> None:
            self.post_release_fn(node, self)

class ConnectedComponent(object):
    def __init__(self, state):
        self.state = state
        self.events = set()
        self.tasks = dict()
        self.parent_tasks = dict()
        self.maps = dict()
        self.closes = set()
        self.acquires = dict()
        self.releases = dict()
        self.partition_ops = dict()
        self.copy_events = dict()
        self.phase_barriers = set()
        self.processors = state.processors
        self.utilities = state.utilities
        self.memories = state.memories
        self.instances = set()

    def add_event(self, event):
        assert event not in self.events
        self.events.add(event)

    def add_task(self, task):
        assert task.uid not in self.tasks
        self.tasks[task.uid] = task
        if task.ctx != None:
            self.parent_tasks[task.uid] = task.ctx

    def add_map(self, mapp):
        assert mapp not in self.maps
        self.maps[mapp.uid] = mapp
        if mapp.ctx != None:
            self.parent_tasks[mapp.uid] = mapp.ctx

    def add_close(self, close):
        assert close not in self.closes
        self.closes.add(close)

    def add_copy_event(self, copy):
        assert copy not in self.copy_events
        self.copy_events[copy.uid] = copy

    def add_acquire(self, acquire):
        assert acquire not in self.acquires
        self.acquires[acquire.uid] = acquire
        if acquire.ctx != None:
            self.parent_tasks[acquire.uid] = acquire.ctx

    def add_release(self, release):
        assert release not in self.releases
        self.releases[release.uid] = release
        if release.ctx != None:
            self.parent_tasks[release.uid] = release.ctx

    def add_partition_op(self, partition_op):
        assert partition_op not in self.partition_ops
        self.partition_ops[partition_op.uid] = partition_op
        if partition_op.ctx != None:
            self.parent_tasks[partition_op.uid] = partition_op.ctx

    def add_phase_barrier(self, phase_barrier):
        assert phase_barrier not in self.phase_barriers
        self.phase_barriers.add(phase_barrier)

    def num_ops(self):
        return len(self.tasks)+len(self.maps)+len(self.closes)+\
                len(self.copy_events)+len(self.acquires)+len(self.releases)+\
                len(self.partition_ops)

    def empty(self):
        total = self.num_ops()
        return (total < 2)

    def unmark_all(self):
        for t in self.tasks.itervalues():
            t.physical_unmark()
        for m in self.maps.itervalues():
            m.physical_unmark()
        for c in self.copy_events.itervalues():
            c.physical_unmark()
        for c in self.closes:
            c.physical_unmark()
        for e in self.events:
            e.physical_unmark()
        for a in self.acquires.itervalues():
            a.physical_unmark()
        for r in self.releases.itervalues():
            r.physical_unmark()
        for p in self.partition_ops.itervalues():
            p.physical_unmark()

    def generate_graph(self, idx, path, simplify):
        name = 'event_graph_'+str(idx)
        printer = GraphPrinter(path,name)
        # Print the nodes
        for t in self.tasks.itervalues():
            t.print_physical_node(printer)
        for m in self.maps.itervalues():
            m.print_physical_node(printer)
        for c in self.closes:
            c.print_physical_node(printer)
        for c in self.copy_events.itervalues():
            c.print_physical_node(printer)
        for a in self.acquires.itervalues():
            a.print_physical_node(printer)
        for r in self.releases.itervalues():
            r.print_physical_node(printer)
        for p in self.partition_ops.itervalues():
            p.print_physical_node(printer)
        for p in self.phase_barriers:
            p.print_physical_node(printer)
        # Now print the dependences
        for t in self.tasks.itervalues():
            t.print_event_dependences(printer)
        for m in self.maps.itervalues():
            m.print_event_dependences(printer)
        for c in self.closes:
            c.print_event_dependences(printer)
        for c in self.copy_events.itervalues():
            c.print_event_dependences(printer) 
        for a in self.acquires.itervalues():
            a.print_event_dependences(printer)
        for r in self.releases.itervalues():
            r.print_event_dependences(printer)
        for p in self.partition_ops.itervalues():
            p.print_event_dependences(printer)
        for e in self.events:
            e.print_event_dependences(printer)
        # Print links to contexts when it's in verbose mode
        if self.state.verbose:
            for uid, ctx in self.parent_tasks.iteritems():
                if uid in self.tasks:
                    op = self.tasks[uid]
                elif uid in self.maps:
                    op = self.maps[uid]
                elif uid in self.acquires:
                    op = self.acquires[uid]
                elif uid in self.releases:
                    op = self.releases[uid]
                elif uid in self.partition_ops:
                    op = self.partition_ops[uid]
                else:
                    continue
                printer.println(ctx.node_name+' -> '+op.node_name+\
                        ' [style=dotted,color=black,penwidth=2];')
                if ctx.uid not in self.tasks:
                    ctx.print_physical_node(printer)

        printer.print_pdf_after_close(simplify)

    def collect_physical_instances(self):
        def collect(op):
            self.instances |= op.get_all_instances()
        for t in self.tasks.itervalues():
            collect(t)
        for m in self.maps.itervalues():
            collect(m)
        for c in self.copy_events.itervalues():
            collect(c)
        for a in self.acquires.itervalues():
            collect(a)
        for r in self.releases.itervalues():
            collect(r)
        self.instances = sorted(self.instances,
                key=lambda op: op.iid, reverse=True)

    def generate_igraph(self, idx, path):
        def get_field_string(pair, fields):
            field_strings = []
            (src_inst, dst_inst) = pair
            for (src_field, dst_field) in fields:
                if src_field == dst_field:
                    field_strings.append(src_inst.get_field_name(src_field))
                else:
                    field_strings.append(src_inst.get_field_name(src_field) + \
                            " \-\> " + \
                            dst_inst.get_field_name(dst_field))
            return ",".join(field_strings)

        def print_igraph_edges_for_multiple_copies(printer, pair, copies):
            (src_inst, dst_inst) = pair
            copy_strings = [
                    'copy '+str(copy.uid)+' (fields: '+\
                            get_field_string(pair, fields)+')'
                    for (copy,fields) in copies]
            dst_inst.print_incoming_edge(printer,
                    src_inst.node_name,
                    edge_type='dashed',
                    edge_label=", ".join(copy_strings))

        self.collect_physical_instances()
        name = 'instance_graph_'+str(idx)
        printer = GraphPrinter(path,name)

        # Print the nodes
        for inst in self.instances:
            inst.print_igraph_node(printer)
        for t in self.tasks.itervalues():
            t.print_igraph_node(printer)
        for m in self.maps.itervalues():
            m.print_igraph_node(printer)

        # Print the edges
        for t in self.tasks.itervalues():
            t.print_igraph_edges(printer)
        for m in self.maps.itervalues():
            m.print_igraph_edges(printer)
        copy_groups = {}
        for c in self.copy_events.itervalues():
            inst_pair = (c.get_instance(0), c.get_instance(1))
            fields = list(zip(c.get_requirement(0).fields,
                              c.get_requirement(1).fields))
            if not inst_pair in copy_groups:
                copy_groups[inst_pair] = []
            copy_groups[inst_pair].append((c, fields))
        for inst_pair in copy_groups:
            print_igraph_edges_for_multiple_copies(printer,
                    inst_pair,
                    copy_groups[inst_pair])

        printer.print_pdf_after_close(True)

class GraphPrinter(object):
    def __init__(self,path,name,dir='LR'):
        self.name = name
        self.filename = path+name+'.dot'
        self.out = open(self.filename,'w')
        self.depth = 0
        self.println('digraph "'+name+'"')
        self.println('{')
        self.down()
        #self.println('aspect = ".00001,100";')
        #self.println('ratio = 1;')
        #self.println('size = "10,10";')
        self.println('compound = true;')
        self.println('rankdir="'+dir+'";')
        self.println('size = "36,36";')
        self.cluster_id = 0

    def close(self):
        self.up()
        self.println('}')
        self.out.close()
        return self.filename

    def print_pdf_after_close(self, simplify):
        dot_file = self.close()
        pdf_file = self.name+".pdf"
        try:
            if simplify:
                tred = subprocess.Popen(['tred', dot_file], stdout=subprocess.PIPE)
                dot = subprocess.Popen(['dot', '-Tpdf', '-o', pdf_file], stdin=tred.stdout)
                if dot.wait() != 0:
                    raise Exception('DOT failed')
            else:
                subprocess.check_call(['dot', '-Tpdf', '-o', pdf_file, dot_file])
        except:
            print "WARNING: DOT failure, image for graph "+str(self.name)+" not generated"
            subprocess.call(['rm', '-f', 'core', pdf_file])

    def up(self):
        assert self.depth > 0
        self.depth = self.depth-1

    def down(self):
        self.depth = self.depth+1

    def start_new_cluster(self):
        self.println('subgraph cluster_' + str(self.cluster_id))
        self.cluster_id += 1
        self.println('{')
        self.down()

    def end_this_cluster(self):
        self.up()
        self.println('}')

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
        self.copies = dict()
        self.slice_index = dict()
        self.slice_slice = dict()
        self.point_slice = dict()
        self.instances = dict()
        self.events = dict()
        self.phase_barriers = dict()
        self.verbose = verbose
        self.top_level_uid = None
        self.traverser_gen = 1
        self.copy_uid = 0
        self.next_logical_mark = long(1)
        self.independent_ispaces = dict()
        self.independent_ipart = dict()

    def get_next_traverser_gen(self):
        result = self.traverser_gen
        self.traverser_gen = result + 1
        return result

    def add_utility(self, pid):
        assert pid not in self.utilities
        self.utilities[pid] = UtilityProcessor(self, pid)
        return True

    def add_processor(self, pid, kind):
        assert pid not in self.processors
        self.processors[pid] = Processor(self, pid, kind)
        return True

    def add_memory(self, mid, capacity, kind):
        assert mid not in self.memories
        self.memories[mid] = Memory(self, mid, capacity, kind)
        return True

    def set_proc_mem(self, pid, mid, bandwidth, latency):
        if pid not in self.processors and pid not in self.utilities:
            return False
        if mid not in self.memories:
            return False
        if pid in self.processors:
            processor = self.processors[pid]
        else: # pid in self.utilities
            processor = self.utilities[pid]
        processor.add_memory(self.memories[mid], bandwidth, latency)
        self.memories[mid].add_processor(processor, bandwidth, latency)
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
        return True

    def add_index_space_name(self, uid, name):
        assert uid in self.index_space_nodes
        self.index_space_nodes[uid].set_name(name)
        return True

    def add_index_partition(self, pid, uid, disjoint, dim, v1, v2, v3):
        assert uid not in self.index_part_nodes
        if pid not in self.index_space_nodes:
            return False
        point = Point(0, dim)
        point.add_value(v1)
        if dim > 1:
            point.add_value(v2)
            if dim > 2:
                point.add_value(v3)
        self.index_part_nodes[uid] = IndexPartNode(self, uid, disjoint, point,
                                                    self.index_space_nodes[pid])
        return True

    def add_index_partition_name(self, uid, name):
        assert uid in self.index_part_nodes
        self.index_part_nodes[uid].set_name(name)
        return True

    def add_index_subspace(self, pid, uid, dim, v1, v2, v3):
        assert uid not in self.index_space_nodes
        if pid not in self.index_part_nodes:
            return False
        point = Point(0, dim)
        point.add_value(v1)
        if dim > 1:
            point.add_value(v2)
            if dim > 2:
                point.add_value(v3)
        self.index_space_nodes[uid] = IndexSpaceNode(self, uid, point,
                                              self.index_part_nodes[pid])
        return True

    def add_field_space(self, uid):
        assert uid not in self.field_space_nodes
        self.field_space_nodes[uid] = FieldSpaceNode(self, uid)
        return True

    def add_field_space_name(self, uid, name):
        assert uid in self.field_space_nodes
        self.field_space_nodes[uid].set_name(name)
        return True

    def add_field(self, uid, fid):
        if uid not in self.field_space_nodes:
            return False
        self.field_space_nodes[uid].add_field(fid)
        return True

    def add_field_name(self, uid, fid, name):
        if uid not in self.field_space_nodes:
            return False
        self.field_space_nodes[uid].set_field_name(fid, name)
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

    def add_region_name(self, iid, fid, tid, name):
        assert tid in self.region_trees
        if iid not in self.index_space_nodes:
            return False
        if fid not in self.field_space_nodes:
            return False
        self.region_trees[tid].set_name(name,
                self.index_space_nodes[iid],
                self.field_space_nodes[fid])
        return True

    def add_partition_name(self, iid, fid, tid, name):
        assert tid in self.region_trees
        if iid not in self.index_part_nodes:
            return False
        if fid not in self.field_space_nodes:
            return False
        self.region_trees[tid].set_name(name,
                self.index_part_nodes[iid],
                self.field_space_nodes[fid])
        return True

    def add_top_task(self, tid, uid, name):
        assert uid not in self.ops
        self.ops[uid] = SingleTask(self, uid, tid, None, name)
        self.top_level_uid = uid
        return True
      
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

    def add_close(self, ctx, uid, is_inter_close_op):
        assert uid not in self.ops
        if ctx not in self.ops:
            return False
        self.ops[uid] = Close(self, uid, self.ops[ctx], is_inter_close_op)
        self.ops[ctx].add_operation(self.ops[uid])
        return True

    def set_close_creator(self, uid, cuid, idx):
        if uid not in self.ops:
            return False
        if cuid not in self.ops:
            return False
        close_op = self.ops[uid]
        assert close_op.is_inter_close_op
        creator_op = self.ops[cuid]
        parent_op = close_op.ctx
        close_op.creator = creator_op
        close_op.creator_req_idx = idx
        parent_op.ops.remove(close_op)
        idx = parent_op.ops.index(creator_op)
        parent_op.ops.insert(idx, close_op)
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

    def add_fill_op(self, ctx, uid):
        assert uid not in self.ops
        if ctx not in self.ops:
            return False
        self.ops[uid] = FillOp(self, uid, self.ops[ctx])
        self.ops[ctx].add_operation(self.ops[uid])
        return True

    def add_acquire_op(self, ctx, uid):
        assert uid not in self.ops
        if ctx not in self.ops:
            return False
        self.ops[uid] = AcquireOp(self, uid, self.ops[ctx])
        self.ops[ctx].add_operation(self.ops[uid])
        return True

    def add_release_op(self, ctx, uid):
        assert uid not in self.ops
        if ctx not in self.ops:
            return False
        self.ops[uid] = ReleaseOp(self, uid, self.ops[ctx])
        self.ops[ctx].add_operation(self.ops[uid])
        return True

    def add_deletion(self, ctx, uid):
        assert uid not in self.ops
        if ctx not in self.ops:
            return False
        self.ops[uid] = Deletion(self, uid, self.ops[ctx])
        self.ops[ctx].add_operation(self.ops[uid])
        return True

    def add_dependent_partition_op(self, ctx, uid, pid, kind):
        assert uid not in self.ops
        if ctx not in self.ops:
            return False
        if pid not in self.index_part_nodes:
            return False
        self.ops[uid] = DependentPartitionOp(self, uid, self.ops[ctx],
                                             self.index_part_nodes[pid], kind)
        self.ops[ctx].add_operation(self.ops[uid])
        return True

    def add_pending_partition_op(self, ctx, uid):
        assert uid not in self.ops
        if ctx not in self.ops:
            return False
        self.ops[uid] = PendingPartitionOp(self, uid, self.ops[ctx])
        self.ops[ctx].add_operation(self.ops[uid])
        return True

    def set_pending_partition_target(self, uid, pid, kind):
        if uid not in self.ops:
            return False
        if pid not in self.index_part_nodes:
            return False
        self.ops[uid].set_index_partition_node(self.index_part_nodes[pid])
        self.ops[uid].set_pending_partition_kind(kind)
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

    def add_phase_barrier(self, uid):
        pb = PhaseBarrier(uid, 0)
        self.phase_barriers[uid] = pb
        assert pb not in self.events
        self.get_event(pb)
        return True

    def add_requirement(self, uid, index, is_reg, ispace, fspace, tid, priv, coher, redop):
        if uid not in self.ops:
            return False
        if tid not in self.region_trees:
            return False
        if is_reg:
            node = self.region_trees[tid].find_node(
                    self.index_space_nodes[ispace],
                    self.field_space_nodes[fspace])
        else:
            node = self.region_trees[tid].find_node(
                    self.index_part_nodes[ispace],
                    self.field_space_nodes[fspace])

        self.ops[uid].add_requirement(index, Requirement(index, is_reg, ispace, 
                                                  fspace, tid, node, priv, coher, redop, self))
        return True

    def add_req_field(self, uid, idx, fid):
        if uid not in self.ops:
            return False
        if not self.ops[uid].add_req_field(idx, fid):
            return False
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

    def add_independent_index_spaces(self, pid, uid1, uid2):
        assert pid in self.index_part_nodes
        self.index_part_nodes[pid].mark_independent(uid1, uid2)
        return True

    def add_independent_index_partitions(self, pid, uid1, uid2):
        assert pid in self.index_space_nodes
        self.index_space_nodes[pid].mark_independent(uid1, uid2)
        return True

    def add_instance_requirement(self, uid, idx, index):
        if uid not in self.ops:
            return False
        if not self.ops[uid].add_instance_requirement(idx, index):
            return False
        return True

    def add_op_events(self, uid, startid, startgen, termid, termgen):
        if uid not in self.ops:
            return False
        e1 = self.get_event_from_id(startid,startgen)
        e2 = self.get_event_from_id(termid,termgen)
        e1.add_physical_outgoing(self.ops[uid])
        e2.add_physical_incoming(self.ops[uid])
        self.ops[uid].add_events(e1, e2)
        return True

    def add_copy_event(self, startid, startgen, termid, termgen):
        e1 = self.get_event_from_id(startid,startgen)
        e2 = self.get_event_from_id(termid,termgen)
        self.copy_uid = self.copy_uid + 1
        copy = RealmCopy(self, self.copy_uid, None)
        self.copies[(e1, e2)] = copy
        copy.add_events(e1, e2)
        e1.add_physical_outgoing(copy)
        e2.add_physical_incoming(copy)
        return True

    def add_copy_req(self, startid, startgen, termid, termgen,
            index, is_reg, ispace, fspace, tid, priv, coher, redop):
        e1 = self.get_event_from_id(startid,startgen)
        e2 = self.get_event_from_id(termid,termgen)
        if (e1, e2) not in self.copies:
            return False
        copy = self.copies[(e1, e2)]

        if tid not in self.region_trees:
            return False
        if is_reg:
            node = self.region_trees[tid].find_node(
                    self.index_space_nodes[ispace],
                    self.field_space_nodes[fspace])
        else:
            node = self.region_trees[tid].find_node(
                    self.index_part_nodes[ispace],
                    self.field_space_nodes[fspace])

        req = Requirement(index, is_reg, ispace, fspace, tid, node, priv,
                coher, redop, self)
        copy.add_requirement(index, req)
        return True

    def add_copy_field(self, startid, startgen, termid, termgen, index, fid):
        e1 = self.get_event_from_id(startid,startgen)
        e2 = self.get_event_from_id(termid,termgen)
        if (e1, e2) not in self.copies:
            return False
        copy = self.copies[(e1, e2)]
        return copy.add_req_field(index, fid)
        return True

    def add_copy_instance(self, startid, startgen, termid, termgen, index, iid):
        e1 = self.get_event_from_id(startid,startgen)
        e2 = self.get_event_from_id(termid,termgen)
        if (e1, e2) not in self.copies:
            return False
        copy = self.copies[(e1, e2)]
        if iid not in self.instances:
            return False
        if not self.instances[iid][-1].add_op_user(copy, index):
            return False
        copy.add_instance(index, self.instances[iid][-1])
        return True

    def add_copy_context(self, startid, startgen, termid, termgen, ctx):
        e1 = self.get_event_from_id(startid,startgen)
        e2 = self.get_event_from_id(termid,termgen)
        if (e1, e2) not in self.copies:
            return False
        copy = self.copies[(e1, e2)]
        if ctx not in self.ops:
            return False
        copy.ctx = self.ops[ctx]
        return True

    def add_physical_instance(self, iid, mem, index, field, tree, blocking):
        #if iid in self.instances:
        #    return True
        if mem not in self.memories:
            return False
        if index not in self.index_space_nodes:
            return False
        if field not in self.field_space_nodes:
            return False
        if tree not in self.region_trees:
            return False
        region = self.get_index_node(True, index).get_instance(tree)
        ver = 0 if iid not in self.instances else len(self.instances[iid])
        inst = PhysicalInstance(self, iid, ver, self.memories[mem], region, blocking)
        if iid in self.instances:
            self.instances[iid].append(inst)
        else:
            self.instances[iid] = [inst]
        return True

    def add_reduction_instance(self, iid, mem, index, field, tree, fold, indirect):
        if mem not in self.memories:
            return False
        if index not in self.index_space_nodes:
            return False
        if field not in self.field_space_nodes:
            return False
        if tree not in self.region_trees:
            return False
        region = self.get_index_node(True, index).get_instance(tree)
        ver = 0 if iid not in self.instances else len(self.instances[iid])
        inst = ReductionInstance(self, iid, ver, self.memories[mem],
                                 region, fold, indirect)
        if iid in self.instances:
            self.instances[iid].append(inst)
        else:
            self.instances[iid] = [inst]
        return True

    def add_instance_field(self, iid, fid):
        if not iid in self.instances:
            return False
        self.instances[iid][-1].add_field(fid)
        return True

    def add_op_user(self, uid, idx, iid):
        if uid not in self.ops:
            return False
        if iid not in self.instances:
            return False
        if not self.instances[iid][-1].add_op_user(self.ops[uid], idx):
            return False
        self.ops[uid].add_instance(idx, self.instances[iid][-1])
        return True

    def add_op_field_user(self, uid, idx, iid, fid):
        if uid not in self.ops:
            return False
        if iid not in self.instances:
            return False
        if not self.instances[iid][-1].add_op_user(self.ops[uid], idx):
            return False
        self.ops[uid].add_partial_instance(idx, self.instances[iid][-1], fid)
        return True

    def add_op_proc_user(self, uid, pid):
        if pid not in self.processors and pid not in self.utilities:
            return False
        if uid not in self.ops:
            return False
        if pid in self.processors:
            self.processors[pid].append_op_instance(self.ops[uid])
        else:
            self.utilities[pid].append_op_instance(self.ops[uid])
        return True

    def add_event_dependence(self, id1, gen1, id2, gen2):
        e1 = self.get_event_from_id(id1,gen1)
        e2 = self.get_event_from_id(id2,gen2)
        e1.add_physical_outgoing(e2)
        e2.add_physical_incoming(e1)
        return True

    def add_implicit_dependence(self, id1, gen1, id2, gen2):
        e1 = self.get_event_from_id(id1,gen1)
        e2 = self.get_event_from_id(id2,gen2)
        e1.add_implicit_outgoing(e2)
        e2.add_implicit_incoming(e1)
        return True

    def find_enclosing_index_id(self, slice_id):
        if slice_id in self.slice_index:
            return self.slice_index[slice_id]
        assert slice_id in self.slice_slice
        return self.find_enclosing_index_id(self.slice_slice[slice_id])

    def get_event(self, handle):
        if handle not in self.events:
            self.events[handle] = Event(self, handle)
        return self.events[handle]

    def get_event_from_id(self, id, gen):
        if id in self.phase_barriers:
            handle = PhaseBarrier(id, gen)
        else:
            handle = EventHandle(id, gen)

        if not handle in self.events:
            self.events[handle] = Event(self, handle)

        ev = self.events[handle]
        #if EventHandle(id, gen - 1) in self.events:
        #    prev_ev = self.events[EventHandle(id, gen - 1)]
        #    prev_ev.add_physical_outgoing(ev)
        #    ev.add_physical_incoming(prev_ev)

        return ev

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
        if len(set(req1.fields) & set(req2.fields)) == 0:
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

    def is_subtree(self, inode1, inode2):
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
        else:
            return False

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
            inode1_prev = inode1
            inode1 = inode1.parent
            inode2_prev = inode2
            inode2 = inode2.parent
        assert inode1 is inode2
        assert inode1_prev.parent == inode2_prev.parent
        # Least common ancestor is a region, so they came from different
        # partitions and are therefore not disjoint
        # TODO: handle when partitions are computed to be disjoint
        if inode1.is_region() or not inode1.disjoint:
            return not inode1.is_independent(inode1_prev.uid, inode2_prev.uid)
        else:
            return False

    def get_next_logical_mark(self):
        result = self.next_logical_mark
        self.next_logical_mark += long(1)
        return result

    def check_data_flow(self):
        for uid,op in self.ops.iteritems():
            op.check_data_flow()

    def check_instance_dependences(self):
        for versions in self.instances.itervalues():
            for instance in versions:
                # TODO: deal with simultaneous properly
                if instance.is_simultaneous():
                    continue
                print "Checking physical instance "+hex(instance.iid)+"..."
                for field, op_users in instance.op_users.iteritems():
                    for op1, reqs1 in op_users.iteritems():
                        for req1 in reqs1:
                            for op2, reqs2 in op_users.iteritems():
                                for req2 in reqs2:
                                    if ((field == 145) and (instance.iid == 16140901064495857727L) and ((op1.uid == 77) or (op2.uid == 77))):
                                        print "HELLO THERE"
                                    if op1 != op2 and \
                                            self.compute_dependence(req1, req2) in \
                                            (TRUE_DEPENDENCE, ANTI_DEPENDENCE):
                                        def traverse_event(node, traverser):
                                            if traverser.found:
                                                return False
                                            return True
                                        def traverse_op(node, traverser):
                                            if traverser.target == node or traverser.found:
                                                traverser.found = True
                                            return not traverser.found
                                        def post_traverse(node, traverser):
                                            pass
                                        traverser = EventGraphTraverser(False, True, True,
                                                self.get_next_traverser_gen(), traverse_event,
                                                traverse_op, traverse_op, traverse_op, traverse_op,
                                                traverse_op, traverse_op, post_traverse, post_traverse,
                                                post_traverse, post_traverse, post_traverse, post_traverse,
                                                post_traverse)
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
                                                        op1.get_name()+" and requirement "+\
                                                        str(req2.index)+" of "+op2.get_name()+\
                                                        " for field "+str(field)
                                                if self.verbose:
                                                    print "      First Requirement:"
                                                    req1.print_requirement()
                                                    print "      Second Requirement:"
                                                    req2.print_requirement()

    def print_dataflow_graphs(self, path, simplify_graphs):
        total_dataflow_graphs = 0
        for uid,op in self.ops.iteritems():
            total_dataflow_graphs += op.print_dataflow(path,simplify_graphs) 
        print "Found "+str(total_dataflow_graphs)+" dataflow graphs"

    def print_pictures(self, path, simplify_graphs):
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
            components[idx].generate_graph(idx,path,simplify_graphs)
            components[idx].unmark_all()

    def print_instance_graphs(self, path):
        components = list()
        for h,e in self.events.iteritems():
            comp = ConnectedComponent(self)
            e.physical_traverse(comp)
            if not comp.empty():
                components.append(comp)
            else:
                comp.unmark_all()
        print "Found "+str(len(components))+" instance graphs"
        for idx in range(len(components)):
            components[idx].generate_igraph(idx,path)
            components[idx].unmark_all()

    def print_partition_graphs(self, path, simplify_graphs):
        index_space_printer = GraphPrinter(path, "index_space_graph")
        for index_space_node in self.index_space_nodes.itervalues():
            if isinstance(index_space_node, IndexSpaceNode) and \
                    index_space_node.parent == None:
                        index_space_node.print_graph(index_space_printer)
        index_space_printer.print_pdf_after_close(simplify_graphs)

        field_space_printer = GraphPrinter(path, "field_space_graph")
        for field_space_node in self.field_space_nodes.itervalues():
            field_space_node.print_graph(field_space_printer)
        field_space_printer.print_pdf_after_close(simplify_graphs)

        region_graph_printer = GraphPrinter(path, "region_graph")
        for region_node in self.region_trees.itervalues():
            if simplify_graphs:
                region_node.mark_named_children()
            region_node.print_graph(region_graph_printer, simplify_graphs)
        region_graph_printer.print_pdf_after_close(simplify_graphs)

    def dump_event_paths(self):
        for uid1,op1 in self.ops.iteritems():
            for uid2,op2 in self.ops.iteritems():
                if op1 <> op2 and not (isinstance(op1, IndexTask) or isinstance(op2, IndexTask)) and\
                        not (isinstance(op1, Deletion) or isinstance(op2, Deletion)) and\
                        not (isinstance(op1, CopyOp) or isinstance(op2, CopyOp)):
                    def traverse_event(node, traverser):
                        if node in traverser.visited:
                            return False
                        else:
                            traverser.path.append(node)
                            return True
                    def traverse_op(node, traverser):
                        traverser.path.append(node)
                        if traverser.target == node:
                            traverser.paths.append(traverser.path)
                            traverser.path = list(traverser.path)
                            return False
                        else:
                            return True
                    def post_traverse(node, traverser):
                        if not (node in traverser.visited):
                            traverser.visited.add(node)
                            traverser.path.pop()
                    traverser = EventGraphTraverser(True, False, True,
                            self.get_next_traverser_gen(), traverse_event,
                            traverse_op, traverse_op, traverse_op, traverse_op,
                            traverse_op, traverse_op, post_traverse, post_traverse,
                            post_traverse, post_traverse, post_traverse, post_traverse,
                            post_traverse)
                    traverser.target = op2
                    traverser.path = list()
                    traverser.paths = list()
                    traverser.visited = set()
                    op1.event_graph_traverse(traverser)
                    if len(traverser.paths) > 0:
                        print("##### paths between "+op1.get_name()+" and "+op2.get_name()+" #####")
                        for p in traverser.paths:
                            print('    ' + ' -> '.join([op.get_name() for op in p]))
