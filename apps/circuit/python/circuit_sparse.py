#!/usr/bin/env python

# Copyright 2019 Stanford University
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

from __future__ import absolute_import, division, print_function

import argparse
from collections import OrderedDict
import math
import numpy as np
import os
import subprocess

import legion
from legion import disjoint_complete, index_launch, print_once, task, Domain, Fspace, Ispace, IndexLaunch, ID, Partition, N, R, Reduce, Region, RW, Trace, WD

root_dir = os.path.dirname(__file__)
circuit_header = subprocess.check_output(
    [
        "gcc", "-D", "__attribute__(x)=", "-E", "-P",
        os.path.join(root_dir, "circuit_config.h")
    ]).decode("utf-8")
ffi = legion.ffi
ffi.cdef(circuit_header)

Config = legion.Type(
    np.dtype([('bytes', np.void, ffi.sizeof('Config'))]),
    'Config')

WIRE_SEGMENTS = 10

def parse_args(argv):
    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument('-l', dest='num_loops', type=int, default=5)
    parser.add_argument('-p', dest='num_pieces', type=int, default=4)
    parser.add_argument('-pps', dest='pieces_per_superpiece', type=int, default=1)
    parser.add_argument('-npp', dest='nodes_per_piece', type=int, default=4)
    parser.add_argument('-wpp', dest='wires_per_piece', type=int, default=8)
    parser.add_argument('-pct', dest='pct_wire_in_piece', type=int, default=80)
    parser.add_argument('-s', dest='random_seed', type=int, default=12345)
    parser.add_argument('-i', dest='steps', type=int, default=10000)
    parser.add_argument('-sync', dest='sync', type=int, default=0)
    parser.add_argument('-prune', dest='prune', type=int, default=0)
    parser.add_argument('-checks', dest='perform_checks', action='store_true')
    parser.add_argument('-dump', dest='dump_values', action='store_true')
    parser.add_argument('-shared', dest='pct_shared_nodes', type=float, default=1.0)
    parser.add_argument('-density', dest='density', type=int, default=20)
    parser.add_argument('-neighbors', dest='num_neighbors', type=int, default=5)
    parser.add_argument('-window', dest='window', type=int, default=3)
    args = parser.parse_args(argv[1:])

    conf = ffi.new('Config *')
    for field, value in vars(args).items():
        setattr(conf, field, value)
    return conf

init_piece = legion.extern_task(
    task_id=10002,
    argument_types=[legion.int32, Config, Region, Region, Region, Region, Region],
    privileges=[None, None, WD, WD, WD, N, WD],
    return_type=legion.void,
    calling_convention='regent')

init_pointers = legion.extern_task(
    task_id=10003,
    argument_types=[Region, Region, Region, Region],
    privileges=[N, N, N, RW('in_ptr', 'in_ptr_r', 'out_ptr', 'out_ptr_r')],
    return_type=legion.void,
    calling_convention='regent')

calculate_new_currents = legion.extern_task(
    task_id=10004,
    argument_types=[legion.bool_, legion.uint32, Region, Region, Region, Region],
    privileges=[
        None,
        None,
        R('node_voltage'),
        R('node_voltage'),
        R('node_voltage'),
        R('in_ptr', 'in_ptr_r', 'out_ptr', 'out_ptr_r', 'inductance', 'resistance', 'wire_cap') + RW(*['current_%d' % i for i in range(10)]) + RW(*['voltage_%d' % i for i in range(9)])],
    return_type=legion.void,
    calling_convention='regent')

distribute_charge = legion.extern_task(
    task_id=10005,
    argument_types=[Region, Region, Region, Region],
    privileges=[
        Reduce('+', 'charge'),
        Reduce('+', 'charge'),
        Reduce('+', 'charge'),
        R('in_ptr', 'in_ptr_r', 'out_ptr', 'out_ptr_r', 'current_0', 'current_9')],
    return_type=legion.void,
    calling_convention='regent')

update_voltages = legion.extern_task(
    task_id=10006,
    argument_types=[legion.bool_, Region, Region],
    privileges=[
        None,
        R('node_cap', 'leakage') + RW('node_voltage', 'charge'),
        R('node_cap', 'leakage') + RW('node_voltage', 'charge')],
    return_type=legion.void,
    calling_convention='regent')

@task(task_id=2, replicable=True) # , inner=True
def main():
    conf = parse_args(legion.input_args(True))

    assert conf.num_pieces % conf.pieces_per_superpiece == 0, "pieces should be evenly distributed to superpieces"
    conf.shared_nodes_per_piece = int(math.ceil(conf.nodes_per_piece * conf.pct_shared_nodes / 100.0))
    print("circuit settings: loops=%d prune=%d pieces=%d (pieces/superpiece=%d) nodes/piece=%d (nodes/piece=%d) wires/piece=%d pct_in_piece=%d seed=%d" % (
        conf.num_loops, conf.prune, conf.num_pieces, conf.pieces_per_superpiece, conf.nodes_per_piece,
        conf.shared_nodes_per_piece, conf.wires_per_piece, conf.pct_wire_in_piece, conf.random_seed))

    num_pieces = conf.num_pieces
    num_superpieces = conf.num_pieces // conf.pieces_per_superpiece
    num_circuit_nodes = num_pieces * conf.nodes_per_piece
    num_circuit_wires = num_pieces * conf.wires_per_piece

    node = Fspace.create(OrderedDict([
        ('node_cap', legion.float32),
        ('leakage', legion.float32),
        ('charge', legion.float32),
        ('node_voltage', legion.float32),
    ]))
    wire = Fspace.create(OrderedDict([
        ('in_ptr', legion.int64),
        ('in_ptr_r', legion.uint8),
        ('out_ptr', legion.int64),
        ('out_ptr_r', legion.uint8),
        ('inductance', legion.float32),
        ('resistance', legion.float32),
        ('wire_cap', legion.float32),
    ] + [
        ('current_%d' % i, legion.float32) for i in range(WIRE_SEGMENTS)
    ] + [
        ('voltage_%d' % i, legion.float32) for i in range(WIRE_SEGMENTS - 1)
    ]))

    all_nodes = Region.create([num_circuit_nodes], node)
    all_wires = Region.create([num_circuit_wires], wire)

    node_size = np.dtype(list(map(lambda x: (x[0], x[1].numpy_type), node.field_types.items())), align=True).itemsize
    wire_size = np.dtype(list(map(lambda x: (x[0], x[1].numpy_type), wire.field_types.items())), align=True).itemsize
    print("Circuit memory usage:")
    print("  Nodes : %10d * %4d bytes = %12d bytes" % (num_circuit_nodes, node_size, num_circuit_nodes * node_size))
    print("  Wires : %10d * %4d bytes = %12d bytes" % (num_circuit_wires, wire_size, num_circuit_wires * wire_size))
    total = ((num_circuit_nodes * node_size) + (num_circuit_wires * wire_size))
    print("  Total                             %12d bytes" % total)

    snpp = conf.shared_nodes_per_piece
    pnpp = conf.nodes_per_piece - conf.shared_nodes_per_piece
    pps = conf.pieces_per_superpiece
    num_shared_nodes = num_pieces * snpp

    privacy_coloring = Region.create([2], {'rect': legion.rect1d})
    np.copyto(
        privacy_coloring.rect,
        np.array([(num_shared_nodes, num_circuit_nodes - 1),
                  (0, num_shared_nodes - 1)],
                 dtype=privacy_coloring.rect.dtype),
        casting='no')
    privacy_part = Partition.create_by_restriction(privacy_coloring, [2], np.eye(1), [1], disjoint_complete)
    all_nodes_part = Partition.create_by_image(all_nodes, privacy_part, 'rect', [2], disjoint_complete)

    all_private = all_nodes_part[0]
    all_shared = all_nodes_part[1]

    launch_domain = Ispace.create([num_superpieces])

    private_part = Partition.create_by_restriction(
        all_private, launch_domain, np.eye(1)*pnpp*pps, Domain.create([pnpp*pps], [num_shared_nodes]), disjoint_complete)
    shared_part = Partition.create_by_restriction(
        all_shared, launch_domain, np.eye(1)*snpp*pps, [snpp*pps], disjoint_complete)

    wires_part = Partition.create_equal(all_wires, launch_domain)

    ghost_ranges = Region.create([num_superpieces], OrderedDict([('rect', legion.rect1d)]))
    ghost_ranges_part = Partition.create_equal(ghost_ranges, launch_domain)

    for i in IndexLaunch(launch_domain):
        init_piece(int(i), conf[0], ghost_ranges_part[i], private_part[i], shared_part[i], all_shared, wires_part[i])

    ghost_part = Partition.create_by_image(all_shared, ghost_ranges_part, 'rect', launch_domain)

    for i in IndexLaunch(launch_domain):
        init_pointers(private_part[i], shared_part[i], ghost_part[i], wires_part[i])

    steps = conf.steps
    prune = conf.prune
    num_loops = conf.num_loops + 2*prune

    trace = Trace()
    for j in range(num_loops):
        if j == prune:
            legion.execution_fence(block=True)
            start_time = legion.c.legion_get_current_time_in_nanos()
        with trace:
            for i in IndexLaunch(launch_domain):
                calculate_new_currents(
                    False, # j == prune,
                    steps, private_part[i], shared_part[i], ghost_part[i], wires_part[i])
            for i in IndexLaunch(launch_domain):
                distribute_charge(private_part[i], shared_part[i], ghost_part[i], wires_part[i])
            for i in IndexLaunch(launch_domain):
                update_voltages(
                    False, # j == num_loops - prune - 1,
                    private_part[i], shared_part[i])
        if j == num_loops - prune - 1:
            legion.execution_fence(block=True)
            stop_time = legion.c.legion_get_current_time_in_nanos()

    sim_time = (stop_time - start_time)/1e9
    print_once('ELAPSED TIME = %7.3f s' % sim_time)

    # Compute the floating point operations per second
    num_circuit_nodes = conf.num_pieces * conf.nodes_per_piece
    num_circuit_wires = conf.num_pieces * conf.wires_per_piece
    # calculate currents
    operations = num_circuit_wires * (WIRE_SEGMENTS*6 + (WIRE_SEGMENTS-1)*4) * conf.steps
    # distribute charge
    operations += (num_circuit_wires * 4)
    # update voltages
    operations += (num_circuit_nodes * 4)
    # multiply by the number of loops
    operations *= conf.num_loops

    # Compute the number of gflops
    gflops = (1e-9*operations)/sim_time
    print_once("GFLOPS = %7.3f GFLOPS" % gflops)

if __name__ == '__legion_main__':
    main()
