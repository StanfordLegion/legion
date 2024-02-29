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

from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict
import numpy as np
import os
import subprocess

import pygion
from pygion import task, print_once, Fspace, Future, IndexLaunch, Ipartition, Ispace, N, Partition, R, Reduce, Region, RW

root_dir = os.path.dirname(__file__)
pennant_header = subprocess.check_output(
    [
        "gcc", "-DLEGION_USE_PYTHON_CFFI", "-E", "-P",
        os.path.join(root_dir, "pennant_config.h")
    ]).decode("utf-8")
ffi = pygion.ffi
ffi.cdef(pennant_header)

mesh_colorings = pygion.Type(
    np.dtype([('bytes', np.void, ffi.sizeof('mesh_colorings'))]),
    'mesh_colorings')

mesh_partitions = pygion.Type(
    np.dtype([('bytes', np.void, ffi.sizeof('mesh_partitions'))]),
    'mesh_partitions')

config = pygion.Type(
    np.dtype([('bytes', np.void, ffi.sizeof('config'))]),
    'config')

def create_partition(is_disjoint, region, c_partition, color_space):
    ipart = Ipartition(c_partition.index_partition, region.ispace, color_space)
    return Partition(region, ipart)

read_config = pygion.extern_task(
    task_id=10005,
    argument_types=[],
    privileges=[],
    return_type=config,
    calling_convention='regent')

read_partitions = pygion.extern_task(
    task_id=10006,
    argument_types=[Region, Region, Region, config],
    privileges=[N, N, N],
    return_type=mesh_partitions,
    calling_convention='regent')

initialize_spans = pygion.extern_task(
    task_id=10007,
    argument_types=[config, pygion.int64, Region, Region, Region, Region],
    privileges=[None, None, RW, RW, RW, RW],
    return_type=pygion.void,
    calling_convention='regent')

initialize_topology = pygion.extern_task(
    task_id=10008,
    argument_types=[config, Region, Region, Region, Region, Region],
    privileges=[
        None,
        RW('znump'),
        RW('px_x', 'px_y', 'has_bcx', 'has_bcy'),
        RW('px_x', 'px_y', 'has_bcx', 'has_bcy'),
        N,
        RW('mapsz', 'mapsp1', 'mapsp1_r', 'mapsp2', 'mapsp2_r', 'mapss3', 'mapss4')],
    return_type=pygion.void,
    calling_convention='regent')

init_pointers = pygion.extern_task(
    task_id=10017,
    argument_types=[Region, Region, Region, Region, Region],
    privileges=[N, N, N, RW('mapsp1', 'mapsp1_r', 'mapsp2', 'mapsp2_r'), R],
    return_type=pygion.void,
    calling_convention='regent')

init_mesh_zones = pygion.extern_task(
    task_id=10018,
    argument_types=[Region, Region],
    privileges=[RW('zx_x', 'zx_y', 'zarea', 'zvol'), R],
    return_type=pygion.void,
    calling_convention='regent')

calc_centers_full = pygion.extern_task(
    task_id=10019,
    argument_types=[Region, Region, Region, Region, Region, pygion.bool_],
    privileges=[
        R('znump') + RW('zx_x', 'zx_y'),
        R('px_x', 'px_y'),
        R('px_x', 'px_y'),
        R('mapsz', 'mapsp1', 'mapsp1_r', 'mapsp2', 'mapsp2_r') + RW('ex_x', 'ex_y'),
        R],
    return_type=pygion.void,
    calling_convention='regent')

calc_volumes_full = pygion.extern_task(
    task_id=10020,
    argument_types=[Region, Region, Region, Region, Region, pygion.bool_],
    privileges=[
        R('zx_x', 'zx_y', 'znump') + RW('zarea', 'zvol'),
        R('px_x', 'px_y'),
        R('px_x', 'px_y'),
        R('mapsz', 'mapsp1', 'mapsp1_r', 'mapsp2', 'mapsp2_r') + RW('sarea'),
        R],
    return_type=pygion.void,
    calling_convention='regent')

init_side_fracs = pygion.extern_task(
    task_id=10021,
    argument_types=[Region, Region, Region, Region, Region],
    privileges=[
        R('zarea'),
        N,
        N,
        R('mapsz', 'sarea') + RW('smf'),
        R],
    return_type=pygion.void,
    calling_convention='regent')

init_hydro = pygion.extern_task(
    task_id=10022,
    argument_types=[Region, Region, pygion.float64, pygion.float64, pygion.float64, pygion.float64, pygion.float64, pygion.float64, pygion.float64, pygion.float64],
    privileges=[
        R('zx_x', 'zx_y', 'zvol') + RW('zr', 'ze', 'zwrate', 'zm', 'zetot'),
        R],
    return_type=pygion.void,
    calling_convention='regent')

init_radial_velocity = pygion.extern_task(
    task_id=10023,
    argument_types=[Region, Region, pygion.float64],
    privileges=[
        R('px_x', 'px_y') + RW('pu_x', 'pu_y'),
        R],
    return_type=pygion.void,
    calling_convention='regent')

adv_pos_half = pygion.extern_task(
    task_id=10024,
    argument_types=[Region, Region, pygion.float64, pygion.bool_, pygion.bool_],
    privileges=[
        R('px_x', 'px_y', 'pu_x', 'pu_y') + RW('px0_x', 'px0_y', 'pxp_x', 'pxp_y', 'pu0_x', 'pu0_y', 'pmaswt', 'pf_x', 'pf_y'),
        R],
    return_type=pygion.void,
    calling_convention='regent')

calc_everything = pygion.extern_task(
    task_id=10025,
    argument_types=[Region, Region, Region, Region, Region, Region, pygion.float64, pygion.float64, pygion.float64, pygion.float64, pygion.float64, pygion.float64, pygion.bool_],
    privileges=[
        R('zm', 'zvol', 'zr', 'znump', 'zwrate', 'ze') + RW('zp', 'zxp_x', 'zuc_y', 'zvol0', 'zvolp', 'z0tmp', 'zxp_y', 'zrp', 'zuc_x', 'zdu', 'zss', 'zareap', 'zdl'),
        R('pu_y', 'pxp_y', 'pxp_x', 'pu_x') + RW('pf_x', 'pmaswt', 'pf_y'),
        R('pu_y', 'pxp_y', 'pxp_x', 'pu_x') + Reduce('+', 'pf_x', 'pmaswt', 'pf_y'),
        R('smf', 'mapsp2', 'mapss4', 'mapsp1_r', 'mapss3', 'mapsp2_r', 'mapsz', 'mapsp1') + RW('sfq_x', 'sfq_y', 'cdiv', 'carea', 'cevol', 'cqe2_x', 'sft_x', 'cqe2_y', 'sareap', 'cqe1_y', 'exp_x', 'sfp_x', 'cdu', 'elen', 'sft_y', 'sfp_y', 'exp_y', 'cqe1_x', 'ccos'),
        R,
        R],
    return_type=pygion.void,
    calling_convention='regent')

adv_pos_full = pygion.extern_task(
    task_id=10026,
    argument_types=[Region, Region, pygion.float64, pygion.bool_],
    privileges=[
        R('px0_x', 'px0_y', 'pmaswt', 'has_bcx', 'has_bcy') + RW('px_x', 'px_y', 'pu_x', 'pu_y', 'pu0_x', 'pu0_y', 'pf_x', 'pf_y'),
        R],
    return_type=pygion.void,
    calling_convention='regent')

calc_everything_full = pygion.extern_task(
    task_id=10027,
    argument_types=[Region, Region, Region, Region, Region, Region, pygion.float64, pygion.bool_],
    privileges=[
        R('zm', 'zvol0', 'znump', 'zp') + RW('ze', 'zx_x', 'zwrate', 'zetot', 'zx_y', 'zw', 'zarea', 'zvol', 'zr'),
        R('px_y', 'pxp_y', 'pxp_x', 'pu0_y', 'pu0_x', 'pu_y', 'pu_x', 'px_x'),
        R('px_y', 'pxp_y', 'pxp_x', 'pu0_y', 'pu0_x', 'pu_y', 'pu_x', 'px_x'),
        R('mapsp2', 'sfq_x', 'mapsp1_r', 'sfp_y', 'sfq_y', 'mapsp1', 'mapsz', 'mapsp2_r', 'sfp_x') + RW('ex_y', 'ex_x', 'sarea'),
        R,
        R],
    return_type=pygion.void,
    calling_convention='regent')

calc_dt_hydro = pygion.extern_task(
    task_id=10028,
    argument_types=[Region, Region, pygion.float64, pygion.float64, pygion.float64, pygion.float64, pygion.bool_, pygion.bool_],
    privileges=[
        R('zdl', 'zvol0', 'zvol', 'zss', 'zdu'),
        R],
    return_type=pygion.float64,
    calling_convention='regent')

calc_global_dt = pygion.extern_task(
    task_id=10029,
    argument_types=[pygion.float64, pygion.float64, pygion.float64, pygion.float64, pygion.float64, pygion.float64, pygion.float64, pygion.int64],
    privileges=[],
    return_type=pygion.float64,
    calling_convention='regent')

validate_output_sequential = pygion.extern_task(
    task_id=10033,
    argument_types=[Region, Region, Region, config],
    privileges=[R, R, R],
    return_type=pygion.void,
    calling_convention='regent')

@task(task_id=2) # , inner=True
def main():
    print_once('Running pennant_fast.py')

    conf = read_config().get()

    zone = Fspace(OrderedDict([
        ('zxp_x', pygion.float64),
        ('zxp_y', pygion.float64),
        ('zx_x', pygion.float64),
        ('zx_y', pygion.float64),
        ('zareap', pygion.float64),
        ('zarea', pygion.float64),
        ('zvol0', pygion.float64),
        ('zvolp', pygion.float64),
        ('zvol', pygion.float64),
        ('zdl', pygion.float64),
        ('zm', pygion.float64),
        ('zrp', pygion.float64),
        ('zr', pygion.float64),
        ('ze', pygion.float64),
        ('zetot', pygion.float64),
        ('zw', pygion.float64),
        ('zwrate', pygion.float64),
        ('zp', pygion.float64),
        ('zss', pygion.float64),
        ('zdu', pygion.float64),
        ('zuc_x', pygion.float64),
        ('zuc_y', pygion.float64),
        ('z0tmp', pygion.float64),
        ('znump', pygion.uint8),
    ]))

    point = Fspace(OrderedDict([
        ('px0_x', pygion.float64),
        ('px0_y', pygion.float64),
        ('pxp_x', pygion.float64),
        ('pxp_y', pygion.float64),
        ('px_x', pygion.float64),
        ('px_y', pygion.float64),
        ('pu0_x', pygion.float64),
        ('pu0_y', pygion.float64),
        ('pu_x', pygion.float64),
        ('pu_y', pygion.float64),
        ('pap_x', pygion.float64),
        ('pap_y', pygion.float64),
        ('pf_x', pygion.float64),
        ('pf_y', pygion.float64),
        ('pmaswt', pygion.float64),
        ('has_bcx', pygion.bool_),
        ('has_bcy', pygion.bool_),
    ]))

    side = Fspace(OrderedDict([
        ('mapsz', pygion.int1d),
        ('mapsp1', pygion.int1d),
        ('mapsp1_r', pygion.uint8),
        ('mapsp2', pygion.int1d),
        ('mapsp2_r', pygion.uint8),
        ('mapss3', pygion.int1d),
        ('mapss4', pygion.int1d),
        ('sareap', pygion.float64),
        ('sarea', pygion.float64),
        ('svolp', pygion.float64),
        ('svol', pygion.float64),
        ('ssurfp_x', pygion.float64),
        ('ssurfp_y', pygion.float64),
        ('smf', pygion.float64),
        ('sfp_x', pygion.float64),
        ('sfp_y', pygion.float64),
        ('sft_x', pygion.float64),
        ('sft_y', pygion.float64),
        ('sfq_x', pygion.float64),
        ('sfq_y', pygion.float64),
        ('exp_x', pygion.float64),
        ('exp_y', pygion.float64),
        ('ex_x', pygion.float64),
        ('ex_y', pygion.float64),
        ('elen', pygion.float64),
        ('carea', pygion.float64),
        ('cevol', pygion.float64),
        ('cdu', pygion.float64),
        ('cdiv', pygion.float64),
        ('ccos', pygion.float64),
        ('cqe1_x', pygion.float64),
        ('cqe1_y', pygion.float64),
        ('cqe2_x', pygion.float64),
        ('cqe2_y', pygion.float64),
    ]))

    span = Fspace(OrderedDict([
        ('start', pygion.int64),
        ('stop', pygion.int64),
        ('internal', pygion.bool_), 
   ]))

    zones = Region([conf.nz], zone)
    points = Region([conf.np], point)
    sides = Region([conf.ns], side)

    assert conf.par_init, 'parallel initialization required'

    old_seq_init = conf.seq_init
    if conf.seq_init:
        print('Warning: Sequential initialization not supported, skipping')
        # Since we aren't actually doing sequential intialization, we
        # have to turn this off or the verification in parallel
        # initialization will fail.
        conf.seq_init = False

    assert conf.par_init
    partitions = read_partitions(zones, points, sides, conf).get()

    conf.nspans_zones = partitions.nspans_zones
    conf.nspans_points = partitions.nspans_points

    pieces = Ispace([conf.npieces])

    zones_part = create_partition(True, zones, partitions.rz_all_p, pieces)

    points_part = create_partition(True, points, partitions.rp_all_p, [2])
    private = points_part[0]
    ghost = points_part[1]

    private_part = create_partition(True, private, partitions.rp_all_private_p, pieces)
    ghost_part = create_partition(False, ghost, partitions.rp_all_ghost_p, pieces)
    shared_part = create_partition(True, ghost, partitions.rp_all_shared_p, pieces)

    sides_part = create_partition(True, sides, partitions.rs_all_p, pieces)

    zone_spans = Region([conf.npieces * conf.nspans_zones], span)
    zone_spans_part = Partition.equal(zone_spans, pieces)

    private_spans = Region([conf.npieces * conf.nspans_points], span)
    private_spans_part = Partition.equal(private_spans, pieces)

    shared_spans = Region([conf.npieces * conf.nspans_points], span)
    shared_spans_part = Partition.equal(shared_spans, pieces)

    side_spans = Region([conf.npieces * conf.nspans_zones], span)
    side_spans_part = Partition.equal(side_spans, pieces)

    for region in [zone_spans, private_spans, shared_spans, side_spans]:
        for field in ['start', 'stop']:
            pygion.fill(region, field, 0)

    if old_seq_init:
        # FIXME: These fields are actually never used, fill them here
        # just to avoid validation errors later.
        pygion.fill(points, 'pap_x', 0)
        pygion.fill(points, 'pap_y', 0)
        pygion.fill(sides, 'svolp', 0)
        pygion.fill(sides, 'svol', 0)
        pygion.fill(sides, 'ssurfp_x', 0)
        pygion.fill(sides, 'ssurfp_y', 0)

    if conf.par_init:
        for i in IndexLaunch(pieces):
            initialize_topology(
                conf,
                zones_part[i],
                private_part[i],
                shared_part[i],
                ghost_part[i],
                sides_part[i])

        for i in IndexLaunch(pieces):
            initialize_spans(
                conf, int(i),
                zone_spans_part[i],
                private_spans_part[i],
                shared_spans_part[i],
                side_spans_part[i])

    for i in IndexLaunch(pieces):
        init_pointers(
            zones_part[i],
            private_part[i],
            ghost_part[i],
            sides_part[i],
            side_spans_part[i])

    for i in IndexLaunch(pieces):
        init_mesh_zones(
            zones_part[i],
            zone_spans_part[i])

    for i in IndexLaunch(pieces):
        calc_centers_full(
            zones_part[i],
            private_part[i],
            ghost_part[i],
            sides_part[i],
            side_spans_part[i],
            True)

    for i in IndexLaunch(pieces):
        calc_volumes_full(
            zones_part[i],
            private_part[i],
            ghost_part[i],
            sides_part[i],
            side_spans_part[i],
            True)

    for i in IndexLaunch(pieces):
        init_side_fracs(
            zones_part[i],
            private_part[i],
            ghost_part[i],
            sides_part[i],
            side_spans_part[i])

    for i in IndexLaunch(pieces):
        init_hydro(
            zones_part[i],
            zone_spans_part[i],
            conf.rinit, conf.einit, conf.rinitsub, conf.einitsub,
            conf.subregion[0], conf.subregion[1], conf.subregion[2], conf.subregion[3])

    for i in IndexLaunch(pieces):
        init_radial_velocity(
            private_part[i],
            private_spans_part[i],
            conf.uinitradial)

    for i in IndexLaunch(pieces):
        init_radial_velocity(
            shared_part[i],
            shared_spans_part[i],
            conf.uinitradial)

    cycle = 0
    cstop = conf.cstop + 2*conf.prune
    time = 0.0
    dt = Future(conf.dtmax, pygion.float64)
    dthydro = conf.dtmax
    while cycle < cstop and time < conf.tstop:
        if cycle == conf.prune:
            pygion.execution_fence(block=True)
            start_time = pygion.c.legion_get_current_time_in_nanos()

        dt = calc_global_dt(dt, conf.dtfac, conf.dtinit, conf.dtmax, dthydro, time, conf.tstop, cycle)

        for i in IndexLaunch(pieces):
            adv_pos_half(
                private_part[i],
                private_spans_part[i],
                dt, True, False)

        for i in IndexLaunch(pieces):
            adv_pos_half(
                shared_part[i],
                shared_spans_part[i],
                dt, True, False)

        for i in IndexLaunch(pieces):
            calc_everything(
                zones_part[i],
                private_part[i],
                ghost_part[i],
                sides_part[i],
                zone_spans_part[i],
                side_spans_part[i],
                conf.alfa, conf.gamma, conf.ssmin, dt, conf.q1, conf.q2, True)

        for i in IndexLaunch(pieces):
            adv_pos_full(
                private_part[i],
                private_spans_part[i],
                dt, True)

        for i in IndexLaunch(pieces):
            adv_pos_full(
                shared_part[i],
                shared_spans_part[i],
                dt, True)

        for i in IndexLaunch(pieces):
            calc_everything_full(
                zones_part[i],
                private_part[i],
                ghost_part[i],
                sides_part[i],
                zone_spans_part[i],
                side_spans_part[i],
                dt, True)

        futures = []
        for i in IndexLaunch(pieces):
            futures.append(
                calc_dt_hydro(
                    zones_part[i],
                    zone_spans_part[i],
                    dt, conf.dtmax, conf.cfl, conf.cflv, True, False))

        dthydro = conf.dtmax
        dthydro = min(dthydro, *list(map(lambda x: x.get(), futures)))

        cycle += 1
        time += dt.get()

        if cycle == conf.cstop - conf.prune:
            pygion.execution_fence(block=True)
            stop_time = pygion.c.legion_get_current_time_in_nanos()

    if old_seq_init:
        validate_output_sequential(zones, points, sides, conf)
    else:
        print_once("Warning: Skipping sequential validation")

    print_once("ELAPSED TIME = %7.3f s" % ((stop_time - start_time)/1e9))

if __name__ == '__legion_main__':
    main()
