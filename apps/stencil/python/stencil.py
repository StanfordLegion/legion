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
import numpy as np

import legion
from legion import disjoint, disjoint_incomplete, disjoint_complete, print_once, task, Fspace, IndexLaunch, Ispace, N, Partition, R, Region, RW

DTYPE = legion.float64
RADIUS = 2

def parse_args(argv):
    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument('-nx', type=int, default=12)
    parser.add_argument('-ny', type=int, default=12)
    parser.add_argument('-ntx', type=int, default=4)
    parser.add_argument('-nty', type=int, default=4)
    parser.add_argument('-tsteps', type=int, default=20)
    parser.add_argument('-tprune', type=int, default=5)
    parser.add_argument('-init', type=int, default=1000)
    return parser.parse_args(argv[1:])

def make_colors_part(tiles):
    colors = Region.create(tiles, {'rect': legion.rect2d})
    colors_part = Partition.create_by_restriction(colors, tiles, np.eye(2), [1, 1], disjoint_complete)
    return colors, colors_part

def make_private_partition(points, tiles, n, nt):
    colors, colors_part = make_colors_part(tiles)
    npoints = n + nt*2*RADIUS
    for tile in np.ndindex(tuple(nt)):
        idx = np.array(tile)
        colors.rect[tile] = (
            idx*npoints/nt,
            (idx+1)*npoints/nt - 1)
    return Partition.create_by_image(points, colors_part, 'rect', tiles, disjoint_complete)

def make_interior_partition(points, tiles, n, nt):
    colors, colors_part = make_colors_part(tiles)
    npoints = n + nt*2*RADIUS
    for tile in np.ndindex(tuple(nt)):
        idx = np.array(tile)
        colors.rect[tile] = (
            idx*npoints/nt + RADIUS,
            (idx+1)*npoints/nt - 1 - RADIUS)
    return Partition.create_by_image(points, colors_part, 'rect', tiles, disjoint_incomplete)

def make_exterior_partition(points, tiles, n, nt):
    colors, colors_part = make_colors_part(tiles)
    npoints = n + nt*2*RADIUS
    for tile in np.ndindex(tuple(nt)):
        idx = np.array(tile)
        loff = (idx != 0) * RADIUS
        hoff = (idx != nt - 1) * RADIUS
        colors.rect[tile] = (
            idx*npoints/nt + loff,
            (idx+1)*npoints/nt - 1 - hoff)
    return Partition.create_by_image(points, colors_part, 'rect', tiles, disjoint_incomplete)

def clamp(val, lo, hi):
  return min(max(val, lo), hi)

def make_ghost_x_partition(points, tiles, n, nt, direction):
    colors, colors_part = make_colors_part(tiles)
    for tile in np.ndindex(tuple(nt)):
        idx = np.array(tile)
        colors.rect[tile] = (
            [clamp((idx[0]+direction)*RADIUS, 0, nt[0]*RADIUS), idx[1]*n[1]/nt[1]],
            [clamp((idx[0]+1+direction)*RADIUS - 1, -1, nt[0]*RADIUS - 1), (idx[1]+1)*n[1]/nt[1] - 1])
    kind = disjoint_complete if direction == 0 else disjoint_incomplete
    return Partition.create_by_image(points, colors_part, 'rect', tiles, kind)

def make_ghost_y_partition(points, tiles, n, nt, direction):
    colors, colors_part = make_colors_part(tiles)
    for tile in np.ndindex(tuple(nt)):
        idx = np.array(tile)
        colors.rect[tile] = (
            [idx[0]*n[0]/nt[0], clamp((idx[1]+direction)*RADIUS, 0, nt[1]*RADIUS)],
            [(idx[0]+1)*n[0]/nt[0] - 1, clamp((idx[1]+1+direction)*RADIUS - 1, -1, nt[1]*RADIUS - 1)])
    kind = disjoint_complete if direction == 0 else disjoint_incomplete
    return Partition.create_by_image(points, colors_part, 'rect', tiles, kind)

stencil = legion.extern_task(
    task_id=10001,
    argument_types=[Region, Region, Region, Region, Region, Region, legion.bool_],
    privileges=[RW, N, R('input'), R('input'), R('input'), R('input')],
    return_type=legion.void,
    calling_convention='regent')

increment = legion.extern_task(
    task_id=10002,
    argument_types=[Region, Region, Region, Region, Region, Region, legion.bool_],
    privileges=[RW('input'), N, RW('input'), RW('input'), RW('input'), RW('input')],
    return_type=legion.void,
    calling_convention='regent')

check = legion.extern_task(
    task_id=10003,
    argument_types=[Region, Region, legion.int64, legion.int64],
    privileges=[R, N],
    return_type=legion.void,
    calling_convention='regent')

@task(task_id=2) # , inner=True
def main():
    conf = parse_args(legion.input_args(True))

    nbloated = np.array([conf.nx, conf.ny])
    nt = np.array([conf.ntx, conf.nty])
    init = conf.init

    n = nbloated - 2*RADIUS
    assert np.all(n >= nt), "grid too small"

    grid = Ispace.create(n + nt*2*RADIUS)
    tiles = Ispace.create(nt)

    point = Fspace.create(OrderedDict([
        ('input', DTYPE),
        ('output', DTYPE),
    ]))

    points = Region.create(grid, point)

    private = make_private_partition(points, tiles, n, nt)
    interior = make_interior_partition(points, tiles, n, nt)
    exterior = make_exterior_partition(points, tiles, n, nt)

    xm = Region.create([nt[0]*RADIUS, n[1]], point)
    xp = Region.create([nt[0]*RADIUS, n[1]], point)
    ym = Region.create([n[0], nt[1]*RADIUS], point)
    yp = Region.create([n[0], nt[1]*RADIUS], point)
    pxm_in = make_ghost_x_partition(xm, tiles, n, nt, -1)
    pxp_in = make_ghost_x_partition(xp, tiles, n, nt,  1)
    pym_in = make_ghost_y_partition(ym, tiles, n, nt, -1)
    pyp_in = make_ghost_y_partition(yp, tiles, n, nt,  1)
    pxm_out = make_ghost_x_partition(xm, tiles, n, nt, 0)
    pxp_out = make_ghost_x_partition(xp, tiles, n, nt, 0)
    pym_out = make_ghost_y_partition(ym, tiles, n, nt, 0)
    pyp_out = make_ghost_y_partition(yp, tiles, n, nt, 0)

    init = conf.init

    for r in [points, xm, xp, ym, yp]:
        for f in ['input', 'output']:
            legion.fill(r, f, init)

    tsteps = conf.tsteps + 2 * conf.tprune
    tprune = conf.tprune

    for t in range(tsteps):
        if t == tprune:
            legion.execution_fence(block=True)
            start_time = legion.c.legion_get_current_time_in_nanos()
        for i in IndexLaunch(tiles):
            stencil(private[i], interior[i], pxm_in[i], pxp_in[i], pym_in[i], pyp_in[i], False) # t == tprune)
        for i in IndexLaunch(tiles):
            increment(private[i], exterior[i], pxm_out[i], pxp_out[i], pym_out[i], pyp_out[i], False) # t == tsteps - tprune - 1)
        if t == tsteps - tprune - 1:
            legion.execution_fence(block=True)
            stop_time = legion.c.legion_get_current_time_in_nanos()

    for i in IndexLaunch(tiles):
        check(private[i], interior[i], tsteps, init)

    print_once('ELAPSED TIME = %7.3f s' % ((stop_time - start_time)/1e9))

if __name__ == '__legion_main__':
    main()
