#!/usr/bin/env python

# Copyright 2020 Stanford University
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

from __future__ import print_function

import pygion
from pygion import index_launch, task, Domain, ID, IndexLaunch, R, Region, Partition

from typing import cast, Callable

import subprocess
import petra as pt

program = pt.Program("module")

# Global variables
from pygion import _max_dim
LEGION_MAX_DIM = _max_dim
MAX_DOMAIN_DIM = 2 * LEGION_MAX_DIM
DIM = 1


# Define types:
legion_region_tree_id_t = pt.Int32_t  # unsigned int
legion_index_partition_id_t = pt.Int32_t  # unsigned int
legion_index_tree_id_t = pt.Int32_t  # unsigned int
legion_type_tag_t = pt.Int32_t  # unsigned int
legion_field_space_id_t = pt.Int32_t  # unsigned int
coord_t = pt.Int64_t  # long long
legion_index_space_id_t = pt.Int32_t  # unsigned int
realm_id_t = pt.Int64_t  # unsigned long long

# Add void type in Types
legion_runtime_t = pt.StructType({"impl": pt.PointerType(pt.Int64_t)})

legion_index_partition_t = pt.StructType(
    {
        "id": legion_index_partition_id_t,
        "tid": legion_index_tree_id_t,
        "type_tag": legion_type_tag_t,
    }
)

legion_field_space_t = pt.StructType({"id": legion_field_space_id_t})

legion_logical_partition_t = pt.StructType(
    {
        "tree_id": legion_region_tree_id_t,
        "index_partition": legion_index_partition_t,
        "field_space": legion_field_space_t,
    }
)

legion_domain_point_t = pt.StructType(
    {"dim": pt.Int32_t, "point_data": pt.ArrayType(coord_t, LEGION_MAX_DIM)}
)

legion_domain_t = pt.StructType(
    {
        "is_id": realm_id_t,
        "dim": pt.Int32_t,
        "rect_data": pt.ArrayType(coord_t, MAX_DOMAIN_DIM),
    }
)

legion_point_1d_array = pt.ArrayType(coord_t, DIM)

legion_point_1d_t = pt.StructType({"x": legion_point_1d_array})

legion_index_space_t = pt.StructType(
    {
        "id": legion_index_space_id_t,
        "tid": legion_index_tree_id_t,
        "type_tag": legion_type_tag_t,
    }
)

legion_logical_region_t = pt.StructType(
    {
        "tree_id": legion_region_tree_id_t,
        "index_space": legion_index_space_t,
        "field_space": legion_field_space_t,
    }
)

# Define functions:
program.add_func_decl(
    "legion_domain_point_get_point_1d", (legion_domain_point_t,), legion_point_1d_t
)
program.add_func_decl(
    "legion_domain_point_from_point_1d", (legion_point_1d_t,), legion_domain_point_t
)
program.add_func_decl(
    "legion_logical_partition_get_logical_subregion_by_color_domain_point",
    (legion_runtime_t, legion_logical_partition_t, legion_domain_point_t,),
    legion_logical_region_t,
)

# Define variables:
runtime_ptr = pt.Symbol(legion_runtime_t, "runtime_ptr")
parent_ptr = pt.Symbol(pt.PointerType(legion_logical_partition_t), "parent_ptr")
point_ptr = pt.Symbol(pt.PointerType(legion_domain_point_t), "point_ptr")
domain_ptr = pt.Symbol(pt.PointerType(legion_domain_t), "domain_ptr")
point1d = pt.Symbol(legion_point_1d_t, "point1d")
point1d_x = pt.Symbol(legion_point_1d_array, "point1d_x")
x = pt.Symbol(coord_t, "x")
x_plus_1 = pt.Symbol(coord_t, "x_plus_1")
point1d_x_plus_1 = pt.Symbol(legion_point_1d_t, "point1d_x_plus_1")
domain_point_x_plus_1 = pt.Symbol(legion_domain_point_t, "domain_point_x_plus_1")
result = pt.Symbol(legion_logical_region_t, "result")
point1d_x_plus_1_x = pt.Symbol(pt.ArrayType(coord_t, DIM), "point1d_x_plus_1_x")

program.add_func(
    "proj_functor",
    (runtime_ptr, parent_ptr, point_ptr, domain_ptr,),
    legion_logical_region_t,
    pt.Block(
        [
            pt.DefineVar(
                point1d, pt.Call("legion_domain_point_get_point_1d", [pt.Deref(pt.Var(point_ptr))])
            ),
            pt.DefineVar(point1d_x, pt.GetElement(pt.Var(point1d), name="x"),),
            pt.DefineVar(x, pt.GetElement(pt.Var(point1d_x), idx=0)),
            pt.DefineVar(x_plus_1, pt.Add(pt.Var(x), pt.Int64(1))),
            pt.DefineVar(point1d_x_plus_1_x),
            pt.Assign(
                pt.Var(point1d_x_plus_1_x),
                pt.SetElement(pt.Var(point1d_x_plus_1_x), pt.Var(x_plus_1), 0),
            ),
            pt.DefineVar(point1d_x_plus_1),
            pt.Assign(
                pt.Var(point1d_x_plus_1),
                pt.SetElement(
                    pt.Var(point1d_x_plus_1), pt.Var(point1d_x_plus_1_x), name="x"
                ),
            ),
            pt.DefineVar(
                domain_point_x_plus_1,
                pt.Call(
                    "legion_domain_point_from_point_1d", [pt.Var(point1d_x_plus_1)]
                ),
            ),
            pt.DefineVar(
                result,
                pt.Call(
                    "legion_logical_partition_get_logical_subregion_by_color_domain_point",
                    [pt.Var(runtime_ptr), pt.Deref(pt.Var(parent_ptr)), pt.Var(domain_point_x_plus_1)],
                ),
            ),
            pt.Return(pt.Var(result)),
        ]
    ),
    attributes=(None, "byval", "byval", "byval"),
)


engine = program.compile()

proj_functor = engine.get_function_address("proj_functor")
print(proj_functor)

from legion_cffi import ffi, lib as c
from pygion import _my

c.legion_runtime_register_projection_functor(
    _my.ctx.runtime,
    100,
    False,
    1,
    ffi.NULL,
    ffi.cast("legion_projection_functor_logical_partition_t", proj_functor))

@task(privileges=[R])
def hello(R, i):
    print("hello from point %s (region %s)" % (i, R.ispace.bounds))

@task
def main():
    R = Region([10], {'x': pygion.float64})
    P = Partition.equal(R, [4])
    pygion.fill(R, 'x', 0)

    for i in IndexLaunch([4]):
        hello(P[i], i)

if __name__ == '__main__':
    main()
