-- Copyright 2024 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- This version relies on the new parallelizer, which has not been merged.

-- runs-with:
-- [["-fparallelize", "0"]]

import "regent"

-------------------------------------------------------------------------------
-- imports
-------------------------------------------------------------------------------

local c = regentlib.c
local std = terralib.includec("stdlib.h")
local cstring = terralib.includec("string.h")
local cmath = terralib.includecstring [[
#include <math.h>
#define DBL_EPSILON 2.2204460492503131e-16
]]

-- import some Legion C APIs to global namespace
local coloring_t            = c.legion_point_coloring_t
local coloring_create       = c.legion_point_coloring_create
local coloring_destroy      = c.legion_point_coloring_destroy
local coloring_add_range    = c.legion_point_coloring_add_range
local ptr_t                 = c.legion_ptr_t

-- utility functions
local abs = regentlib.fabs(double)
local sqrt = regentlib.sqrt(double)

-------------------------------------------------------------------------------
-- build mapper
-------------------------------------------------------------------------------

local launcher = require("std/launcher")
local cmapper = launcher.build_library("miniaero_sequential")

-------------------------------------------------------------------------------
-- constants
-------------------------------------------------------------------------------

local BC_INTERIOR = 0
local BC_TANGENT = 1
local BC_EXTRAPOLATE = 2
local BC_INFLOW = 3
local BC_NOSLIP = 4
local BC_CATEGORY_COUNT = 5

local X = 0
local Y = 1
local Z = 2

-------------------------------------------------------------------------------
-- field spaces & macros
-------------------------------------------------------------------------------

Solution = double[5]

__demand(__inline)
task ComputePrimitives(U : Solution) : Solution
  var gamma : double = 1.4
  var Rgas : double  = 287.05

  var r  = U[0]
  var ri = 1.0 / r
  var u  = U[1] * ri
  var v  = U[2] * ri
  var w  = U[3] * ri
  var k  = 0.5 * (u * u + v * v + w * w)
  var e  = U[4] * ri - k
  var T  = e * (gamma - 1.0) / Rgas

  var V : Solution
  V[0] = r
  V[1] = u
  V[2] = v
  V[3] = w
  V[4] = T
  return V
end

__demand(__inline)
task ComputePressure(V : Solution) : double
  var Rgas = 287.05
  var rho = V[0]
  var T = V[4]

  return rho * Rgas * T
end

__demand(__inline)
task ComputeEnthalpy(V : Solution) : double
  var Cp = 1004.0
  var T = V[4]
  return Cp * T
end

__demand(__inline)
task ComputeViscosity(temperature : double)
  var sutherland_0 : double = 1.458e-6
  var sutherland_1 : double = 110.4
  return sutherland_0 * temperature * sqrt(temperature) / (temperature + sutherland_1)
end

__demand(__inline)
task ComputeThermalConductivity(viscosity : double)
  var Pr : double = 0.71
  var Cp : double = 1006.0
  return viscosity * Cp / Pr
end

Vec3 = double[3]

Gradient = double[15]

struct CellConnect {
  _0 : int,
  _1 : int,
  _2 : int,
  _3 : int,
  _4 : int,
  _5 : int,
  _6 : int,
  _7 : int,
}

fspace Cell {
  solution_n    : Solution,
  solution_np1  : Solution,
  solution_temp : Solution,
  residual      : Solution,
  cell_flux     : Solution,
  stencil_min   : Solution,
  stencil_max   : Solution,
  limiter       : Solution,
  cell_gradients : Gradient,
  cell_connectivity : CellConnect,
  cell_centroid : Vec3,
  volume : double,
}

struct FaceConnect {
  _0 : int,
  _1 : int,
  _2 : int,
  _3 : int,
}

fspace Face {
  -- flattened the face_cell_conn field
  left  : ptr, -- boundary cells only have this pointer set
  right : ptr,
  face_connectivity : FaceConnect,
  face_centroid : Vec3,
  area     : Vec3,
  tangent  : Vec3,
  binormal : Vec3,
}

terra Vec3Cross(a : Vec3, b : Vec3)
  var c : Vec3
  c[0] = a[1] * b[2] - b[1] * a[2]
  c[1] = -a[0] * b[2] + b[0] * a[2]
  c[2] = a[0] * b[1] - b[0] * a[1]
  return c
end

terra Vec3Average(a : Vec3, b : Vec3)
  var c : Vec3
  c[0] = 0.5 * (a[0] + b[0])
  c[1] = 0.5 * (a[1] + b[1])
  c[2] = 0.5 * (a[2] + b[2])
  return c
end

struct Interface {
  problem_type : int,
  num_blocks : int,
  interval_x : int,
  interval_y : int,
  interval_z : int,

  output_frequency : int,
  time_steps : int,
  prune : int,

  length_x : double,
  length_y : double,
  length_z : double,

  ramp_angle : double,
  dt : double,

  wait : bool,
  viscous : bool,
  second_order_space : bool,
  output_results : bool,
  memoize : bool,
}

struct RungaKutta4 {
  stages_ : int,
  alpha_ : double[4],
  beta_ : double[4],
}

-------------------------------------------------------------------------------
-- option parser
-------------------------------------------------------------------------------

terra default_options() : Interface
  return Interface {
    problem_type = 0,
    num_blocks = 2,
    interval_x = 10,
    interval_y = 10,
    interval_z = 10,

    output_frequency = 1,
    time_steps = 1,
    prune = 0,

    length_x = 4.0,
    length_y = 1.0,
    length_z = 0.5,

    ramp_angle = 25.0,
    dt = 5.0e-8,

    wait = false,
    viscous = false,
    second_order_space = false,
    output_results = false,
  }
end
default_options.replicable = true

terra parse_options(interface : Interface) : Interface
  var args = c.legion_runtime_get_input_args()
  for i = 0, args.argc do
    if cstring.strcmp(args.argv[i], "-problem_type") == 0 then
      i = i + 1
      interface.problem_type = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-blocks") == 0 then
      i = i + 1
      interface.num_blocks = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-mesh") == 0 then
      i = i + 1
      var mesh_len = cstring.strlen(args.argv[i])
      var mesh : rawstring = [rawstring](c.malloc(sizeof(int8) * (mesh_len + 1)))
      cstring.strcpy(mesh, args.argv[i])
      var token : rawstring = cstring.strtok(mesh, "x")
      interface.interval_x = std.atoi(token)
      token = cstring.strtok([rawstring](0), "x")
      interface.interval_y = std.atoi(token)
      token = cstring.strtok([rawstring](0), "x")
      interface.interval_z = std.atoi(token)
      c.free(mesh)
    elseif cstring.strcmp(args.argv[i], "-output_frequency") == 0 then
      i = i + 1
      interface.output_frequency = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-time_steps") == 0 then
      i = i + 1
      interface.time_steps = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-prune") == 0 then
      i = i + 1
      interface.prune = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-x_length") == 0 then
      i = i + 1
      interface.length_x = std.atof(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-y_length") == 0 then
      i = i + 1
      interface.length_y = std.atof(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-z_length") == 0 then
      i = i + 1
      interface.length_z = std.atof(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-ramp") == 0 then
      i = i + 1
      interface.ramp_angle = std.atof(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-dt") == 0 then
      i = i + 1
      interface.dt = std.atof(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-wait") == 0 then
      interface.wait = true
    elseif cstring.strcmp(args.argv[i], "-viscous") == 0 then
      interface.viscous = true
    elseif cstring.strcmp(args.argv[i], "-second_order") == 0 then
      interface.second_order_space = true
    elseif cstring.strcmp(args.argv[i], "-output_results") == 0 then
      interface.output_results = true
    end
  end

  return interface
end
parse_options.replicable = true

task print_options(color : int, interface : Interface)
  if color ~= 0 then return end
  if interface.problem_type == 0 then
    c.printf("problem_type = SOD\n")
  elseif interface.problem_type == 1 then
    c.printf("problem_type = Viscous_Flat_Plate\n")
  elseif interface.problem_type == 2 then
    c.printf("problem_type = Inviscid_Ramp\n")
  else
    regentlib.assert(false, "invalid problem type")
  end

  c.printf("num_blocks = %d\n", interface.num_blocks)
  c.printf("interval_x = %d\n", interface.interval_x)
  c.printf("interval_y = %d\n", interface.interval_y)
  c.printf("interval_z = %d\n", interface.interval_z)
  c.printf("output_frequency = %d\n", interface.output_frequency)
  c.printf("time_steps = %d\n", interface.time_steps)
  c.printf("length_x = %f\n", interface.length_x)
  c.printf("length_y = %f\n", interface.length_y)
  c.printf("length_z = %f\n", interface.length_z)
  c.printf("ramp_angle = %f\n", interface.ramp_angle)
  c.printf("dt = %g\n", interface.dt)
  if interface.wait then
    c.printf("wait = true\n")
  else
    c.printf("wait = false\n")
  end
  if interface.viscous then
    c.printf("viscous = true\n")
  else
    c.printf("viscous = false\n")
  end
  if interface.second_order_space then
    c.printf("second_order_space = true\n")
  else
    c.printf("second_order_space = false\n")
  end
  if interface.output_results then
    c.printf("output_results = true\n")
  else
    c.printf("output_results = false\n")
  end
end

-------------------------------------------------------------------------------
-- mesh topology
-------------------------------------------------------------------------------

fspace Block {
  id_ : int,
  location_ : int[3],
  size_ : int[3],
  offset_ : int[3],
  cellOffset_ : int,
  faceOffset_ : int,
}

terra block_array(num_blocks : int) : &Block
  return [&Block](std.malloc(num_blocks * sizeof(Block)))
end

terra block_set_size(block : &Block, i : int, j : int, k : int)
  block.size_[X] = i
  block.size_[Y] = j
  block.size_[Z] = k
end

terra block_set_location(block : &Block, i : int, j : int, k : int)
  block.location_[X] = i
  block.location_[Y] = j
  block.location_[Z] = k
end

terra block_set_offset(block : &Block, i : int, j : int, k : int)
  block.offset_[X] = i
  block.offset_[Y] = j
  block.offset_[Z] = k
end

terra block_cell_count(block : &Block) : int
  return block.size_[X] * block.size_[Y] * block.size_[Z]
end

terra block_face_count(block : &Block) : int
  var face_count = 3 * block_cell_count(block)
  if block.location_[X] == 0 then
    face_count = face_count + (block.size_[Y] * block.size_[Z])
  end
  if block.location_[Y] == 0 then
    face_count = face_count + (block.size_[Z] * block.size_[X])
  end
  if block.location_[Z] == 0 then
    face_count = face_count + (block.size_[X] * block.size_[Y])
  end
  return face_count
end

terra block_global_cell_id(block : &Block,
                           loc_i : int, loc_j : int, loc_k : int) : int
  return block.cellOffset_ +
         loc_i +
         loc_j * block.size_[X] +
         loc_k * block.size_[X] * block.size_[Y]
end

terra block_global_index(block       : &Block,
                         XYZ         : int,
                         local_index : int)
  return local_index + block.offset_[XYZ]
end

terra block_print(block : &Block)
  c.printf(["Block id: %4d\tLocation: %d, %d, %d,\tSize: %dx%dx%d"..
            "\tOffset: %3d, %3d, %3d\tCells: %8d\tCell Offset: %10d"..
            "\tFace Offset: %10d\n"],
            block.id_,
            block.location_[X], block.location_[Y], block.location_[Z],
            block.size_[X], block.size_[Y], block.size_[Z],
            block.offset_[X], block.offset_[Y], block.offset_[Z],
            block_cell_count(block), block.cellOffset_, block.faceOffset_)
end

local block_null = `[&Block](0)

struct MeshTopology {
  num_blocks_ : int,
  problem_type_ : int,
  nblock_ : int[3],
  global_nx_ : int,
  global_ny_ : int,
  global_nz_ : int,
  blocks_ : &Block, -- has num_blocks elmts, not safe on general llr
  length_ : double[3],
  ramp_angle_ : double,
}

terra mesh_compute_mesh_arrangement(mesh : &MeshTopology)
  var blocks_left = mesh.num_blocks_
  var nx_temp = mesh.global_nx_
  var ny_temp = mesh.global_ny_
  var nz_temp = mesh.global_nz_

  mesh.nblock_[X] = 1
  mesh.nblock_[Y] = 1
  mesh.nblock_[Z] = 1

  while blocks_left ~= 1 do
    -- Check if number of processors is power of 2
    regentlib.assert(blocks_left % 2 == 0, "ERROR: number of blocks must be a power of 2.")
    blocks_left = blocks_left / 2

    var max_size = nx_temp
    if ny_temp > max_size then max_size = ny_temp end
    if nz_temp > max_size then max_size = nz_temp end

    if nx_temp == max_size then
      mesh.nblock_[X] = mesh.nblock_[X] * 2
      nx_temp = mesh.global_nx_ / mesh.nblock_[X]
    elseif(ny_temp == max_size) then
      mesh.nblock_[Y] = mesh.nblock_[Y] * 2
      ny_temp = mesh.global_ny_ / mesh.nblock_[Y]
    elseif(nz_temp == max_size) then
      mesh.nblock_[Z] = mesh.nblock_[Z] * 2
      nz_temp = mesh.global_nz_ / mesh.nblock_[Z]
    end
  end
end

terra mesh_block_id(mesh : &MeshTopology, i : int, j : int, k : int) : int
  return i +
         mesh.nblock_[X] * j +
         mesh.nblock_[X] * mesh.nblock_[Y] * k
end

terra mesh_node_id(mesh : &MeshTopology, i : int, j : int, k : int) : int
  return i +
         (mesh.global_nx_ + 1) *
         (j + k * (mesh.global_ny_ + 1))
end

terra mesh_global_cell_count(mesh : &MeshTopology) : int
  return mesh.global_nx_ * mesh.global_ny_ * mesh.global_nz_
end
mesh_global_cell_count.replicable = true

terra mesh_global_face_count(mesh : &MeshTopology) : int
  return (mesh.global_nx_+1) *  mesh.global_ny_    *  mesh.global_nz_
       +  mesh.global_nx_    * (mesh.global_ny_+1) *  mesh.global_nz_
       +  mesh.global_nx_    *  mesh.global_ny_    * (mesh.global_nz_+1)
end
mesh_global_face_count.replicable = true

terra mesh_compute_block_properties(mesh : &MeshTopology)
  var cells_per_x : int = mesh.global_nx_ / mesh.nblock_[X]
  var extra_x     : int = mesh.global_nx_ % mesh.nblock_[X]
  var cells_per_y : int = mesh.global_ny_ / mesh.nblock_[Y]
  var extra_y     : int = mesh.global_ny_ % mesh.nblock_[Y]
  var cells_per_z : int = mesh.global_nz_ / mesh.nblock_[Z]
  var extra_z     : int = mesh.global_nz_ % mesh.nblock_[Z]

  var iblk : int = 0
  var start_x : int = 0
  var start_y : int = 0
  var start_z : int = 0

  var cell_count : int = 0
  var face_count : int = 0
  for k = 0, mesh.nblock_[Z] do
    var size_z = cells_per_z if k < extra_z then size_z = size_z + 1 end
    start_y = 0
    for j = 0, mesh.nblock_[Y] do
      var size_y = cells_per_y if j < extra_y then size_y = size_y + 1 end
      start_x = 0
      for i = 0, mesh.nblock_[X] do
        var size_x = cells_per_x if i < extra_x then size_x = size_x + 1 end
        var block : &Block = &mesh.blocks_[iblk]
        block.id_ = mesh_block_id(mesh, i, j, k)
        regentlib.assert(block.id_ == iblk, "numbering is inconsistent")
        block_set_size(block, size_x, size_y, size_z)
        block_set_location(block, i, j, k)
        block_set_offset(block, start_x, start_y, start_z)
        block.cellOffset_ = cell_count
        block.faceOffset_ = face_count
        start_x = start_x + size_x
        cell_count = cell_count + block_cell_count(block)
        face_count = face_count + block_face_count(block)
        iblk = iblk + 1
      end
      start_y = start_y + size_y
    end
    start_z = start_z + size_z
  end
  regentlib.assert(cell_count == mesh_global_cell_count(mesh),
                   "ERROR: cell count does not match")
  regentlib.assert(face_count == mesh_global_face_count(mesh),
                   "ERROR: face count does not match")
end

terra mesh_init(num_blocks : int,
                global_num_x : int,
                global_num_y : int,
                global_num_z : int,
                len_x : double,
                len_y : double,
                len_z : double,
                ramp_angle : double,
                problem_type : int) : &MeshTopology

  var mesh : &MeshTopology = [&MeshTopology](std.malloc(sizeof(MeshTopology)))
  mesh.num_blocks_ = num_blocks
  mesh.problem_type_ = problem_type
  mesh.nblock_[X] = 0 mesh.nblock_[Y] = 0 mesh.nblock_[Z] = 0
  mesh.length_[X] = len_x mesh.length_[Y] = len_y mesh.length_[Z] = len_z
  mesh.global_nx_ = global_num_x
  mesh.global_ny_ = global_num_y
  mesh.global_nz_ = global_num_z
  mesh.ramp_angle_ = ramp_angle
  mesh.blocks_ = block_array(num_blocks)
  mesh_compute_mesh_arrangement(mesh)
  mesh_compute_block_properties(mesh)
  return mesh
end
mesh_init.replicable = true

terra mesh_get_block_by_id(mesh : &MeshTopology, block_id : int) : &Block
  return &mesh.blocks_[block_id]
end

terra mesh_get_block_by_coord(mesh : &MeshTopology,
                              i : int,
                              j : int,
                              k : int) : &Block
  if i >= 0 and i < mesh.nblock_[X] and
     j >= 0 and j < mesh.nblock_[Y] and
     k >= 0 and k < mesh.nblock_[Z] then
    return &mesh.blocks_[mesh_block_id(mesh, i, j, k)]
  else
    return [block_null]
  end
end

terra mesh_node_id_to_ijk(mesh : &MeshTopology,
                          node_id : int,
                          i : &int,
                          j : &int,
                          k : &int)
  @k = node_id / ((mesh.global_nx_ + 1) * (mesh.global_ny_ + 1))
  node_id = node_id % ((mesh.global_nx_ + 1) * (mesh.global_ny_ + 1))
  @j = node_id / (mesh.global_nx_ + 1)
  node_id = node_id % (mesh.global_nx_ + 1)
  @i = node_id
end

terra mesh_node_coordinate(mesh : &MeshTopology,
                           node_id : int,
                           x : &double,
                           y : &double,
                           z : &double)
  var i : int
  var j : int
  var k : int
  mesh_node_id_to_ijk(mesh, node_id, &i, &j, &k)

  @x = [double](i) / mesh.global_nx_ * mesh.length_[X]
  @z = [double](k) / mesh.global_nz_ * mesh.length_[Z]

  if @x <= mesh.length_[X] / 2.0 then
    @y = [double](j) / mesh.global_ny_ * mesh.length_[Y]
  else
    var y_ramp =
      (@x - (mesh.length_[X] / 2.0)) *
      cmath.tan(mesh.ramp_angle_ * cmath.M_PI / 180.0)
    var ly_scaled = mesh.length_[Y] - y_ramp
    @y = y_ramp + [double](j) / mesh.global_ny_ * ly_scaled
  end
end

terra mesh_node_coordinate_array(mesh : &MeshTopology,
                                 coord : &double,
                                 node_id : int)
  var x : double
  var y : double
  var z : double
  mesh_node_coordinate(mesh, node_id, &x, &y, &z)
  coord[0] = x
  coord[1] = y
  coord[2] = z
end

terra mesh_get_hex_nodal_coordinates(mesh : MeshTopology,
                                     connect : CellConnect) : double[24]
  var idx = 0
  var coord : double[24]
  mesh_node_coordinate_array(&mesh, &coord[idx], connect._0) idx = idx + 3
  mesh_node_coordinate_array(&mesh, &coord[idx], connect._1) idx = idx + 3
  mesh_node_coordinate_array(&mesh, &coord[idx], connect._2) idx = idx + 3
  mesh_node_coordinate_array(&mesh, &coord[idx], connect._3) idx = idx + 3
  mesh_node_coordinate_array(&mesh, &coord[idx], connect._4) idx = idx + 3
  mesh_node_coordinate_array(&mesh, &coord[idx], connect._5) idx = idx + 3
  mesh_node_coordinate_array(&mesh, &coord[idx], connect._6) idx = idx + 3
  mesh_node_coordinate_array(&mesh, &coord[idx], connect._7)
  return coord
end

terra mesh_hex_cell_volume(coord : &double)
  var gradop : double[8]
  var x1 = coord[ 0]
  var x2 = coord[ 3]
  var x3 = coord[ 6]
  var x4 = coord[ 9]
  var x5 = coord[12]
  var x6 = coord[15]
  var x7 = coord[18]
  var x8 = coord[21]

  var y1 = coord[ 1]
  var y2 = coord[ 4]
  var y3 = coord[ 7]
  var y4 = coord[10]
  var y5 = coord[13]
  var y6 = coord[16]
  var y7 = coord[19]
  var y8 = coord[22]

  var z1 = coord[ 2]
  var z2 = coord[ 5]
  var z3 = coord[ 8]
  var z4 = coord[11]
  var z5 = coord[14]
  var z6 = coord[17]
  var z7 = coord[20]
  var z8 = coord[23]

  var z24 = z2 - z4
  var z52 = z5 - z2
  var z45 = z4 - z5
  gradop[0] = ( y2*(z6-z3-z45) + y3*z24 + y4*(z3-z8-z52) +
                y5*(z8-z6-z24) + y6*z52 + y8*z45 ) / 12.
  var z31 = z3 - z1
  var z63 = z6 - z3
  var z16 = z1 - z6
  gradop[1] = ( y3*(z7-z4-z16) + y4*z31 + y1*(z4-z5-z63) +
                y6*(z5-z7-z31) + y7*z63 + y5*z16 ) / 12.
  var z42 = z4 - z2
  var z74 = z7 - z4
  var z27 = z2 - z7
  gradop[2] = ( y4*(z8-z1-z27) + y1*z42 + y2*(z1-z6-z74) +
                y7*(z6-z8-z42) + y8*z74 + y6*z27 ) / 12.
  var z13 = z1 - z3
  var z81 = z8 - z1
  var z38 = z3 - z8
  gradop[3] = ( y1*(z5-z2-z38) + y2*z13 + y3*(z2-z7-z81) +
                y8*(z7-z5-z13) + y5*z81 + y7*z38 ) / 12.
  var z86 = z8 - z6
  var z18 = z1 - z8
  var z61 = z6 - z1
  gradop[4] = ( y8*(z4-z7-z61) + y7*z86 + y6*(z7-z2-z18) +
                y1*(z2-z4-z86) + y4*z18 + y2*z61 ) / 12.
  var z57 = z5 - z7
  var z25 = z2 - z5
  var z72 = z7 - z2
  gradop[5] = ( y5*(z1-z8-z72) + y8*z57 + y7*(z8-z3-z25) +
                y2*(z3-z1-z57) + y1*z25 + y3*z72 ) / 12.
  var z68 = z6 - z8
  var z36 = z3 - z6
  var z83 = z8 - z3
  gradop[6] = ( y6*(z2-z5-z83) + y5*z68 + y8*(z5-z4-z36) +
                y3*(z4-z2-z68) + y2*z36 + y4*z83 ) / 12.
  var z75 = z7 - z5
  var z47 = z4 - z7
  var z54 = z5 - z4
  gradop[7] = ( y7*(z3-z6-z54) + y6*z75 + y5*(z6-z1-z47) +
                y4*(z1-z3-z75) + y3*z47 + y1*z54 ) / 12.

  var volume =
    x1 * gradop[0] + x2 * gradop[1] + x3 * gradop[2] + x4 * gradop[3] +
    x5 * gradop[4] + x6 * gradop[5] + x7 * gradop[6] + x8 * gradop[7]
  return volume
end

terra mesh_hex_cell_centroid(coord : &double) : Vec3
  var v : Vec3
  v[0] = coord[0]
  v[1] = coord[1]
  v[2] = coord[2]
  for i = 1, 8 do
    v[0] = v[0] + coord[3 * i + 0]
    v[1] = v[1] + coord[3 * i + 1]
    v[2] = v[2] + coord[3 * i + 2]
  end
  v[0] = v[0] / 8.0
  v[1] = v[1] / 8.0
  v[2] = v[2] / 8.0
  return v
end

terra mesh_quad_face_centroid(mesh : MeshTopology,
                              connect : FaceConnect)
  var x_sum = 0.0
  var y_sum = 0.0
  var z_sum = 0.0
  var x : double
  var y : double
  var z : double
  do
    mesh_node_coordinate(&mesh, connect._0, &x, &y, &z)
    x_sum = x_sum + x
    y_sum = y_sum + y
    z_sum = z_sum + z
    mesh_node_coordinate(&mesh, connect._1, &x, &y, &z)
    x_sum = x_sum + x
    y_sum = y_sum + y
    z_sum = z_sum + z
    mesh_node_coordinate(&mesh, connect._2, &x, &y, &z)
    x_sum = x_sum + x
    y_sum = y_sum + y
    z_sum = z_sum + z
    mesh_node_coordinate(&mesh, connect._3, &x, &y, &z)
    x_sum = x_sum + x
    y_sum = y_sum + y
    z_sum = z_sum + z
  end
  var v : Vec3
  v[0] = x_sum / 4.0
  v[1] = y_sum / 4.0
  v[2] = z_sum / 4.0
  return v
end

terra mesh_print(mesh : &MeshTopology)
  c.printf("\"Structured\" Unstructured mesh of size %dx%dx%d\n",
           mesh.global_nx_, mesh.global_ny_, mesh.global_nz_)
  c.printf("\tMesh Size: %d cells, %d faces (duplicated at block boundaries).\n",
           mesh_global_cell_count(mesh), mesh_global_face_count(mesh))
  c.printf("\tMesh Extent: %f, %f, %f. With ramp angle = %f degrees.\n",
           mesh.length_[X], mesh.length_[Y], mesh.length_[Z], mesh.ramp_angle_)
  if mesh.problem_type_ == 0 then
    c.printf("\tRunning problem type: SOD\n")
  elseif mesh.problem_type_ == 1 then
    c.printf("\tRunning problem type: Viscous_Flat_Plate\n")
  elseif mesh.problem_type_ == 2 then
    c.printf("\tRunning problem type: Inviscid_Ramp\n")
  end
  c.printf("\nDecomposed into %d blocks: %dx%dx%d\n",
           mesh.num_blocks_, mesh.nblock_[X], mesh.nblock_[Y], mesh.nblock_[Z])
  for i = 0, mesh.num_blocks_ do
    block_print(&mesh.blocks_[i])
  end
end

terra mesh_deref(mesh : &MeshTopology) : MeshTopology
  return @mesh
end
mesh_deref.replicable = true

-------------------------------------------------------------------------------
-- coloring
-------------------------------------------------------------------------------

struct Colorings {
  cell_coloring : coloring_t,
  face_coloring : coloring_t,
  face_category : coloring_t,
}

terra category_name(category : int)
  if category == BC_INTERIOR then
    return "Interior"
  elseif category == BC_TANGENT then
    return "Tangent"
  elseif category == BC_EXTRAPOLATE then
    return "Extrapolate"
  elseif category == BC_INFLOW then
    return "Inflow"
  elseif category == BC_NOSLIP then
    return "Noslip"
  end
end

terra categorize_faces_for_block(mesh: &MeshTopology,
                                 block : &Block,
                                 face_category : coloring_t)
  --           type 0      | type 1      | type 2
  --           ------      | ------      | ------
  -- left      extrapolate | inflow      | inflow
  -- right     extrapolate | extrapolate | extrapolate
  -- down      tangent     | noslip      | tangent
  -- up        tangent     | extrapolate | tangent
  -- back      tangent     | tangent     | tangent
  -- front     tangent     | tangent     | tangent

  var left : int, right : int, down : int, up : int, back : int, front : int
  if mesh.problem_type_ == 0 then
    left  = BC_EXTRAPOLATE
    right = BC_EXTRAPOLATE
    down  = BC_TANGENT
    up    = BC_TANGENT
    back  = BC_TANGENT
    front = BC_TANGENT
  elseif mesh.problem_type_ == 1 then
    left  = BC_INFLOW
    right = BC_EXTRAPOLATE
    down  = BC_NOSLIP
    up    = BC_EXTRAPOLATE
    back  = BC_TANGENT
    front = BC_TANGENT
  else
    left  = BC_INFLOW
    right = BC_EXTRAPOLATE
    down  = BC_TANGENT
    up    = BC_TANGENT
    back  = BC_TANGENT
    front = BC_TANGENT
  end

  --c.printf("Face Categorization calculations for block %d\n", block.id_)

  var block_L = mesh_get_block_by_coord(mesh, block.location_[X]-1, block.location_[Y], block.location_[Z])
  var block_R = mesh_get_block_by_coord(mesh, block.location_[X]+1, block.location_[Y], block.location_[Z])

  var block_D = mesh_get_block_by_coord(mesh, block.location_[X], block.location_[Y]-1, block.location_[Z])
  var block_U = mesh_get_block_by_coord(mesh, block.location_[X], block.location_[Y]+1, block.location_[Z])

  var block_B = mesh_get_block_by_coord(mesh, block.location_[X], block.location_[Y], block.location_[Z]-1)
  var block_F = mesh_get_block_by_coord(mesh, block.location_[X], block.location_[Y], block.location_[Z]+1)

  var face_offset = block.faceOffset_

  -- 2d example of coloring:
  -- base region (6x4 cells, 7x4 x-faces, 6x5 y-faces):
  --   +-+-+-+-+-+-+
  --   | | | | | | |
  --   +-+-+-+-+-+-+
  --   | | | | | | |
  --   +-+-+-+-+-+-+
  --   | | | | | | |
  --   +-+-+-+-+-+-+
  --   | | | | | | |
  --   +-+-+-+-+-+-+
  -- cell coloring, by block id (2x2 blocks):
  --   +-+-+-+-+-+-+
  --   |2|2|2|3|3|3|
  --   +-+-+-+-+-+-+
  --   |2|2|2|3|3|3|
  --   +-+-+-+-+-+-+
  --   |0|0|0|1|1|1|
  --   +-+-+-+-+-+-+
  --   |0|0|0|1|1|1|
  --   +-+-+-+-+-+-+
  -- face coloring, by block id (2x2 blocks):
  --   +2+2+2+3+3+3+
  --   2 2 2 2 3 3 3
  --   +2+2+2+3+3+3+
  --   2 2 2 2 3 3 3
  --   +0+0+0+1+1+1+
  --   0 0 0 0 1 1 1
  --   +0+0+0+1+1+1+
  --   0 0 0 0 1 1 1
  --   +0+0+0+1+1+1+
  -- face coloring, by type (2x2 blocks):
  --   +U+U+U+U+U+U+
  --   L I I I I I R
  --   +I+I+I+I+I+I+
  --   L I I I I I R
  --   +I+I+I+I+I+I+
  --   L I I I I I R
  --   +I+I+I+I+I+I+
  --   L I I I I I R
  --   +D+D+D+D+D+D+

  -- XXX: Make sure these are kept in sync with create_face_connectivity_raw
  var blockXY = block.size_[X] * block.size_[Y]
  var blockYZ = block.size_[Y] * block.size_[Z]
  var blockZX = block.size_[Z] * block.size_[X]
  var interior_faces = 3 * block_cell_count(block)
  if block_R == [block_null] then
    interior_faces = interior_faces - blockYZ
  end
  if block_U == [block_null] then
    interior_faces = interior_faces - blockZX
  end
  if block_F == [block_null] then
    interior_faces = interior_faces - blockXY
  end

  coloring_add_range(face_category, [int1d](BC_INTERIOR),
      ptr_t { value = face_offset                      },
      ptr_t { value = face_offset + interior_faces - 1 })
  --c.printf("interior: %d -- %d\n", face_offset, face_offset + interior_faces - 1)
  face_offset = face_offset + interior_faces

  if block_L == [block_null] then
    coloring_add_range(face_category, [int1d](left),
        ptr_t { value = face_offset                      },
        ptr_t { value = face_offset + blockYZ - 1 })
    --c.printf("left: %d -- %d\n", face_offset, face_offset + blockYZ - 1)
    face_offset = face_offset + blockYZ
  end
  if block_R == [block_null] then
    coloring_add_range(face_category, [int1d](right),
        ptr_t { value = face_offset                      },
        ptr_t { value = face_offset + blockYZ - 1 })
    --c.printf("right: %d -- %d\n", face_offset, face_offset + blockYZ - 1)
    face_offset = face_offset + blockYZ
  end
  if block_D == [block_null] then
    coloring_add_range(face_category, [int1d](down),
        ptr_t { value = face_offset                      },
        ptr_t { value = face_offset + blockZX - 1 })
    --c.printf("down: %d -- %d\n", face_offset, face_offset + blockZX - 1)
    face_offset = face_offset + blockZX
  end
  if block_U == [block_null] then
    coloring_add_range(face_category, [int1d](up),
        ptr_t { value = face_offset                      },
        ptr_t { value = face_offset + blockZX - 1 })
    --c.printf("up: %d -- %d\n", face_offset, face_offset + blockZX - 1)
    face_offset = face_offset + blockZX
  end
  if block_B == [block_null] then
    coloring_add_range(face_category, [int1d](back),
        ptr_t { value = face_offset                      },
        ptr_t { value = face_offset + blockXY - 1 })
    --c.printf("back: %d -- %d\n", face_offset, face_offset + blockXY - 1)
    face_offset = face_offset + blockXY
  end
  if block_F == [block_null] then
    coloring_add_range(face_category, [int1d](front),
        ptr_t { value = face_offset                      },
        ptr_t { value = face_offset + blockXY - 1 })
    --c.printf("front: %d -- %d\n", face_offset, face_offset + blockXY - 1)
    face_offset = face_offset + blockXY
  end

  regentlib.assert(face_offset - block.faceOffset_ == block_face_count(block),
                   "ERROR: miscount in categorize_faces_for_block")
end

terra categorize_faces(mesh : &MeshTopology,
                       face_category : coloring_t)
  for i = 0, mesh.num_blocks_ do
    categorize_faces_for_block(mesh, mesh_get_block_by_id(mesh, i),
                               face_category)
  end
end

terra create_colorings(mesh : &MeshTopology) : Colorings
  var colorings : Colorings
  colorings.cell_coloring = coloring_create()
  colorings.face_coloring = coloring_create()
  colorings.face_category = coloring_create()

  for iblk = 0, mesh.num_blocks_ do
    var block = mesh_get_block_by_id(mesh, iblk)
    var start_cell = block.cellOffset_
    var end_cell   = start_cell + block_cell_count(block) - 1
    coloring_add_range(colorings.cell_coloring, [int1d](iblk),
                       ptr_t { value = start_cell },
                       ptr_t { value = end_cell })

    var start_face = block.faceOffset_
    var end_face   = start_face + block_face_count(block) - 1
    coloring_add_range(colorings.face_coloring, [int1d](iblk),
                       ptr_t { value = start_face },
                       ptr_t { value = end_face })
  end

  categorize_faces(mesh, colorings.face_category)

  return colorings
end
create_colorings.replicable = true

terra init_rk4() : RungaKutta4
  var rk4 : RungaKutta4
  rk4.stages_ = 4
  rk4.alpha_[0] = 0.0
  rk4.alpha_[1] = 1.0/2.0
  rk4.alpha_[2] = 1.0/2.0
  rk4.alpha_[3] = 1.0
  rk4.beta_[0] = 1.0/6.0
  rk4.beta_[1] = 1.0/3.0
  rk4.beta_[2] = 1.0/3.0
  rk4.beta_[3] = 1.0/6.0
  return rk4
end
init_rk4.replicable = true

-------------------------------------------------------------------------------
-- initialization
-------------------------------------------------------------------------------

task init_blocks(pmesh : &MeshTopology, blocks : region(Block))
where reads writes(blocks)
do
  for n = 0, pmesh.num_blocks_ do
    var block_ptr = dynamic_cast(ptr(Block, blocks), [ptr](n))
    var block = mesh_get_block_by_id(pmesh, n)
    block_ptr.id_         = block.id_
    block_ptr.location_   = block.location_
    block_ptr.size_       = block.size_
    block_ptr.offset_     = block.offset_
    block_ptr.cellOffset_ = block.cellOffset_
    block_ptr.faceOffset_ = block.faceOffset_
  end
  return 1
end

terra create_face_connectivity_raw(my_blk_id : int,
                                   mesh      : MeshTopology,
                                   faces     : c.legion_physical_region_t[6],
                                   faces_fields : c.legion_field_id_t[6])

  var block = mesh_get_block_by_id(&mesh, my_blk_id)

  var block_L = mesh_get_block_by_coord(&mesh, block.location_[X]-1, block.location_[Y], block.location_[Z])
  var block_R = mesh_get_block_by_coord(&mesh, block.location_[X]+1, block.location_[Y], block.location_[Z])

  var block_D = mesh_get_block_by_coord(&mesh, block.location_[X], block.location_[Y]-1, block.location_[Z])
  var block_U = mesh_get_block_by_coord(&mesh, block.location_[X], block.location_[Y]+1, block.location_[Z])

  var block_B = mesh_get_block_by_coord(&mesh, block.location_[X], block.location_[Y], block.location_[Z]-1)
  var block_F = mesh_get_block_by_coord(&mesh, block.location_[X], block.location_[Y], block.location_[Z]+1)

  var fa_left =
    c.legion_physical_region_get_field_accessor_array_1d(faces[0], faces_fields[0])
  var fa_right =
    c.legion_physical_region_get_field_accessor_array_1d(faces[1], faces_fields[1])
  var fa_connectivity : &c.legion_accessor_array_1d_t =
    [&c.legion_accessor_array_1d_t](std.malloc(4 * sizeof(c.legion_accessor_array_1d_t)))
  for i = 0, 4 do
    fa_connectivity[i] =
      c.legion_physical_region_get_field_accessor_array_1d(faces[2 + i], faces_fields[2 + i])
  end

  -- XXX: Make sure these are kept in sync with categorize_faces_for_block
  var blockXY = block.size_[X] * block.size_[Y]
  var blockYZ = block.size_[Y] * block.size_[Z]
  var blockZX = block.size_[Z] * block.size_[X]
  var interior_faces = 3 * block_cell_count(block)
  if block_R == [block_null] then
    interior_faces = interior_faces - blockYZ
  end
  if block_U == [block_null] then
    interior_faces = interior_faces - blockZX
  end
  if block_F == [block_null] then
    interior_faces = interior_faces - blockXY
  end

  var first_x_face = 1
  if block_L == [block_null] then
    first_x_face = 0
  end
  var first_y_face = 1
  if block_D == [block_null] then
    first_y_face = 0
  end
  var first_z_face = 1
  if block_B == [block_null] then
    first_z_face = 0
  end

  var interior_face_offset = 0
  var left_face_offset  = interior_faces
  var right_face_offset = interior_faces
  var down_face_offset  = interior_faces
  var up_face_offset    = interior_faces
  var back_face_offset  = interior_faces
  var front_face_offset = interior_faces
  var face_count        = interior_faces

  if block_L == [block_null] then
    right_face_offset = right_face_offset + blockYZ
    down_face_offset  = down_face_offset  + blockYZ
    up_face_offset    = up_face_offset    + blockYZ
    back_face_offset  = back_face_offset  + blockYZ
    front_face_offset = front_face_offset + blockYZ
    face_count        = face_count        + blockYZ
  end
  if block_R == [block_null] then
    down_face_offset  = down_face_offset  + blockYZ
    up_face_offset    = up_face_offset    + blockYZ
    back_face_offset  = back_face_offset  + blockYZ
    front_face_offset = front_face_offset + blockYZ
    face_count        = face_count        + blockYZ
  end
  if block_D == [block_null] then
    up_face_offset    = up_face_offset    + blockZX
    back_face_offset  = back_face_offset  + blockZX
    front_face_offset = front_face_offset + blockZX
    face_count        = face_count        + blockZX
  end
  if block_U == [block_null] then
    back_face_offset  = back_face_offset  + blockZX
    front_face_offset = front_face_offset + blockZX
    face_count        = face_count        + blockZX
  end
  if block_B == [block_null] then
    front_face_offset = front_face_offset + blockXY
    face_count        = face_count        + blockXY
  end
  if block_F == [block_null] then
    face_count        = face_count + blockXY
  end
  regentlib.assert(face_count == block_face_count(block),
                   "ERROR: miscount in create_face_connectivity_raw")

  var interior_face_end = left_face_offset
  var left_face_end     = right_face_offset
  var right_face_end    = down_face_offset
  var down_face_end     = up_face_offset
  var up_face_end       = back_face_offset
  var back_face_end     = front_face_offset
  var front_face_end    = face_count

  --c.printf("Connectivity calculations for block %d\n", my_blk_id)
  -- Left/Right Faces...
  for k = 0, block.size_[Z] do
    var g_k = block_global_index(block, Z, k)
    for j = 0, block.size_[Y] do
      var g_j = block_global_index(block, Y, j)
      for i = first_x_face, block.size_[X]+1 do
        var g_i = block_global_index(block, X, i)
        var cell_L : ptr_t = ptr_t { value = -1 }
        var cell_R : ptr_t = ptr_t { value = -1 }
        var local_face_offset = -1

        if i == 0 and block_L == [block_null] then
          -- left boundary face
          cell_L.value = block_global_cell_id(block, i, j, k)
          local_face_offset = left_face_offset
          left_face_offset = left_face_offset + 1
        elseif i > 0 and i < block.size_[X] then
          -- interior face
          cell_L.value = block_global_cell_id(block, i-1, j, k)
          cell_R.value = block_global_cell_id(block, i, j, k)
          local_face_offset = interior_face_offset
          interior_face_offset = interior_face_offset + 1
        elseif i == block.size_[X] and block_R ~= [block_null] then
          -- interior face, at the edge of the block
          cell_L.value = block_global_cell_id(block, i-1, j, k)
          cell_R.value = block_global_cell_id(block_R, 0, j, k)
          local_face_offset = interior_face_offset
          interior_face_offset = interior_face_offset + 1
        elseif i == block.size_[X] and block_R == [block_null] then
          -- right boundary face
          cell_L.value = block_global_cell_id(block, i-1, j, k)
          local_face_offset = right_face_offset
          right_face_offset = right_face_offset + 1
        else
          regentlib.assert(false, 'missing case in switch')
        end

        var face_id : ptr_t = ptr_t { value = block.faceOffset_ + local_face_offset }

        var ids : int[4]
        ids[0] = mesh_node_id(&mesh, g_i, g_j,   g_k)
        ids[1] = mesh_node_id(&mesh, g_i, g_j+1, g_k)
        ids[2] = mesh_node_id(&mesh, g_i, g_j+1, g_k+1)
        ids[3] = mesh_node_id(&mesh, g_i, g_j,   g_k+1)
        for i = 0, 4 do
          @[&int](c.legion_accessor_array_1d_ref(fa_connectivity[i], face_id)) = ids[i]
        end
        @[&ptr_t](c.legion_accessor_array_1d_ref(fa_left, face_id)) = cell_L
        @[&ptr_t](c.legion_accessor_array_1d_ref(fa_right, face_id)) = cell_R

        --c.printf("face=%4d\tleft=%4d\tright=%4d\towner_l=%4d\towner_r=%4d\n",
        --  face_id.value, cell_L.value, cell_R.value, owner_L, owner_R)
      end
    end
  end

  -- Down/Up Faces...
  for k = 0, block.size_[Z] do
    var g_k = block_global_index(block, Z, k)
    for j = first_y_face, block.size_[Y]+1 do
      var g_j = block_global_index(block, Y, j)
      for i = 0, block.size_[X] do
        var g_i = block_global_index(block, X, i)
        var cell_D : ptr_t = ptr_t { value = -1 }
        var cell_U : ptr_t = ptr_t { value = -1 }
        var local_face_offset = -1

        if j == 0 and block_D == [block_null] then
          -- down boundary face
          cell_D.value = block_global_cell_id(block, i, j, k)
          local_face_offset = down_face_offset
          down_face_offset = down_face_offset + 1
        elseif j > 0 and j < block.size_[Y] then
          -- interior face
          cell_D.value = block_global_cell_id(block, i, j-1, k)
          cell_U.value = block_global_cell_id(block, i, j, k)
          local_face_offset = interior_face_offset
          interior_face_offset = interior_face_offset + 1
        elseif j == block.size_[Y] and block_U ~= [block_null] then
          -- interior face, at the edge of the block
          cell_D.value = block_global_cell_id(block, i, j-1, k)
          cell_U.value = block_global_cell_id(block_U, i, 0, k)
          local_face_offset = interior_face_offset
          interior_face_offset = interior_face_offset + 1
        elseif j == block.size_[Y] and block_U == [block_null] then
          -- up boundary face
          cell_D.value = block_global_cell_id(block, i, j-1, k)
          local_face_offset = up_face_offset
          up_face_offset = up_face_offset + 1
        else
          regentlib.assert(false, 'missing case in switch')
        end

        var face_id : ptr_t = ptr_t { value = block.faceOffset_ + local_face_offset }

        var ids : int[4]
        ids[0] = mesh_node_id(&mesh, g_i,   g_j, g_k)
        ids[1] = mesh_node_id(&mesh, g_i,   g_j, g_k+1)
        ids[2] = mesh_node_id(&mesh, g_i+1, g_j, g_k+1)
        ids[3] = mesh_node_id(&mesh, g_i+1, g_j, g_k)
        for i = 0, 4 do
          @[&int](c.legion_accessor_array_1d_ref(fa_connectivity[i], face_id)) = ids[i]
        end
        @[&ptr_t](c.legion_accessor_array_1d_ref(fa_left, face_id)) = cell_D
        @[&ptr_t](c.legion_accessor_array_1d_ref(fa_right, face_id)) = cell_U

        --c.printf("face=%4d\tleft=%4d\tright=%4d\towner_l=%4d\towner_r=%4d\n",
        --  face_id.value, cell_D.value, cell_U.value, owner_D, owner_U)
      end
    end
  end

  ---- Front/Back Faces...
  for k = first_z_face, block.size_[Z]+1 do
    var g_k = block_global_index(block, Z, k)
    for j = 0, block.size_[Y] do
      var g_j = block_global_index(block, Y, j)
      for i = 0, block.size_[X] do
        var g_i = block_global_index(block, X, i)
        var cell_B : ptr_t = ptr_t { value = -1 }
        var cell_F : ptr_t = ptr_t { value = -1 }
        var local_face_offset = -1

        if k == 0 and block_B == [block_null] then
          -- back boundary face
          cell_B.value = block_global_cell_id(block, i, j, k)
          local_face_offset = back_face_offset
          back_face_offset = back_face_offset + 1
        elseif k > 0 and k < block.size_[Z] then
          -- interior face
          cell_B.value = block_global_cell_id(block, i, j, k-1)
          cell_F.value = block_global_cell_id(block, i, j, k)
          local_face_offset = interior_face_offset
          interior_face_offset = interior_face_offset + 1
        elseif k == block.size_[Z] and block_F ~= [block_null] then
          -- interior face, at the edge of the block
          cell_B.value = block_global_cell_id(block, i, j, k-1)
          cell_F.value = block_global_cell_id(block_F, i, j, 0)
          local_face_offset = interior_face_offset
          interior_face_offset = interior_face_offset + 1
        elseif k == block.size_[Z] and block_F == [block_null] then
          -- front boundary face
          cell_B.value = block_global_cell_id(block, i, j, k-1)
          local_face_offset = front_face_offset
          front_face_offset = front_face_offset + 1
        else
          regentlib.assert(false, 'missing case in switch')
        end

        var face_id : ptr_t = ptr_t { value = block.faceOffset_ + local_face_offset }

        var ids : int[4]
        ids[0] = mesh_node_id(&mesh, g_i,   g_j,   g_k)
        ids[1] = mesh_node_id(&mesh, g_i+1, g_j,   g_k)
        ids[2] = mesh_node_id(&mesh, g_i+1, g_j+1, g_k)
        ids[3] = mesh_node_id(&mesh, g_i,   g_j+1, g_k)
        for i = 0, 4 do
          @[&int](c.legion_accessor_array_1d_ref(fa_connectivity[i], face_id)) = ids[i]
        end
        @[&ptr_t](c.legion_accessor_array_1d_ref(fa_left, face_id)) = cell_B
        @[&ptr_t](c.legion_accessor_array_1d_ref(fa_right, face_id)) = cell_F

        --c.printf("face=%4d\tleft=%4d\tright=%4d\towner_l=%4d\towner_r=%4d\n",
        --  face_id.value, cell_B.value, cell_F.value, owner_B, owner_F)
      end
    end
  end

  regentlib.assert(interior_face_offset == interior_face_end and
                   left_face_offset     == left_face_end     and
                   right_face_offset    == right_face_end    and
                   down_face_offset     == down_face_end     and
                   up_face_offset       == up_face_end       and
                   back_face_offset     == back_face_end     and
                   front_face_offset    == front_face_end,
                   "ERROR: miscount in create_face_connectivity_raw")

  c.legion_accessor_array_1d_destroy(fa_left)
  c.legion_accessor_array_1d_destroy(fa_right)
  for i = 0, 4 do
    c.legion_accessor_array_1d_destroy(fa_connectivity[i])
  end
  std.free(fa_connectivity)
end

__demand(__inline)
task create_face_connectivity(my_blk_id   : int,
                              mesh        : MeshTopology,
                              blocks      : region(Block),
                              rface       : region(Face))
where
  reads(blocks),
  writes(rface.{left, right, face_connectivity})
do
  var idx = 0
  var num_blocks = mesh.num_blocks_
  mesh.blocks_ = block_array(num_blocks)
  for block in blocks do
    mesh.blocks_[idx] = @block
    idx = idx + 1
  end

  create_face_connectivity_raw(
      my_blk_id, mesh,
      __physical(rface.{left, right, face_connectivity.{_0, _1, _2, _3}}),
      __fields(rface.{left, right, face_connectivity.{_0, _1, _2, _3}}))
  std.free(mesh.blocks_)
end

terra create_cell_connectivity_raw(my_blk_id : int,
                                   mesh      : MeshTopology,
                                   cells     : c.legion_physical_region_t[8],
                                   cells_fields : c.legion_field_id_t[8])
  var fa_connectivity : &c.legion_accessor_array_1d_t =
    [&c.legion_accessor_array_1d_t](std.malloc(8 * sizeof(c.legion_accessor_array_1d_t)))
  for i = 0, 8 do
    fa_connectivity[i] =
      c.legion_physical_region_get_field_accessor_array_1d(cells[i], cells_fields[i])
  end

  var block = mesh_get_block_by_id(&mesh, my_blk_id)
  for k = 0, block.size_[Z] do
    var g_k = block_global_index(block, Z, k)
    for j = 0, block.size_[Y] do
      var g_j = block_global_index(block, Y, j)
      for i = 0, block.size_[X] do
        var g_i = block_global_index(block, X, i)

        var cell_id = ptr_t { value = block_global_cell_id(block, i, j, k) }
        @[&int](c.legion_accessor_array_1d_ref(fa_connectivity[0], cell_id)) =
          mesh_node_id(&mesh, g_i,   g_j,   g_k)
        @[&int](c.legion_accessor_array_1d_ref(fa_connectivity[1], cell_id)) =
          mesh_node_id(&mesh, g_i+1, g_j,   g_k)
        @[&int](c.legion_accessor_array_1d_ref(fa_connectivity[2], cell_id)) =
          mesh_node_id(&mesh, g_i+1, g_j+1, g_k)
        @[&int](c.legion_accessor_array_1d_ref(fa_connectivity[3], cell_id)) =
          mesh_node_id(&mesh, g_i,   g_j+1, g_k)
        @[&int](c.legion_accessor_array_1d_ref(fa_connectivity[4], cell_id)) =
          mesh_node_id(&mesh, g_i,   g_j,   g_k+1)
        @[&int](c.legion_accessor_array_1d_ref(fa_connectivity[5], cell_id)) =
          mesh_node_id(&mesh, g_i+1, g_j,   g_k+1)
        @[&int](c.legion_accessor_array_1d_ref(fa_connectivity[6], cell_id)) =
          mesh_node_id(&mesh, g_i+1, g_j+1, g_k+1)
        @[&int](c.legion_accessor_array_1d_ref(fa_connectivity[7], cell_id)) =
          mesh_node_id(&mesh, g_i,   g_j+1, g_k+1)
      end
    end
  end

  for i = 0, 8 do
    c.legion_accessor_array_1d_destroy(fa_connectivity[i])
  end
  std.free(fa_connectivity)
end

__demand(__inline)
task create_cell_connectivity(my_blk_id : int,
                              mesh      : MeshTopology,
                              blocks    : region(Block),
                              rcell     : region(Cell))
where
  reads(blocks),
  writes(rcell.cell_connectivity)
do
  var idx = 0
  var num_blocks = mesh.num_blocks_
  mesh.blocks_ = block_array(num_blocks)
  for block in blocks do
    mesh.blocks_[idx] = @block
    idx = idx + 1
  end

  create_cell_connectivity_raw(
      my_blk_id, mesh,
      __physical(rcell.cell_connectivity.{_0, _1, _2, _3, _4, _5, _6, _7}),
      __fields(rcell.cell_connectivity.{_0, _1, _2, _3, _4, _5, _6, _7}))
  std.free(mesh.blocks_)
end

__demand(__inline)
task cell_volume(mesh  : MeshTopology,
                 rcell : region(Cell))
where
  reads(rcell.cell_connectivity),
  writes(rcell.volume, rcell.cell_centroid)
do
  for cell in rcell do
    var coord : double[24] =
      mesh_get_hex_nodal_coordinates(mesh, cell.cell_connectivity)
    var volume = mesh_hex_cell_volume(coord)
    cell.volume = volume
    var center : Vec3 = mesh_hex_cell_centroid(coord)
    cell.cell_centroid = center
    --c.printf("Cell: %lld\tCentroid: %10.3e\t%10.3e\t%10.3e\tVolume: %10.3e\n",
    --          c.coord_t(cell), center[0], center[1], center[2], volume)
  end
end

task cell_total_volume(my_blk_id : int,
                       rcell     : region(Cell))
where
  reads(rcell.volume)
do
  var volume = 0.0
  for cell in rcell do
    volume += cell.volume
  end
  c.printf("Volume for block %d = %12.5e\n", my_blk_id, volume);
end

__demand(__inline)
task face_centroid(mesh  : MeshTopology,
                   rface : region(Face))
where
  reads(rface.face_connectivity),
  writes(rface.face_centroid)
do
  for face in rface do
    var center : Vec3 =
      mesh_quad_face_centroid(mesh, face.face_connectivity)
    face.face_centroid = center
    --c.printf("Face: %lld\tCentroid: %10.3e\t%10.3e\t%10.3e\n",
    --         c.coord_t(face),
    --         center.x, center.y, center.z)
  end
end

terra calculate_area_vector(mesh    : MeshTopology,
                            connect : FaceConnect) : Vec3
  var num_face_nodes = 4
  var n_xyz : double[5 * 3]
  var x : double
  var y : double
  var z : double
  var idx = 0

  mesh_node_coordinate(&mesh, connect._0, &x, &y, &z)
  n_xyz[idx + X] = x n_xyz[idx + Y] = y n_xyz[idx + Z] = z idx = idx + 3
  mesh_node_coordinate(&mesh, connect._1, &x, &y, &z)
  n_xyz[idx + X] = x n_xyz[idx + Y] = y n_xyz[idx + Z] = z idx = idx + 3
  mesh_node_coordinate(&mesh, connect._2, &x, &y, &z)
  n_xyz[idx + X] = x n_xyz[idx + Y] = y n_xyz[idx + Z] = z idx = idx + 3
  mesh_node_coordinate(&mesh, connect._3, &x, &y, &z)
  n_xyz[idx + X] = x n_xyz[idx + Y] = y n_xyz[idx + Z] = z idx = idx + 3

  var vec1 : Vec3
  vec1[0] = n_xyz[3 + X] - n_xyz[X]
  vec1[1] = n_xyz[3 + Y] - n_xyz[Y]
  vec1[2] = n_xyz[3 + Z] - n_xyz[Z]
  var vec2 : Vec3
  vec2[0] = n_xyz[6 + X] - n_xyz[X]
  vec2[1] = n_xyz[6 + Y] - n_xyz[Y]
  vec2[2] = n_xyz[6 + Z] - n_xyz[Z]
  var vec3 : Vec3
  vec3[0] = n_xyz[9 + X] - n_xyz[X]
  vec3[1] = n_xyz[9 + Y] - n_xyz[Y]
  vec3[2] = n_xyz[9 + Z] - n_xyz[Z]

  var normal1 = Vec3Cross(vec1, vec2)
  var normal2 = Vec3Cross(vec2, vec3)
  var area = Vec3Average(normal1, normal2)

  return area
end

terra max_index(arr : &double, size_arr : int) : int
  var idx = 0
  var v : double = arr[0]
  for i = 1, size_arr do
    if arr[i] > v then
      v = arr[i]
      idx = i
    end
  end
  return idx
end

terra calculate_tangent_vector(area : Vec3) : Vec3
  var abs_a_vec : double[3]
  abs_a_vec[0] = cmath.fabs(area[0])
  abs_a_vec[1] = cmath.fabs(area[1])
  abs_a_vec[2] = cmath.fabs(area[2])
  var i1 = max_index([&double](abs_a_vec), 3)
  var i2 = (i1 + 1) % 3
  var i3 = (i1 + 2) % 3

  var denom = cmath.sqrt(area[i1] * area[i1] + area[i3] * area[i3])

  var t_vec : double[3]
  t_vec[i2] =  0.0
  t_vec[i1] =  area[i3] / denom
  t_vec[i3] = -area[i1] / denom

  return t_vec
end

__demand(__inline)
task face_normal(my_blk_id : int,
                 mesh  : MeshTopology,
                 rface : region(Face))
where
  reads(rface.face_connectivity),
  writes(rface.{area, tangent, binormal})
do
  for face in rface do
    var connect  : FaceConnect = face.face_connectivity
    var area     : Vec3 = calculate_area_vector(mesh, connect)
    var tangent  : Vec3 = calculate_tangent_vector(area)
    var binormal : Vec3 = Vec3Cross(area, tangent)
    face.area     = area
    face.tangent  = tangent
    face.binormal = binormal
    --c.printf("Normal %lld: %d %d %d %d, %12.5e\t%12.5e\t%12.5e\n",
    --         c.coord_t(face),
    --         connect._0, connect._1, connect._2, connect._3,
    --         area.x, area.y, area.z)
  end
end

__demand(__inline)
task initialize_problem(problem_type : int,
                        extentX : double,
                        rcell : region(Cell))
where
  reads(rcell.{solution_n, cell_centroid}),
  writes(rcell.{solution_n, solution_np1, solution_temp})
do
  if problem_type == 0 then
    var midx = extentX / 2.0

    var Rgas = 287.05
    var gamma = 1.4
    var Cv = Rgas / (gamma - 1.0)

    var P1 = 68947.57
    var P2 = 6894.757
    var T1 = 288.889
    var T2 = 231.11

    var density1 = P1 / (Rgas * T1)
    var rhoE1 = density1 * (Cv * T1)

    var density2 = P2 / (Rgas * T2)
    var rhoE2 = density2 * (Cv * T2)

    for cell in rcell do
      var x = cell.cell_centroid[0]

      var solution_n : Solution

      solution_n[1] = 0.0
      solution_n[2] = 0.0
      solution_n[3] = 0.0

      if x < midx then
        solution_n[0] = density1
        solution_n[4] = rhoE1
      else
        solution_n[0] = density2
        solution_n[4] = rhoE2
      end
      cell.solution_n = solution_n
    end
  else
    for cell in rcell do
      var solution_n : Solution
      solution_n[0] = 0.5805
      solution_n[1] = 503.96
      solution_n[2] = 0.0
      solution_n[3] = 0.0
      solution_n[4] = 343750.0
      cell.solution_n = solution_n
    end
  end

  for cell in rcell do
    cell.solution_np1 = cell.solution_n
    cell.solution_temp = cell.solution_n
  end
end

task initialize(n : int,
                mesh : MeshTopology,
                problem_type : int,
                extentX : int,
                blocks : region(Block),
                rcell : region(Cell),
                rface : region(Face))
where
  reads(blocks),
  reads writes(rcell, rface)
do
  create_face_connectivity(n, mesh, blocks, rface)
  create_cell_connectivity(n, mesh, blocks, rcell)
  cell_volume(mesh, rcell)
  face_centroid(mesh, rface)
  face_normal(n, mesh, rface)
  initialize_problem(problem_type, extentX, rcell)
end

-------------------------------------------------------------------------------
-- main computation
-------------------------------------------------------------------------------

__demand(__parallel, __cuda)
task update_rk_stage_alpha_and_initialize_solution_fields(rcell : region(Cell),
                                                          rk : int,
                                                          alpha_ : double[4],
                                                          init_minmax : bool)
where
  reads(rcell.{residual, solution_n,solution_temp, residual, cell_flux, limiter, cell_gradients,
                stencil_min, stencil_max}),
  writes(rcell.{solution_temp, residual, cell_flux, limiter, cell_gradients,
                stencil_min, stencil_max})
do
  var alpha = alpha_[rk]
  for cell in rcell do
    var sol_n = cell.solution_n
    var res = cell.residual
    var s : Solution
    for i = 0, 5 do
      s[i] = sol_n[i] + alpha * res[i]
    end
    cell.solution_temp = s
    --var cellptr = c.coord_t(cell)
    --c.printf("Res: Cell %lld\t%12.5e\t%12.5e\t%12.5e\t%12.5e\t%12.5e\n",
    --         cellptr, res[0], res[1], res[2], res[3], res[4])
    --c.printf("Sol: Cell %lld\t%12.5e\t%12.5e\t%12.5e\t%12.5e\t%12.5e\n",
    --         cellptr, s[0], s[1], s[2], s[3], s[4])
  end

  for cell in rcell do
    cell.residual = array(0.0, 0.0, 0.0, 0.0, 0.0)
  end
  for cell in rcell do
    cell.cell_flux = array(0.0, 0.0, 0.0, 0.0, 0.0)
  end
  for cell in rcell do
    cell.limiter = array(1.0, 1.0, 1.0, 1.0, 1.0)
  end
  for cell in rcell do
    for i = 0, 15 do
      cell.cell_gradients[i] = 0.0
    end
  end

  if init_minmax then
    for cell in rcell do
      cell.stencil_min = array(1.0e300, 1.0e300, 1.0e300, 1.0e300, 1.0e300)
    end
    for cell in rcell do
      cell.stencil_max = array(-1.0e300, -1.0e300, -1.0e300, -1.0e300, -1.0e300)
    end
  end
end

__demand(__parallel, __cuda)
task face_gradient(rcell : region(Cell),
                   rface_all : region(Face),
                   rface_interior : region(Face),
                   rface_tangent : region(Face),
                   rface_extrapolate : region(Face),
                   rface_inflow : region(Face),
                   rface_noslip : region(Face))
where
  reads(rcell.{cell_gradients, solution_temp, volume},
        rface_all.{left, right, area}),
  writes(rcell.cell_gradients),
  rface_interior <= rface_all,
  rface_tangent <= rface_all,
  rface_extrapolate <= rface_all,
  rface_inflow <= rface_all,
  rface_noslip <= rface_all
do
  do
    for face in rface_interior do
      var left = rface_all[face].left
      var right = rface_all[face].right

      var primitives_l : Solution = ComputePrimitives(rcell[left].solution_temp)
      var primitives_r : Solution = ComputePrimitives(rcell[right].solution_temp)

      var face_normal = rface_all[face].area
      var gradients : Gradient
      for i = 0, 5 do
        for dir = 0, 3 do
          gradients[i * 3 + dir] =
            0.5 * (primitives_l[i] + primitives_r[i]) * face_normal[dir]
        end
      end

      do
        var volume = rcell[left].volume
        for i = 0, 15 do
          rcell[left].cell_gradients[i] += gradients[i] / volume
        end
      end

      do
        var volume = rcell[right].volume
        for i = 0, 15 do
          rcell[right].cell_gradients[i] += -gradients[i] / volume
        end
      end
    end
  end

  [(function() local quotes = rquote end
    for _,rface in ipairs({rface_tangent,rface_extrapolate,rface_inflow,rface_noslip}) do
      quotes = rquote [quotes];
        for face in rface do
          var left = rface_all[face].left
          var primitives : Solution = ComputePrimitives(rcell[left].solution_temp)

          var face_normal = rface_all[face].area
          face_normal[0] = -face_normal[0]
          face_normal[1] = -face_normal[1]
          face_normal[2] = -face_normal[2]
          var volume = rcell[left].volume

          for i = 0, 5 do
            for dir = 0, 3 do
              rcell[left].cell_gradients[i * 3 + dir] +=
              primitives[i] * face_normal[dir] / volume
            end
          end
        end
      end
    end
  return quotes end)()]
end

task print_gradients(warmup: bool, rcell : region(Cell))
where
  reads(rcell.cell_gradients)
do
  --if warmup then return end
  for cell in rcell do
    var cellptr = c.coord_t(cell)
    c.printf("Gradient: Cell %lld\t%12.5e, %12.5e, %12.5e\t%12.5e, %12.5e, %12.5e\t%12.5e, %12.5e, %12.5e\t%12.5e, %12.5e, %12.5e\t%12.5e, %12.5e, %12.5e\n",
    cellptr,
    cell.cell_gradients[0], cell.cell_gradients[1], cell.cell_gradients[2],
    cell.cell_gradients[3], cell.cell_gradients[4], cell.cell_gradients[5],
    cell.cell_gradients[6], cell.cell_gradients[7], cell.cell_gradients[8],
    cell.cell_gradients[9], cell.cell_gradients[10], cell.cell_gradients[11],
    cell.cell_gradients[12], cell.cell_gradients[13], cell.cell_gradients[14])
  end
end

__demand(__parallel, __cuda)
task compute_min_max(rcell       : region(Cell),
                     rface_all   : region(Face),
                     rface_interior : region(Face),
                     rface_tangent : region(Face),
                     rface_extrapolate : region(Face),
                     rface_inflow : region(Face),
                     rface_noslip : region(Face))
where
  reads(rcell.{solution_temp, stencil_min, stencil_max},
        rface_all.{left, right}),
  writes(rcell.{stencil_min, stencil_max}),
  rface_interior <= rface_all,
  rface_tangent <= rface_all,
  rface_extrapolate <= rface_all,
  rface_inflow <= rface_all,
  rface_noslip <= rface_all
do
  -- for interior faces
  do
    for face in rface_interior do
      var left = rface_all[face].left
      var right = rface_all[face].right

      var primitives_l : Solution = ComputePrimitives(rcell[left].solution_temp)
      var primitives_r : Solution = ComputePrimitives(rcell[right].solution_temp)

      var face_min : Solution
      var face_max : Solution
      for i = 0, 5 do
        face_min[i] = min(primitives_r[i], primitives_l[i])
      end
      for i = 0, 5 do
        face_max[i] = max(primitives_r[i], primitives_l[i])
      end

      --c.printf("Stencil Min:\t%12.5e\t%12.5e\t%12.5e\t%12.5e\t%12.5e\n",
      --         face_min[0], face_min[1], face_min[2], face_min[3], face_min[4])
      --c.printf("Stencil Max:\t%12.5e\t%12.5e\t%12.5e\t%12.5e\t%12.5e\n",
      --         face_max[0], face_max[1], face_max[2], face_max[3], face_max[4])

      do
        for i = 0, 5 do
          rcell[left].stencil_min[i] min= face_min[i]
        end
        for i = 0, 5 do
          rcell[left].stencil_max[i] max= face_max[i]
        end
      end
      do
        for i = 0, 5 do
          rcell[right].stencil_min[i] min= face_min[i]
        end
        for i = 0, 5 do
          rcell[right].stencil_max[i] max= face_max[i]
        end
      end
    end
  end

  [(function() local quotes = rquote end
    for _,rface in ipairs({rface_tangent,rface_extrapolate,rface_inflow,rface_noslip}) do
      quotes = rquote [quotes];
        for face in rface do
          var left = rface_all[face].left
          var primitives_l : Solution = ComputePrimitives(rcell[left].solution_temp)
          for i = 0, 5 do
            rcell[left].stencil_min[i] min= primitives_l[i]
          end
          for i = 0, 5 do
            rcell[left].stencil_max[i] max= primitives_l[i]
          end
        end
      end
    end
  return quotes end)()]
end

__demand(__inline)
task venkat_limit(dumax : double, dumin : double,
                  du : double, deltax3 : double) : double
  var res : double
  var beta = 1.0
  var epstilde2 = deltax3 * beta * beta * beta

  if du > cmath.DBL_EPSILON then
    var num =   dumax*dumax + epstilde2 + 2.0*du*dumax
    var denom = dumax*dumax + epstilde2 + 2.0*du*du + dumax*du
    res = num / denom
  elseif du < -cmath.DBL_EPSILON then
    var num  =  dumin*dumin + epstilde2 + 2.0*du*dumin
    var denom = dumin*dumin + epstilde2 + 2.0*du*du + dumin*du
    res = num / denom
  else
    res = 1.0
  end

  return res
end

__demand(__inline)
task _compute_limiter(conservatives : Solution,
                      face_coordinates : Vec3,
                      cell_coordinates : Vec3,
                      cell_gradients : Gradient,
                      cell_min : Solution,
                      cell_max : Solution) : Solution
  var primitives : Solution = ComputePrimitives(conservatives)

  var displacement : Vec3
  var distance : double = 0.0
  for i = 0, 3 do
    displacement[i] = face_coordinates[i] - cell_coordinates[i]
  end
  distance = displacement[0] * displacement[0] +
             displacement[1] * displacement[1] +
             displacement[2] * displacement[2]

  var dU : Solution
  for i = 0, 5 do
    dU[i] =
      displacement[0] * cell_gradients[3 * i + 0] +
      displacement[1] * cell_gradients[3 * i + 1] +
      displacement[2] * cell_gradients[3 * i + 2]
  end

  var limiter : Solution
  for i = 0, 5 do
    limiter[i] = venkat_limit(cell_max[i] - primitives[i],
                              cell_min[i] - primitives[i],
                              dU[i], distance)
  end
  return limiter
end

__demand(__parallel, __cuda)
task compute_limiter(rcell       : region(Cell),
                     rface_all   : region(Face),
                     rface_interior : region(Face))
where
  reads(rcell.{solution_temp, stencil_min, stencil_max, limiter,
               cell_centroid, cell_gradients},
        rface_all.{left, right, face_centroid}),
  writes(rcell.limiter),
  rface_interior <= rface_all
do
  for face in rface_all do
    var p = rface_all[face].left
    var limiter = _compute_limiter(rcell[p].solution_temp, face.face_centroid,
                                   rcell[p].cell_centroid, rcell[p].cell_gradients,
                                   rcell[p].stencil_min, rcell[p].stencil_max)
    for i = 0, 5 do
      rcell[p].limiter[i] min= limiter[i]
    end
    --c.printf("Limiter:\t%12.5e\t%12.5e\t%12.5e\t%12.5e\t%12.5e\n",
    --         limiter[0], limiter[1], limiter[2], limiter[3], limiter[4])
  end

  for face in rface_interior do
    var p = rface_all[face].right
    var limiter = _compute_limiter(rcell[p].solution_temp, rface_all[face].face_centroid,
                                   rcell[p].cell_centroid, rcell[p].cell_gradients,
                                   rcell[p].stencil_min, rcell[p].stencil_max)
    for i = 0, 5 do
      rcell[p].limiter[i] min= limiter[i]
    end
    --c.printf("Limiter:\t%12.5e\t%12.5e\t%12.5e\t%12.5e\t%12.5e\n",
    --         limiter[0], limiter[1], limiter[2], limiter[3], limiter[4])
  end
end

__demand(__inline)
task roe_flux_compute_flux(primitives_l : Solution,
                           primitives_r : Solution,
                           a_vec        : Vec3,
                           t_vec        : Vec3,
                           a_x_t        : Vec3) : Solution
  var efix_u = 0.1
  var efix_c = 0.1

  var gm1 = 0.4

  -- Left state
  var rl = primitives_l[0]
  var ul = primitives_l[1]
  var vl = primitives_l[2]
  var wl = primitives_l[3]

  var pl = ComputePressure(primitives_l)
  var hl = ComputeEnthalpy(primitives_l)

  var kel = 0.5 * (ul * ul + vl * vl + wl * wl)
  var htl = hl + kel
  var ubl = a_vec[0] * ul + a_vec[1] * vl + a_vec[2] * wl

  -- Right state
  var rr = primitives_r[0]
  var ur = primitives_r[1]
  var vr = primitives_r[2]
  var wr = primitives_r[3]

  var pr = ComputePressure(primitives_r)
  var hr = ComputeEnthalpy(primitives_r)

  var ker = 0.5 * (ur * ur + vr * vr + wr * wr)
  var htr = hr + ker
  var ubr = a_vec[0] * ur + a_vec[1] * vr + a_vec[2] * wr

  var mdotl = rl * ubl
  var mdotr = rr * ubr

  var pl_plus_pr = pl + pr

  -- Central part
  var flux : Solution
  flux[0] = 0.5 * (mdotl + mdotr)
  flux[1] = 0.5 * (mdotl * ul + mdotr * ur + a_vec[0] * pl_plus_pr)
  flux[2] = 0.5 * (mdotl * vl + mdotr * vr + a_vec[1] * pl_plus_pr)
  flux[3] = 0.5 * (mdotl * wl + mdotr * wr + a_vec[2] * pl_plus_pr)
  flux[4] = 0.5 * (mdotl * htl + mdotr * htr)

  -- Upwinded part
  var a_vec_norm = sqrt(a_vec[0] * a_vec[0] + a_vec[1] * a_vec[1] + a_vec[2] * a_vec[2])
  var t_vec_norm = sqrt(t_vec[0] * t_vec[0] + t_vec[1] * t_vec[1] + t_vec[2] * t_vec[2])
  var a_x_t_norm = sqrt(a_x_t[0] * a_x_t[0] + a_x_t[1] * a_x_t[1] + a_x_t[2] * a_x_t[2])

  var a_vec_unit : Vec3
  for i = 0, 3 do a_vec_unit[i] = a_vec[i] / a_vec_norm end
  var t_vec_unit : Vec3
  for i = 0, 3 do t_vec_unit[i] = t_vec[i] / t_vec_norm end
  var a_x_t_unit : Vec3
  for i = 0, 3 do a_x_t_unit[i] = a_x_t[i] / a_x_t_norm end

  var denom = 1.0 / (sqrt(rl) + sqrt(rr))
  var alpha = sqrt(rl) * denom
  var beta = 1.0 - alpha

  var ua = alpha * ul + beta * ur
  var va = alpha * vl + beta * vr
  var wa = alpha * wl + beta * wr
  var ha = alpha * hl + beta * hr
    + 0.5 * alpha * beta
    * ((ur-ul)*(ur-ul) + (vr-vl)*(vr-vl) + (wr-wl)*(wr-wl))
  var Ca = sqrt(gm1 * ha)

  -- Compute flux matrices
  var roe_mat_l : double[25]
  var roe_mat_r : double[25]

  var ub = ua * a_vec_unit[0] + va * a_vec_unit[1] + wa * a_vec_unit[2]
  var vb = ua * t_vec_unit[0] + va * t_vec_unit[1] + wa * t_vec_unit[2]
  var wb = ua * a_x_t_unit[0] + va * a_x_t_unit[1] + wa * a_x_t_unit[2]
  var keb = 0.5 * (ua * ua + va * va + wa * wa)
  var c2i = 1.0 / (Ca * Ca)
  var hc2 = 0.5 * c2i

  -- Left matrix
  roe_mat_l[0] = gm1 * (keb - ha) + Ca * (Ca - ub)
  roe_mat_l[1] = Ca * a_vec_unit[0] - gm1 * ua
  roe_mat_l[2] = Ca * a_vec_unit[1] - gm1 * va
  roe_mat_l[3] = Ca * a_vec_unit[2] - gm1 * wa
  roe_mat_l[4] = gm1

  roe_mat_l[5] = gm1 * (keb - ha) + Ca * (Ca + ub)
  roe_mat_l[6] = -Ca * a_vec_unit[0] - gm1 * ua
  roe_mat_l[7] = -Ca * a_vec_unit[1] - gm1 * va
  roe_mat_l[8] = -Ca * a_vec_unit[2] - gm1 * wa
  roe_mat_l[9] = gm1

  roe_mat_l[10] = keb - ha
  roe_mat_l[11] = -ua
  roe_mat_l[12] = -va
  roe_mat_l[13] = -wa
  roe_mat_l[14] = 1.0

  roe_mat_l[15] = -vb
  roe_mat_l[16] = t_vec_unit[0]
  roe_mat_l[17] = t_vec_unit[1]
  roe_mat_l[18] = t_vec_unit[2]
  roe_mat_l[19] = 0.0

  roe_mat_l[20] = -wb
  roe_mat_l[21] = a_x_t_unit[0]
  roe_mat_l[22] = a_x_t_unit[1]
  roe_mat_l[23] = a_x_t_unit[2]
  roe_mat_l[24] = 0.0

  -- Right matrix
  roe_mat_r[0] = hc2
  roe_mat_r[1] = hc2
  roe_mat_r[2] = -gm1 * c2i
  roe_mat_r[3] = 0.0
  roe_mat_r[4] = 0.0

  roe_mat_r[5] = (ua + a_vec_unit[0] * Ca) * hc2
  roe_mat_r[6] = (ua - a_vec_unit[0] * Ca) * hc2
  roe_mat_r[7] = -gm1 * ua * c2i
  roe_mat_r[8] = t_vec_unit[0]
  roe_mat_r[9] = a_x_t_unit[0]

  roe_mat_r[10] = (va + a_vec_unit[1] * Ca) * hc2
  roe_mat_r[11] = (va - a_vec_unit[1] * Ca) * hc2
  roe_mat_r[12] = -gm1 * va * c2i
  roe_mat_r[13] = t_vec_unit[1]
  roe_mat_r[14] = a_x_t_unit[1]

  roe_mat_r[15] = (wa + a_vec_unit[2] * Ca) * hc2
  roe_mat_r[16] = (wa - a_vec_unit[2] * Ca) * hc2
  roe_mat_r[17] = -gm1 * wa * c2i
  roe_mat_r[18] = t_vec_unit[2]
  roe_mat_r[19] = a_x_t_unit[2]

  roe_mat_r[20] = (ha + keb + Ca * ub) * hc2
  roe_mat_r[21] = (ha + keb - Ca * ub) * hc2
  roe_mat_r[22] = (Ca * Ca - gm1 * (ha + keb)) * c2i
  roe_mat_r[23] = vb
  roe_mat_r[24] = wb

  -- Conservative variable jumps
  var U_jmp : double[5]
  U_jmp[0] = rr - rl
  U_jmp[1] = rr * ur - rl * ul
  U_jmp[2] = rr * vr - rl * vl
  U_jmp[3] = rr * wr - rl * wl
  U_jmp[4] = (rr * htr - pr) - (rl * htl - pl)

  -- Compute CFL number
  var cbar = Ca * a_vec_norm
  var ubar = ua * a_vec[0] + va * a_vec[1] + wa * a_vec[2]
  var cfl = abs(ubar) + cbar

  -- Eigenvalue fix
  var eig1 = ubar + cbar
  var eig2 = ubar - cbar
  var eig3 = ubar

  var abs_eig1 = abs(eig1)
  var abs_eig2 = abs(eig2)
  var abs_eig3 = abs(eig3)

  var epuc = efix_u * cfl
  var epcc = efix_c * cfl

  -- Original Roe eigenvalue fix
  var cond1 = [int](abs_eig1 < epcc)
  var cond2 = [int](abs_eig2 < epcc)
  var cond3 = [int](abs_eig3 < epuc)
  abs_eig1 = cond1 * 0.5 * (eig1 * eig1 + epcc * epcc) / epcc + (1 - cond1) * abs_eig1
  abs_eig2 = cond2 * 0.5 * (eig2 * eig2 + epcc * epcc) / epcc + (1 - cond2) * abs_eig2
  abs_eig3 = cond3 * 0.5 * (eig3 * eig3 + epuc * epuc) / epuc + (1 - cond3) * abs_eig3

  var eigp : double[5]
  eigp[0] = 0.5 * (eig1 + abs_eig1)
  eigp[1] = 0.5 * (eig2 + abs_eig2)
  eigp[2] = 0.5 * (eig3 + abs_eig3)
  eigp[3] = eigp[2]
  eigp[4] = eigp[2]

  var eigm : double[5]
  eigm[0] = 0.5 * (eig1 - abs_eig1)
  eigm[1] = 0.5 * (eig2 - abs_eig2)
  eigm[2] = 0.5 * (eig3 - abs_eig3)
  eigm[3] = eigm[2]
  eigm[4] = eigm[2]

  -- Compute upwind flux
  var ldq : double[5]
  var lldq : double[5]
  var rlldq : double[5]
  for i = 0, 5 do ldq[i] = 0.0 lldq[i] = 0.0 rlldq[i] = 0.0 end

  for i = 0, 5 do
    ldq[i] = 0.0
    for j = 0, 5 do
      ldq[i] = ldq[i] + roe_mat_l[5 * i + j] * U_jmp[j];
    end
  end

  for j = 0, 5 do
    lldq[j] = (eigp[j] - eigm[j]) * ldq[j]
  end

  for i = 0, 5 do
    rlldq[i] = 0.0
    for j = 0, 5 do
      rlldq[i] = rlldq[i] + roe_mat_r[5 * i + j] * lldq[j];
    end
  end

  for i = 0, 5 do
    flux[i] -= 0.5 * rlldq[i]
  end

  return flux
end

__demand(__inline)
task newtonian_viscous_flux_compute_flux(grad_primitive : double[15],
                                         primitive      : double[5],
                                         a_vec          : double[3]) : Solution
  var viscosity = ComputeViscosity(primitive[4])
  var thermal_conductivity = ComputeThermalConductivity(viscosity)

  var vflux : Solution
  for icomp = 0, 5 do
    vflux[icomp] = 0.0
  end

  var divergence_velocity : double = 0.0
  for i = 0, 3 do
    divergence_velocity =
      divergence_velocity + grad_primitive[3 * (i + 1) + i]
  end

  for i = 0, 3 do
    for j = 0, 3 do
      var delta_ij : int = [int](i == j)
      var S_ij : double = 0.5 * (grad_primitive[3 * (i + 1) + j] +
                                 grad_primitive[3 * (j + 1) + i])
      var t_ij : double = S_ij - divergence_velocity * delta_ij / 3.0
      vflux[1 + i] =
        vflux[1 + i] + (2 * viscosity * t_ij) * a_vec[j]
      vflux[4] =
        vflux[4] + (2 * viscosity * t_ij) * primitive[i + 1] * a_vec[j]
    end
    vflux[4] =
      vflux[4] + thermal_conductivity * grad_primitive[12 + i] * a_vec[i]
  end
  return vflux
end

__demand(__parallel, __cuda)
task compute_face_flux(is_viscous   : bool,
                       second_order : bool,
                       rcell        : region(Cell),
                       rface_all    : region(Face),
                       rface_interior : region(Face),
                       rface_tangent : region(Face),
                       rface_extrapolate : region(Face),
                       rface_inflow : region(Face),
                       rface_noslip : region(Face))
where
  reads(rcell.{solution_temp, cell_centroid, cell_gradients, limiter, cell_flux},
        rface_all.{left, right, area, tangent, binormal, face_centroid}),
  writes(rcell.cell_flux),
  rface_interior <= rface_all,
  rface_tangent <= rface_all,
  rface_extrapolate <= rface_all,
  rface_inflow <= rface_all,
  rface_noslip <= rface_all
do
  do
    for face in rface_interior do
      var left = rface_all[face].left
      var right = rface_all[face].right

      var primitives_l : Solution = ComputePrimitives(rcell[left].solution_temp)
      var primitives_r : Solution = ComputePrimitives(rcell[right].solution_temp)

      var cell_gradients_l = rcell[left].cell_gradients
      var cell_gradients_r = rcell[right].cell_gradients

      if second_order then
        var cell_coordinates_l = rcell[left].cell_centroid
        var cell_coordinates_r = rcell[right].cell_centroid

        var distance_l : Vec3
        var distance_r : Vec3
        var face_coordinates = rface_all[face].face_centroid

        for i = 0, 3 do distance_l[i] = face_coordinates[i] - cell_coordinates_l[i] end
        for i = 0, 3 do distance_r[i] = face_coordinates[i] - cell_coordinates_r[i] end

        var gradient_primitives_l : Solution
        var gradient_primitives_r : Solution

        for i = 0, 5 do
          gradient_primitives_l[i] =
            distance_l[0] * cell_gradients_l[3 * i + 0] +
            distance_l[1] * cell_gradients_l[3 * i + 1] +
            distance_l[2] * cell_gradients_l[3 * i + 2]
        end

        for i = 0, 5 do
          gradient_primitives_r[i] =
            distance_r[0] * cell_gradients_r[3 * i + 0] +
            distance_r[1] * cell_gradients_r[3 * i + 1] +
            distance_r[2] * cell_gradients_r[3 * i + 2]
        end

        var cell_limiters_l = rcell[left].limiter
        var cell_limiters_r = rcell[right].limiter

        for i = 0, 5 do
          primitives_l[i] += gradient_primitives_l[i] * cell_limiters_l[i]
        end
        for i = 0, 5 do
          primitives_r[i] += gradient_primitives_r[i] * cell_limiters_r[i]
        end
      end

      var normal   = rface_all[face].area
      var tangent  = rface_all[face].tangent
      var binormal = rface_all[face].binormal
      var flux : Solution = roe_flux_compute_flux(primitives_l, primitives_r,
                                                  normal, tangent, binormal)

      if is_viscous then
        var primitives_face : double[5]
        for i = 0, 5 do
          primitives_face[i] = 0.5 * (primitives_l[i] + primitives_r[i])
        end

        var gradients_face : double[5 * 3]
        for i = 0, 15 do
          gradients_face[i]  = 0.5 * (cell_gradients_l[i] + cell_gradients_r[i])
        end

        var vflux : Solution =
          newtonian_viscous_flux_compute_flux(gradients_face,
                                              primitives_face,
                                              normal)
        for i = 0, 5 do flux[i] -= vflux[i] end
      end

      rcell[right].cell_flux += flux

      for i = 0, 5 do flux[i] = -flux[i] end
      rcell[left].cell_flux += flux
    end
  end
  do
    for face in rface_tangent do
      var left = rface_all[face].left

      var primitives_l = ComputePrimitives(rcell[left].solution_temp)

      var normal   = rface_all[face].area
      var tangent  = rface_all[face].tangent
      var binormal = rface_all[face].binormal
      normal[0] = -normal[0]
      normal[1] = -normal[1]
      normal[2] = -normal[2]

      var area_norm = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])

      var uboundary : double = 0.0
      uboundary += primitives_l[1] * normal[0] / area_norm
      uboundary += primitives_l[2] * normal[1] / area_norm
      uboundary += primitives_l[3] * normal[2] / area_norm

      var primitives_r : Solution
      primitives_r[0] = primitives_l[0]
      primitives_r[1] = primitives_l[1] - 2.0 * uboundary * normal[0] / area_norm
      primitives_r[2] = primitives_l[2] - 2.0 * uboundary * normal[1] / area_norm
      primitives_r[3] = primitives_l[3] - 2.0 * uboundary * normal[2] / area_norm
      primitives_r[4] = primitives_l[4]

      var flux : Solution = roe_flux_compute_flux(primitives_l, primitives_r,
                                                  normal, tangent, binormal)
      do
        for i = 0, 5 do flux[i] = -flux[i] end
        rcell[left].cell_flux += flux
      end

      --c.printf("Tangent Sol: Cell %lld\t%9.2e\t%9.2e\t%9.2e\t\t%12.5e\t%12.5e\t%12.5e\t%12.5e\t%12.5e\n",
      --         c.coord_t(rcell[left]),
      --         normal[0], normal[1], normal[2], primitives_l[0], primitives_l[1], primitives_l[2], primitives_l[3], primitives_l[4])
      --c.printf("Tangent Flux: Cell %lld\t%9.2e\t%9.2e\t%9.2e\t\t%12.5e\t%12.5e\t%12.5e\t%12.5e\t%12.5e\n",
      --         c.coord_t(rcell[left]),
      --         normal[0], normal[1], normal[2], flux[0], flux[1], flux[2], flux[3], flux[4])
    end
  end
  do
    for face in rface_extrapolate do
      var left = rface_all[face].left

      var normal   = rface_all[face].area
      var tangent  = rface_all[face].tangent
      var binormal = rface_all[face].binormal
      for i = 0, 3 do normal[i] = -normal[i] end

      var primitives = ComputePrimitives(rcell[left].solution_temp)
      var flux : Solution = roe_flux_compute_flux(primitives, primitives,
                                                  normal, tangent, binormal)
      for i = 0, 5 do flux[i] = -flux[i] end
      rcell[left].cell_flux += flux

      --c.printf("Extrapolate Flux: Cell %lld\t%9.2e\t%9.2e\t%9.2e\t\t%12.5e\t%12.5e\t%12.5e\t%12.5e\t%12.5e\n",
      --         c.coord_t(rcell[left]),
      --         normal[0], normal[1], normal[2], flux[0], flux[1], flux[2], flux[3], flux[4])
    end
  end
  do
    for face in rface_inflow do
      var left = rface_all[face].left

      var primitives_l = ComputePrimitives(rcell[left].solution_temp)

      var conservatives_r : Solution
      conservatives_r[0] = 0.5805
      conservatives_r[1] = 503.96
      conservatives_r[2] = 0.0
      conservatives_r[3] = 0.0
      conservatives_r[4] = 343750.0

      var primitives_r = ComputePrimitives(conservatives_r)

      var normal   = rface_all[face].area
      var tangent  = rface_all[face].tangent
      var binormal = rface_all[face].binormal
      for i = 0, 3 do normal[i] = -normal[i] end

      var flux : Solution = roe_flux_compute_flux(primitives_l, primitives_r,
                                                  normal, tangent, binormal)
      for i = 0, 5 do flux[i] = -flux[i] end
      rcell[left].cell_flux += flux

      --c.printf("Inflow Flux: Cell %lld\t%9.2e\t%9.2e\t%9.2e\t\t%12.5e\t%12.5e\t%12.5e\t%12.5e\t%12.5e\n",
      --         c.coord_t(rcell[left]),
      --         normal[0], normal[1], normal[2], flux[0], flux[1], flux[2], flux[3], flux[4])
    end
  end
  do
    for face in rface_noslip do
      var left = rface_all[face].left

      var normal = rface_all[face].area
      for i = 0, 3 do normal[i] = -normal[i] end

      var area_norm = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])

      var primitives_l = ComputePrimitives(rcell[left].solution_temp)

      var uboundary : double = 0.0
      for i = 0, 3 do
        uboundary += primitives_l[i + 1] * normal[i] / area_norm
      end

      var primitives_r : Solution
      primitives_r[0] = primitives_l[0]
      for i = 0, 3 do
        primitives_r[i + 1] =
          primitives_l[i + 1] - 2.0 * uboundary * normal[i] / area_norm
      end
      primitives_r[4] = primitives_l[4]

      var tangent  = rface_all[face].tangent
      var binormal = rface_all[face].binormal
      var iflux : Solution = roe_flux_compute_flux(primitives_l, primitives_r,
                                                   normal, tangent, binormal)
      var vflux : Solution
      if is_viscous then
        var primitives_face : double[5]
        primitives_face[0] = primitives_l[0]
        primitives_face[1] = 0.0
        primitives_face[2] = 0.0
        primitives_face[3] = 0.0
        primitives_face[4] = primitives_l[4]

        var face_coordinates = rface_all[face].face_centroid
        var cell_coordinates = rcell[left].cell_centroid
        var delta : Vec3
        for i = 0, 3 do delta[i] = face_coordinates[i] - cell_coordinates[i] end
        var distance_to_wall : double =
          delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
        var inv_distance_to_wall = 1.0 / sqrt(distance_to_wall)

        var unit_normal : Vec3
        for i = 0, 3 do unit_normal[i] = normal[i] / area_norm end

        var gradients_face : Gradient
        for i = 0, 5 do
          for dir = 0, 3 do
            gradients_face[i * 3 + dir] =
              (primitives_face[i] - primitives_l[i]) * unit_normal[dir] * inv_distance_to_wall
          end
        end

        vflux =
          newtonian_viscous_flux_compute_flux(gradients_face,
                                              primitives_face,
                                              normal)
      else
        for i = 0, 5 do vflux[i] = 0.0 end
      end

      var flux : Solution
      do
        for i = 0, 5 do flux[i] = -iflux[i] + vflux[i] end
      end
      rcell[left].cell_flux += flux

      --c.printf("Noslip Flux: Cell %lld\t%9.2e\t%9.2e\t%9.2e\t\t%12.5e\t%12.5e\t%12.5e\t%12.5e\t%12.5e\n",
      --         c.coord_t(rcell[left]),
      --         normal[0], normal[1], normal[2],
      --         -iflux[0] + vflux[0],
      --         -iflux[1] + vflux[1],
      --         -iflux[2] + vflux[2],
      --         -iflux[3] + vflux[3],
      --         -iflux[4] + vflux[4])
    end
  end
end

task print_bc_flux(warmup: bool, rcell : region(Cell))
where
  reads(rcell.cell_flux)
do
  --if warmup then return end
  for cell in rcell do
    var cellptr = c.coord_t(cell)
    var sol = cell.cell_flux
      c.printf("Cell Flux: Cell %lld\t%12.5e\t%12.5e\t%12.5e\t%12.5e\t%12.5e\n",
               cellptr, sol[0], sol[1], sol[2], sol[3], sol[4])
  end
end

__demand(__parallel, __cuda)
task compute_residuals_and_update_rk_stage_beta(dt       : double,
                                                rk       : int,
                                                beta_    : double[4],
                                                rcell    : region(Cell))
where
  reads(rcell.{volume, cell_flux, residual, solution_np1}),
  writes(rcell.{residual, solution_np1, solution_n})
do
  var beta = beta_[rk]
  for cell in rcell do
    var volume = cell.volume
    var c_flux = cell.cell_flux
    var res : Solution
    var sol_np1 = cell.solution_np1
    for i = 0, 5 do
      res[i] = dt * c_flux[i] / volume
      sol_np1[i] += beta * res[i]
    end
    cell.residual = res
    cell.solution_np1 = sol_np1
    --var cellptr = c.coord_t(cell)
    --c.printf("Res: Cell %lld\t%12.5e\t%12.5e\t%12.5e\t%12.5e\t%12.5e\n",
    --         cellptr, res[0], res[1], res[2], res[3], res[4])
    --c.printf("Sol: Cell %lld\t%12.5e\t%12.5e\t%12.5e\t%12.5e\t%12.5e\n",
    --         cellptr, sol_np1[0], sol_np1[1], sol_np1[2], sol_np1[3], sol_np1[4])
  end
  if rk == 3 then
    for cell in rcell do
      cell.solution_n = cell.solution_np1
    end
  end
end

task output_solution(rcell : region(Cell))
where
  reads(rcell.{solution_n, cell_centroid})
do
  var outfile = c.fopen("results", "w")
  for cell in rcell do
    var centroid = cell.cell_centroid
    var sol_n = cell.solution_n
    c.fprintf(outfile,
        "%13e\t%13e\t%13e\t%13e\t%13e\t%13e\t%13e\t%13e\t\n",
        centroid[0], centroid[1], centroid[2],
        sol_n[0], sol_n[1], sol_n[2], sol_n[3], sol_n[4])
  end
  c.fclose(outfile)
end

-------------------------------------------------------------------------------
-- toplevel task
-------------------------------------------------------------------------------

terra wait_for(x : int)
  return 1
end
wait_for.replicable = true

task print_setup_time(color : int, setup_time : double)
  if color == 0 then
    c.printf("\nSetup Time = %10.6f s\n\n", setup_time)
  end
end

task print_exec_time(color : int, sim_time : double)
  if color == 0 then
    c.printf("ELAPSED TIME = %7.3f s\n", sim_time)
  end
end

__demand(__inner, __replicable)
task toplevel()
  var interface : Interface = default_options()
  interface = parse_options(interface)
  for i = 0, interface.num_blocks do
    print_options(i, interface)
  end

  --c.printf("Main task start\n")
  var num_blocks = interface.num_blocks
  var global_num_x = interface.interval_x
  var global_num_y = interface.interval_y
  var global_num_z = interface.interval_z

  var len_x = interface.length_x
  var len_y = interface.length_y
  var len_z = interface.length_z
  var ramp_angle = interface.ramp_angle
  var problem_type = interface.problem_type

  var ts_setup_start = c.legion_get_current_time_in_micros()
  var pmesh : &MeshTopology = mesh_init(num_blocks, global_num_x, global_num_y, global_num_z,
                                        len_x, len_y, len_z, ramp_angle, problem_type)
  --mesh_print(pmesh)

  var cells = region(ispace(ptr, mesh_global_cell_count(pmesh)), Cell)
  var faces = region(ispace(ptr, mesh_global_face_count(pmesh)), Face)
  var colorings : Colorings = create_colorings(pmesh)
  var is_blocks = ispace(int1d, num_blocks)

  var cell_partition  = partition(disjoint, cells, colorings.cell_coloring, is_blocks)
  var face_partition  = partition(disjoint, faces, colorings.face_coloring, is_blocks)

  var is_categories = ispace(int1d, BC_CATEGORY_COUNT)
  var faces_by_category  = partition(disjoint, faces, colorings.face_category, is_categories)
  var faces_interior = faces_by_category[BC_INTERIOR]
  var faces_tangent = faces_by_category[BC_TANGENT]
  var faces_extrapolate = faces_by_category[BC_EXTRAPOLATE]
  var faces_inflow = faces_by_category[BC_INFLOW]
  var faces_noslip = faces_by_category[BC_NOSLIP]

  var blocks = region(ispace(ptr, num_blocks), Block)
  var _ = init_blocks(pmesh, blocks)
  wait_for(_)

  --
  --build_mesh_datastructure
  --
  var mesh : MeshTopology = mesh_deref(pmesh)
  c.free(pmesh.blocks_)
  c.free(pmesh)

  var block_coloring = coloring_create()
  for n = 0, num_blocks do
    coloring_add_range(block_coloring, [int1d](n), ptr_t { value = 0 },
                       ptr_t { value = num_blocks - 1} )
  end
  var block_partition = partition(aliased, blocks, block_coloring, is_blocks)
  var extentX = mesh.length_[X]

  var rk4 : RungaKutta4 = init_rk4()
  var dt = interface.dt
  var prune = interface.prune
  var max_its = interface.time_steps + 2 * prune
  var time : double = 0.0
  var is_viscous = interface.viscous
  var second_order = interface.second_order_space
  var init_minmax = is_viscous or second_order

  var stages = rk4.stages_
  var alpha = rk4.alpha_
  var beta = rk4.beta_

  for n = 0, num_blocks do
    initialize(n, mesh, problem_type, extentX,
               block_partition[n], cell_partition[n], face_partition[n])
  end

  __fence(__execution, __block)
  var ts_setup_end = c.legion_get_current_time_in_micros()
  var setup_time = 1e-6 * (ts_setup_end - ts_setup_start)
  for i = 0, num_blocks do
    print_setup_time(i, setup_time)
  end

  var ts_start = c.legion_get_current_time_in_micros()
  var ts_end = ts_start
  __demand(__spmd) do __parallelize_with is_blocks do
    for time_it = 0, max_its do

      if time_it == prune then
        __fence(__execution, __block)
        ts_start = c.legion_get_current_time_in_micros()
      end

      --__demand(__trace)
      for rk = 0, stages do
        update_rk_stage_alpha_and_initialize_solution_fields(cells,
                                                             rk,
                                                             alpha,
                                                             init_minmax)
        if second_order or is_viscous then
            face_gradient(cells,
                          faces,
                          faces_interior,
                          faces_tangent,
                          faces_extrapolate,
                          faces_inflow,
                          faces_noslip)
            compute_min_max(cells,
                            faces,
                            faces_interior,
                            faces_tangent,
                            faces_extrapolate,
                            faces_inflow,
                            faces_noslip)
            compute_limiter(cells,
                            faces,
                            faces_interior)
        end -- if second_order or is_viscous
        compute_face_flux(is_viscous,
                          second_order,
                          cells,
                          faces,
                          faces_interior,
                          faces_tangent,
                          faces_extrapolate,
                          faces_inflow,
                          faces_noslip)
        compute_residuals_and_update_rk_stage_beta(dt,
                                                   rk,
                                                   beta,
                                                   cells)
      end -- for rk = 0, 4

      if time_it == max_its - 1 - prune then
        __fence(__execution, __block)
        ts_end = c.legion_get_current_time_in_micros()
      end
    end -- for time_it = 0, max_its + 1
  end end

  --c.printf("Main task end\n")
  var sim_time : double = 1e-6 * (ts_end - ts_start)
  for i = 0, num_blocks do
    print_exec_time(i, sim_time)
  end

  if interface.output_results then
    output_solution(cells)
  end
end

launcher.launch(toplevel, "miniaero_sequential", cmapper.register_mappers, {"-lminiaero_sequential"})
