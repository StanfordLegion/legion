-- Copyright 2023 Stanford University
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

-- fails-with:
-- undefined_field_type.rg:43: field type is undefined or nil
--   solution_n    : Solution,
--            ^

import "regent"

-- This test ensures that the compiler complains about fields
-- referencing types that are undefined.

-- Solution = double[5]

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

task face_gradient(rcell : region(Cell))
where reads writes(rcell.{cell_gradients, solution_temp, volume}) do
end
face_gradient:compile()
