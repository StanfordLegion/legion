-- Copyright 2018 Stanford University, Los Alamos National Laboratory
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

import "regent"
local c = regentlib.c
local hdf5 = terralib.includec(os.getenv("HDF_HEADER") or "hdf5.h")

local NX = 180
local HALO_WIDTH = 2

local LEFT_HALO = 0
local DOMAIN = 1
local RIGHT_HALO = 2

local filename="oops.h5"

fspace GridData { x : double }

hdf5.H5F_ACC_TRUNC = 2
hdf5.H5T_IEEE_F64LE = hdf5.H5T_IEEE_F64LE_g
hdf5.H5P_DEFAULT = 0

terra init_hdf5_file(dims : int1d)

  var status : hdf5.herr_t

  var file_id = hdf5.H5Fcreate(filename, hdf5.H5F_ACC_TRUNC, hdf5.H5P_DEFAULT, hdf5.H5P_DEFAULT)

  var h_dims : hdf5.hsize_t[1]
  h_dims[0] = dims.__ptr

  var dataspace_id = hdf5.H5Screate_simple(1, h_dims, [&uint64](0))

  var x_dataset_id = hdf5.H5Dcreate2(file_id, "x", hdf5.H5T_IEEE_F64LE, dataspace_id, hdf5.H5P_DEFAULT, hdf5.H5P_DEFAULT, hdf5.H5P_DEFAULT)
  status = hdf5.H5Dclose(x_dataset_id)

  status = hdf5.H5Sclose(dataspace_id)

  status = hdf5.H5Fclose(file_id)

end

__demand(__inline)
task writeGrid(grid : region(ispace(int1d), GridData))
where reads(grid.x) do

  var limits : rect1d = grid.bounds
  var grid_dims = limits.hi - limits.lo + 1

  init_hdf5_file(grid_dims)
  var write_data = region(grid.ispace, GridData)
  attach(hdf5, write_data.x, filename, regentlib.file_read_write)
  copy(grid.x, write_data.x)
  detach(hdf5, write_data.x)

end

task main()
  var grid = region(ispace(int1d, NX + 2 * HALO_WIDTH), GridData)

  var coloring = c.legion_domain_point_coloring_create()
  c.legion_domain_point_coloring_color_domain(coloring, [int1d](LEFT_HALO), rect1d { 0, HALO_WIDTH - 1 })
  c.legion_domain_point_coloring_color_domain(coloring, [int1d](RIGHT_HALO),
                                              rect1d { NX + HALO_WIDTH, NX + 2 * HALO_WIDTH -1 })
  c.legion_domain_point_coloring_color_domain(coloring, [int1d](DOMAIN),
                                              rect1d { HALO_WIDTH, NX + HALO_WIDTH -1 })
  var halo_partition = partition(disjoint, grid, coloring, ispace(int1d, 3))
  c.legion_domain_point_coloring_destroy(coloring)

  writeGrid(halo_partition[DOMAIN])
end

regentlib.start(main)
