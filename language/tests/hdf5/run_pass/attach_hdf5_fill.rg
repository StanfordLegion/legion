-- Copyright 2018 Stanford University
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

local c = terralib.includec("assert.h")

local hdf5 = terralib.includec(os.getenv("HDF_HEADER") or "hdf5.h")
-- there's some funny business in hdf5.h that prevents terra from being able to
--  see some of the #define's, so we fix it here, and hope the HDF5 folks don't
--  change the internals very often...
hdf5.H5F_ACC_TRUNC = 2
hdf5.H5T_STD_I32LE = hdf5.H5T_STD_I32LE_g
hdf5.H5T_STD_I64LE = hdf5.H5T_STD_I64LE_g
hdf5.H5T_IEEE_F64LE = hdf5.H5T_IEEE_F64LE_g
hdf5.H5P_DEFAULT = 0

fspace t {
  a : int32,
  b : int64,
  c : double,
}

local filename = os.tmpname() .. ".hdf"

terra generate_hdf5_file(filename : rawstring, dims : int3d)
  var fid = hdf5.H5Fcreate(filename, hdf5.H5F_ACC_TRUNC, hdf5.H5P_DEFAULT, hdf5.H5P_DEFAULT)
  --c.assert(fid > 0)

  -- Legion defaults to Fortran-style (column-major) layout, so we have to reverse
  --  the dimensions when calling directly into the HDF5 C API
  var h_dims : hdf5.hsize_t[3]
  h_dims[2] = dims.__ptr.x
  h_dims[1] = dims.__ptr.y
  h_dims[0] = dims.__ptr.z
  var did = hdf5.H5Screate_simple(3, h_dims, [&uint64](0))
  --c.assert(did > 0)

  var ds1id = hdf5.H5Dcreate2(fid, "a", hdf5.H5T_STD_I32LE, did,
                              hdf5.H5P_DEFAULT, hdf5.H5P_DEFAULT, hdf5.H5P_DEFAULT)
  --c.assert(ds1id > 0)
  hdf5.H5Dclose(ds1id)

  var ds2id = hdf5.H5Dcreate2(fid, "b", hdf5.H5T_STD_I64LE, did,
                              hdf5.H5P_DEFAULT, hdf5.H5P_DEFAULT, hdf5.H5P_DEFAULT)
  --c.assert(ds2id > 0)
  hdf5.H5Dclose(ds2id)

  var ds3id = hdf5.H5Dcreate2(fid, "c", hdf5.H5T_IEEE_F64LE, did,
                              hdf5.H5P_DEFAULT, hdf5.H5P_DEFAULT, hdf5.H5P_DEFAULT)
  --c.assert(ds3id > 0)
  hdf5.H5Dclose(ds3id)

  hdf5.H5Sclose(did)
  hdf5.H5Fclose(fid)
end

task main()
  var dims : int3d = { 2, 3, 4 }
  var is = ispace(int3d, dims)
  var r1 = region(is, t)


  generate_hdf5_file(filename, dims)
  
  attach(hdf5, r1.{a, b, c}, filename, regentlib.file_read_write)
  fill(r1.a, 1)
  fill(r1.b, -1)
  fill(r1.c, 2.0)

  -- subregion fills
  var cs1 = ispace(int3d, {2, 1, 1})
  var p1 = partition(equal, r1, cs1)
  var i1 : int3d = { 1, 0, 0 }
  fill((p1[i1]).a, 5)

  var cs2 = ispace(int3d, {1, 3, 1})
  var p2 = partition(equal, r1, cs2)
  var i2 : int3d = { 0, 1, 0 }
  fill((p2[i2]).b, 6)

  var cs3 = ispace(int3d, {1, 1, 4})
  var p3 = partition(equal, r1, cs3)
  var i3 : int3d = { 0, 0, 1 }
  fill((p3[i3]).c, 7)

  detach(hdf5, r1.{a, b, c})
end

regentlib.start(main)
