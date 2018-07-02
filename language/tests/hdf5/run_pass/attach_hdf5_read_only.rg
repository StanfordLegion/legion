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

task fill_region(r : region(ispace(int3d), t), seed : int32)
where writes(r.{a,b,c}) do
  for p in r do
    r[p].a = 1000 * seed + p.x + 10 * p.y + 100 * p.z
    r[p].b = 1000 * seed + 5 * p.x + 50 * p.y + 500 * p.z
    r[p].c = 1000 * seed + 0.1 * p.x + 0.01 * p.y + 0.001 * p.z
  end
end

task compare_regions(is : ispace(int3d), r1 : region(is, t), r2 : region(is, t))
where reads(r1.{a,b,c}), reads(r2.{a,b,c}) do
  var errors = 0
  for p in is do
    if(r1[p].a ~= r2[p].a) then
      errors += 1
      regentlib.c.printf("[%d,%d,%d]: a mismatch - %d %d\n", p.x, p.y, p.z, r1[p].a, r2[p].a)
    end
  end
  regentlib.assert(errors == 0, "test failed")
end

task main()
  var dims : int3d = { 2, 3, 4 }
  var is = ispace(int3d, dims)
  var r1 = region(is, t)
  var r2 = region(is, t)

  generate_hdf5_file(filename, dims)

  -- test 1: attach in read-only mode and acquire/release
  --  (should make a local copy)
  if true then
    regentlib.c.printf("test 1\n")
    fill_region(r1, 2)
    for x in r2 do x.{a, b, c} = 1 end -- force an inline mapping
    attach(hdf5, r2.{a, b, c}, filename, regentlib.file_read_only)
    acquire(r2)
    copy(r2.a, r1.a)
    copy(r2.b, r1.b)
    copy(r2.c, r1.c)
    compare_regions(is, r1, r2)
    release(r2)
    detach(hdf5, r2.{a, b, c})
  end
end

regentlib.start(main)
