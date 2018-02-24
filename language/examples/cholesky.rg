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

-- runs-with:
-- [
--   ["-verify", "-ll:cpu", "4", "-fflow", "0"],
--   ["-p", "1", "-verify", "-fflow", "0"]
-- ]

-- FIXME: Breaks RDIR

import "regent"

local blas = terralib.includecstring [[
extern void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha,
                   double* A, int* lda, double* B, int* ldb, double* beta,
                   double* C, int* ldc);

extern void dpotrf_(char *uplo, int *n, double *A, int *lda, int *info);

extern void dtrsm_(char* side, char *uplo, char* transa, char* diag,
                   int* m, int* n, double* alpha,
                   double *A, int *lda, double *B, int *ldb);

extern void dsyrk_(char *uplo, char* trans, int* n, int* k,
                   double* alpha, double *A, int *lda,
                   double* beta, double *C, int *ldc);

]]

if os.execute("bash -c \"[ `uname` == 'Darwin' ]\"") == 0 then
  terralib.linklibrary("libblas.dylib")
  terralib.linklibrary("liblapack.dylib")
else
  terralib.linklibrary("libblas.so")
  terralib.linklibrary("liblapack.so")
end

local c = regentlib.c
local cstr = terralib.includec("string.h")
local cmath = terralib.includec("math.h")
local std = terralib.includec("stdlib.h")
rawset(_G, "drand48", std.drand48)
rawset(_G, "srand48", std.srand48)

-- declare fortran-order 2D indexspace
local struct __f2d { y : int, x : int }
local f2d = regentlib.index_type(__f2d, "f2d")

task make_pds_matrix(p : f2d, n : int, rA : region(ispace(f2d), double))
where reads writes(rA)
do
  var is = rA.ispace
  for p in rA.ispace do
    if p.x <= p.y then
      rA[p] = [int](10.0 * drand48())
      if p.x == p.y then
        rA[p] += n * 10
      end
      rA[f2d { x = p.y, y = p.x }] = rA[p]
    end
  end
end

task make_random_matrix(p : f2d, rA : region(ispace(f2d), double))
where reads writes(rA)
do
  for p in rA.ispace do
    rA[p] = [int](10.0 * drand48())
  end
end

task transpose_copy(rSrc : region(ispace(f2d), double), rDst : region(ispace(f2d), double))
where reads(rSrc), writes(rDst)
do
  for p in rSrc.ispace do
    rDst[f2d { y = p.x, x = p.y }] = rSrc[p]
  end
end

function raw_ptr_factory(ty)
  local struct raw_ptr
  {
    ptr : &ty,
    offset : int,
  }
  return raw_ptr
end

local raw_ptr = raw_ptr_factory(double)

terra get_raw_ptr(y : int, x : int, bn : int,
                  pr : c.legion_physical_region_t,
                  fld : c.legion_field_id_t)
  var fa = c.legion_physical_region_get_field_accessor_array_2d(pr, fld)
  var rect : c.legion_rect_2d_t
  var subrect : c.legion_rect_2d_t
  var offsets : c.legion_byte_offset_t[2]
  rect.lo.x[0] = y * bn
  rect.lo.x[1] = x * bn
  rect.hi.x[0] = (y + 1) * bn - 1
  rect.hi.x[1] = (x + 1) * bn - 1
  var ptr = c.legion_accessor_array_2d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[1].offset / sizeof(double) }
end

terra dpotrf_terra(x : int, n : int, bn : int,
                   pr : c.legion_physical_region_t,
                   fld : c.legion_field_id_t)
  var uplo : rawstring = 'L'
  var n_ : int[1], bn_ : int[1]
  n_[0], bn_[0] = n, bn
  var info : int[1]

  var rawA = get_raw_ptr(x, x, bn, pr, fld)

  blas.dpotrf_(uplo, bn_, rawA.ptr, &(rawA.offset), info)
end

task dpotrf(x : int, n : int, bn : int, rA : region(ispace(f2d), double))
where reads writes(rA)
do
  dpotrf_terra(x, n, bn, __physical(rA)[0], __fields(rA)[0])
end

terra dtrsm_terra(x : int, y : int, n : int, bn : int,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t)

  var side : rawstring = 'R'
  var uplo : rawstring = 'L'
  var transa : rawstring = 'T'
  var diag : rawstring = 'N'
  var n_ : int[1], bn_ : int[1]
  n_[0], bn_[0] = n, bn
  var alpha : double[1] = array(1.0)

  var rawA = get_raw_ptr(y, x, bn, prA, fldA)
  var rawB = get_raw_ptr(x, x, bn, prB, fldB)

  blas.dtrsm_(side, uplo, transa, diag, bn_, bn_, alpha,
              rawB.ptr, &(rawB.offset), rawA.ptr, &(rawA.offset))
end

task dtrsm(x : int, y : int, n : int, bn : int,
           rA : region(ispace(f2d), double),
           rB : region(ispace(f2d), double))
where reads writes(rA), reads(rB)
do
  dtrsm_terra(x, y, n, bn, __physical(rA)[0], __fields(rA)[0],
              __physical(rB)[0], __fields(rB)[0])
end

terra dsyrk_terra(x : int, k : int, n : int, bn : int,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t)

  var uplo : rawstring = 'L'
  var trans : rawstring = 'N'
  var n_ : int[1], bn_ : int[1]
  n_[0], bn_[0] = n, bn
  var alpha : double[1] = array(-1.0)
  var beta : double[1] = array(1.0)

  var rawA = get_raw_ptr(k, k, bn, prA, fldA)
  var rawB = get_raw_ptr(k, x, bn, prB, fldB)

  blas.dsyrk_(uplo, trans, bn_, bn_,
              alpha, rawB.ptr, &(rawB.offset),
              beta, rawA.ptr, &(rawA.offset))
end

task dsyrk(x : int, k : int, n : int, bn : int,
           rA : region(ispace(f2d), double),
           rB : region(ispace(f2d), double))
where reads writes(rA), reads(rB)
do
  dsyrk_terra(x, k, n, bn, __physical(rA)[0], __fields(rA)[0],
              __physical(rB)[0], __fields(rB)[0])
end

terra dgemm_terra(x : int, y : int, k : int,
                  n : int, bn : int,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t,
                  prC : c.legion_physical_region_t,
                  fldC : c.legion_field_id_t)

  var transa : rawstring = 'N'
  var transb : rawstring = 'T'
  var n_ : int[1], bn_ : int[1]
  n_[0], bn_[0] = n, bn
  var alpha : double[1] = array(-1.0)
  var beta : double[1] = array(1.0)

  var rawA = get_raw_ptr(y, k, bn, prA, fldA)
  var rawB = get_raw_ptr(y, x, bn, prB, fldB)
  var rawC = get_raw_ptr(k, x, bn, prC, fldC)

  blas.dgemm_(transa, transb, bn_, bn_, bn_,
              alpha, rawB.ptr, &(rawB.offset),
              rawC.ptr, &(rawC.offset),
              beta, rawA.ptr, &(rawA.offset))
end


task dgemm(x : int, y : int, k : int, n : int, bn : int,
           rA : region(ispace(f2d), double),
           rB : region(ispace(f2d), double),
           rC : region(ispace(f2d), double))
where reads writes(rA), reads(rB, rC)
do
  dgemm_terra(x, y, k, n, bn,
              __physical(rA)[0], __fields(rA)[0],
              __physical(rB)[0], __fields(rB)[0],
              __physical(rC)[0], __fields(rC)[0])
end

task verify_result(n : int,
                   org : region(ispace(f2d), double),
                   res : region(ispace(f2d), double))
where reads(org, res)
do
  c.printf("verifying results...\n")
  for x = 0, n do
    for y = x, n do
      var v = org[f2d { x = x, y = y }]
      var sum : double = 0
      for k = 0, x + 1 do
        sum += res[f2d { x = k, y = y }] * res[f2d { x = k, y = x }]
      end
      if cmath.fabs(sum - v) > 1e-6 then
        c.printf("error at (%d, %d) : %.3f, %.3f\n", y, x, sum, v)
      end
    end
  end
end

task cholesky(n : int, np : int, verify : bool)
  regentlib.assert(n % np == 0, "tile sizes should be uniform")
  var is = ispace(f2d, { x = n, y = n })
  var cs = ispace(f2d, { x = np, y = np })
  var rA = region(is, double)
  var rB = region(is, double)

  var pA = partition(equal, rA, cs)
  var pB = partition(equal, rB, cs)

  for x = 0, np do
    make_pds_matrix(f2d { x = x, y = x }, n, pA[f2d { x = x, y = x }])
    for y = x + 1, np do
      make_random_matrix(f2d { x = x, y = y }, pA[f2d { x = x, y = y }])
      var src = pA[f2d { x = x, y = y }]
      var dst = pA[f2d { x = y, y = x }]
      transpose_copy(src, dst)
    end
  end

  for c in cs do
    var src = pA[c]
    var dst = pB[c]
    copy(src, dst)
  end

  __fence(__execution, __block)
  var ts_start = c.legion_get_current_time_in_micros()

  var bn = n / np
  for x = 0, np do
    dpotrf(x, n, bn, pB[f2d { x = x, y = x }])
    for y = x + 1, np do
      dtrsm(x, y, n, bn, pB[f2d { x = x, y = y }], pB[f2d { x = x, y = x }])
    end
    for k = x + 1, np do
      dsyrk(x, k, n, bn, pB[f2d { x = k, y = k }], pB[f2d { x = x, y = k }])
      for y = k + 1, np do
        dgemm(x, y, k, n, bn,
              pB[f2d { x = k, y = y }],
              pB[f2d { x = x, y = y }],
              pB[f2d { x = x, y = k }])
      end
    end
  end

  __fence(__execution, __block)
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("ELAPSED TIME = %7.3f ms\n", 1e-3 * (ts_end - ts_start))

  if verify then verify_result(n, rA, rB) end
end

task toplevel()
  var n = 8
  var np = 4
  var verify = false

  var args = c.legion_runtime_get_input_args()
  for i = 0, args.argc do
    if cstr.strcmp(args.argv[i], "-n") == 0 then
      n = std.atoi(args.argv[i + 1])
    elseif cstr.strcmp(args.argv[i], "-p") == 0 then
      np = std.atoi(args.argv[i + 1])
    elseif cstr.strcmp(args.argv[i], "-verify") == 0 then
      verify = true
    end
  end

  cholesky(n, np, verify)
end

regentlib.start(toplevel)
