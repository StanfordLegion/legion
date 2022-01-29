-- Copyright 2022 Stanford University
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
-- [["-fmapping", "0"]]

-- FIXME: This test breaks inline optimization.

import "regent"

-- This code has not been optimized and is not high performance.

terra abs(a : double) : double
  if a < 0 then
    return -a
  else
    return a
  end
end

task dgemm(isl : ispace(int1d),
           ismn : ispace(int2d),
           a : region(ispace(int2d), double),
           b : region(ispace(int2d), double),
           c : region(ismn, double))
where
  reads(a, b, c),
  writes(c)
do
  for mn in ismn do
    for l in isl do
      var ia = int2d {x = l, y = mn.y}
      var ib = int2d {x = mn.x, y = l}
      var ic = mn
      c[ic] += a[ia] * b[ib]
    end
  end
end

task test(l: int, m : int, n : int, bl : int, bm : int, bn : int)
  var isl, ism, isn = ispace(int1d, l), ispace(int1d, m), ispace(int1d, n)
  var isln = ispace(int2d, { x = l, y = n })
  var isml = ispace(int2d, { x = m, y = l })
  var ismn = ispace(int2d, { x = m, y = n })
  var a = region(isln, double)
  var b = region(isml, double)
  var c = region(ismn, double)

  var cln = ispace(int2d, { x = bl, y = bn })
  var cml = ispace(int2d, { x = bm, y = bl })
  var cmn = ispace(int2d, { x = bm, y = bn })
  var pa = partition(equal, a, cln)
  var pb = partition(equal, b, cml)
  var pc = partition(equal, c, cmn)

  var rl = region(isl, bool) -- FIXME: Need index space partitioning.
  var cl = ispace(int1d, bl)
  var pl = partition(equal, rl, cl)

  fill(a, 1.0)
  fill(b, 1.0)
  fill(c, 0.0)

  for mn in cmn do
    for l in cl do
      var ia = int2d {x = l, y = mn.y}
      var ib = int2d {x = mn.x, y = l}
      var ic = mn
      var sl = pl[l]
      var sa = pa[ia]
      var sb = pb[ib]
      var sc = pc[ic]

      dgemm(sl.ispace, sc.ispace, sa, sb, sc)
    end
  end

  for ic in c do
    regentlib.assert(abs(c[ic] - l) < 0.00001, "test failed")
  end
end

task main()
  test(3, 4, 5, 1, 2, 3)
  test(10, 10, 20, 2, 5, 4)
end
regentlib.start(main)
