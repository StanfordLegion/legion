-- Copyright 2015 Stanford University
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

-- This code has not been optimized and is not high performance.

terra abs(a : double) : double
  if a < 0 then
    return -a
  else
    return a
  end
end

task dgemm(isa : ispace(int2d), a: region(isa, double),
           isb : ispace(int2d), b: region(isb, double),
           isc : ispace(int2d), c: region(isc, double),
           isx : ispace(int1d))
where
  reads(a, b, c),
  writes(c)
do
  for ic in isc do
    for ix in isx do
      var ia = int2d {x = ix, y = ic.y}
      var ib = int2d {x = ic.x, y = ix}
      c[ic] += a[ia] * b[ib]
    end
  end
end

task test(l: int, m : int, n : int)
  var isa = ispace(int2d, { x = l, y = n })
  var isb = ispace(int2d, { x = m, y = l })
  var isc = ispace(int2d, { x = m, y = n })
  var isx = ispace(int1d, l)

  var a = region(isa, double)
  var b = region(isb, double)
  var c = region(isc, double)

  for ia in isa do
    a[ia] = 1.0
  end

  for ib in isb do
    b[ib] = 1.0
  end

  for ic in isc do
    c[ic] = 0.0
  end

  dgemm(isa, a, isb, b, isc, c, isx)

  for ic in isc do
    regentlib.assert(abs(c[ic] - l) < 0.00001, "test failed")
  end
end

task main()
  test(3, 4, 5)
  test(10, 10, 20)
end
regentlib.start(main)
