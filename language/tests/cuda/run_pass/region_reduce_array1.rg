-- Copyright 2024 Stanford University, NVIDIA Corporation
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
-- [["-ll:gpu", "1" ]]

import "regent"

fspace fs
{
  a : int[5],
  b : int,
}

task init(r : region(ispace(int2d), fs))
where reads writes(r)
do
  for e in r do
    e.a = array(1, 1, 1, 1, 1)
    e.b = 2
  end
end

__demand(__cuda)
task reduce(r : region(ispace(int2d), fs))
where reduces+(r.a)
do
  var inc : int[5]
  for i = 0, 5 do inc[i] = 2 + i end
  for e in r do e.a += inc end
end

-- task on device with read/write privileges forces the reductions to happen
--   in GPU memory
__demand(__cuda)
task update(r : region(ispace(int2d), fs))
where reads writes(r)
do
  for e in r do
    e.a += array(1, 1, 1, 1, 1)
    e.b += 2
  end
end

task check(r : region(ispace(int2d), fs))
where reads(r)
do
  for e in r do
    for i = 0, 5 do
      regentlib.assert(e.a[i] == 2 + (i + 2) * 2, "test failed")
    end
    regentlib.assert(e.b == 4, "test failed")
  end
end

task main()
  var r = region(ispace(int2d, {4, 4}), fs)
  var p = partition(equal, r, ispace(int2d, {2, 2}))
  for c in p.colors do init(p[c]) end
  for c in p.colors do reduce(p[c]) end
  for c in p.colors do reduce(p[c]) end
  for c in p.colors do update(p[c]) end
  for c in p.colors do check(p[c]) end
end

regentlib.start(main)
