-- Copyright 2024 Stanford University, NVIDIA Coporation
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
-- [["-fflow", "0"]]

import "regent"

fspace fs
{
  c : int1d;
  v : int;
}

task init(r : region(ispace(int2d), fs))
where reads writes(r)
do
  for e in r do
    e.c = int1d((e.x / 8) % 2)
    e.v = 0
  end
end

task update(r : region(ispace(int2d), int))
where reads(r), writes(r)
do
  var v : int32 = 1
  __demand(__vectorize)
  for e in r do
    @e = @e + v
  end
end

task check(r : region(ispace(int2d), fs))
where reads(r.v)
do
  for e in r do
    regentlib.assert(e.v == 1, "test failed")
  end
end

task main()
  var r = region(ispace(int2d, {64, 1}), fs)
  init(r)
  var p = partition(r.c, ispace(int1d, 2))
  for point in p.colors do
    update(p[point].{v})
  end
  check(r)
end

regentlib.start(main)
