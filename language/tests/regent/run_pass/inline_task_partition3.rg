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
--   [ "-fflow", "0" ]
-- ]
import "regent"

local c = regentlib.c

fspace ret(parent_region : region(ispace(int1d), int))
{
  v : int,
  id : uint64,
  part : partition(disjoint, parent_region, ispace(int1d)),
}

__demand(__inline)
task foo(id_main : int64,
         x : region(ispace(int1d), int),
         part : partition(disjoint, x, ispace(int1d)),
         k : int)
where reads writes(x) do
  for c in part.colors do
    fill((part[c]), k)
    var r = part[c]
    for p in r.ispace do
      r[p] += [int](c)
    end
  end

  var id_foo = c.legion_context_get_unique_id(__context())
  regentlib.assert(id_main == id_foo, "test failed")

  return partition(equal, x, ispace(int1d, 4))
end

task main()
  var id_main = c.legion_context_get_unique_id(__context())
  var size : int1d = 4
  var is = ispace(int1d, size)
  var r = region(is, int)
  var part = partition(equal, r, ispace(int1d, 4))

  var part_foo1 = foo(id_main, r, part, 10)
  do
    var sum = 0
    for e in r do
      sum += @e
    end
    regentlib.assert(sum == 46, "test failed")
  end
  var part_foo2 = foo(id_main, r, part_foo1, 20)
  do
    var sum = 0
    for e in r do
      sum += @e
    end
    regentlib.assert(sum == 86, "test failed")
  end
end

regentlib.start(main)

