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

fspace ret(parent_region : region(ispace(int1d), int),
           color_space : ispace(int1d))
{
  v : int,
  id : uint64,
  part : partition(disjoint, parent_region, color_space),
}

__demand(__inline)
task foo(x : region(ispace(int1d), int),
         cs : ispace(int1d),
         part : partition(disjoint, x, cs),
         k : int)
where reads writes(x) do
  for c in cs do
    fill((part[c]), k)
    var r = part[c]
    for p in r.ispace do
      r[p] += [int](c)
    end
  end
  var sum = 0
  for e in x do
    sum += @e
  end
  return [ret(x, cs)] {
    v = sum,
    id = c.legion_context_get_unique_id(__context()),
    part = partition(equal, x, cs),
  }
end

task main()
  var id_main = c.legion_context_get_unique_id(__context())
  var size : int1d = 4
  var is = ispace(int1d, size)
  var cs = ispace(int1d, size)
  var r = region(is, int)
  var part = partition(equal, r, cs)

  var x = 0
  var ret_foo = foo(r, cs, part, 10)
  regentlib.assert(id_main == ret_foo.id, "test failed")
  x += ret_foo.v
  ret_foo = foo(r, cs, ret_foo.part, 20)
  regentlib.assert(id_main == ret_foo.id, "test failed")
  x += ret_foo.v
  regentlib.assert(x == 132, "test failed")
end

regentlib.start(main)

