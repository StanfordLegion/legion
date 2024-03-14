-- Copyright 2024 Stanford University
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

struct t {
  a : int,
  b : int,
  c : int,
}

task f(r : region(t), p : partition(disjoint, r),
       is : regentlib.list(int), rs : regentlib.list(region(t)))
where reads writes(r, rs) do

  copy(rs.a, rs.b, *)
  copy(r.b, rs.b, +)
  for i in is do
    -- Because rs indexing is zero-based, we have to subtract the
    -- offset.
    copy((rs[i-3]).b, (p[i]).a, +)
  end
end

task main()
  var r = region(ispace(ptr, 7), t)
  var rc = c.legion_coloring_create()
  for i = 0, 7 do
    var x = dynamic_cast(ptr(t, r), i)
    x.a = i
    x.b = 10*i
    x.c = i
    c.legion_coloring_add_point(rc, i, __raw(x))
  end
  var p = partition(disjoint, r, rc)
  c.legion_coloring_destroy(rc)

  var is = list_range(3, 7)
  var rs = list_duplicate_partition(p, is)
  copy(r, rs)
  f(r, p, is, rs)

  for x in r do
    if x.c >= 3 then
      regentlib.assert(x.a == x.c * x.c * 10 + x.c * 10 + x.c, "test failed")
    else
      regentlib.assert(x.a == x.c, "test failed")
    end
  end
end
regentlib.start(main)
