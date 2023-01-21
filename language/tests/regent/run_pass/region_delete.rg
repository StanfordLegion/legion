-- Copyright 2023 Stanford University
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

task f() : int
  var r = region(ispace(ptr, 5), int)
  var rc = c.legion_coloring_create()
  c.legion_coloring_add_point(rc, 0, c.legion_ptr_t { value = 0 })
  var p = partition(disjoint, r, rc)
  c.legion_coloring_destroy(rc)
  var r0 = p[0]

  var x = dynamic_cast(ptr(int, r0), 0)
  @x = 5
  var v = @x
  __delete(p)
  __delete(r)
  return v
end

task g() : int
  var r = region(ispace(ptr, 5), int)
  var rc = c.legion_coloring_create()
  c.legion_coloring_add_point(rc, 0, c.legion_ptr_t { value = 0 })
  var p = partition(disjoint, r, rc)
  c.legion_coloring_destroy(rc)
  var r0 = p[0]
  var r0c = c.legion_coloring_create()
  c.legion_coloring_add_point(r0c, 0, c.legion_ptr_t { value = 0 })
  var p0 = partition(disjoint, r0, r0c)
  c.legion_coloring_destroy(r0c)
  var r1 = p0[0]

  var x = dynamic_cast(ptr(int, r1), 0)
  @x = 5
  var v = @x
  __delete(p)
  __delete(p0)
  __delete(r)
  return v
end

task main()
  regentlib.assert(f() == 5, "test failed")
  regentlib.assert(g() == 5, "test failed")
end
regentlib.start(main)
