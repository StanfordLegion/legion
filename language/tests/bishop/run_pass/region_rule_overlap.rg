-- Copyright 2023 Stanford University, NVIDIA Corporation
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
import "bishop"

local c = bishoplib.c

mapper

$p = processors[isa=x86][0]

task#foo region#r {
  create : demand;
}

task#foo region#r {
  create : forbid;
}

task {
  target : $p;
}

end

__demand(__inline)
task get_ptr(r : region(int))
where reads writes(r)
do
  var pr = __physical(r)[0]
  var fd = __fields(r)[0]
  var accessor =
    c.legion_physical_region_get_field_accessor_array_1d(pr, fd)
  var p =
    c.legion_accessor_array_1d_ref(accessor,
                                c.legion_ptr_t { value = 0 })
  c.legion_accessor_array_1d_destroy(accessor)
  return p
end

task foo(r : region(int), prev : &opaque)
where reads writes(r)
do
  var p = get_ptr(r)

  -- the second rule should win
  bishoplib.assert(p == prev,
                   "r shouldn't be mapped to a different instance")
  return p
end

task toplevel()
  var r = region(ispace(ptr, 10), int)
  for e in r do @e = 10 end -- this will create an instance
  var p = get_ptr(r)
  p = foo(r, p)
  p = foo(r, p)
  p = foo(r, p)
end

regentlib.start(toplevel, bishoplib.make_entry())
