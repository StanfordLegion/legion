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

task#foo region#r1 {
  create : demand;
}

task#foo region#r2 {
  create : forbid;
}

task {
  target : $p;
}

end

struct ptrs
{
  ptr1 : &opaque,
  ptr2 : &opaque,
}

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

task foo(r1 : region(int), r2 : region(int), prev : ptrs)
where reads writes(r1, r2)
do
  var ptr1 = get_ptr(r1)
  var ptr2 = get_ptr(r2)

  bishoplib.assert(ptr1 ~= prev.ptr1,
                   "r1 should be mapped to a fresh instance")
  bishoplib.assert(ptr2 == prev.ptr2,
                   "r2 shouldn't be mapped to a different instance")
  return ptrs { ptr1 = ptr1, ptr2 = ptr2 }
end

task toplevel()
  var r1 = region(ispace(ptr, 10), int)
  var r2 = region(ispace(ptr, 10), int)
  for e in r2 do @e = 10 end -- this will create an instance
  var ref = ptrs { ptr1 = 0, ptr2 = get_ptr(r2) }
  ref = foo(r1, r2, ref)
  ref = foo(r1, r2, ref)
  ref = foo(r1, r2, ref)
end

regentlib.start(toplevel, bishoplib.make_entry())
