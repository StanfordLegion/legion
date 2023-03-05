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
local cstdio = terralib.includec("stdio.h")

fspace fs {
  a : int64,
  b : int64,
  c : int64,
  d : int64,
}

terra edit(runtime : c.legion_runtime_t,
           ctx : c.legion_context_t,
           r_physical : c.legion_physical_region_t[1],
           r_fields : c.legion_field_id_t[1],
           s_physical : c.legion_physical_region_t[3],
           s_fields : c.legion_field_id_t[3])
  cstdio.printf("region r:\n")
  cstdio.printf("  physical region: %p\n", r_physical[0].impl)
  cstdio.printf("  field id: %d\n", r_fields[0])
  cstdio.printf("region s:\n")
  cstdio.printf("  physical regions: %p, %p, %p\n", s_physical[0].impl, s_physical[1].impl, s_physical[2].impl)
  cstdio.printf("  field ids: %d, %d, %d\n", s_fields[0], s_fields[1], s_fields[2])

  var r = c.legion_physical_region_get_field_accessor_array_1d(
    r_physical[0], r_fields[0])
  var s_a = c.legion_physical_region_get_field_accessor_array_1d(
    s_physical[1], s_fields[1])
  var s_b = c.legion_physical_region_get_field_accessor_array_1d(
    s_physical[2], s_fields[2])
  var s_c = c.legion_physical_region_get_field_accessor_array_1d(
    s_physical[0], s_fields[0])

  var r_logical = c.legion_physical_region_get_logical_region(
    r_physical[0])
  var r_iterator = c.legion_rect_in_domain_iterator_create_1d(
    c.legion_index_space_get_domain(runtime, r_logical.index_space))
  var sum : int64 = 0
  while c.legion_rect_in_domain_iterator_valid_1d(r_iterator) do
    var rect = c.legion_rect_in_domain_iterator_get_rect_1d(r_iterator)
    c.legion_rect_in_domain_iterator_step_1d(r_iterator)
    for idx = rect.lo.x[0], rect.hi.x[0] + 1 do
      cstdio.printf("attempting to read r at pointer %d\n", idx)
      var x : int64 = 9999999
      c.legion_accessor_array_1d_read(r, c.legion_ptr_t { value = idx }, &x, sizeof(int64))
      cstdio.printf("  value is %lld\n", x)
      sum = sum + x
    end
  end
  c.legion_rect_in_domain_iterator_destroy_1d(r_iterator)

  c.legion_accessor_array_1d_destroy(r)
  c.legion_accessor_array_1d_destroy(s_a)
  c.legion_accessor_array_1d_destroy(s_b)
  c.legion_accessor_array_1d_destroy(s_c)
  return sum
end

task intermediate(r : region(int64),
                  s : region(fs),
                  x : ptr(int64, r),
                  y : ptr(fs, s))
where
  reads(r),
  reads(s.a, s.b),
  writes(s.a, s.c)
do
  return edit(__runtime(), __context(),
              __physical(r), __fields(r),
              __physical(s.{c, a, b}), __fields(s.{c, a, b}))
end

task main()
  var r = region(ispace(ptr, 1), int64)
  var s = region(ispace(ptr, 1), fs)
  var x = dynamic_cast(ptr(int64, r), 0)
  var y = dynamic_cast(ptr(fs, s), 0)

  @x = 1
  y.a = 20
  y.b = 300
  y.c = 4000

  regentlib.assert(intermediate(r, s, x, y) == 1, "test failed")
end
regentlib.start(main)
