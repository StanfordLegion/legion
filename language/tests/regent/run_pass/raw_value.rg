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

import "regent"

local c = regentlib.c

terra test(is : c.legion_index_space_t,
           r : c.legion_logical_region_t,
           p : c.legion_logical_partition_t,
           cp : c.legion_terra_index_cross_product_t,
           x : c.legion_ptr_t)
end

task main()
  var is = ispace(ptr, 5)
  var r = region(is, int)
  var x = dynamic_cast(ptr(int, r), 0)

  var colors = c.legion_coloring_create()
  c.legion_coloring_ensure_color(colors, 0)
  var part0 = partition(disjoint, r, colors)
  var part1 = partition(disjoint, r, colors)
  c.legion_coloring_destroy(colors)

  var prod = cross_product(part0, part1)

  test(__raw(is), __raw(r), __raw(part0), __raw(prod), __raw(x))
end
regentlib.start(main)
