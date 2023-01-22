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

-- runs-with:
-- [["-ll:cpu", "4"]]

import "regent"

-- A test of various language features need for expressing the
-- automatic SPMD optimization.

local c = regentlib.c

struct elt {
  src : int,
  dst : int,
}

task init(r : region(ispace(int1d), elt))
where reads writes(r)
do
  for e in r do
    e.src = 1000
    e.dst = 0
  end
end

task copy_src_to_dst(r : region(ispace(int1d), elt))
where reads(r.src), reads writes(r.dst)
do
  for e in r do
    e.dst = e.src
  end
end

task check(r : region(ispace(int1d), elt))
where reads(r)
do
  for e in r do
    regentlib.assert(e.src == 1000, "test failed")
    regentlib.assert(e.dst == 1000, "test failed")
  end
end

task shard(r : regentlib.list(region(ispace(int1d), elt)),
           s : regentlib.list(region(ispace(int1d), elt)),
           qs : regentlib.list(regentlib.list(region(ispace(int1d), elt))),
           barrier : phase_barrier)
where
  reads(r),
  reads writes simultaneous(s),
  reads writes simultaneous(qs),
  no_access_flag(qs)
do
  var rs_product = list_cross_product_complete(r, qs)
  barrier = adjust(barrier, qs:num_leaves())
  var barriers = list_from_element(qs, barrier)
  regentlib.assert(barriers:num_leaves() == 1, "test failed")
  copy(r, rs_product, arrives(barriers))
  do
    arrive(barrier)
    barrier = advance(barrier)
    await(barrier)
  end

  for i = 0, 1 do copy_src_to_dst(s[i]) end
end

task main()
  var size = 1000000
  var num_blocks = 4
  var block_size = size / num_blocks

  var r = region(ispace(int1d, size), elt)
  var cs = ispace(int1d, num_blocks)
  var c1 = c.legion_domain_point_coloring_create()
  for i = 0, num_blocks do
    var color = [int1d](i)
    c.legion_domain_point_coloring_color_domain(c1, color, [rect1d] { i * block_size, (i + 1) * block_size - 1 })
  end
  var p = partition(disjoint, r, c1, cs)
  c.legion_domain_point_coloring_destroy(c1)
  var c2 = c.legion_domain_point_coloring_create()
  for i = 0, num_blocks do
    var color = [int1d]((i + 1) % num_blocks)
    c.legion_domain_point_coloring_color_domain(c2, color, [rect1d] { i * block_size, (i + 1) * block_size - 1 })
  end
  var q = partition(disjoint, r, c2, cs)
  c.legion_domain_point_coloring_destroy(c2)

  for color in cs do init(p[color]) end

  var rs_p = list_slice_partition(p, list_range(0, num_blocks))
  var rs_q = list_duplicate_partition(q, list_range(0, num_blocks))
  var rs_product = list_cross_product(rs_p, rs_q, true)
  var start_barrier = phase_barrier(num_blocks)

  must_epoch
    for i = 0, num_blocks do
      var l = list_range(i, i + 1)
      var rs_p = rs_p[l]
      var rs_q = rs_q[l]
      var rs_qs = rs_product[l]
      shard(rs_p, rs_q, rs_qs, start_barrier)
    end
  end

  for color in cs do check(rs_q[color]) end
end
regentlib.start(main)
