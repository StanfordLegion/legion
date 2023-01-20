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
-- [[]]

-- FIXME: Breaks SPMD optimization
-- [["-ll:cpu", "4", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "3"]]

-- This test triggered a bug in RDIR (even without SPMD) that caused a
-- cycle due to the application of reductions to multiple partitions.

import "regent"

local c = regentlib.c

fspace node { nodal_mass : double }

fspace elem(rn_ghost : region(node), rn_owned : region(node)) { }

local wild_elem = elem(wild, wild)

task init_nodes(rn : region(node))
where
  reads writes(rn.{ nodal_mass })
do
end

task init_elems(rn_ghost  : region(node),
                rn_owned  : region(node),
                re_owned  : region(elem(rn_ghost, rn_owned)))
where
  reduces +(rn_owned.nodal_mass),
  reduces +(rn_ghost.nodal_mass)
do
end

task main()
  var num_elems = 42
  var num_nodes = 137

  var rn = region(ispace(ptr, num_nodes), node)
  var re = region(ispace(ptr, num_elems), wild_elem)

  var colors = ispace(int1d, 10)
  var p_owned_nodes = partition(equal, rn, colors)
  var p_ghost_nodes = p_owned_nodes | p_owned_nodes
  var p_owned_elems = partition(equal, re, colors)

  __demand(__spmd)
  do
    for part_id = 0, 10 do
      init_nodes(p_owned_nodes[part_id])
    end

    for part_id = 0, 10 do
      init_elems(p_ghost_nodes[part_id], p_owned_nodes[part_id],
                 p_owned_elems[part_id])
    end
  end
end
regentlib.start(main)
