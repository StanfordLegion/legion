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

fspace fs
{
  input : int,
  output : int,
  output2 : int,
}

local c = regentlib.c

task toplevel()
  var r = region(ispace(int1d, 100), fs)
  fill(r.output, 0)
  __forbid(__vectorize)
  for e in r do
    e.input = e
  end

  var coloring = c.legion_domain_coloring_create()
  c.legion_domain_coloring_color_domain(coloring, 0,
    rect1d { r.bounds.lo + 1, r.bounds.hi - 1 })
  var p_interior = partition(disjoint, r, coloring)
  c.legion_domain_coloring_destroy(coloring)
  var r_interior = p_interior[0]
  __demand(__vectorize)
  for e in r_interior do
    r[e].output = (r[e - 1].input + r[e].input + r[e + 1].input) / 3
  end
  __forbid(__vectorize)
  for e in r_interior do
    r[e].output2 = (r[e - 1].input + r[e].input + r[e + 1].input) / 3
  end
  __forbid(__vectorize)
  for e in r_interior do
    regentlib.assert(r[e].output == r[e].output2, "test failed")
  end
end

regentlib.start(toplevel)
