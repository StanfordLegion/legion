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

fspace node(ghost : region(node(wild, wild)),
            owned : region(node(wild, wild))) {
  p : ptr(node(wild, wild), ghost, owned),
  v : int,
}

task init(ghost : region(node(wild, wild)),
          owned : region(node(ghost, owned)))
where reads writes(owned) do
  for x in owned do
    @x = [node(ghost, owned)] {
      p = dynamic_cast(ptr(node(wild, wild), ghost, owned), x.p),
      v = x.v,
    }
    x.p = dynamic_cast(ptr(node(wild, wild), ghost, owned), x.p)
  end
end

task update(ghost : region(node(wild, wild)),
            owned : region(node(ghost, owned)))
where reads(owned.p), reduces+(owned.v, ghost.v) do
  for x in owned do
    x.p.v += int(x)
  end
end

task main()
  var nodes = region(ispace(ptr, 4), node(wild, wild))
  var x0 = dynamic_cast(ptr(node(wild, wild), nodes), 0)
  var x1 = dynamic_cast(ptr(node(wild, wild), nodes), 1)
  var x2 = dynamic_cast(ptr(node(wild, wild), nodes), 2)
  var x3 = dynamic_cast(ptr(node(wild, wild), nodes), 3)

  @x0 = [node(wild, wild)]{ p = static_cast(ptr(node(wild, wild), nodes, nodes), x1), v = 0 }
  @x1 = [node(wild, wild)]{ p = static_cast(ptr(node(wild, wild), nodes, nodes), x2), v = 0 }
  @x2 = [node(wild, wild)]{ p = static_cast(ptr(node(wild, wild), nodes, nodes), x3), v = 0 }
  @x3 = [node(wild, wild)]{ p = static_cast(ptr(node(wild, wild), nodes, nodes), x0), v = 0 }

  var colors = ispace(int1d, 2)
  var owned = partition(equal, nodes, colors)
  var ghost = image(nodes, owned, nodes.p)

  for c in colors do
    init(ghost[c], owned[c])
  end

  for c in colors do
    update(ghost[c], owned[c])
  end

  for x in nodes do
    regentlib.c.printf("x%d: %d\n", x, x.v)
  end

  regentlib.assert(x0.v == 3, "test failed")
  regentlib.assert(x1.v == 0, "test failed")
  regentlib.assert(x2.v == 1, "test failed")
  regentlib.assert(x3.v == 2, "test failed")
end
regentlib.start(main)
