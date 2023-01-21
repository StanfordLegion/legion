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

fspace node(private : region(node(private, shared, ghost)),
            shared : region(node(private, shared, ghost)),
            ghost : region(node(private, shared, ghost))) {
  x : ptr(node(private, shared, ghost), private, shared, ghost),
  n : int,
}

task use_node(private : region(node(private, shared, ghost)),
              shared : region(node(private, shared, ghost)),
              ghost : region(node(private, shared, ghost)))
where reads(private, shared, ghost) do
  for i in private do
    var x = i.x
    -- Can't actually do this since the mesh wasn't initialized.
    -- var n = x.n
  end
end

task main()
  var r = region(ispace(ptr, 10), node(wild, wild, wild))
  var p_private = partition(equal, r, ispace(int1d, 3))
  var p_shared = partition(equal, r, ispace(int1d, 3))
  var p_ghost = partition(equal, r, ispace(int1d, 3))

  -- Hack: need to initialize properly.
  fill(r.x, r[0].x) -- Definitely don't do this at home....
  fill(r.n, 0)

  for i = 0, 3 do
    use_node(p_private[i], p_shared[i], p_ghost[i])
  end
end
regentlib.start(main)
