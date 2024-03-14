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

fspace list(a : region(list(a))) {
  data : int,
  next : ptr(list(a), a),
}

task flat(b : region(list(b)), c : list(b)) : int
where reads(b) do
  return c.data
end

task deep(d : region(list(d)), e : ptr(list(d), d)) : int
where reads(d) do
  var f : list(d) = @e
  return flat(d, f)
end

task top() : int
  var g = region(ispace(ptr, 5), list(g))
  var h = dynamic_cast(ptr(list(g), g), 0)
  @h = { data = 5, next = h }
  return deep(g, h)
end

task main()
  regentlib.assert(top() == 5, "test failed")
end
regentlib.start(main)
