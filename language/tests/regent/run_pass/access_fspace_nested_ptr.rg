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

fspace subsubspace {
  a : int,
}

fspace subspace(r : region(subsubspace)) {
  b : ptr(subsubspace, r),
}

fspace space(s : region(subsubspace), t : region(subspace(s))) {
  c : ptr(subspace(s), t)
}

task f(u : region(subsubspace), v : region(subspace(u)),
       w : region(space(u, v)), x : ptr(space(u, v), w)) : int
where reads(u, v, w) do

  return x.c.b.a
end

task g()
  var m = region(ispace(ptr, 5), subsubspace)
  var n = region(ispace(ptr, 5), subspace(m))
  var o = region(ispace(ptr, 5), space(m, n))
  var p = dynamic_cast(ptr(subsubspace, m), 0)
  var q = dynamic_cast(ptr(subspace(m), n), 0)
  var r = dynamic_cast(ptr(space(m, n), o), 0)

  r.c = q
  r.c.b = p
  r.c.b.a = 17

  return f(m, n, o, r)
end

task main()
  regentlib.assert(g() == 17, "test failed")
end
regentlib.start(main)
