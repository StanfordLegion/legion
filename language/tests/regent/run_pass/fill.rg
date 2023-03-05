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

struct t {
  a : int,
  b : float,
  c : uint8,
}

task k2(r : region(t))
where writes(r.{a, b}) do
  fill(r.{a, b}, 25)
end

task k() : int
  var r = region(ispace(ptr, 5), t)
  var x = dynamic_cast(ptr(t, r), 0)

  x.a = 123
  x.b = 3.14
  x.c = 48

  fill(r.c, 25)
  k2(r)

  return x.a + x.b + x.c
end

task main()
  regentlib.assert(k() == 75, "test failed")
end
regentlib.start(main)
