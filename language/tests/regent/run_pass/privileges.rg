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

-- Tests privilege and coherence aggregation.

struct t {
  a : int,
  b : int,
  c : int,
}

task f(s : region(t))
where
  reads(s.a), reads(s.{b, c}), writes(s.{a, b}, s.c),
  exclusive(s.{a, b}), atomic(s.c)
do
  for x in s do
    x.a += 1
    x.b += 20
    x.c += 300
  end
end

task g(s : region(t))
where
  reads writes simultaneous(s.a), reads relaxed writes(s.b),
  writes reads reads reads writes(s.{}, s.{a, c}),
  relaxed relaxed(s.b), simultaneous(s.a)
do
  for x in s do
    x.a *= 4
    x.b *= 2
    x.c *= 1
  end
end

task k() : int
  var r = region(ispace(ptr, 1), t)
  var x = dynamic_cast(ptr(t, r), 0)

  x.a = 1000
  x.b = 10000
  x.c = 100000

  f(r)
  g(r)
  return x.a + x.b + x.c
end

task main()
  regentlib.assert(k() == 124344, "test failed")
end
regentlib.start(main)
