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

fspace a {
  b : int,
  c : int,
}

task d(e : region(a))
where reads(e.b), writes(e.b) do
  for f in e do
    f.b += 10
  end
end

task g(h : region(a))
where reads(h.c), writes(h.c) do
  for i in h do
    i.c += 100
  end
end

task j(k : region(a))
where reads(k.{b, c}), writes(k.{b, c}) do
  for l in k do
    l.b *= 2
    l.c *= 2
  end
end

task main()
  var m = 5
  var n = region(ispace(ptr, m), a)
  for o in n do
    o.b = 1
    o.c = 2
  end

  d(n)
  g(n)
  j(n)
  d(n)
  j(n)
  g(n)

  for p in n do
    regentlib.assert(p.b == ((1+10)*2 + 10)*2, "test failed")
    regentlib.assert(p.c == (2+100)*2*2 + 100, "test failed")
  end
end
regentlib.start(main)
