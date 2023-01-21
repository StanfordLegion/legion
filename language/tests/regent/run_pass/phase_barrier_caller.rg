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
-- [["-ll:cpu", "2"]]

import "regent"

local c = terralib.includec("unistd.h")

task g1(s : region(int))
where reads writes(s) do
  for x in s do
    @x *= 2
  end
end

task g2(s : region(int))
where reads writes(s) do
  for x in s do
    @x *= 3
  end
end

task g(s : region(int), n : phase_barrier)
where reads writes simultaneous(s) do
  c.sleep(1) -- Make sure we're really synchronizing with h.
  var n2 = advance(advance(n))
  g1(s, arrives(n))
  g2(s, awaits(n2))
end

task h1(s : region(int))
where reads writes(s) do
  for x in s do
    @x += 10
  end
end

task h(s : region(int), n : phase_barrier)
where reads writes simultaneous(s) do
  var n1 = advance(n)
  h1(s, awaits(n1), arrives(n1))
end

task k() : int
  var r = region(ispace(ptr, 5), int)
  var x = dynamic_cast(ptr(int, r), 0)
  var p = phase_barrier(1)
  @x = 1
  must_epoch
    g(r, p)
    h(r, p)
  end
  return @x
end

task main()
  regentlib.assert(k() == 36, "test failed")
end
regentlib.start(main)
