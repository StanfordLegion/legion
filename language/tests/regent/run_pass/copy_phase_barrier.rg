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

-- runs-with:
-- [["-ll:cpu", "2"]]

import "regent"

task inc(r : region(int), y : int, t : phase_barrier)
where reads writes(r), awaits(t) do
  for x in r do
    @x += y
  end
end

task mul(r : region(int), y : int, t : phase_barrier)
where reads writes(r), awaits(t) do
  for x in r do
    @x *= y
  end
end

task f(r : region(int), s : region(int), t : phase_barrier)
where reads writes simultaneous(r, s), no_access_flag(s) do
  copy(r, s, arrives(t))
  var t2 = advance(advance(t))
  inc(r, 7, t2)
end

task g(r : region(int), s : region(int), t : phase_barrier)
where reads writes simultaneous(r, s), no_access_flag(r) do
  var t1 = advance(t)
  mul(s, 20, t1)
  copy(s, r, arrives(t1))
end

task k() : int
  var r = region(ispace(ptr, 1), int)
  var x = dynamic_cast(ptr(int, r), 0)
  var s = region(ispace(ptr, 1), int)
  var y = dynamic_cast(ptr(int, s), 0)

  @x = 123
  @y = 456

  var t = phase_barrier(1)
  must_epoch
    f(r, s, t)
    g(r, s, t)
  end

  return @x
end

task main()
  regentlib.assert(k() == 2467, "test failed")
end
regentlib.start(main)
