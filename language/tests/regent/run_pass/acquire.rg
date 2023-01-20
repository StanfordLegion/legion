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

task inc(r : region(int))
where reduces +(r) do
  for x in r do
    @x += 4000
  end
end

task check(r : region(int))
where reads(r) do
  for x in r do
    regentlib.assert(@x == 4123, "test failed")
  end
end

task first(r : region(int), b : phase_barrier)
where reads writes simultaneous(r) do
  acquire(r)
  inc(r)
  release(r, arrives(b))
end

task second(r : region(int), b : phase_barrier)
where reads writes simultaneous(r) do
  var b1 = advance(b)
  acquire(r, awaits(b1))
  check(r)
  release(r)
end

task main()
  var r = region(ispace(ptr, 5), int)
  fill(r, 123)

  var b = phase_barrier(1)

  must_epoch
    first(r, b)
    second(r, b)
  end
end
regentlib.start(main)
