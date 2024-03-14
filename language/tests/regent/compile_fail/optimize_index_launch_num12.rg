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

-- fails-with:
-- optimize_index_launch_num12.rg:39: loop optimization failed: argument 1 is not provably invariant
--     f(g(x))
--     ^

import "regent"

-- This tests the various loop optimizations supported by the
-- compiler.

local c = regentlib.c

task f(x : phase_barrier)
where arrives(x) do
end

terra g(x : phase_barrier) : phase_barrier
  return x
end

task h(n : int, x : phase_barrier)
  -- not optimized: argument is not invariant
  __demand(__index_launch)
  for i = 0, n do
    f(g(x))
  end
end
h:compile()
