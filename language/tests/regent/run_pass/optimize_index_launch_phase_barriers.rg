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
-- [["-ll:cpu", "5"]]

import "regent"

-- This tests the various loop optimizations supported by the
-- compiler.

task k1(x : phase_barrier)
where arrives(x) do
end

task k2(x : phase_barrier)
where awaits(x) do
end

task k3(x : phase_barrier)
end

terra g(x : phase_barrier) : phase_barrier
  return x
end

task main()
  var n = 2
  var z = phase_barrier(1)
  must_epoch
    __demand(__index_launch)
    for i = 0, n do
      k1(z)
    end
    z = advance(z)
    __demand(__index_launch)
    for i = n, 2*n do
      k2(z)
    end
    -- Sanity check this is actually working by launching a single
    -- task too. If it fails, the test will hang.
    k2(z)
  end

  var a = phase_barrier(1)
  __demand(__index_launch)
  for i = 0, n do
    -- This is opaque to the compiler, but that's ok because we're not
    -- actually using the barrier as an arrival or wait.
    k3(g(a))
  end
end
regentlib.start(main)
