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
-- [["-fflow", "0", "-foverride-demand-index-launch", "1"]]

import "regent"

task f(r : region(ispace(int1d), int))
where reads writes(r) do
end

task g(r : region(ispace(int1d), int), s : region(ispace(int1d), int))
where reads(r), reads writes(s) do
end

task main()
  var r = region(ispace(int1d, 10), int)
  var N = 4
  var pr = partition(equal, r, ispace(int1d, N))

  fill(r, 0)

  -- These are checks the compiler couldn't normally do because the
  -- optimizer isn't sophisticated enough.
  __demand(__index_launch)
  for i = 0, N do
    f(pr[(i + 1) % N])
  end

  __demand(__index_launch)
  for i = 1, N do
    g(pr[0], pr[i])
  end

  -- This loop cannot be index launched, it has a true dependence.
  -- IMPORTANT: Using the flag to override __demand(__index_launch)
  -- should not cause this to get index launched.
  for i = 1, N do
    f(r)
  end
end
regentlib.start(main)
