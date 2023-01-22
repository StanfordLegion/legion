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

local c = regentlib.c

struct a {
  b : int,
  c : int,
}

task g(s : region(a))
-- Split the privileges here between two coherence modes to make sure
-- that the compiler emits two (not one) region requirements.
where reads(s), simultaneous(s.b), exclusive(s.c) do
end

task h(s : region(a))
where reads writes simultaneous(s.b), reads exclusive(s.c) do
end

task k() : int
  var r = region(ispace(ptr, 5), a)
  fill(r.{b, c}, 0)
  must_epoch
    g(r)
    h(r)
  end
  return 5
end

task main()
  regentlib.assert(k() == 5, "test failed")
end
regentlib.start(main)
