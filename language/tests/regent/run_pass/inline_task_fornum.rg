-- Copyright 2018 Stanford University
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

local c = regentlib.c

struct ret
{
  v : int,
  id : uint64,
}

__demand(__inline)
task inc1(x : int) : int
  return x + 1
end

__demand(__inline)
task dec1(x : int) : ret
  return ret { v = x - 1, id = c.legion_context_get_unique_id(__context()) }
end

__demand(__inline)
task f(x : int) : ret
  return dec1(inc1(x + 5))
end

__forbid(__inline)
task g(x : int) : ret
  return ret { v = x + 5, id = c.legion_context_get_unique_id(__context()) }
end

__demand(__inline)
task h()
  regentlib.c.printf("called h\n")
  return c.legion_context_get_unique_id(__context())
end

task main()
  for i = 0, [int](f(0).id == g(0).id) do
    regentlib.assert(false, "test failed")
  end
end

regentlib.start(main)
