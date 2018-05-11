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
  var id_main = c.legion_context_get_unique_id(__context())
  var id_h = h()
  regentlib.assert(id_h == id_main, "test failed")
  for i = 0, 10 do
    var ret_f, ret_g = f(i), g(i)
    regentlib.assert(ret_f.v == ret_g.v, "test failed")
    regentlib.assert(id_main == ret_f.id, "test failed")
    regentlib.assert(id_main ~= ret_g.id, "test failed")
  end
end

regentlib.start(main)
