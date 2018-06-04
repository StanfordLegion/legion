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

struct st
{
  v1 : ret,
  v2 : ret,
}

__demand(__inline)
task dec1(x : int) : ret
  return ret { v = x - 1, id = c.legion_context_get_unique_id(__context()) }
end

__forbid(__inline)
task inc1(x : int) : ret
  return ret { v = x + 1, id = c.legion_context_get_unique_id(__context()) }
end

__demand(__inline)
task check()
  var id_check = c.legion_context_get_unique_id(__context())
  var s : st
  s = st { v1 = inc1(10), v2 = dec1(10) }
  var { v1 = v1, v2 = v2 } = s
  regentlib.assert(v1.v == 11, "test failed")
  regentlib.assert(v2.v == 9, "test failed")
  regentlib.assert(v1.id ~= id_check, "task inc1 is inlined though forbidden")
  regentlib.assert(v2.id == id_check, "task dec1 is not inlined though demanded")
end

task main()
  check()
  check()
end

regentlib.start(main)
