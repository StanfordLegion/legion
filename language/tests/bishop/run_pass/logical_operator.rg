-- Copyright 2024 Stanford University, NVIDIA Corporation
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
import "bishop"

local c = bishoplib.c

mapper

$CPUS = processors[isa=x86]

task#foo[index=$p] {
  target : $p[0] % 2 == 0 or $p[0] > 2 and $p[0] < 6 ? $CPUS[1] : $CPUS[0];
}

end

task foo(x : int)
  var current_proc =
    c.legion_runtime_get_executing_processor(__runtime(), __context())
  var procs = c.bishop_all_processors()
  var target_proc = procs.list[0]
  if x == 0 then target_proc = procs.list[1]
  elseif x == 1 then target_proc = procs.list[0]
  elseif x == 2 then target_proc = procs.list[1]
  elseif x == 3 then target_proc = procs.list[1]
  elseif x == 4 then target_proc = procs.list[1]
  elseif x == 5 then target_proc = procs.list[1]
  elseif x == 6 then target_proc = procs.list[1]
  elseif x == 7 then target_proc = procs.list[0]
  else regentlib.assert(false, "unreachable")
  end
  regentlib.assert(target_proc.id == current_proc.id, "assertion failed")
end

task toplevel()
  __demand(__index_launch)
  for i = 0, 8 do
    foo(i)
  end
end

regentlib.start(toplevel, bishoplib.make_entry())
