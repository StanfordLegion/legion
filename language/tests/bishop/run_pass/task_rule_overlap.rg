-- Copyright 2023 Stanford University, NVIDIA Corporation
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
-- [["-ll:cpu", "3"]]

import "regent"
import "bishop"

local c = bishoplib.c

mapper

$procs = processors[isa=x86]

task#ta {
  target : $procs[1];
}

task#ta {
  target : $procs[2];
}

end

function get_proc()
  return rexpr
    c.legion_runtime_get_executing_processor(__runtime(), __context())
  end
end

fspace fs
{
  x : int,
  y : int,
}

task tc(r : region(fs))
where reads(r.x) do
end

task tb(r : region(fs))
where reads(r.x), reads writes(r.y) do
  tc(r)
end

task ta(r : region(fs))
where reads(r.x), reads writes(r.y) do
  var proc = [get_proc()]
  var procs = c.bishop_all_processors()
  tb(r)
  -- the second rule should win
  regentlib.assert(procs.list[2].id == proc.id, "test failed in ta")
end

task toplevel()
  var r = region(ispace(ptr, 10), fs)
  fill(r.{x, y}, 0)
  ta(r)
end

regentlib.start(toplevel, bishoplib.make_entry())
