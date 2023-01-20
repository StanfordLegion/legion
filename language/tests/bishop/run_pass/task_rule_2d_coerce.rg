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
-- [["-ll:cpu", "4"]]

import "regent"
import "bishop"

local c = bishoplib.c

mapper

task task#foo[index=$p] {
  target : processors[isa=x86][$p % processors[isa=x86].size];
}

task {
  target : processors[isa=x86][1];
}

end

task foo(i : int2d)
  var proc =
    c.legion_runtime_get_executing_processor(__runtime(), __context())
  var procs = c.bishop_all_processors()
  regentlib.assert(procs.list[(i.x + i.y * 4) % procs.size].id == proc.id,
    "test failed in foo")
end

task toplevel()
  var proc =
    c.legion_runtime_get_executing_processor(__runtime(), __context())
  var procs = c.bishop_all_processors()

  var is = ispace(int2d, { x = 4, y = 4 })
  __demand(__index_launch)
  for i in is do
    foo(i)
  end
  regentlib.assert(procs.list[1].id == proc.id, "test failed in toplevel")
end

regentlib.start(toplevel, bishoplib.make_entry())
