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
-- [["-ll:rsize", "10"]]

import "regent"
import "bishop"

local c = bishoplib.c

mapper

$HAS_REGMEM = memories[kind=regmem].size > 0

task#foo[target=$proc] region#r2 {
  target : $HAS_REGMEM ? $proc.memories[kind=regmem] :
                         $proc.memories[kind=sysmem];
}

end

task bar(r : region(int))
end

task foo(r1 : region(int), r2 : region(int))
where reduces+(r2)
do
  var memories_r2 = c.bishop_physical_region_get_memories(__physical(r2)[0])
  var kind_r2 = c.legion_memory_kind(memories_r2.list[0])
  var all_memories = c.bishop_all_memories()
  var regmems = c.bishop_filter_memories_by_kind(all_memories, c.REGDMA_MEM)
  regentlib.assert(regmems.size == 0 or kind_r2 == c.REGDMA_MEM, "test failed")
  c.bishop_delete_memory_list(regmems)
  c.bishop_delete_memory_list(all_memories)
  for e in r2 do @e += 10 end
end

task toplevel()
  var r1 = region(ispace(ptr, 10), int)
  var r2 = region(ispace(ptr, 10), int)
  for e in r2 do
    @e = 0
  end
  for i = 0, 10 do
    foo(r1, r2)
  end
  bar(r2)
  for e in r2 do
    regentlib.assert(@e == 100, "test failed")
  end
end

regentlib.start(toplevel, bishoplib.make_entry())

