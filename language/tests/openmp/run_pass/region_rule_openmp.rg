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
-- [["-ll:rsize", "10", "-fopenmp", "1", "-ll:ocpu", "1"]]

import "regent"
import "bishop"

local c = bishoplib.c

mapper

$HAS_OPENMP = processors[kind=openmp].size > 0
$HAS_REGMEM = memories[kind=regmem].size > 0

task#foo {
  target : $HAS_OPENMP ? processors[kind=openmp] : processors[kind=loc];
}

task#foo[target=$proc] region#r2 {
  target : $HAS_REGMEM ? $proc.memories[kind=regmem] :
                         $proc.memories[kind=sysmem];
}

task[target=$proc] region {
  target : $proc.memories[kind=sysmem];
}

end

task foo(r1 : region(int), r2 : region(int))
where reads(r1, r2)
do
  var memories_r1 = c.bishop_physical_region_get_memories(__physical(r1)[0])
  var memories_r2 = c.bishop_physical_region_get_memories(__physical(r2)[0])
  var kind_r1 = c.legion_memory_kind(memories_r1.list[0])
  var kind_r2 = c.legion_memory_kind(memories_r2.list[0])
  var all_memories = c.bishop_all_memories()
  var regmems = c.bishop_filter_memories_by_kind(all_memories, c.REGDMA_MEM)
  regentlib.assert(kind_r1 == c.SYSTEM_MEM,
      "r1 should be allocated on system memory")
  regentlib.assert(regmems.size == 0 or kind_r2 == c.REGDMA_MEM,
      "r2 should be allocated on register memory unless register memory doesn't exist")
  c.bishop_delete_memory_list(regmems)
  c.bishop_delete_memory_list(all_memories)

  __demand(__openmp)
  for e in r1 do
  end
end

task toplevel()
  var r1 = region(ispace(ptr, 10), int)
  fill(r1,1)
  var r2 = region(ispace(ptr, 10), int)
  fill(r2,2)
  foo(r1, r2)
end

regentlib.start(toplevel, bishoplib.make_entry())
