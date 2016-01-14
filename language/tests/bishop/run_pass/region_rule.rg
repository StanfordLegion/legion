-- Copyright 2016 Stanford University, NVIDIA Corporation
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
                         $proc.memories[kind=l1cache];
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
  regentlib.assert(kind_r1 == c.SYSTEM_MEM, "test failed")
  regentlib.assert(
    kind_r2 == c.REGDMA_MEM or kind_r2 == c.LEVEL1_CACHE,
    "test failed")
end

task toplevel()
  var r1 = region(ispace(ptr, 10), int)
  var r2 = region(ispace(ptr, 10), int)
  new(ptr(int, r1), 10)
  new(ptr(int, r2), 10)
  foo(r1, r2)
end

bishoplib.register_bishop_mappers()
regentlib.start(toplevel)
