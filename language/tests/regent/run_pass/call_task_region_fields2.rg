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

import "regent"

fspace fs
{
  x : int,
  y : int,
}

task ta(r : region(fs))
where reads(r.x), reads writes(r.y) do
  for e in r do
    e.y += e.x
  end
end

task tb(r : region(fs))
where reads(r.x), reads writes(r.y) do
  ta(r)
end

task toplevel()
  var r = region(ispace(ptr, 10), fs)
  for e in r do
    e.x = 10
    e.y = 0
  end
  tb(r)
  for e in r do
    regentlib.assert(e.y == 10, "test failed")
  end
end

regentlib.start(toplevel)
