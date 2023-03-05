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

import "regent"
import "bishop"

mapper
end

task foo(r : region(ispace(int1d), int))
where reads writes(r) do
end

task bar(r : region(ispace(int1d), int))
where reduces+(r) do
end

--task baz(r : region(ispace(int1d), int))
--where reduces-(r) do
--end

task toplevel()
  var r = region(ispace(int1d, 5), int)

  var ld1 = ispace(int1d, 5)
  var ld2 = ispace(int1d, 3, 1)

  var p1 = partition(equal, r, ld1)
  var p2 = partition(equal, r, ld1)
  var p3 = partition(equal, r, ld1)

  -- FIXME: The following four task launches exercise the case when
  --        some of the reduction instances are not covered by the
  --        normal instances within the same composite instance,
  --        hence the nested instances should not be pruned.
  --        However, Legion Spy cannot handle this case properly.
  --        Put back the following code once Legion Spy gets fixed.
  --__demand(__index_launch)
  --for c in ld1 do foo(p1[c]) end

  --__demand(__index_launch)
  --for c in ld2 do foo(p2[c]) end

  --__demand(__index_launch)
  --for c in ld1 do bar(p2[c]) end

  --__demand(__index_launch)
  --for c in ld1 do bar(p3[c]) end

  __demand(__index_launch)
  for c in ld1 do foo(p1[c]) end

  __demand(__index_launch)
  for c in ld1 do bar(p2[c]) end

  __demand(__index_launch)
  for c in ld1 do foo(p3[c]) end
end

regentlib.start(toplevel, bishoplib.make_entry())
