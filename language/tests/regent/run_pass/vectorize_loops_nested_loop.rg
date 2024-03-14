-- Copyright 2024 Stanford University
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

fspace fs1
{
  v1 : float,
  v2 : float,
}

task f(r : region(fs1))
where
  reads(r.v1, r.v2),
  writes(r.v2, r.v1)
do
  --var x : float = 10.0

  __demand(__vectorize)
  for e in r do
    for i = 0, 3 do
      for j = 0, 3 do
        var x : float = [float](i == j)
        e.v1 += x
      end
    end
  end

  __forbid(__vectorize)
  for e in r do
    for i = 0, 3 do
      for j = 0, 3 do
        var x : float = [float](i == j)
        e.v2 += x
      end
    end
  end
end

task toplevel()
  var n = 10
  var r = region(ispace(ptr, n), fs1)
  for p in r do
    p.v1 = 1.0 + [int](p)
    p.v2 = 1.0 + [int](p)
  end
  f(r)
  for e in r do
    regentlib.assert(e.v1 == e.v2, "test failed")
  end
end

regentlib.start(toplevel)
