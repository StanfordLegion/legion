-- Copyright 2024 Stanford University, NVIDIA Coporation
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

fspace fs {
  f : double,
  g : double,
}

task test_reduce_add(r: region(ispace(int1d), fs))
where reduces +(r.f), reads(r.g) do
  var val = 3.14
  __demand(__vectorize)
  for c in r do
    c.f += c.g + val
  end
end

task test_reduce_sub(r: region(ispace(int1d), fs))
where reduces -(r.f), reads(r.g) do
  var val = 3.14
  __demand(__vectorize)
  for c in r do
    c.f -= c.g - val
  end
end

task test_reduce_min(r: region(ispace(int1d), fs))
where reduces min(r.f), reads(r.g) do
  var val = 3.14
  __demand(__vectorize)
  for c in r do
    c.f min= min(c.g, val)
  end
end

task test_reduce_max(r: region(ispace(int1d), fs))
where reduces max(r.f), reads(r.g) do
  var val = 3.14
  __demand(__vectorize)
  for c in r do
    c.f max= max(c.g, val)
  end
end

task main()
  var x = region(ispace(int1d, 5), fs)
  fill(x.f, 2.0)
  fill(x.g, 3.14)
  test_reduce_add(x)
  test_reduce_sub(x)
  test_reduce_min(x)
  test_reduce_max(x)
  regentlib.assert(x[0].f == x[4].f, "test failed")
end

regentlib.start(main)
