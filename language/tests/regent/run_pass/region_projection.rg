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

-- runs-with:
-- [["-fflow", "0"]]

import "regent"

fspace fs
{
  a : int;
  b : int;
}

task g(r : region(fs), p : ptr(fs, r)) : int
where reads(r) do
  return p.a
end

task h(r : region(int), p : ptr(int, r)) : int
where reads(r) do
  return @p
end

task f() : int
  var r = region(ispace(ptr, 5), fs)
  var s = r.{a}
  var p = dynamic_cast(ptr(fs, r), 0)
  var q = dynamic_cast(ptr(int, s), 0)
  p.a = 5
  var v1 = g(r, p)
  @q = 3
  var v2 = h(s, q)
  return (v1 + v2) / 2
end

task main()
  regentlib.assert(f() == 4, "test failed")
end
regentlib.start(main)
