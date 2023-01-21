-- Copyright 2023 Stanford University
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

-- This test attempts to leverage the substitution principle of fspace
-- types.

fspace s(a : region(int)) {
  x : ptr(int, a),
}

task f(b : region(int), c : region(s(b)), y : ptr(s(b), c)) : int
where reads(b, c) do
  return @(y.x)
end

task g() : int
  var d = region(ispace(ptr, 5), int)
  var e = region(ispace(ptr, 5), s(d))
  var w = dynamic_cast(ptr(int, d), 0)
  var u = dynamic_cast(ptr(s(d), e), 0)
  @w = 7
  @u = [s(d)] { x = w }
  return f(d, e, u)
end

task main()
  regentlib.assert(g() == 7, "test failed")
end
regentlib.start(main)
