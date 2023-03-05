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

task f(x : double) : int
  return [int](x)
end

struct s {
  x : double
}

struct t {
  y : int
}

s.metamethods.__cast = function(from, to, expr)
  regentlib.assert(from == s and to == t, "test failed")
  return `(t { y = [expr].x })
end

task g(x : double) : int
  var z = [s]({ x = x })
  return ([t](z)).y
end

task main()
  regentlib.assert(f(4.1) == 4, "test failed")
  regentlib.assert(g(5.7) == 5, "test failed")
end
regentlib.start(main)
