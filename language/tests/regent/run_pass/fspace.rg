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

fspace s0 {}

task s0t() var x = s0 {} end

fspace s1 {
  x : int
}

task s1t() : s1 var x = s1 { x = 1 } return x end

task e1t() : s1 var x = s1 { [ "x" ] = 1 } return x end

task u1t() : s1 var x = s1 { 1 } return x end

fspace s1c {
  x : int,
}

task s1ct() : s1c var x = s1c { x = 1, } return x end

task e1ct() : s1c var x = s1c { [ "x" ] = 1, } return x end

task u1ct() : s1c var x = s1c { 1, } return x end

fspace s2 {
  x : double,
  y : double
}

task s2t() : s2 var x = s2 { x = 1, y = 2 } return x end

task e2t() : s2 var x = s2 { [ "x" ] = 1, ["y"] = 2 } return x end

task u2t() : s2 var x = s2 { 1, 2 } return x end

fspace s2c {
  x : double,
  y : double,
}

task s2ct() : s2c var x = s2c { y = 2, x = 1, } return x end

task e2ct() : s2c var x = s2c { ["y"] = 2, [ "x" ] = 1, } return x end

task u2ct() : s2c var x = s2c { 2, 1, } return x end

task main()
  s0t()
  regentlib.assert(s1t().x == 1, "test failed")
  regentlib.assert(e1t().x == 1, "test failed")
  regentlib.assert(u1t().x == 1, "test failed")
  regentlib.assert(s1ct().x == 1, "test failed")
  regentlib.assert(e1ct().x == 1, "test failed")
  regentlib.assert(u1ct().x == 1, "test failed")
  var s2r = s2t()
  regentlib.assert(s2r.x == 1 and s2r.y == 2, "test failed")
  var e2r = e2t()
  regentlib.assert(e2r.x == 1 and e2r.y == 2, "test failed")
  var u2r = u2t()
  regentlib.assert(u2r.x == 1 and u2r.y == 2, "test failed")
  var s2cr = s2ct()
  regentlib.assert(s2cr.x == 1 and s2cr.y == 2, "test failed")
  var e2cr = e2ct()
  regentlib.assert(e2cr.x == 1 and e2cr.y == 2, "test failed")
  var u2cr = u2ct()
  regentlib.assert(u2cr.x == 2 and u2cr.y == 1, "test failed")
end
regentlib.start(main)
