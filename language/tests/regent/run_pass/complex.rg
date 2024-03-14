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

task main()
  var x = complex { 1.0, 2.0 }
  regentlib.assert(x.real == 1.0, "test failed")
  regentlib.assert(x.imag == 2.0, "test failed")

  var y = complex { 3.0, 4.0 }
  var z = x + y
  regentlib.assert(z.real == 4.0, "test failed")
  regentlib.assert(z.imag == 6.0, "test failed")

  var w = x - y
  regentlib.assert(w.real == -2.0, "test failed")
  regentlib.assert(w.imag == -2.0, "test failed")

  var v = x * y
  regentlib.assert(v.real == -5.0, "test failed")
  regentlib.assert(v.imag == 10.0, "test failed")

  var v2 = x / y
  regentlib.assert((v2.real - 0.44) < 0.000001, "test failed")
  regentlib.assert((v2.imag - 0.8)  < 0.000001, "test failed")

  var u = x + 2
  regentlib.assert(u.real == 3.0, "test failed")
  regentlib.assert(u.imag == 2.0, "test failed")
end
regentlib.start(main)
