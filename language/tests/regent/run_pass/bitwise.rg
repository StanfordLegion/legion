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

task main()
  var x = 0xAD
  var y = 0xD4
  var z = 3

  regentlib.assert((x and y) == 0x84, "test failed")
  regentlib.assert((x or y) == 0xFD, "test failed")
  regentlib.assert((x ^ y) == 0x79, "test failed")
  regentlib.assert(((x << z) ^ y) == 0x5BC, "test failed")
end
regentlib.start(main)
