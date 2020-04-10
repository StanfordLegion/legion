-- Copyright 2020 Stanford University
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

local printf = require("std/printf")

task main()
  var x : int32 = 1
  var y : uint64 = 1234
  printf.printf("Hello {} {} world!", x, y)

  var z : float = 1.23
  var w : double = 3.45
  printf.printf("Floats: {} {}", z, w)

  var i1 = int1d(1)
  var i2 = int2d { 1, 2 }
  var i3 = int3d { 1, 2, 3 }
  printf.printf("int1d {}", i1)
  printf.printf("int2d {}", i2)
  printf.printf("int3d {}", i3)
end
regentlib.start(main)
