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

task g(x : int)
  return x
end

task f()
  var i : int = 5
  var i32 : int32 = 5
  var i16 : int16 = 5
  var i8 : int8 = 5
  var d : double = 5.5
  var f : float = 5.5f

  var x0 = i + d
  var x1 = d + i
  var x2 = i + f
  var x3 = f + i
  var x4 = i + i32
  var x5 = i32 + i
  var x6 = i + i16
  var x7 = i16 + i
  var x8 = i + i8
  var x9 = i8 + i
  var xa = i16 + i8
  var xb = i8 + i16

  g(i)
  g(i32)
  g(i16)
  g(i8)
  g(d)
  g(f)
end

task main()
  f()
end
regentlib.start(main)

