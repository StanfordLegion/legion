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

fspace s { a : int, b : bool, c : double }

task f()
  var x : int = 6
  var y : bool = false
  var z : double = 3.14
  var w = s { [terralib.newlist({x, rexpr not y end, rexpr [z] * 2 end})] }

  if w.b then
    return w.a
  else
    return w.c
  end
end

task main()
  regentlib.assert(f() == 6, "test failed")
end
regentlib.start(main)
