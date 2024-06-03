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
-- [["-ll:cpu", "2"]]

import "regent"

task f(x : int)
  return x * x
end

task g(x : int, y : int, z : dynamic_collective(int))
  arrive(z, y)
  var w = f(x)
  arrive(z, w)
end

task main()
  var c = dynamic_collective(int, +, 4)
  must_epoch
    g(2, 300, c)
    g(5, 4000, c)
  end
  c = advance(c)
  var d = dynamic_collective_get_result(c)

  regentlib.assert(d == 4329, "test failed")
end
regentlib.start(main)
