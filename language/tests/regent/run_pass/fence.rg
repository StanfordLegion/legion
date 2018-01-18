-- Copyright 2018 Stanford University
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

task f(x : int)
  var y = x * 2
  regentlib.c.printf("f returned %d\n", y)
  return y
end

task g(x : int)
  regentlib.c.printf("g got %d\n", x)
end

task main()
  var y = f(1)
  __fence(__execution)
  g(y)

  var z = f(10)
  __fence(__mapping)
  g(z)
end
regentlib.start(main)
