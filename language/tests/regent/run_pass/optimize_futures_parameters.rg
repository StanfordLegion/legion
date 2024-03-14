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

task f() return 3 end

task g(x : int)
  -- Parameters are tricky. Either the compiler has to deoptimize
  -- futures (because the parameter itself cannot be promoted to a
  -- future), or the compiler has to redefine the parameter in the
  -- task body allow its use as a future.
  x += f()
  x += f()
  return x
end

task main()
  regentlib.assert(g(10) == 16, "test failed")
end
regentlib.start(main)
