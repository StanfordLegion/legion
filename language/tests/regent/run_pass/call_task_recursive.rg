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

task fib(n : int) : int
  if n <= 1 then
    return 1
  end
  return fib(n-1) + fib(n-2)
end

task main()
  regentlib.assert(fib(0) == 1, "test failed")
  regentlib.assert(fib(1) == 1, "test failed")
  regentlib.assert(fib(2) == 2, "test failed")
  regentlib.assert(fib(3) == 3, "test failed")
  regentlib.assert(fib(4) == 5, "test failed")
  regentlib.assert(fib(5) == 8, "test failed")
  regentlib.assert(fib(6) == 13, "test failed")
  regentlib.assert(fib(7) == 21, "test failed")
end
regentlib.start(main)
