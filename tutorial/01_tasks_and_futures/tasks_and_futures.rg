-- Copyright 2016 Stanford University
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

local c = terralib.includec("stdio.h")

task fibonacci(n : int) : int
  if n == 0 then return 0 end
  if n == 1 then return 1 end

  var f1 = fibonacci(n - 1) -- A task call implicitly returns a future.
  var f2 = fibonacci(n - 2)

  return f1 + f2 -- Arithmetic is automatically promoted to operate on futures.
end

task print_result(n : int, result : int)
  c.printf("Fibonacci(%d) = %d\n", n, result)
end

task main()
  var num_fibonacci = 7
  for i = 0, num_fibonacci do
    print_result(i, fibonacci(i))
  end
end
regentlib.start(main)
