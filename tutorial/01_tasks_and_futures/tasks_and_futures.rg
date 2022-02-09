-- Copyright 2022 Stanford University
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

local format = require("std/format")

-- Tasks may take declare arguments and return values. The following
-- task takes one int argument and returns an int.
task fibonacci(n : int) : int
  if n == 0 then return 0 end
  if n == 1 then return 1 end

  -- Task calls implicitly return futures, allowing the two calls
  -- below to execute in parallel. Futures are implicitly coerced to
  -- concrete values when needed.
  var f1 = fibonacci(n - 1)
  var f2 = fibonacci(n - 2)

  -- Arithmetic is automatically promoted to operate on futures.
  return f1 + f2
end

-- A second task, this time with two arguments and no return value.
task print_result(n : int, result : int)
  format.println("Fibonacci({}) = {}", n, result)
end

task main()
  var num_fibonacci = 7
  -- Futures can be passed between tasks. Thus the tasks called in the
  -- iterations of the following loop may execute in parallel.
  for i = 0, num_fibonacci do
    print_result(i, fibonacci(i))
  end
end
regentlib.start(main)
