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

local c = regentlib.c

fspace input {
  x : double,
  y : double,
}

fspace output {
  z : double,
}

-- The vast majority of task arguments are pass-by-value. Regions are
-- instead pass-by-reference. This means that tasks may modify the
-- contents of their region arguments.
task init(is : ispace(int1d),
          input_lr : region(is, input))
-- Tasks declare privileges on region arguments to indicate what
-- regions they will read or write. This task will write to its region
-- argument.
where writes(input_lr) do
  for i in is do
    -- Privileges are enforced by the type system. These region
    -- accesses are legal because of the write privilege declared
    -- above.
    input_lr[i].x = c.drand48()
    input_lr[i].y = c.drand48()
  end
end

task daxpy(is : ispace(int1d),
           input_lr : region(is, input),
           output_lr : region(is, output),
           alpha : double)
-- Multiple privileges may be specified at once. Privileges may also
-- apply to specific fields. (Multiple fields can be named with braces.)
where reads writes(output_lr.z), reads(input_lr.{x, y}) do
  for i in is do
    output_lr[i].z = alpha*input_lr[i].x + input_lr[i].y
  end
end

task check(is : ispace(int1d),
           input_lr : region(is, input),
           output_lr : region(is, output),
           alpha : double)
where reads(input_lr, output_lr) do
  for i in is do
    var expected = alpha*input_lr[i].x + input_lr[i].y
    var received = output_lr[i].z
    regentlib.assert(expected == received, "check failed")
  end
end

task main()
  var num_elements = 1024
  var is = ispace(int1d, num_elements)
  var input_lr = region(is, input)
  var output_lr = region(is, output)

  -- Privileges are also required for task calls. The main task has
  -- read-write privileges on any regions it created.
  init(is, input_lr)

  var alpha = c.drand48()
  daxpy(is, input_lr, output_lr, alpha)

  check(is, input_lr, output_lr, alpha)
end
regentlib.start(main)
