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

task init(input_lr : region(ispace(int1d), input))
where writes(input_lr.{x, y}) do
  for i in input_lr do
    input_lr[i].x = c.drand48()
    input_lr[i].y = c.drand48()
  end
end

task daxpy(input_lr : region(ispace(int1d), input),
           output_lr : region(ispace(int1d), output),
           alpha : double)
where reads writes(output_lr.z), reads(input_lr.{x, y}) do
  for i in input_lr do
    output_lr[i].z = alpha*input_lr[i].x + input_lr[i].y
  end
end

task check(input_lr : region(ispace(int1d), input),
           output_lr : region(ispace(int1d), output),
           alpha : double)
where reads(input_lr, output_lr) do
  for i in input_lr do
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

  -- Data parallelism in Regent is achieved by partioning a reigon
  -- into subregions. Regent provides a number of partitioning
  -- operators; the code below uses an equal partitioning to create
  -- simple blocked subregions.
  --
  -- The subregions in a partition are also associated with points in
  -- an index space. The ispace ps below names the respective subregions.
  var num_subregions = 4
  var ps = ispace(int1d, num_subregions)
  var input_lp = partition(equal, input_lr, ps)
  var output_lp = partition(equal, output_lr, ps)

  -- Loops may now call tasks on the subregions of the partitions
  -- declared above. (Note as before the __demand annotation is
  -- optional and causes the compiler to issue an error if the loop
  -- iterations cannot be guarranteed to execute in parallel.)
  __demand(__index_launch)
  for i = 0, num_subregions do
    init(input_lp[i])
  end

  var alpha = c.drand48()
  __demand(__index_launch)
  for i = 0, num_subregions do
    daxpy(input_lp[i], output_lp[i], alpha)
  end

  __demand(__index_launch)
  for i = 0, num_subregions do
    check(input_lp[i], output_lp[i], alpha)
  end
end
regentlib.start(main)
