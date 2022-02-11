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

task main()
  var num_elements = 1024
  var is = ispace(int1d, num_elements)
  var input_lr = region(is, input)
  var output_lr = region(is, output)

  -- Regent provides two distinct syntaxes for accessing regions.

  -- 1. Iteration over regions.
  --
  -- The loop below iterates over pointers to the elements in the
  -- region. The loop index type (which is optional and shown only for
  -- pedagogical purposes) records both the type that the pointer
  -- points to **AND** the region that it points to. The dereference
  -- operator @ is used to access the element.
  for input_ptr : int1d(input, input_lr) in input_lr do
    (@input_ptr).x = c.drand48()
    input_ptr.y = c.drand48() -- The dereference operator is also optional.
  end

  var alpha = c.drand48()

  -- 2. Iteration over ispaces.
  --
  -- The loop below iterates over points in the index space. Since the
  -- loop variable is typed on the index space rather than a region,
  -- the index access operator [] must be used to access the
  -- respective regions.
  for i : int1d(is) in is do
    output_lr[i].z = alpha*input_lr[i].x + input_lr[i].y
  end

  -- Again, with type inference.
  for i in is do
    var expected = alpha*input_lr[i].x + input_lr[i].y
    var received = output_lr[i].z
    regentlib.assert(expected == received, "check failed")
  end
end
regentlib.start(main)
