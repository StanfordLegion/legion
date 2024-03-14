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

task eq(a : int, b : int) : bool
  return a == b
end

task ne(a : int, b : int) : bool
  return a ~= b
end

task lt(a : int, b : int) : bool
  return a < b
end

task le(a : int, b : int) : bool
  return a <= b
end

task gt(a : int, b : int) : bool
  return a > b
end

task ge(a : int, b : int) : bool
  return a >= b
end

task main()
  regentlib.assert(false == eq(5, 3), "test failed")
  regentlib.assert(eq(5, 5), "test failed")
  regentlib.assert(false == eq(3, 5), "test failed")
  regentlib.assert(ne(5, 3), "test failed")
  regentlib.assert(false == ne(5, 5), "test failed")
  regentlib.assert(ne(3, 5), "test failed")
  regentlib.assert(false == lt(5, 3), "test failed")
  regentlib.assert(false == lt(5, 5), "test failed")
  regentlib.assert(lt(3, 5), "test failed")
  regentlib.assert(false == le(5, 3), "test failed")
  regentlib.assert(le(5, 5), "test failed")
  regentlib.assert(le(3, 5), "test failed")
  regentlib.assert(gt(5, 3), "test failed")
  regentlib.assert(false == gt(5, 5), "test failed")
  regentlib.assert(false == gt(3, 5), "test failed")
  regentlib.assert(ge(5, 3), "test failed")
  regentlib.assert(ge(5, 5), "test failed")
  regentlib.assert(false == ge(3, 5), "test failed")
end
regentlib.start(main)
