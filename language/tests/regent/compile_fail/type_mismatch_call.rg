-- Copyright 2021 Stanford University
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

-- fails-with:
-- type_mismatch_call.rg:31: type mismatch in argument 1: expected &int32 but got int32
--   f(y, x)
--   ^

import "regent"

terra null_rawptr() : &int
  return [&int](0)
end

task f(x : &int, y : int) end

task g()
  var y = 0
  var x = null_rawptr()
  f(y, x)
end
g:compile()
