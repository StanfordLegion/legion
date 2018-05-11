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

__demand(__inline)
task rawptr_test(str : &int8, p : &int32)
  var x = [regentlib.string](str)
  regentlib.c.memset(p, 0, [sizeof(int32)])
  p[0] = 101
  return regentlib.c.atoi([&int8](x)) == p[0]
end

task toplevel()
  var str : int8[256]
  var p : int32[1]
  regentlib.c.sprintf([&int8](str), "101")
  regentlib.assert(rawptr_test(str, p), "test failed")
end

regentlib.start(toplevel)
