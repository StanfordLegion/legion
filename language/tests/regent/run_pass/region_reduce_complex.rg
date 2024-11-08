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
local launcher = require("std/launcher")
local cmapper = launcher.build_library("reduc_mapper")
local c = regentlib.c

task g_plus(r : region(complex32), p : ptr(complex32, r))
where reduces+(r) do
  @p += complex32 { 2, 0 }
  @p += complex32 { 13, 0 }
  @p += complex32 { 5, 0 }
end

task g_times(r : region(complex32), p : ptr(complex32, r))
where reduces*(r) do
  @p *= complex32 { 2, 0 }
  @p *= complex32 { 1, 0 }
  @p *= complex32 { 2, 0 }
end

task g_minus(r : region(complex32), p : ptr(complex32, r))
where reduces-(r) do
  @p -= complex32 { 3, 0 }
  @p -= complex32 { 10, 0 }
  @p -= complex32 { 7, 0 }
end

-- You can also do reductions with read-write privileges.
task h_plus(r : region(complex32), p : ptr(complex32, r))
where reads(r), writes(r) do
  @p += complex32 { 100, 0 }
  @p += complex32 { 100, 0 }
  @p += complex32 { 100, 0 }
end

task h_minus(r : region(complex32), p : ptr(complex32, r))
where reads(r), writes(r) do
  @p -= complex32 { 500, 0 }
  @p -= complex32 { 1200, 0 }
  @p -= complex32 { 1300, 0 }
end

task f() : complex32
  var r = region(ispace(ptr, 5), complex32)
  var p = dynamic_cast(ptr(complex32, r), 0)
  @p = complex32 { 1, 0 }
  g_plus(r, p)
  g_times(r, p)
  g_minus(r, p)
  h_plus(r, p)
  h_minus(r, p)
  return @p
end

task main()
  var v = complex32 { 1, 0 }
  v += complex32 { 2, 0 }
  v += complex32 { 13, 0 }
  v += complex32 { 5, 0 }
  v *= complex32 { 2, 0 }
  v *= complex32 { 1, 0 }
  v *= complex32 { 2, 0 }
  v -= complex32 { 3, 0 }
  v -= complex32 { 10, 0 }
  v -= complex32 { 7, 0 }
  v += complex32 { 100, 0 }
  v += complex32 { 100, 0 }
  v += complex32 { 100, 0 }
  v -= complex32 { 500, 0 }
  v -= complex32 { 1200, 0 }
  v -= complex32 { 1300, 0 }
  regentlib.assert(f().real == v.real, "test failed")
end
launcher.launch(main, "region_reduce_complex", cmapper.register_mappers, {"-lreduc_mapper"})

