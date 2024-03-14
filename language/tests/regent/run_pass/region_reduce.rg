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

-- Do each of these reductions multiple times so we know they're
-- folding properly.
task g_plus(r : region(int), p : ptr(int, r))
where reduces+(r) do
  @p += 2
  @p += 13
  @p += 5
end

task g_times(r : region(int), p : ptr(int, r))
where reduces*(r) do
  @p *= 2
  @p *= 1
  @p *= 2
end

task g_minus(r : region(int), p : ptr(int, r))
where reduces-(r) do
  @p -= 3
  @p -= 10
  @p -= 7
end

task g_divide(r : region(int), p : ptr(int, r))
where reduces/(r) do
  @p /= 2
  @p /= 1
end

task g_max(r : region(int), p : ptr(int, r))
where reduces max(r) do
  @p max= 1
  @p max= 0
end

task g_min(r : region(int), p : ptr(int, r))
where reduces min(r) do
  @p min= 10000
  @p min= 100000
  @p min= 1000000
end

-- You can also do reductions with read-write privileges.
task h_plus(r : region(int), p : ptr(int, r))
where reads(r), writes(r) do
  @p += 100
  @p += 100
  @p += 100
end

task h_minus(r : region(int), p : ptr(int, r))
where reads(r), writes(r) do
  @p -= 500
  @p -= 1200
  @p -= 1300
end

task f() : int
  var r = region(ispace(ptr, 5), int)
  var p = dynamic_cast(ptr(int, r), 0)
  @p = 1
  g_plus(r, p)
  g_times(r, p)
  g_minus(r, p)
  -- FIXME: Divide is currently broken.
  -- g_divide(r, p)
  g_max(r, p)
  g_min(r, p)
  h_plus(r, p)
  h_minus(r, p)
  return @p
end

task main()
  regentlib.assert(f() == -2636, "test failed")
end
regentlib.start(main)
