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

task g_plus(r : region(complex64), p : ptr(complex64, r))
where reduces+(r) do
  @p += complex64 { 2, 0 }
  @p += complex64 { 13, 0 }
  @p += complex64 { 5, 0 }
end

task g_minus(r : region(complex64), p : ptr(complex64, r))
where reduces-(r) do
  @p -= complex64 { 3, 0 }
  @p -= complex64 { 20, 0 }
  @p -= complex64 { 7, 0 }
end

-- You can also do reductions with read-write privileges.
task h_plus(r : region(complex64), p : ptr(complex64, r))
where reads(r), writes(r) do
  @p += complex64 { 100, 0 }
  @p += complex64 { 100, 0 }
  @p += complex64 { 100, 0 }
end

task h_minus(r : region(complex64), p : ptr(complex64, r))
where reads(r), writes(r) do
  @p -= complex64 { 500, 0 }
  @p -= complex64 { 1200, 0 }
  @p -= complex64 { 1300, 0 }
end

task f() : complex64
  var r = region(ispace(ptr, 5), complex64)
  var p = dynamic_cast(ptr(complex64, r), 0)
  @p = complex64 { 1, 0 }
  g_plus(r, p)
  g_minus(r, p)
  h_plus(r, p)
  h_minus(r, p)
  return @p
end

task main()
  var v = complex64 { 1, 0 }
  v += complex64 { 2, 0 }
  v += complex64 { 13, 0 }
  v += complex64 { 5, 0 }
  v -= complex64 { 3, 0 }
  v -= complex64 { 20, 0 }
  v -= complex64 { 7, 0 }
  v += complex64 { 100, 0 }
  v += complex64 { 100, 0 }
  v += complex64 { 100, 0 }
  v -= complex64 { 500, 0 }
  v -= complex64 { 1200, 0 }
  v -= complex64 { 1300, 0 }
  regentlib.assert(f().real == v.real, "test failed")
end

regentlib.start(main)
