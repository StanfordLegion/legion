-- Copyright 2019 Stanford University
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

local function gen_test(complex_type)

  local task g_plus(r : region(complex_type), p : ptr(complex_type, r))
  where reduces+(r) do
    @p += complex_type { 2, 0 }
    @p += complex_type { 13, 0 }
    @p += complex_type { 5, 0 }
  end

  local task g_times(r : region(complex_type), p : ptr(complex_type, r))
  where reduces*(r) do
    @p *= complex_type { 2, 0 }
    @p *= complex_type { 1, 0 }
    @p *= complex_type { 2, 0 }
  end

  local task g_minus(r : region(complex_type), p : ptr(complex_type, r))
  where reduces-(r) do
    @p -= complex_type { 3, 0 }
    @p -= complex_type { 10, 0 }
    @p -= complex_type { 7, 0 }
  end

  -- You can also do reductions with read-write privileges.
  local task h_plus(r : region(complex_type), p : ptr(complex_type, r))
  where reads(r), writes(r) do
    @p += complex_type { 100, 0 }
    @p += complex_type { 100, 0 }
    @p += complex_type { 100, 0 }
  end

  local task h_minus(r : region(complex_type), p : ptr(complex_type, r))
  where reads(r), writes(r) do
    @p -= complex_type { 500, 0 }
    @p -= complex_type { 1200, 0 }
    @p -= complex_type { 1300, 0 }
  end

  local task f() : complex_type
    var r = region(ispace(ptr, 5), complex_type)
    var p = dynamic_cast(ptr(complex_type, r), 0)
    @p = complex_type { 1, 0 }
    g_plus(r, p)
    g_times(r, p)
    g_minus(r, p)
    h_plus(r, p)
    h_minus(r, p)
    return @p
  end

  local task tsk()
    var v = complex_type { 1, 0 }
    v += complex_type { 2, 0 }
    v += complex_type { 13, 0 }
    v += complex_type { 5, 0 }
    v *= complex_type { 2, 0 }
    v *= complex_type { 1, 0 }
    v *= complex_type { 2, 0 }
    v -= complex_type { 3, 0 }
    v -= complex_type { 10, 0 }
    v -= complex_type { 7, 0 }
    v += complex_type { 100, 0 }
    v += complex_type { 100, 0 }
    v += complex_type { 100, 0 }
    v -= complex_type { 500, 0 }
    v -= complex_type { 1200, 0 }
    v -= complex_type { 1300, 0 }
    regentlib.assert(f().real == v.real, "test failed")
  end
  return tsk
end

task main()
  [gen_test(complex32)]()
end

regentlib.start(main)
