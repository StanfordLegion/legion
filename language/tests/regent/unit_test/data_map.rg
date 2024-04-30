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

local data = require("common/data")

local m = data.newmap()

local obj1 = {}
local obj2 = {}
local obj3 = {}

-- Insert some keys, including a few that might cause ordering issues.
m["a"] = 1
m["b"] = 2
m["c"] = 3
m[true] = 4
m[5] = 5
m[obj1] = 6
m[obj2] = 7
m[obj3] = 8
m["d"] = 9

-- Look up all keys.
assert(m["a"] == 1)
assert(m["b"] == 2)
assert(m["c"] == 3)
assert(m[true] == 4)
assert(m[5] == 5)
assert(m[obj1] == 6)
assert(m[obj2] == 7)
assert(m[obj3] == 8)
assert(m["d"] == 9)

-- Again, with different syntax.
assert(m:get("a") == 1)
assert(m:has("a") == 1)

-- Check some lookups that should fail.
assert(m["e"] == nil)
assert(m[4] == nil)
assert(m[false] == nil)
assert(m[nil] == nil)
assert(m[{}] == nil)

-- Delete a key.
m["b"] = nil
assert(m["b"] == nil)

-- Same, with a key that didn't exist in the first place.
m["e"] = nil
assert(m["e"] == nil)

-- Add a few more keys.
m[""] = obj1
m[false] = "definitely false"

-- No keys went missing.
assert(m["a"] == 1)
assert(m["b"] == nil) -- was deleted
assert(m["c"] == 3)
assert(m[true] == 4)
assert(m[5] == 5)
assert(m[obj1] == 6)
assert(m[obj2] == 7)
assert(m[obj3] == 8)
assert(m["d"] == 9)
assert(m[""] == obj1)
assert(m[false] == "definitely false")

-- Modify some key values.
m["c"] = "asdf"
m[obj3] = true
m["d"] = false

-- Put the key "b" back: it should appear at the end now.
m["b"] = 1234

-- Iterate keys.
do
  local it, t, init = m:keys()
  local i1, k1 = it(t, init)
  assert(k1 == "a")
  local i2, k2 = it(t, i1)
  assert(k2 == "c")
  local i3, k3 = it(t, i2)
  assert(k3 == true)
  local i4, k4 = it(t, i3)
  assert(k4 == 5)
  local i5, k5 = it(t, i4)
  assert(k5 == obj1)
  local i6, k6 = it(t, i5)
  assert(k6 == obj2)
  local i7, k7 = it(t, i6)
  assert(k7 == obj3)
  local i8, k8 = it(t, i7)
  assert(k8 == "d")
  local i9, k9 = it(t, i8)
  assert(k9 == "")
  local i10, k10 = it(t, i9)
  assert(k10 == false)
  local i11, k11 = it(t, i10)
  assert(k11 == "b")
  local i12, k12 = it(t, i11)
  assert(i12 == nil and k12 == nil)

  -- These are stateless, so I should be able to pick a random point
  -- in the middle to rerun.
  local i5b, k5b = it(t, i4)
  assert(k5b == obj1)
end

-- Iterate values.
do
  local it, t, init = m:values()
  local i1, v1 = it(t, init)
  assert(v1 == 1)
  local i2, v2 = it(t, i1)
  assert(v2 == "asdf")
  local i3, v3 = it(t, i2)
  assert(v3 == 4)
  local i4, v4 = it(t, i3)
  assert(v4 == 5)
  local i5, v5 = it(t, i4)
  assert(v5 == 6)
  local i6, v6 = it(t, i5)
  assert(v6 == 7)
  local i7, v7 = it(t, i6)
  assert(v7 == true)
  local i8, v8 = it(t, i7)
  assert(v8 == false)
  local i9, v9 = it(t, i8)
  assert(v9 == obj1)
  local i10, v10 = it(t, i9)
  assert(v10 == "definitely false")
  local i11, v11 = it(t, i10)
  assert(v11 == 1234)
  local i12, v12 = it(t, i11)
  assert(i12 == nil and v12 == nil)

  -- These are stateless, so I should be able to pick a random point
  -- in the middle to rerun.
  local i5b, v5b = it(t, i4)
  assert(v5b == 6)
end

-- Iterate items.
do
  local it, t, init = m:items()
  local k1, v1 = it(t, init)
  assert(k1 == "a" and v1 == 1)
  local k2, v2 = it(t, k1)
  assert(k2 == "c" and v2 == "asdf")
  local k3, v3 = it(t, k2)
  assert(k3 == true and v3 == 4)
  local k4, v4 = it(t, k3)
  assert(k4 == 5 and v4 == 5)
  local k5, v5 = it(t, k4)
  assert(k5 == obj1 and v5 == 6)
  local k6, v6 = it(t, k5)
  assert(k6 == obj2 and v6 == 7)
  local k7, v7 = it(t, k6)
  assert(k7 == obj3 and v7 == true)
  local k8, v8 = it(t, k7)
  assert(k8 == "d" and v8 == false)
  local k9, v9 = it(t, k8)
  assert(k9 == "" and v9 == obj1)
  local k10, v10 = it(t, k9)
  assert(k10 == false and v10 == "definitely false")
  local k11, v11 = it(t, k10)
  assert(k11 == "b" and v11 == 1234)
  local k12, v12 = it(t, k11)
  assert(k12 == nil and v12 == nil)

  -- These are stateless, so I should be able to pick a random point
  -- in the middle to rerun.
  local k5b, v5b = it(t, k4)
  assert(k5b == obj1 and v5b == 6)
end

-- Equality.
do
  local m1 = data.newmap()
  m1["a"] = 1
  m1["b"] = 2
  m1["c"] = 3

  local m2 = data.newmap()
  m2["a"] = 1
  m2["b"] = 2
  m2["c"] = 3

  -- A differently ordered map will not match.
  local m3 = data.newmap()
  m3["c"] = 3
  m3["a"] = 1
  m3["b"] = 2

  -- A map with a subset of elements will not match.
  local m4 = data.newmap()
  m4["a"] = 1
  m4["b"] = 2

  -- A map with a superset of elements will not match.
  local m5 = data.newmap()
  m5["a"] = 1
  m5["b"] = 2
  m5["c"] = 3
  m5["d"] = 4

  -- A copy will match.
  local m6 = m1:copy()

  assert(m1 == m2)
  assert(m2 == m1)

  assert(m1 ~= m3)
  assert(m3 ~= m1)

  assert(m1 ~= m4)
  assert(m4 ~= m1)

  assert(m1 ~= m5)
  assert(m5 ~= m1)

  assert(m1 == m6)
  assert(m6 == m1)
end
