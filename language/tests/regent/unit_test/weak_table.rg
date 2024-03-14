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

local w = data.new_weak_key_table()

local k1 = {}
local k2 = {}
local k3 = {}

w[k1] = 123
w[k2] = 456
w[k3] = 789

assert(w[k1] == 123)
assert(w[k2] == 456)
assert(w[k3] == 789)

-- Clear a key and make sure it goes away after GC.
k2 = nil
collectgarbage("collect")
collectgarbage("collect") -- Need to run twice to trigger full collection.

-- Should only have 2 keys in the table now.
local i = 0
for k, v in pairs(w) do
  i = i + 1
  assert(v == 123 or v == 789)
end
assert(i == 2)

-- Clear a key and make sure it goes away after GC.
k1 = nil
collectgarbage("collect")
collectgarbage("collect") -- Need to run twice to trigger full collection.

-- Should only have 1 keys in the table now.
local j = 0
for k, v in pairs(w) do
  j = j + 1
  assert(v == 789)
end
assert(j == 1)
