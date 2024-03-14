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

local state = 0
local assign_id = data.weak_memoize(
  function(k)
    state = state + 1
    return state
  end)

local k1 = {}
local k2 = {}
local k3 = {}

local v1 = assign_id(k1)
local v2 = assign_id(k2)
local v3 = assign_id(k3)

assert(assign_id(k1) == v1)
assert(assign_id(k2) == v2)
assert(assign_id(k3) == v3)

-- Clear a key and make sure it goes away after GC.
k2 = nil
collectgarbage("collect")
collectgarbage("collect") -- Need to run twice to trigger full collection.

-- Make sure the remaining keys are still there.
assert(assign_id(k1) == v1)
assert(assign_id(k3) == v3)

-- Clear a key and make sure it goes away after GC.
k1 = nil
collectgarbage("collect")
collectgarbage("collect") -- Need to run twice to trigger full collection.

-- Make sure the remaining keys are still there.
assert(assign_id(k3) == v3)
