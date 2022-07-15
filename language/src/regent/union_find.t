-- Copyright 2022 Stanford University
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

-- Union Find

local data = require("common/data")

local union_find = {}
union_find.__index = union_find

function union_find.new()
  return setmetatable({forest = data.newmap()}, union_find)
end

function union_find:union_keys(a, b)
  local root_a = self:find_key(a)
  local root_b = self:find_key(b)
  self.forest[root_b] = root_a
  return root_a
end

function union_find:find_key(k)
  if not self.forest[k] then
    self.forest[k] = k
  end

  -- Find the root.
  local root = k
  while root ~= self.forest[root] do
    root = self.forest[root]
  end

  -- Path compression.
  local node = k
  while node ~= self.forest[node] do
    self.forest[node] = root
    node = self.forest[node]
  end

  return root
end

function union_find:keys()
  local result = terralib.newlist()
  for _, k in self.forest:keys() do
    result:insert(k)
  end
  return result
end

return union_find
