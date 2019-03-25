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

local data = require("common/data")

local hash_set = {}

hash_set.__index = hash_set

function hash_set.new()
  local s = {
    __set = data.newmap(),
    __len = false,
    __hash = false,
  }
  return setmetatable(s, hash_set)
end

function hash_set.from_list(l)
  local m = data.newmap()
  l:map(function(v) m[v] = true end)
  local s = {
    __set = m,
    __len = #l,
    __hash = false,
  }
  return setmetatable(s, hash_set)
end

function hash_set:copy()
  local s = {
    __set = self.__set:copy(),
    __len = self.__len,
    __hash = self.__hash,
  }
  return setmetatable(s, hash_set)
end

function hash_set:insert(v)
  self.__set[v] = true
  self.__len = false
  self.__hash = false
end

function hash_set:insert_all(other)
  for v, _ in other.__set:items() do
    self.__set[v] = true
  end
  self.len = false
  self.__hash = false
end

function hash_set:remove(v)
  self.__set[v] = nil
  self.__len = false
  self.__hash = false
end

function hash_set:remove_all(other)
  for v, _ in other.__set:items() do
    self.__set[v] = nil
  end
  self.len = false
  self.__hash = false
end

function hash_set:has(v)
  return self.__set[v]
end

function hash_set:__le(other)
  for v, _ in self.__set:items() do
    if not other:has(v) then return false end
  end
  return true
end

function hash_set:__mul(other)
  local result = hash_set.new()
  for v, _ in other.__set:items() do
    if self:has(v) then
      result.__set[v] = true
    end
  end
  return result
end

function hash_set:__add(other)
  local copy = self:copy()
  copy:insert_all(other)
  return copy
end

function hash_set:__sub(other)
  local copy = self:copy()
  copy:remove_all(other)
  return copy
end

function hash_set:map(fn)
  local result = hash_set.new()
  for v, _ in self.__set:items() do
    result.__set[fn(v)] = true
  end
  return result
end

function hash_set:map_table(tbl)
  return self:map(function(k) return tbl[k] end)
end

function hash_set:foreach(fn)
  for v, _ in self.__set:items() do
    fn(v)
  end
end

function hash_set:filter(fn)
  local result = hash_set.new()
  for v, _ in self.__set:items() do
    if fn(v) then result.__set[v] = true end
  end
  return result
end
function hash_set:to_list()
  local l = terralib.newlist()
  for v, _ in self.__set:items() do l:insert(v) end
  return l
end

function hash_set:__tostring()
  local str = nil
  for v, _ in self.__set:items() do
    if str == nil then
      str = tostring(v)
    else
      str = str .. "," .. tostring(v)
    end
  end
  return str or ""
end

function hash_set:hash()
  if not self.__hash then
    self:canonicalize()
    self.__hash = tostring(self)
  end
  return self.__hash
end

function hash_set:canonicalize(fn)
  local l = self:to_list()
  local fn = fn or function(v1, v2) return v1:hash() < v2:hash() end
  table.sort(l, fn)
  local s = hash_set.from_list(l)
  self.__set = s.__set
end

function hash_set:size()
  if not self.__len then
    local len = 0
    for v, _ in self.__set:items() do len = len + 1 end
    self.__len = len
  end
  return self.__len
end

function hash_set:is_empty()
  return self.__set:is_empty()
end

return hash_set
