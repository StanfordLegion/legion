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

-- Data Structures

-- Elliott: The more I write in Lua, the more I feel like I'm
-- reimplementing Python piece by piece. Several limitations in Lua
-- (e.g. no __hash metamethod) can bite hard if you try to go without
-- properly engineered data structures. (Exercise for the reader: try
-- to write a fully general purpose memoize function in plain Lua.)
--
-- Note: In the code below, whenever an interface exists in plain Lua
-- (and can be hijacked with existing Lua metamethods), those
-- metamethods are used. In all other cases, the metatable is left
-- alone, and regular functions and methods are used instead. So for
-- example:
--
--   * Using metamethods:
--     x + y        -- addition (via __add)
--     x .. y       -- concatenation (via __concat)
--     x[k]         -- key lookup (via __index)
--     x[k] = v     -- key assignment (via __newindex)
--     tostring(x)  -- stringification (via __tostring)
--
--   * Using regular methods and functions:
--     x:items()    -- key, value iterator (__pairs does not exist in LuaJIT)
--     x:keys()     -- key iterator
--     x:values()   -- value iterator
--     data.hash(x) -- calls x:hash() if supported, otherwise returns x
--
-- This means that, for example the [] operator, and the pairs
-- function, may or may not work with a given data structure, since it
-- depends on the internal representation of that data structure.

local data = {}

-- Utility for figuring out if we're in LuaJIT or PUC Lua.
function data.is_luajit()
  return type(rawget(_G,"jit")) == "table"
end

-- #####################################
-- ## Hashing
-- #################

-- Note: Unlike a normal hash function, it is important that the
-- values returned by this function are unique. This is because the
-- hash values are frequently used in Lua tables as the keys
-- themselves---thus, they get treated as if they were the values. Lua
-- itself has no alternative (there is no __hash metamethod), so this
-- is the best you can do without building your own separate-chaining
-- hash map, which would either be slow (if you did it in Lua) or
-- would require extensions (if you did it in C).

function data.hash(x)
  if type(x) == "table" and x.hash then
    return x:hash()
  else
    return x
  end
end

-- #####################################
-- ## Numbers
-- #################

function data.min(a, b)
  if a < b then
    return a
  else
    return b
  end
end

function data.max(a, b)
  if a > b then
    return a
  else
    return b
  end
end

-- #####################################
-- ## Booleans
-- #################

function data.any(...)
  for _, elt in ipairs({...}) do
    if elt then
      return true
    end
  end
  return false
end

function data.all(...)
  for _, elt in ipairs({...}) do
    if not elt then
      return false
    end
  end
  return true
end

-- #####################################
-- ## Lists
-- #################

-- The following methods work on Terra lists, or regular Lua tables
-- with 1-based consecutive numeric indices.

function data.range(start, stop) -- zero-based, exclusive (as in Python)
  if stop == nil then
    stop = start
    start = 0
  end
  local result = terralib.newlist()
  for i = start, stop - 1, 1 do
    result:insert(i)
  end
  return result
end

function data.mapi(fn, list)
  local result = terralib.newlist()
  for i, elt in ipairs(list) do
    result:insert(fn(i, elt))
  end
  return result
end

function data.flatmap(fn, list)
  local result = terralib.newlist()
  for i, elt in ipairs(list) do
    elt = fn(elt)
    if terralib.islist(elt) then
      result:insertall(elt)
    else
      result:insert(elt)
    end
  end
  return result
end

function data.filter(fn, list)
  local result = terralib.newlist()
  for _, elt in ipairs(list) do
    if fn(elt) then
      result:insert(elt)
    end
  end
  return result
end

function data.filteri(fn, list)
  local result = terralib.newlist()
  for i, elt in ipairs(list) do
    if fn(elt) then
      result:insert(i)
    end
  end
  return result
end

function data.reduce(fn, list, init)
  local result = init
  for i, elt in ipairs(list) do
    if i == 1 and result == nil then
      result = elt
    else
      result = fn(result, elt)
    end
  end
  return result
end

function data.zip(...)
  local lists = terralib.newlist({...})
  local len = data.reduce(
    data.min,
    lists:map(function(list) return #list or 0 end)) or 0
  local result = terralib.newlist()
  for i = 1, len do
    result:insert(lists:map(function(list) return list[i] end))
  end
  return result
end

function data.flatten(list)
  local result = terralib.newlist()
  for _, sublist in ipairs(list) do
    result:insertall(sublist)
  end
  return result
end

function data.take(n, list)
  local result = terralib.newlist()
  for i, elt in ipairs(list) do
    result:insert(elt)
    if i >= n then
      break
    end
  end
  return result
end

function data.dict(list)
  local result = {}
  for _, pair in ipairs(list) do
    result[pair[1]] = pair[2]
  end
  return result
end

function data.find_key(dict, val)
  -- given a value find the key
  for k, v in pairs(dict) do
    if v == val then
      return k
    end
  end
  return nil
end

function data.set(list)
  local result = {}
  for _, k in ipairs(list) do
    result[k] = true
  end
  return result
end

-- #####################################
-- ## Tuples
-- #################

data.tuple = {}
setmetatable(data.tuple, { __index = terralib.newlist })
data.tuple.__index = data.tuple

function data.tuple.__eq(a, b)
  if not data.is_tuple(a) or not data.is_tuple(b) then
    return false
  end
  if #a ~= #b then
    return false
  end
  for i, v in ipairs(a) do
    if v ~= b[i] then
      return false
    end
  end
  return true
end

function data.tuple.__concat(a, b)
  assert(data.is_tuple(a) and (not b or data.is_tuple(b)))
  local result = data.newtuple()
  result:insertall(a)
  if not b then
    return result
  end
  result:insertall(b)
  return result
end

function data.tuple:slice(start --[[ inclusive ]], stop --[[ inclusive ]])
  local result = data.newtuple()
  for i = start, stop do
    result:insert(self[i])
  end
  return result
end

function data.tuple:starts_with(t)
  assert(data.is_tuple(t))
  return self:slice(1, data.min(#self, #t)) == t
end

function data.tuple:contains(f)
  for i = 1, #self do
    if self[i] == f then return true end
  end
  return false
end

function data.tuple:mkstring(first, sep, last)
  if first and sep and last then
    return first .. self:map(tostring):concat(sep) .. last
  elseif first and sep then
    return first .. self:map(tostring):concat(sep)
  else
    return self:map(tostring):concat(first)
  end
end

function data.tuple:__tostring()
  return self:mkstring("<", ".", ">")
end

function data.tuple:hash()
  return "data.tuple" .. tostring(self)
end

function data.newtuple(...)
  return setmetatable({...}, data.tuple)
end

function data.is_tuple(x)
  return getmetatable(x) == data.tuple
end

-- #####################################
-- ## Vectors
-- #################

data.vector = {}
setmetatable(data.vector, { __index = data.tuple })
data.vector.__index = data.vector

function data.vector.__eq(a, b)
  assert(data.is_vector(a) and data.is_vector(b))
  for i, v in ipairs(a) do
    if v ~= b[i] then
      return false
    end
  end
  return true
end

-- Since Lua doesn't support mixed-type equality, we have to expose this as a method.
function data.vector.eq(a, b)
  if data.is_vector(a) then
    if type(b) == "number" then
      for i, v in ipairs(a) do
        if v ~= b then
          return false
        end
      end
      return true
    end
  elseif data.is_vector(b) then
    return data.vector.eq(b, a)
  end
  return a == b
end

function data.vector.__add(a, b)
  if data.is_vector(a) then
    if data.is_vector(b) then
      local result = data.newvector()
      for i, v in ipairs(a) do
        result:insert(v + b[i])
      end
      return result
    elseif type(b) == "number" then
      local result = data.newvector()
      for i, v in ipairs(a) do
        result:insert(v + b)
      end
      return result
    end
  elseif data.is_vector(b) then
    return data.vector.__add(b, a)
  end
  assert(false) -- At least one should have been a vector
end

function data.vector.__mul(a, b)
  if data.is_vector(a) then
    if data.is_vector(b) then
      local result = data.newvector()
      for i, v in ipairs(a) do
        result:insert(v * b[i])
      end
      return result
    elseif type(b) == "number" then
      local result = data.newvector()
      for i, v in ipairs(a) do
        result:insert(v * b)
      end
      return result
    end
  elseif data.is_vector(b) then
    return data.vector.__mul(b, a)
  end
  assert(false) -- At least one should have been a vector
end

function data.vector:__tostring()
  return self:mkstring("[", ",", "]")
end

function data.vector:hash()
  return "data.vector" .. tostring(self)
end

function data.newvector(...)
  return setmetatable({...}, data.vector)
end

function data.is_vector(x)
  return getmetatable(x) == data.vector
end

-- #####################################
-- ## Maps
-- #################

-- This is an insertion ordered map data structure. It supports the
-- following operations:
--
--  * Put: O(1)
--  * Get: O(1)
--  * Iterate keys: O(M)
--  * Iterate values: O(M)
--  * Iterate key/value pairs: O(M)
--
-- Where N is the number of live entries, and M is the number of total
-- (live + dead) entries. Right now there isn't any effort to do
-- compaction (to reduce iteration from O(M) to O(N)), but this could
-- be added in the future if it is determined to be an important use
-- case.
--
-- Keys may be any value that supports data.hash().
--
-- The following APIs are supported on a map M:
--
--     M[k]                   -- equivalent to M:get(k)
--     M[k] = v               -- equivalent to M:put(k, v)
--     M:get(k)
--     M:has(k)               -- equivalent to M:get(k) (but see default_map)
--     M:put(k, v)
--     for _, k in M:keys()   -- note that the _ is undefined
--     for _, v in M:values() -- note that the _ is undefined
--     for k, v in M:items()
--     M:is_empty()
--     M:copy()               -- note: a shallow copy
--     M:map(fn)              -- map fn over k, v pairs and return a map
--     M:map_list(fn)         -- map fn over k, v pairs and return a list
--     tostring(M)            -- human readable representation of the map
--
-- The following APIs are NOT SUPPORTED:
--
--     for k, v in pairs(M)   -- DON'T DO THIS: IT WON'T WORK
--     for k, v in ipairs(M)  -- DON'T DO THIS: IT WON'T WORK

data.map = {}

do
-- Sentinels that won't accidentally collide with keys supplied by the user.
local keys_by_hash = {}
local values_by_hash = {}
local next_insertion_index = {}
local hash_by_insertion_index = {}
local insertion_index_by_hash = {}

function data.newmap()
  return setmetatable(
    {
      [keys_by_hash] = {},
      [values_by_hash] = {},
      [next_insertion_index] = 1,
      [hash_by_insertion_index] = {},
      [insertion_index_by_hash] = {},
    }, data.map)
end

function data.map_from_table(t)
  local result = data.newmap()
  for k, v in pairs(t) do
    result[k] = v
  end
  return result
end

function data.is_map(x)
  return getmetatable(x) == data.map
end

function data.map:__index(k)
  return self[values_by_hash][data.hash(k)] or data.map[k]
end

function data.map:__newindex(k, v)
  self:put(k, v)
end

function data.map:has(k)
  return self[values_by_hash][data.hash(k)]
end

function data.map:get(k)
  return self[values_by_hash][data.hash(k)]
end

function data.map:put(k, v, verbose)
  local kh = data.hash(k)
  if v == nil then
    k = nil
  end

  local idx = self[insertion_index_by_hash][kh]
  if not idx and v ~= nil then
    idx = self[next_insertion_index]
    self[next_insertion_index] = self[next_insertion_index] + 1
  end

  self[keys_by_hash][kh] = k
  self[values_by_hash][kh] = v
  self[insertion_index_by_hash][kh] = v ~= nil and idx or nil
  if idx then
    if v ~= nil then
      self[hash_by_insertion_index][idx] = kh
    else
      self[hash_by_insertion_index][idx] = nil
    end
  end
end

function data.map:next_item(last_k, verbose)
  local idx = last_k ~= nil and
    self[insertion_index_by_hash][data.hash(last_k)] or 0

  local k, v
  repeat
    idx = idx + 1
    local kh = self[hash_by_insertion_index][idx]
    if kh ~= nil then
      k = self[keys_by_hash][kh]
      v = self[values_by_hash][kh]
    else
      k, v = nil, nil
    end
  until k ~= nil or idx + 1 >= self[next_insertion_index]
  return k, v
end

function data.map:items()
  return data.map.next_item, self, nil
end

function data.map:next_key(idx)
  local k
  repeat
    idx = idx + 1
    local kh = self[hash_by_insertion_index][idx]
    if kh ~= nil then
      k = self[keys_by_hash][kh]
    else
      k = nil
    end
  until k ~= nil or idx + 1 >= self[next_insertion_index]
  return k ~= nil and idx or nil, k
end

function data.map:keys()
  return data.map.next_key, self, 0
end

function data.map:next_value(idx)
  local v
  repeat
    idx = idx + 1
    local kh = self[hash_by_insertion_index][idx]
    if kh ~= nil then
      v = self[values_by_hash][kh]
    else
      v = nil
    end
  until v ~= nil or idx + 1 >= self[next_insertion_index]
  return v ~= nil and idx or nil, v
end

function data.map:values()
  return data.map.next_value, self, 0
end

function data.map:is_empty()
  return next(self[values_by_hash]) == nil
end

function data.map:copy()
  return self:map(function(k, v) return v end)
end

function data.map:map(fn)
  local result = data.newmap()
  for k, v in self:items() do
    result:put(k, fn(k, v))
  end
  return result
end

function data.map:map_list(fn)
  local result = terralib.newlist()
  for k, v in self:items() do
    result:insert(fn(k, v))
  end
  return result
end

function data.map:__tostring()
  return "{" .. self:map_list(
    function(k, v)
      return tostring(k) .. "=" .. tostring(v)
    end):concat(",") .. "}"
end

function data.map:inspect()
  print("keys_by_hash:")
  for k, v in pairs(self[keys_by_hash]) do
    print("", k, v)
  end
  print("values_by_hash:")
  for k, v in pairs(self[values_by_hash]) do
    print("", k, v)
  end
  print("insertion_index_by_hash:")
  for k, v in pairs(self[insertion_index_by_hash]) do
    print("", k, v)
  end
  print("hash_by_insertion_index:")
  for k, v in pairs(self[hash_by_insertion_index]) do
    print("", k, v)
  end
  print("next_insertion_index:", self[next_insertion_index])
end
end -- data.map

-- #####################################
-- ## Default Maps
-- #################

-- Default maps are like maps with the following differences:
--
-- The factory, supplied to data.new_default_map is called to
-- construct new keys on the following API calls:
--
--     M:get(k) -- returns factory(k) if value does not exist
--     M[k]     -- equivalent to M:get(k)
--
-- The following API calls DO NOT construct default values:
--
--     M:has(k) -- returns nil if the value does not exist

do
-- Sentinels that won't accidentally collide with keys supplied by the user.
local default_factory = {}

data.default_map = setmetatable(
  {
    -- So, apparently for this to work you must re-list any metamethods.
    __tostring = data.map.__tostring,
    __newindex = data.map.__newindex,
  }, {
    __index = data.map,
})

function data.new_default_map(factory)
  local map = data.newmap()
  rawset(map, default_factory, factory)
  return setmetatable(map, data.default_map)
end

local function make_recursive_map(depth)
  return function()
    if depth > 0 then
      return data.new_recursive_map(depth - 1)
    end
  end
end

function data.new_recursive_map(depth)
  return data.new_default_map(make_recursive_map(depth))
end

function data.is_default_map(x)
  return getmetatable(x) == data.default_map
end

function data.default_map:__index(k)
  local lookup = data.map.get(self, k) or data.default_map[k]
  if lookup == nil then
    lookup = self[default_factory](k)
    if lookup ~= nil then self:put(k, lookup) end
  end
  return lookup
end

function data.default_map:has(k)
  return data.map.get(self, k)
end

function data.default_map:get(k)
  local lookup = data.map.get(self, k)
  if lookup == nil then
    lookup = self[default_factory](k)
    if lookup ~= nil then self:put(k, lookup) end
  end
  return lookup
end
end -- data.default_map

return data
