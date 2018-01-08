-- Copyright 2018 Stanford University
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

local std = terralib.includec("stdlib.h")
local ffi = require("ffi")

-- utility functions

function printf(...)
  print(string.format(...))
end

function ite(a, b, c)
  if a then return b else return c end
end

function inherit(table, metatable)
  local metatable_ = getmetatable(table)
  local combined = {
    __index = function(child, idx)
      return rawget(metatable_, idx) or
             rawget(metatable, idx) or
             rawget(child, idx)
    end
  }
  setmetatable(table, combined)
  return table
end

sizeof = terralib.sizeof

-- gcmalloc allocates GCable memory blocks
local malloc_functions = {}

function gcmalloc(c_type, size)
  if not malloc_functions[c_type] then
    malloc_functions[c_type] = terra(size : uint) : &c_type
      return [&c_type](std.malloc(sizeof(c_type) * size))
    end
  end
  return ffi.gc(malloc_functions[c_type](size), std.free)
end

-- Simple array implementation
Array = {}

function Array:new(values)
  values = values or {}
  local array = { size = 0, values = {} }
  if values then
    local size = 0
    for _, v in pairs(values) do
      array.values[size] = v
      size = size + 1
    end
    array.size = size
  end
  setmetatable(array, self)
  self.__index = function(child, idx)
    return rawget(self, idx) or
           rawget(rawget(child, "values"), idx) or
           error("Error: invalid access to array: " .. idx)
  end
  self.__newindex = function(child, idx, value)
    rawset(rawget(child, "values"), idx, value)
  end
  return array
end

function Array:init(num, value)
  local values = {}
  for i = 1, num do
    values[i] = value
  end
  return Array:new(values)
end

function Array:insert(v)
  self[self.size] = v
  self.size = self.size + 1
end

function Array:clear()
  self.size = 0
  self.values = {}
end

function Array:itr(f)
  for i = 0, self.size - 1 do
    f(self[i])
  end
end

function Array:__tostring()
  local str = ""
  for i = 0, self.size - 1 do
    if i > 0 then
      str = str .. ", "
    end
    str = str .. tostring(self.values[i])
  end
  return str
end

-- Simple set implementation
Set = {}
PrimSet = {}
PrimSet.__index = PrimSet
ObjSet = {}
ObjSet.__index = ObjSet

function Set:new(l, elem_type)
  l = l or {}
  local set = { set = {}, size = 0 }

  if not elem_type or not elem_type.__hash then
    setmetatable(set, PrimSet)
  else
    setmetatable(set, ObjSet)
  end
  for _, v in pairs(l) do
    set:insert(v)
  end

  return set
end

function PrimSet:insert(v)
  self.set[v] = true
  self.size = self.size + 1
end

function PrimSet:erase(v)
  self.set[v] = nil
  self.size = self.size - 1
end

function PrimSet:elem(v)
  return self.set[v]
end

function PrimSet:itr(f)
  for v, _ in pairs(self.set) do
    f(v)
  end
end

function PrimSet:__tostring()
  local str = ""
  local first = true
  for k, _ in pairs(self.set) do
    if not first then
      str = str .. ", "
    end
    first = false
    str = str .. tostring(k)
  end
  return "{" .. str .. "}"
end

function ObjSet:insert(v)
  local h = v:__hash()
  if self.set[h] then
    self.set[h]:insert(v)
  else
    self.set[h] = Array:new{v}
  end
  self.size = self.size + 1
end

local function equivalent(a,b)
  if type(a) ~= 'table' then return a == b end
  local counta, countb = 0, 0
  for k,va in pairs(a) do
    if not equivalent(va, b[k]) then return false end
    counta = counta + 1
  end
  for _,_ in pairs(b) do countb = countb + 1 end
  return counta == countb
end

function ObjSet:erase(v)
  local h = v:__hash()
  local arr = self.set[h]
  if arr then
    if arr.size == 1 then
      self.set[h] = nil
      self.size = self.size - 1
    else
      local arr_ = Array:new {}
      for i = 0, arr.size - 1 do
        if not equivalent(v, arr[i]) then
          arr_:insert(arr[i])
        end
      end
      self.set[h] = arr_
      self.size = self.size - 1
    end
  end
end

function ObjSet:elem(v)
  local h = v:__hash()
  local arr = self.set[h]
  if arr then
    for i = 0, arr.size - 1 do
      if equivalent(v, arr[i]) then
        return true
      end
    end
  end
  return false
end

function ObjSet:itr(f)
  for _, l in pairs(self.set) do
    for i = 0, l.size - 1 do
      f(l.values[i])
    end
  end
end

function ObjSet:__tostring()
  local str = ""
  local first = true
  for k, _ in pairs(self.set) do
    if not first then
      str = str .. ", "
    end
    first = false
    str = str .. tostring(k)
  end
  return "{" .. str .. "}"
end

