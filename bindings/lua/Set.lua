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

Set = {}

-- create a new set with the values of the given list
function Set:new(l)
   local l_ = l or {}
   local set = { __set = {}, __size = 0 }
   for _, v in ipairs(l_) do set.__set[v] = true end
   set.__size = #l_
   setmetatable(set, self)
   self.__add = Set.union
   self.__mul = Set.intersection
   self.__tostring = Set.tostring
   self.__index = self
   return set
end

-- addition and deletion functions are mutable
function Set:add(v)
   self.__set[v] = true
   self.__size = self.__size + 1
end

function Set:del(v)
   self.__set[v] = nil
   self.__size = self.__size - 1   
end

function Set:mem(v)
   if not (self.__set[v] == true) then return false
   else return true
   end
end

function Set:size()
   return self.__size
end

function Set:get(n)
   for v in pairs(self.__set)
   do
      if n == 0 then return v
      else n = n - 1
      end
   end
   assert(false, n .. " is bigger than the size of the set")
end

-- set union and intersection are immutable functions
function Set:union(a)
   local res = Set:new {}
   for k in pairs(self.__set) do res:add(k) end
   for k in pairs(a.__set) do res:add(k) end
   return res
end

function Set:intersection(a)
   local res = Set:new {}
   for k in pairs(self.__set) do
      if a:mem(k) then res:add(k) end
   end
   return res
end

function Set:tostring()
   local l = self:to_list()
   return "{" .. table.concat(l, ", ") .. "}"
end

function Set:print()
   print(self:tostring())
end

function Set:to_list()
   local l = {}     -- list to put all elements from the set
   for e in pairs(self.__set) do
      l[#l + 1] = e
   end
   return l
end
