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

require('legionlib')

local x = Point:new {1, 2}
local y = Point:new {3, 4}
local z = Point:new {1, 3}
local w = Point:new {2, 4}

print(tostring(x) .. " == " .. tostring(y) .. " : " .. tostring(x == y))
print(tostring(x) .. " == " .. tostring(x) .. " : " .. tostring(x == x))

print(tostring(x) .. " <= " .. tostring(x) .. " : " .. tostring(x <= x))
print(tostring(x) .. " <= " .. tostring(z) .. " : " .. tostring(x <= z))
print(tostring(y) .. " <= " .. tostring(x) .. " : " .. tostring(y <= x))
print(tostring(x) .. " <= " .. tostring(w) .. " : " .. tostring(x <= w))

print(tostring(x) .. " < " .. tostring(x) .. " : " .. tostring(x < x))
print(tostring(x) .. " < " .. tostring(z) .. " : " .. tostring(x < z))
print(tostring(y) .. " < " .. tostring(x) .. " : " .. tostring(y < x))
print(tostring(x) .. " < " .. tostring(w) .. " : " .. tostring(x < w))

print(tostring(x) .. " + " .. tostring(y) .. " : " .. tostring(x + y))
print(tostring(x) .. " - " .. tostring(y) .. " : " .. tostring(x - y))
print(tostring(x) .. " * " .. tostring(y) .. " : " .. tostring(x * y))
print(tostring(x) .. " / " .. tostring(y) .. " : " .. tostring(x / y))

print("-" .. tostring(x) .. " : " .. tostring(-x))

print(tostring(x) .. " dot " .. tostring(y) .. " : " .. x:dot(y))

local a = Rect:new(x, y)
local b = Rect:new(z, y)
local c = Rect:new({4, 5, 6, 7})

print(tostring(a) .. " == " .. tostring(a) .. " : " ..
         tostring(tostring(a) == tostring(a)))
print(tostring(a) .. " == " .. tostring(b) .. " : " ..
         tostring(tostring(a) == tostring(b)))

print(tostring(a) .. " overlaps " .. tostring(b) .. " : " ..
         tostring(a:overlaps(b)))
print(tostring(a) .. " overlaps " .. tostring(c) .. " : " ..
         tostring(a:overlaps(c)))

print(tostring(a) .. " contains " .. tostring(a) .. " : " ..
         tostring(a:contains(a)))
print(tostring(a) .. " contains " .. tostring(z) .. " : " ..
         tostring(a:contains(z)))
print(tostring(a) .. " * " .. tostring(b) .. " : " ..
         tostring(a * b))

a = Rect:new {1, 10}
b = Rect:new {3, 7}
print(tostring(a) .. " contains " .. tostring(a) .. " : " ..
         tostring(a:contains(a)))
print(tostring(a) .. " contains " .. tostring(b) .. " : " ..
         tostring(a:contains(b)))

print(tostring(a) .. " * " .. tostring(b) .. " : " ..
         tostring(a * b))

a = Rect:new({1, 1, 1, 6, 8, 10})

print("volume of " .. tostring(a) .. " : " .. tostring(a:volume()))

local q = Blockify:new(y)

print(tostring(q) .. " image " .. tostring(x) .. " : " ..
         tostring(q:image(x)))

print(tostring(q) .. " preimage " .. tostring(x) .. " : " ..
         tostring(q:preimage(x)))

local low = Point:new {0, 0}
local high = Point:new {7, 32}
local itr = PointInRectIterator:new(Rect:new(low, high))
while(itr:has_next())
do
   local point = itr:next()
   print(point)
end

local blockify = Blockify:new {32, 2}
local range = blockify:preimage(Point:new {3, 4})
itr = PointInRectIterator:new(range)
while(itr:has_next())
do
   local point = itr:next()
   print(point)
end

local blockify_in_c = blockify:to_c_object()
local rect_in_c = Rect:new(low, high):to_c_object()
local point_in_c = low:to_c_object()
