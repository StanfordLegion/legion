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

local base = require("regent/std_base")

local partition_info = {}

partition_info.__index = partition_info

function partition_info.new(region_symbol, disjoint, complete)
  assert(base.is_symbol(region_symbol))
  local tuple = {
    region = region_symbol,
    disjoint = disjoint or false,
    complete = complete or false,
  }
  return setmetatable(tuple, partition_info)
end

function partition_info:clone(region_mapping)
  return partition_info.new(region_mapping(self.region),
                            self.disjoint,
                            self.complete)
end

function partition_info:meet_disjointness(disjoint)
  self.disjoint = self.disjoint or disjoint
end

function partition_info:meet_completeness(complete)
  self.complete = self.complete or complete
end

function partition_info:__tostring()
  local disjoint = (self.disjoint and "D") or "A"
  local complete = (self.complete and "C") or "I"
  return "partition(" .. tostring(self.region) ..  "," ..
    disjoint .. "," .. complete .. ")"
end

function partition_info:unifiable(other)
  return self.region == other.region
end

function partition_info:__eq(other)
  return self.region == other.region and
         self.disjoint == other.disjoint and
         self.complete == other.complete
end

return partition_info
