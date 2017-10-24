-- Copyright 2017 Stanford University
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

local ffi = require("ffi")
local std = terralib.includec("stdlib.h")
local cstr = terralib.includec("string.h")
local cstdio = terralib.includec("stdio.h")

legion = {}
legion.__index = legion

require 'legionlib-terra'
-- require 'legionlib-mapper'
require 'legionlib-util'

local legion_c = terralib.includec("legion.h")
local legion_terra = terralib.includecstring([[
#include "legion_terra.h"
#include "legion_terra_tasks.h"
]])
local terra terra_null() : &opaque return [&opaque](0) end

-- coercion functions

local function make_coercion_op(c_type)
  local terra coerce(ptr : &opaque, idx : uint)
    var proc = [&c_type](ptr)
    return proc[idx]
  end
  return coerce
end

coerce_machine = make_coercion_op(legion_c.legion_machine_t)
coerce_processor = make_coercion_op(legion_c.legion_processor_t)
coerce_domain = make_coercion_op(legion_c.legion_domain_t)

-- initializing binding library

function legion:set_top_level_script(name)
  self.script_name = name
end

if not rawget(_G, "initialized") then
  -- local flag = package.loadlib("liblegion_terra.so", "*")
  -- if not flag then
  --   print "Error: failed to load the shared binding library"
  --   os.exit(-1)
  -- end
  terralib.linklibrary("liblegion_terra.so")
  if not rawget(_G, "arg") then
    print "Error: legionlib requires a top-level script as a command line argument"
    os.exit(-1)
  end
  legion:set_top_level_script(arg[0])
end

-- imports constants in legion_c.h to the legion table
for k, v in pairs(legion_c) do
    if type(v) == "number" then
      legion[k] = v
    end
  end

-- Type declarations
Ptr = {}
Ptr.__index = Ptr

Pair = {}
Pair.__index = Pair

Coloring = {}
Coloring.__index = Coloring

Point = {}

Rect = {}
Rect.__index = Rect

Blockify = {}
Blockify.__index = Blockify

Domain = {}
Domain.__index = Domain

DomainColoring = {}

DomainPoint = {}
DomainPoint.__index = DomainPoint

IndexSpace = {}
IndexSpace.__index = IndexSpace

IndexPartition = {}
IndexPartition.__index = IndexPartition

IndexAllocator = {}
IndexAllocator.__index = IndexAllocator

FieldSpace = {}
FieldSpace.__index = FieldSpace

FieldAllocator = {}
FieldAllocator.__index = FieldAllocator

LogicalRegion = {}
LogicalRegion.__index = LogicalRegion

LogicalPartition = {}
LogicalPartition.__index = LogicalPartition

RegionAccessor = {}
RegionAccessor.__index = RegionAccessor

PhysicalRegion = {}
PhysicalRegion.__index = PhysicalRegion

RegionRequirement = {}
RegionRequirement.__index = RegionRequirement

ArgumentMap = {}
ArgumentMap.__index = ArgumentMap

Future = {}
Future.__index = Future

FutureMap = {}
FutureMap.__index = FutureMap

Task = {}
Task.__index = Task

Inline = {}
Inline.__index = Inline

TaskArgument = {}
TaskArgument.__index = TaskArgument

TaskLauncher = {}
TaskLauncher.__index = TaskLauncher

IndexLauncher = {}
IndexLauncher.__index = IndexLauncher

InlineLauncher = {}
InlineLauncher.__index = InlineLauncher

GenericPointInRectIterator = {}
GenericPointInRectIterator.__index = GenericPointInRectIterator

IndexIterator = {}
IndexIterator.__index = IndexIterator

TaskConfigOptions = {}
TaskConfigOptions.__index = TaskConfigOptions

Predicate = {}
Predicate.__index = Predicate
Predicate.TRUE_PRED = legion_c.legion_predicate_true()
Predicate.FALSE_PRED = legion_c.legion_predicate_false()

legion.PLUS_OP = 1
legion.MINUS_OP = 2
legion.TIMES_OP = 3

ReductionOp = {}
ReductionOp.__index = ReductionOp

ReductionOp.op_names = {"plus", "minus", "times"}
ReductionOp.op_types = {float, double, int}
ReductionOp.register_funcs = {}
ReductionOp.reduce_funcs = {}
ReductionOp.safe_reduce_funcs = {}

Machine = {}
Machine.__index = Machine

Processor = {}
Processor.__index = Processor

MachineQueryInterface = {}
MachineQueryInterface.__index = MachineQueryInterface

DomainSplit = {}
DomainSplit.__index = DomainSplit

DomainSplitVector = {}
DomainSplitVector.__index = DomainSplitVector

for idx, name in pairs(ReductionOp.op_names) do
  do
    local funcs = {}
    for _, ty in pairs(ReductionOp.op_types) do
      local register_func_name =
        string.format("register_reduction_%s_%s", name, tostring(ty))
      funcs[ty] = legion_terra[register_func_name]
    end
    ReductionOp.register_funcs[idx] = funcs
  end
  do
    local funcs = {}
    for _, ty in pairs(ReductionOp.op_types) do
      funcs[ty] = {}
      local reduce_func_name =
        string.format("reduce_%s_%s", name, tostring(ty))
      funcs[ty] = legion_terra[reduce_func_name]
    end
    ReductionOp.reduce_funcs[idx] = funcs
  end
  do
    local funcs = {}
    for _, ty in pairs(ReductionOp.op_types) do
      funcs[ty] = {}
      local reduce_func_name =
        string.format("safe_reduce_%s_%s", name, tostring(ty))
      funcs[ty] = legion_terra[reduce_func_name]
    end
    ReductionOp.safe_reduce_funcs[idx] = funcs
  end
end

function ReductionOp:register_reduction_op(op_type, elem_type, redop)
  local func = ReductionOp.register_funcs[op_type][elem_type]
  assert(func,
    string.format("Error: undefined reduction operator %s for type %s",
    ReductionOp.op_names[op_type], tostring(elem_type)))
  func(redop)
end

function ReductionOp:get_reduce_func(op_type, elem_type)
  return ReductionOp.reduce_funcs[op_type][elem_type]
end

function ReductionOp:get_safe_reduce_func(op_type, elem_type)
  return ReductionOp.safe_reduce_funcs[op_type][elem_type]
end

-- auxiliary function that converts cdata members to lua wrappers
local function convert_cdata(value)
  if type(value) ~= "cdata" then
    return value
  end
  local value_type = terralib.typeof(value)
  if value_type == legion_c.legion_ptr_t then
    return Ptr:from_cobj(value)
  end
  if terralib.typeof(value) == legion_c.legion_logical_region_t then
    return LogicalRegion:from_cobj(value)
  end
  if terralib.typeof(value) == legion_c.legion_logical_partition_t then
    return LogicalPartition:from_cobj(value)
  end
  if terralib.typeof(value) == legion_c.legion_index_space_t then
    return IndexSpace:from_cobj(value)
  end
  if terralib.typeof(value) == legion_c.legion_index_partition_t then
    return IndexPartition:from_cobj(value)
  end
  if terralib.typeof(value) == legion_c.legion_field_space_t then
    return FieldSpace:from_cobj(value)
  end
  if value_type:isstruct() then
    local new_value = {}
    for _, v in pairs(value_type.entries) do
      new_value[v.field] = convert_cdata(value[v.field])
    end
    return new_value
  else
    return value
  end
end

-- Point and rect types
function Ptr:from_cobj(cobj)
  local ptr = { value = cobj.value }
  ptr.is_ptr = true
  setmetatable(ptr, self)
  return ptr
end

function Ptr:is_null()
  return legion_c.legion_ptr_is_null(self)
end

function Ptr:__tostring()
  return tostring(self.value)
end

function Ptr:__hash()
  return self.value
end

function Pair:new(a, b)
  local pair = { fst = a, snd = b }
  pair.is_pair = true
  setmetatable(pair, self)
  return pair
end

function Pair:__tostring()
  return "(" .. self.fst .. "," .. self.snd .. ")"
end

function Coloring:new()
  local c = { coloring = {}, is_coloring = true }
  setmetatable(c, self)
  self.__index = function(child, color)
    if type(color) ~= "number" then
      return rawget(self, color) or rawget(child, color)
    end
    local coloring = rawget(child, "coloring")
    local coloredPoints = rawget(coloring, color)
    if not coloredPoints then
      coloredPoints = { points = Set:new({}, Ptr),
                        ranges = Set:new({}, Ptr) }
      rawset(coloring, color, coloredPoints)
    end
    return coloredPoints
  end
  return c
end

function Coloring:ensure_color(color)
  return self[color]
end

function Coloring:to_cobj()
  local coloring = legion_c.legion_coloring_create()

  for color, coloredPoints in pairs(self.coloring) do
    legion_c.legion_coloring_ensure_color(coloring, color)
    coloredPoints.points:itr(function(ptr)
      legion_c.legion_coloring_add_point(coloring, color, ptr)
    end)
    coloredPoints.ranges:itr(function(pair)
      legion_c.legion_coloring_add_range(coloring, color, pair.fst, pair.snd)
    end)
  end

  return coloring
end

function Coloring:__tostring()
  local str = ""
  for color, coloredPoints in pairs(self.coloring) do
    if coloredPoints.points.size > 0 then
      str = str .. color .. " => " .. tostring(coloredPoints.points) .. "\n"
    end
    if coloredPoints.ranges.size > 0 then
      str = str .. color .. " => " .. tostring(coloredPoints.ranges) .. "\n"
    end
  end
  return str
end

function Point:new(coords)
  local point = {}
  point.is_point = true
  local x = {}
  for i = 1, #coords do
    x[i - 1] = coords[i]
  end
  point.x = x
  point.dim = #coords
  setmetatable(point, self)
  self.__index = function(child, idx)
    return rawget(self, idx) or
           rawget(rawget(child, "x"), idx) or
           error("Error: invalid access to Point: " .. idx)
  end
  return point
end

function Point:clone()
  local coords = {}
  for i = 1, self.dim do
    coords[i] = self.x[i - 1]
  end
  return Point:new(coords)
end

function Point:from_cobj(cobj, dim)
  local coords = {}
  for i = 1, dim do
    coords[i] = cobj.x[i - 1]
  end
  return Point:new(coords)
end

local point_to_cobj = {}
point_to_cobj[1] = terra(x0 : int)
  var p : legion_c.legion_point_1d_t
  p.x[0] = x0
  return p
end
point_to_cobj[2] = terra(x0 : int, x1 : int)
  var p : legion_c.legion_point_2d_t
  p.x[0] = x0
  p.x[1] = x1
  return p
end
point_to_cobj[3] = terra(x0 : int, x1 : int, x2 : int)
  var p : legion_c.legion_point_3d_t
  p.x[0] = x0
  p.x[1] = x1
  p.x[2] = x2
  return p
end

function Point:to_cobj()
  assert(self.is_point, "Error: not a valid Point object")
  local to_cobj = point_to_cobj[self.dim]
  if self.dim == 1 then
    return to_cobj(self.x[0])
  else
    if self.dim == 2 then
      return to_cobj(self.x[0], self.x[1])
    else -- self.dim == 3
      return to_cobj(self.x[0], self.x[1], self.x[2])
    end
  end
  assert(false,
    "Error: dimension of the point exceeds the maximum dimension")
end

function Point.__eq(a, b)
  if not a.is_point or not b.is_point then return false end
  for i = 0, a.dim - 1 do
    if not (a.x[i] == b.x[i]) then
      return false
    end
  end
  return true
end

function Point.__le(a, b)
  if not a.is_point or not b.is_point then return false end
  for i = 0, a.dim - 1 do
    if a.x[i] > b.x[i] then
      return false
    end
  end
  return true
end

function Point.__add(a, b)
  assert(a.is_point and b.is_point)
  local coords = {}
  for i = 0, a.dim - 1 do
    coords[i + 1] = a.x[i] + b.x[i]
  end
  return Point:new(coords)
end

function Point.__sub(a, b)
  assert(a.is_point and b.is_point)
  local coords = {}
  for i = 0, a.dim - 1 do
    coords[i + 1] = a.x[i] - b.x[i]
  end
  return Point:new(coords)
end

function Point:__tostring()
  if self.dim == 1 then
    return tostring(self.x[0])
  end
  if self.dim == 2 then
    return "(" .. self.x[0] .. "," ..
                  self.x[1] .. ")"
  end
  if self.dim == 3 then
    return "(" .. self.x[0] .. "," ..
                  self.x[1] .. "," ..
                  self.x[2] .. ")"
  end
end

function Rect:new(lo, hi)
  assert(lo.dim == hi.dim,
    "Error: two bounds have the same dimension")
  local rect = {}
  rect.is_rect = true
  rect.lo = lo
  rect.hi = hi
  setmetatable(rect, self)
  return rect
end

function Rect:from_cobj(cobj, dim)
  return Rect:new(Point:from_cobj(cobj.lo, dim),
                  Point:from_cobj(cobj.hi, dim))
end

local rect_to_cobj = {}
for i = 1, legion_c.MAX_POINT_DIM do
  local point_type_name = string.format("legion_point_%dd_t", i)
  local point_type = legion_c[point_type_name]
  local rect_type_name = string.format("legion_rect_%dd_t", i)
  local rect_type = legion_c[rect_type_name]
  local terra convert(lo : point_type, hi : point_type)
    var rect : rect_type
    rect.lo = lo
    rect.hi = hi
    return rect
  end
  rect_to_cobj[i] = convert
end

function Rect:to_cobj()
  assert(self.is_rect, "Error: not a valid Rect object")
  assert(self.lo.dim <= legion_c.MAX_POINT_DIM,
    "Error: the dimension of the rectangle exceeds the maximum dimension")
  return rect_to_cobj[self.lo.dim](self.lo:to_cobj(), self.hi:to_cobj())
end

function Rect.__eq(a, b)
  if not a.is_rect or not b.is_rect then return false end
  return a.lo == b.lo and a.hi == b.hi
end

function Rect:__tostring()
  return "[" .. tostring(self.lo) .. " : " ..
                tostring(self.hi) .. "]"
end

local blockify_to_cobj = {}
for i = 1, legion_c.MAX_POINT_DIM do
  local point_type_name = string.format("legion_point_%dd_t", i)
  local point_type = legion_c[point_type_name]
  local blockify_type_name = string.format("legion_blockify_%dd_t", i)
  local blockify_type = legion_c[blockify_type_name]
  local terra convert(p : point_type)
    var b : blockify_type
    b.block_size = p
    return b
  end
  blockify_to_cobj[i] = convert
end

function Blockify:new(coords)
  local blockify = { block_size = Point:new(coords) }
  setmetatable(blockify, self)
  blockify.is_blockify = true
  return blockify
end

function Blockify:to_cobj()
  assert(self.is_blockify, "Error: invalid Blockify object")
  assert(self.block_size.dim <= legion_c.MAX_POINT_DIM,
    "Error: the dimension of the block size exceeds the maximum dimension")
  return blockify_to_cobj[self.block_size.dim](self.block_size:to_cobj())
end

-- Lua wrapper for Legion::Domain
function Domain:from_cobj(cobj)
  local domain = { cobj = cobj }
  domain.is_domain = true
  setmetatable(domain, self)
  return domain
end

function Domain:from_rect(r)
  if r.lo and r.hi then
    local func_name =
      string.format("legion_domain_from_rect_%dd", r.lo.dim)
    local func = legion_c[func_name]
    local r_cobj = r:to_cobj()
    return self:from_cobj(func(r_cobj))
  end
  assert(false, "Error: not a valid call to Domain:from_rect")
end

function Domain:get_rect()
  local func_name =
    string.format("legion_domain_get_rect_%dd", self.cobj.dim)
  local func = legion_c[func_name]
  return Rect:from_cobj(func(self.cobj), self.cobj.dim)
end

function Domain:__tostring()
  return tostring(self:get_rect())
end

-- Lua wrapper for Legion::DomainColoring
function DomainColoring:new()
  local dc = { coloring = {}, is_domain_coloring = true }
  setmetatable(dc, self)
  self.__index = function(child, idx)
    return rawget(self, idx) or
           rawget(rawget(child, "coloring"), idx) or
           error("Error: invalid access to array: " .. idx)
  end
  self.__newindex = function(child, idx, value)
    rawget(child, "coloring")[idx] = value
  end
  return dc
end

function DomainColoring:to_cobj()
  local coloring = legion_c.legion_domain_coloring_create()
  for color, domain in pairs(self.coloring) do
    legion_c.legion_domain_coloring_color_domain(coloring,
      color, domain.cobj)
  end
  return coloring
end

function DomainColoring:__tostring()
  local str = ""
  for idx, dom in pairs(self.coloring) do
    str = str .. idx .. " => " .. tostring(dom) .. "\n"
  end
  return str
end

-- Lua wrapper for Legion::DomainPoint
function DomainPoint:from_point(p)
  assert(p.is_point, "Error: DomainPoint:from_point takes only a Point")
  local point_data = {}
  for i = 0, p.dim - 1 do
    point_data[i] = p.x[i]
  end
  local dp = { dim = p.dim, point_data = point_data,
               is_domain_point = true }
  setmetatable(dp, self)
  return dp
end

function DomainPoint:from_cobj(cobj)
  local point_data = {}
  for i = 0, cobj.dim - 1 do
    point_data[i] = cobj.point_data[i]
  end
  local dp = { dim = cobj.dim, point_data = point_data,
               is_domain_point = true }
  setmetatable(dp, self)
  return dp
end

local dp_to_cobj = {}
dp_to_cobj[1] = terra(x0 : int)
  var dp : legion_c.legion_domain_point_t
  dp.dim = 1
  dp.point_data[0] = x0
  return dp
end
dp_to_cobj[2] = terra(x0 : int, x1 : int)
  var dp : legion_c.legion_domain_point_t
  dp.dim = 2
  dp.point_data[0] = x0
  dp.point_data[1] = x1
  return dp
end
dp_to_cobj[3] = terra(x0 : int, x1 : int, x2 : int)
  var dp : legion_c.legion_domain_point_t
  dp.dim = 3
  dp.point_data[0] = x0
  dp.point_data[1] = x1
  dp.point_data[2] = x2
  return dp
end

function DomainPoint:to_cobj()
  assert(self.is_domain_point, "Error: not a valid DomainPoint object")
  local convert = dp_to_cobj[self.dim]
  if self.dim == 1 then
    return convert(self.point_data[0])
  end
  if self.dim == 2 then
    return convert(self.point_data[0], self.point_data[1])
  end
  if self.dim == 3 then
    return convert(self.point_data[0], self.point_data[1], self.point_data[2])
  end
  assert(false,
    "Error: dimension of the domain point exceeds the maximum dimension")
end

function DomainPoint:__tostring()
  if self.dim == 1 then
    return tostring(self.point_data[0])
  end
  if self.dim == 2 then
    return "(" .. self.point_data[0] .. "," ..
                  self.point_data[1] .. ")"
  end
  if self.dim == 3 then
    return "(" .. self.point_data[0] .. "," ..
                  self.point_data[1] .. "," ..
                  self.point_data[2] .. ")"
  end
end

-- Lua wrapper for Legion::IndexSpace
function IndexSpace:from_cobj(cobj)
  local is = { id = cobj.id, tid = cobj.tid, is_index_space = true }
  setmetatable(is, self)
  return is
end

function IndexSpace:clone()
  local is = { id = self.id, tid = self.tid, is_index_space = true }
  setmetatable(is, IndexSpace)
  return is
end

-- Lua wrapper for Legion::IndexPartition
function IndexPartition:from_cobj(cobj)
  local ip = { id = cobj.id, tid = cobj.tid, is_index_partition = true }
  setmetatable(ip, self)
  return ip
end

function IndexPartition:clone()
  local ip = { id = self.id, tid = self.tid, is_index_partition = true }
  setmetatable(ip, self)
  return ip
end

-- Lua wrapper for Legion::FieldSpace
function FieldSpace:from_cobj(cobj)
  local is = { id = cobj.id, is_field_space = true }
  setmetatable(is, self)
  return is
end

function FieldSpace:clone()
  local is = { id = self.id, is_field_space = true }
  setmetatable(is, FieldSpace)
  return is
end

-- Lua wrapper for Legion::IndexAllocator
function IndexAllocator:from_cobj(cobj)
  local allocator = { cobj = cobj }
  allocator.is_index_allocator = true
  setmetatable(allocator, self)
  return allocator
end

function IndexAllocator:alloc(num_elements)
  return
    Ptr:from_cobj(
      legion_c.legion_index_allocator_alloc(self.cobj, num_elements))
end

-- Lua wrapper for Legion::FieldAllocator
function FieldAllocator:from_cobj(cobj)
  local allocator = { cobj = cobj }
  allocator.is_field_allocator = true
  setmetatable(allocator, self)
  return allocator
end

function FieldAllocator:allocate_field(size, fid)
  fid = fid or -1
  return legion_c.legion_field_allocator_allocate_field(self.cobj, size, fid)
end

-- Lua wrapper for Legion::LogicalRegion
function LogicalRegion:from_cobj(cobj)
  local lr = { index_space = IndexSpace:from_cobj(cobj.index_space),
               field_space = FieldSpace:from_cobj(cobj.field_space),
               tree_id = cobj.tree_id }
  lr.is_logical_region = true
  setmetatable(lr, self)
  return lr
end

function LogicalRegion:get_index_space()
  return self.index_space
end

function LogicalRegion:get_field_space()
  return self.field_space
end

function LogicalRegion:get_tree_id()
  return self.tree_id
end

function LogicalRegion:__tostring()
  return "LR{" .. self.index_space.id .. "," ..
                  self.field_space.id .. "," ..
                  self.tree_id .. "}"
end

-- Lua wrapper for Legion::LogicalPartition
function LogicalPartition:from_cobj(cobj)
  local lp =
    { index_partition = IndexPartition:from_cobj(cobj.index_partition),
      field_space = FieldSpace:from_cobj(cobj.field_space),
      tree_id = cobj.tree_id }
  lp.is_logical_partition = true
  setmetatable(lp, self)
  return lp
end

function LogicalPartition:to_cobj()
  return { index_partition = self.index_partition,
           field_space = self.field_space,
           tree_id = self.tree_id }
end

function LogicalPartition:get_index_partition()
  return self.index_partition
end

function LogicalPartition:get_field_space()
  return self.field_space
end

function LogicalPartition:get_tree_id()
  return self.tree_id
end

function LogicalPartition:__tostring()
  return "LP{" .. self.index_partition.id .. "," ..
                  self.field_space.id .. "," ..
                  self.tree_id .. "}"
end

-- Lua wrapper for Legion::PhysicalRegion
function PhysicalRegion:from_cobj(cobj)
  local pr = { impl = cobj.impl }
  pr.is_physical_region = true
  setmetatable(pr, self)
  return pr
end

function PhysicalRegion:is_mapped()
  return legion_c.legion_physical_region_is_mapped(self)
end

function PhysicalRegion:wait_until_valid()
  legion_c.legion_physical_region_wait_until_valid(self)
end

function PhysicalRegion:is_valid()
  return legion_c.legion_physical_region_is_valid(self)
end

function PhysicalRegion:get_accessor()
  return RegionAccessor:new(self)
end

function PhysicalRegion:get_field_accessor(fid)
  return RegionAccessor:new(self, fid)
end

-- Lua wrapper for Legion::RegionRequirement
function RegionRequirement:new(arg0, arg1, arg2, arg3,
                               arg4, arg5, arg6)
  assert(arg0.is_logical_region or arg0.is_logical_partition)

  local req = {}
  if arg0.is_logical_region then
    req.region = arg0
  else -- arg0.is_logical_partition
    req.partition = arg0
  end
  if not (type(arg4) == "table") then
    assert(type(arg3) == "table" and arg3.is_logical_region,
      "Error: RegionRequirement should be initialized with a parent region")
    req.privilege = arg1
    req.prop = arg2
    req.parent = arg3
    req.region_tag = arg4 or 0
    req.verified = arg5 or false
    req.handle_type = legion_c.SINGULAR
  else -- arg4.is_logical_region
    assert(arg4.is_logical_region,
      "Error: RegionRequirement should be initialized with a parent region")
    req.projection = arg1
    req.privilege = arg2
    req.prop = arg3
    req.parent = arg4
    req.region_tag = arg5 or 0
    req.verified = arg6 or false
    if arg0.is_logical_region then
      req.handle_type = legion_c.REG_PROJECTION
    else -- arg0.is_logical_partition
      req.handle_type = legion_c.PART_PROJECTION
    end
  end
  req.privilege_fields = Array:new {}
  req.instance_fields = Array:new {}
  req.is_region_requirement = true
  setmetatable(req, self)
  return req
end

function RegionRequirement:add_field(fid, inst)
  inst = inst or true
  if inst then self.instance_fields:insert(fid) end
  self.privilege_fields:insert(fid)
end

function RegionRequirement:set_reduction()
  self.is_reduction = true
  self.redop = self.privilege
  self.privilege = legion_c.REDUCE
end

-- TODO: RegionRequirement:from_cobj does not initialize several fields properly
local attrs_to_get =
  Array:new
  { "privilege", "prop", "redop", "tag", "handle_type", "projection",
    "virtual_map", "early_map", "enable_WAR_optimization",
    "reduction_list", "make_persistent", "blocking_factor",
    "max_blocking_factor" }
local attrs_to_refresh =
  Array:new
  { "virtual_map", "early_map", "enable_WAR_optimization",
    "reduction_list", "make_persistent", "blocking_factor" }
local function get_req_attr(req, attr)
  return legion_c["legion_region_requirement_get_" .. attr](req);
end

function RegionRequirement:refresh_fields()
  attrs_to_refresh:itr(function(attr)
    self[attr] = get_req_attr(self.cobj, attr)
  end)
end

function RegionRequirement:from_cobj(req_cobj)
  local req =
    { cobj = req_cobj,
      region = LogicalRegion:from_cobj(get_req_attr(req_cobj, "region")),
      privilege_fields = Array:new {},
      instance_fields = Array:new {},
      parent = LogicalRegion:from_cobj(get_req_attr(req_cobj, "parent")),
      target_ranking = Array:new{}
    }
  attrs_to_get:itr(function(attr)
    req[attr] = get_req_attr(req_cobj, attr)
  end)
  local num_privilege_fields =
    get_req_attr(req_cobj, "privilege_fields_size")
  for j = 0, num_privilege_fields - 1 do
    req.privilege_fields:insert(
      legion_c.legion_region_requirement_get_privilege_field(
        req_cobj, j))
  end
  local num_instance_fields =
    get_req_attr(req_cobj, "instance_fields_size")
  for j = 0, num_instance_fields - 1 do
    req.instance_fields:insert(
      legion_c.legion_region_requirement_get_instance_field(
        req_cobj, j))
  end
  setmetatable(req, self)
  return req
end

-- Lua wrapper for Legion::ArgumentMap
function ArgumentMap:new()
  local arg_map = {}
  arg_map.is_argument_map = true
  setmetatable(arg_map, self)
  local cobj =
    ffi.gc(legion_c.legion_argument_map_create(),
           legion_c.legion_argument_map_destroy)
  arg_map.cobj = cobj
  return arg_map
end

function ArgumentMap:set_point(dp, arg, replace)
  replace = replace or true
  assert(dp.is_domain_point,
    "Error: ArgumentMap:set_point takes only a DomainPoint")
  dp = dp:to_cobj()
  legion_c.legion_argument_map_set_point(self.cobj, dp, arg, replace)
end

-- Lua wrapper for Legion::Future
function Future:from_cobj(cobj)
  local future = { cobj = ffi.gc(cobj, legion_c.legion_future_destroy) }
  future.is_future = true
  setmetatable(future, Future)
  return future
end

function Future:get_result(value_type)
  assert(self.is_future)
  local terra call_capi(future : legion_c.legion_future_t)
    var result = legion_c.legion_future_get_result(future)
    var return_value : value_type =
      @[&value_type](result.value)
    legion_c.legion_task_result_destroy(result)
    return return_value
  end
  return call_capi(self.cobj)
end

function Future:get_void_result()
  assert(self.is_future)
  legion_c.legion_future_get_void_result(self.cobj)
end

-- Lua wrapper for Legion::FutureMap
function FutureMap:from_cobj(cobj)
  local fm = { cobj = ffi.gc(cobj, legion_c.legion_future_map_destroy) }
  fm.is_future_map = true
  setmetatable(fm, FutureMap)
  return fm
end

function FutureMap:get_future(dp)
  assert(self.is_future_map)
  return Future:from_cobj(legion_c.legion_future_map_get_future(self.cobj, dp))
end

function FutureMap:get_result(value_type, dp)
  assert(self.is_future_map)
  local future = self:get_future(dp)
  local result = future:get_result(value_type)
  return result
end

function FutureMap:wait_all_results()
  assert(self.is_future_map)
  legion_c.legion_future_map_wait_all_results(self.cobj)
end

-- Lua wrapper for Legion::Task
function Task:from_cobj(cobj)
  local task = { impl = cobj.impl }
  task.is_task = true
  task.task_id = legion_c.legion_task_get_task_id(cobj)
  task.target_proc =
    Processor:from_cobj(legion_c.legion_task_get_target_proc(cobj))

  -- initialize RegionRequirement objects
  task.regions = Array:new {}
  local num_regions = legion_c.legion_task_get_regions_size(cobj)
  for i = 0, num_regions - 1 do
    local req =
      RegionRequirement:from_cobj(
        legion_c.legion_task_get_region(cobj, i))
    task.regions:insert(req)
  end

  -- initialize Future objects
  task.futures = Array:new {}
  local num_futures = legion_c.legion_task_get_futures_size(cobj)
  for i = 0, num_futures - 1 do
    local future_cobj =
      legion_c.legion_task_get_future(cobj, i)
    task.futures:insert(Future:from_cobj(future_cobj))
  end

  -- initialize the index point
  local index_point_cobj = legion_c.legion_task_get_index_point(cobj)
  if index_point_cobj.dim <= 1 and
    index_point_cobj.dim <= legion_c.MAX_POINT_DIM then
    task.index_point = DomainPoint:from_cobj(index_point_cobj)
  end
  setmetatable(task, self)
  return task
end

function Task:get_args(arg_type)
  local terra call_capi(task : legion_c.legion_task_t)
    var args: arg_type
    args =
      @[&arg_type](legion_c.legion_task_get_args(task))
    return args
  end
  return convert_cdata(call_capi(self))
end

function Task:get_local_args(arg_type)
  local terra call_capi(task : legion_c.legion_task_t)
    var args: arg_type
    args =
      @[&arg_type](legion_c.legion_task_get_local_args(task))
    return args
  end
  return convert_cdata(call_capi(self))
end

-- Lua wrapper for Legion::Inline
function Inline:from_cobj(cobj)
  local inline = { impl = cobj.impl }
  inline.is_inline = true
  inline.requirement =
    RegionRequirement:from_cobj(
      legion_c.legion_inline_get_requirement(cobj))
  setmetatable(inline, self)
  return inline
end

-- Lua wrapper for Legion::TaskArgument
function TaskArgument:new(arg, arg_type)
  assert(arg and arg_type)
  local terra alloc(arg : arg_type)
    var args: &arg_type
    args = [&arg_type](std.malloc(sizeof(arg_type)))
    @args = [arg]
    return args
  end
  local ptr = ffi.gc(alloc(arg), std.free)
  local tbl = { args = ptr, arglen = sizeof(arg_type) }
  setmetatable(tbl, self)
  return tbl
end

-- Lua wrapper for Legion::TaskLauncher
function TaskLauncher:new(task_id, task_arg, pred, id, tag)
  local launcher = {}
  launcher.task_id = task_id
  launcher.task_arg = task_arg or { args = terra_null(), arglen = 0 }
  launcher.pred = pred or Predicate.TRUE_PRED
  launcher.id = id or 0
  launcher.tag = tag or 0
  launcher.is_task_launcher = true
  launcher.region_requirements = Array:new {}
  launcher.futures = Array:new {}
  setmetatable(launcher, self)
  return launcher
end

function TaskLauncher:add_field(idx, fid, inst)
  self.region_requirements[idx]:add_field(fid, inst)
end

function TaskLauncher:add_future(f)
  self.futures:insert(f)
end

function TaskLauncher:add_region_requirement(req)
  self.region_requirements:insert(req)
end

-- Lua wrapper for Legion::IndexLauncher
function IndexLauncher:new(task_id, domain, global_arg, map,
                           pred, must, id, tag)
  local launcher = {}
  launcher.task_id = task_id
  launcher.launch_domain = domain
  launcher.global_arg = global_arg or { args = terra_null(), arglen = 0 }
  launcher.argument_map = map
  launcher.pred = pred or Predicate.TRUE_PRED
  launcher.must = must or false
  launcher.id = id or 0
  launcher.tag = tag or 0
  launcher.is_index_launcher = true
  launcher.region_requirements = Array:new {}
  setmetatable(launcher, self)
  return launcher
end

function IndexLauncher:add_field(idx, fid, inst)
  self.region_requirements[idx]:add_field(fid, inst)
end

function IndexLauncher:add_region_requirement(req)
  self.region_requirements:insert(req)
end

-- Lua wrapper for Legion::InlineLauncher
function InlineLauncher:new(req, id, launcher_tag)
  assert(req.is_region_requirement,
    "Error: InlineLauncher needs a RegionRequirement object to create")
  local launcher = {}
  launcher.is_inline_launcher = true
  launcher.requirement = req
  launcher.id = id or 0
  launcher.launcher_tag = launcher_tag or 0
  setmetatable(launcher, self)
  return launcher
end

function InlineLauncher:add_field(fid, inst)
  self.requirement:add_field(fid, inst)
end

-- Lua wrapper for Legion::Runtime

function legion:create_index_space(ctx, arg)
  assert(ctx.is_context,
    "Error: create_index_space takes a Context object as the first argument")
  if type(arg) == "number" then
    local is_cobj = legion_c.legion_index_space_create(self, ctx, arg)
    return IndexSpace:from_cobj(is_cobj)
  end
  if arg.is_domain then
    local is_cobj = legion_c.legion_index_space_create_domain(self, ctx, arg.cobj)
    return IndexSpace:from_cobj(is_cobj)
  end
  assert(false,
    "Error: unsupported argument for create_index_space")
end

function legion:get_index_space_domain(ctx, handle)
  assert(ctx.is_context,
    "Error: get_index_space_domain takes a Context object as the first argument")
  assert(handle.is_index_space,
    "Error: get_index_space_domain takes an index space as the second argument")
  return
    Domain:from_cobj(legion_c.legion_index_space_get_domain(self, ctx, handle))
end

function legion:destroy_index_space(ctx, handle)
  assert(ctx.is_context,
    "Error: destroy_index_space takes a Context object as the first argument")
  assert(handle.is_index_space,
    "Error: destroy_index_space takes an index space as the second argument")
  legion_c.legion_index_space_destroy(self, ctx, handle)
end

function legion:create_index_partition(ctx, parent, arg1, arg2, arg3, arg4)
  assert(ctx.is_context,
    "Error: create_index_partition takes a Context object as the first argument")
  assert(parent.is_index_space,
    "Error: create_index_partition takes an index space as the second argument")
  if arg1.is_domain then
    local domain = arg1
    if arg2.is_domain_coloring then
      local domain_coloring = arg2
      local disjoint = arg3
      local part_color = arg4 or -1
      local dc_cobj = domain_coloring:to_cobj()
      local ip = IndexPartition:from_cobj(
        legion_c.legion_index_partition_create_domain_coloring(self,
          ctx, parent, domain.cobj, dc_cobj,
          disjoint, part_color))
      legion_c.legion_domain_coloring_destroy(dc_cobj)
      return ip
    end
  end
  if arg1.is_blockify then
    local blockify = arg1
    local part_color = arg2 or -1
    local dim = blockify.block_size.dim
    local func_name =
      string.format("legion_index_partition_create_blockify_%dd", dim)
    local func = legion_c[func_name]
    local ip = IndexPartition:from_cobj(
      func(self, ctx, parent, blockify:to_cobj(), part_color))
    return ip
  end
  if arg1.is_coloring then
    local coloring = arg1
    local disjoint = arg2 or true
    local part_color = arg3 or -1
    local coloring_cobj = coloring:to_cobj()
    local ip = IndexPartition:from_cobj(
      legion_c.legion_index_partition_create_coloring(self,
        ctx, parent, coloring_cobj, disjoint, part_color))
    legion_c.legion_coloring_destroy(coloring_cobj)
    return ip
  end
  assert(false)
end

function legion:get_index_subspace(ctx, part, color)
  assert(ctx.is_context,
    "Error: get_index_subspace takes a Context object as the first argument")
  assert(part.is_index_partition,
    "Error: get_index_subspace takes an index partition as the second argument")
  return IndexSpace:from_cobj(
    legion_c.legion_index_partition_get_index_subspace(self, ctx,
                                                       part.id, color))
end

function legion:create_field_space(ctx)
  assert(ctx.is_context,
    "Error: create_field_space takes a Context object as the first argument")
  local fs_cobj = legion_c.legion_field_space_create(self, ctx)
  return FieldSpace:from_cobj(fs_cobj)
end

function legion:destroy_field_space(ctx, handle)
  assert(ctx.is_context,
    "Error: destroy_field_space takes a Context object as the first argument")
  assert(handle.is_field_space,
    "Error: destroy_index_space takes a field space as the second argument")
  return legion_c.legion_field_space_destroy(self, ctx, handle)
end

function legion:create_logical_region(ctx, is, fs)
  assert(ctx.is_context,
    "Error: create_logical_region takes a Context object as the first argument")
  assert(is.is_index_space,
    "Error: create_logical_region takes an index space as the second argument")
  assert(fs.is_field_space,
    "Error: create_logical_region takes a field space as the third argument")
  local cobj = legion_c.legion_logical_region_create(self, ctx, is, fs)
  return LogicalRegion:from_cobj(cobj)
end

function legion:get_logical_subregion_by_color(ctx, lp, color)
  assert(ctx.is_context,
    "Error: get_logical_subregion_by_color takes a Context object as the first argument")
  assert(lp.is_logical_partition,
    "Error: get_logical_subregion_by_color takes a LogicalPartition object as the second argument")
  local cobj =
    legion_c.legion_logical_partition_get_logical_subregion_by_color(self, ctx, lp:to_cobj(), color)
  return LogicalRegion:from_cobj(cobj)
end

function legion:destroy_logical_region(ctx, lr)
  assert(ctx.is_context,
    "Error: destroy_logical_region takes a Context object as the first argument")
  assert(lr.is_logical_region,
    "Error: destroy_logical_region takes a LogicalRegion object as the second argument")
  legion_c.legion_logical_region_destroy(self, ctx, lr)
end

function legion:get_logical_partition(ctx, parent, ip)
  assert(ctx.is_context,
    "Error: get_logical_partition takes a Context object as the first argument")
  assert(parent.is_logical_region,
    "Error: get_logical_partition takes a LogicalRegion object as the second argument")
  assert(ip.is_index_partition,
    "Error: get_logical_partition takes an IndexPartition object as the third argument")
  local cobj = legion_c.legion_logical_partition_create(self, ctx, parent, ip)
  return LogicalPartition:from_cobj(cobj)
end

function legion:get_logical_partition_by_tree(ctx, ip, fs, tree_id)
  assert(ctx.is_context,
    "Error: get_logical_partition_by_tree takes a Context object as the first argument")
  assert(ip.is_index_partition,
    "Error: get_logical_partition_by_tree takes an IndexPartition object as the second argument")
  assert(fs.is_field_space,
    "Error: create_logical_region takes a field space as the third argument")
  local cobj = legion_c.legion_logical_partition_create_by_tree(self, ctx, ip.id, fs, tree_id)
  return LogicalPartition:from_cobj(cobj)
end

function legion:destroy_logical_partition(ctx, lp)
  assert(ctx.is_context,
    "Error: destroy_logical_partition takes a Context object as the first argument")
  assert(lp.is_logical_partition,
    "Error: destroy_logical_partition takes a LogicalRegion object as the second argument")
  legion_c.legion_logical_partition_destroy(self, ctx, lp:to_cobj())
end

function legion:create_index_allocator(ctx, handle)
  assert(ctx.is_context,
    "Error: create_index_allocator takes a Context object as the first argument")
  assert(handle.is_index_space,
    "Error: create_index_allocator takes an index space as the second argument")
  return
    IndexAllocator:from_cobj(
      ffi.gc(
        legion_c.legion_index_allocator_create(self, ctx, handle),
        legion_c.legion_index_allocator_destroy))
end

function legion:create_field_allocator(ctx, handle)
  assert(ctx.is_context,
    "Error: create_field_allocator takes a Context object as the first argument")
  assert(handle.is_field_space,
    "Error: create_field_allocator takes an field space as the second argument")
  return
    FieldAllocator:from_cobj(
      ffi.gc(
        legion_c.legion_field_allocator_create(self, ctx, handle),
        legion_c.legion_field_allocator_destroy))
end

function legion:execute_task(ctx, launcher)
  assert(ctx.is_context,
    "Error: execute_task takes a Context object as the first argument")
  assert(launcher.is_task_launcher,
    "Error: execute_task takes a TaskLauncher object as the second argument")

  local launcher_cobj =
    legion_c.legion_task_launcher_create(
      launcher.task_id, launcher.task_arg, launcher.pred, launcher.id, launcher.tag)
  for i = 0, launcher.region_requirements.size - 1 do
    local req = launcher.region_requirements[i]
    if req.handle_type == legion_c.SINGULAR then
      assert(req.region and req.region.is_logical_region)
      legion_c.legion_task_launcher_add_region_requirement_logical_region(
        launcher_cobj,
        req.region, req.privilege, req.prop, req.parent,
        req.region_tag, req.verified)
    end
    local set = {}
    for j = 0, req.instance_fields.size - 1 do
      local fid = req.instance_fields[j]
      legion_c.legion_task_launcher_add_field(launcher_cobj,
        i, fid, true)
      set[fid] = true
    end
    for j = 0, req.privilege_fields.size - 1 do
      local fid = req.privilege_fields[j]
      if not set[fid] then
        legion_c.legion_task_launcher_add_field(launcher_cobj,
          i, fid, false)
      end
    end
  end
  for i = 0, launcher.futures.size - 1 do
    local f = launcher.futures[i]
    legion_c.legion_task_launcher_add_future(launcher_cobj, f.cobj)
  end
  local cobj = legion_c.legion_task_launcher_execute(self, ctx, launcher_cobj)
  legion_c.legion_task_launcher_destroy(launcher_cobj)
  return Future:from_cobj(cobj)
end

function legion:execute_index_space(ctx, launcher)
  assert(ctx.is_context,
    "Error: execute_index_space takes a Context object as the first argument")
  assert(launcher.is_index_launcher,
    "Error: execute_index_space takes a IndexLauncher object as the second argument")

  local launcher_cobj =
    legion_c.legion_index_launcher_create(
      launcher.task_id, launcher.launch_domain.cobj, launcher.global_arg,
      launcher.argument_map.cobj, launcher.pred, launcher.must,
      launcher.id, launcher.tag)
  for i = 0, launcher.region_requirements.size - 1 do
    local req = launcher.region_requirements[i]
    if req.handle_type == legion_c.PART_PROJECTION then
      assert(req.partition and req.partition.is_logical_partition)
      if not req.is_reduction then
        legion_c.legion_index_launcher_add_region_requirement_logical_partition(
          launcher_cobj,
          req.partition:to_cobj(), req.projection, req.privilege, req.prop,
          req.parent, req.region_tag, req.verified)
      else
        legion_c
        .legion_index_launcher_add_region_requirement_logical_partition_reduction(
          launcher_cobj,
          req.partition:to_cobj(), req.projection, req.redop, req.prop,
          req.parent, req.region_tag, req.verified)
      end
    end
    if req.handle_type == legion_c.REG_PROJECTION then
      if not req.is_reduction then
        legion_c.legion_index_launcher_add_region_requirement_logical_region(
          launcher_cobj,
          req.region, req.projection, req.privilege, req.prop,
          req.parent, req.region_tag, req.verified)
      else
        legion_c
        .legion_index_launcher_add_region_requirement_logical_region_reduction(
          launcher_cobj,
          req.region, req.projection, req.redop, req.prop,
          req.parent, req.region_tag, req.verified)
      end
    end
    local set = {}
    for j = 0, req.instance_fields.size - 1 do
      local fid = req.instance_fields[j]
      legion_c.legion_index_launcher_add_field(launcher_cobj,
        i, fid, true)
      set[fid] = true
    end
    for j = 0, req.privilege_fields.size - 1 do
      local fid = req.privilege_fields[j]
      if not set[fid] then
        legion_c.legion_index_launcher_add_field(launcher_cobj,
          i, fid, false)
      end
    end
  end

  local cobj = legion_c.legion_index_launcher_execute(self, ctx, launcher_cobj)
  legion_c.legion_index_launcher_destroy(launcher_cobj)
  return FutureMap:from_cobj(cobj)
end

function legion:map_region(ctx, arg0, arg1, arg2)
  assert(ctx.is_context,
    "Error: map_region takes a Context object as the first argument")
  local req, id, tag
  if arg0.is_inline_launcher then
    req = arg0.requirement
    id = arg0.id
    tag = arg0.launcher_tag
  else
    if arg0.is_region_requirement then
      req = arg0
      id = arg1 or 0
      tag = arg2 or 0
    else
      assert(false,
        "Error: map_region takes either a InlineLauncher object or " ..
        "RegionRequirement object as the second argument")
    end
  end
  local launcher =
    legion_c.legion_inline_launcher_create_logical_region(
      req.region, req.privilege, req.prop, req.parent,
      req.region_tag, req.verified, id, tag)
  local set = {}
  for j = 0, req.instance_fields.size - 1 do
    local fid = req.instance_fields[j]
    legion_c.legion_inline_launcher_add_field(launcher,
      fid, true)
    set[fid] = true
  end
  for j = 0, req.privilege_fields.size - 1 do
    local fid = req.privilege_fields[j]
    if not set[fid] then
      legion_c.legion_inline_launcher_add_field(launcher,
        fid, false)
    end
  end

  local cobj = legion_c.legion_inline_launcher_execute(self, ctx, launcher)
  legion_c.legion_inline_launcher_destroy(launcher)
  return PhysicalRegion:from_cobj(cobj)
end

function legion:unmap_region(ctx, pr)
  assert(ctx.is_context,
    "Error: unmap_region takes a Context object as the first argument")
  assert(pr.is_physical_region,
    "Error: unmap_region takes a PhysicalRegion object as the second argument")
  legion_c.legion_runtime_unmap_region(self, ctx, pr)
  legion_c.legion_physical_region_destroy(pr)
end

function legion:get_input_args()
  local args = legion_c.legion_runtime_get_input_args()
  local converted = {}
  for i = 0, args.argc - 1 do
      converted[i] = ffi.string(args.argv[i])
    end
  return converted
end

function legion:register_lua_task(return_type, task_name, task_id, proc, single,
                                  index, vid, opt)
  assert(return_type.name,
    "the first argument of register_lua_task should be a return type of the task")
  vid = vid or -1
  opt = opt or { leaf = false, inner = false, idempotent = false }
  local ptr =
    legion_terra.lua_task_wrapper:getdefinitions()[1]:getpointer()
  -- hack: pass the top-level script name to the next interpreter launched
  local qualified_task_name =
    legion.script_name .. "/" .. task_name .. ":" .. return_type.name
  legion_c.legion_runtime_register_task(task_id, proc, single, index,
                                        vid, opt, qualified_task_name, ptr)
end

function legion:register_lua_task_void(task_name, task_id, proc, single, index,
                                      vid, opt)
  vid = vid or -1
  opt = opt or { leaf = false, inner = false, idempotent = false }
  local ptr =
    legion_terra.lua_task_wrapper_void:getdefinitions()[1]:getpointer()
  -- hack: pass the top-level script name to the next interpreter launched
  local qualified_task_name = legion.script_name .. "/" .. task_name
  legion_c.legion_runtime_register_task_void(task_id, proc, single, index,
                                             vid, opt, qualified_task_name, ptr)
end

function legion:register_reduction_op(op_type, elem_type, redop)
  ReductionOp:register_reduction_op(op_type, elem_type, redop)
end

function legion:set_registration_callback(func_name)
  local qualified_callback_name = legion.script_name .. "/" .. func_name
  legion_terra.set_lua_registration_callback_name(qualified_callback_name)
  local ptr =
    legion_terra.lua_registration_callback_wrapper:getdefinitions()[1]
                                                  :getpointer()
  legion_c.legion_runtime_set_registration_callback(ptr)
end

function legion:create_mapper(mapper_name, machine, runtime, local_proc)
  local qualified_mapper_name = legion.script_name .. "/" .. mapper_name
  return legion_terra.create_mapper(qualified_mapper_name,
                                    machine, runtime, local_proc)
end

function legion:replace_default_mapper(mapper, local_proc)
  legion_c.legion_runtime_replace_default_mapper(self, mapper, local_proc)
end

function legion:set_top_level_task_id(task_id)
  legion_c.legion_runtime_set_top_level_task_id(task_id)
end

function legion:start(args)
  local argc = #args + 1
  local argv = terralib.newsymbol((&int8)[#args], "argv")
  local argv_setup = terralib.newlist({quote var [argv] end})
  for i, arg in pairs(args) do
    if i >= 0 then
      argv_setup:insert(quote
        [argv][ [i-1] ] = [arg]
      end)
    end
  end

  local terra main()
    [argv_setup]
    var res = legion_c.legion_runtime_start(argc, argv, true)
    legion_c.legion_runtime_wait_for_shutdown()
  end
  main()
end

function legion:get_current_time_in_micros()
  return tonumber(legion_c.legion_get_current_time_in_micros())
end

-- wrapper function for registration callbacks in Lua
function lua_registration_callback_wrapper_in_lua(script_name, func_name,
                                                  machine, runtime,
                                                  local_procs_, num_local_procs)
  legion:set_top_level_script(script_name)
  local local_procs = Array:new{}
  for i = 0, num_local_procs - 1 do
    local_procs:insert(coerce_processor(local_procs_, i))
  end
  setmetatable(runtime, legion)
  _G[func_name](machine, runtime, local_procs)
end

-- wrapper functions for legion tasks in Lua
function lua_task_wrapper_in_lua(script_name, return_type_name,
                                 task_name, task, regions, ctx, runtime)
  legion:set_top_level_script(script_name)
  task = Task:from_cobj(task)
  local regions_lua = Array:new {}
  for i = 1, #regions do
    regions_lua:insert(PhysicalRegion:from_cobj(regions[i]))
  end
  ctx.is_context = true
  runtime.is_runtime = true
  setmetatable(runtime, legion)
  local result_from_task = _G[task_name](task, regions_lua, ctx, runtime)
  local return_type = _G[return_type_name]

  local terra alloc_task_result(result_from_task : return_type)
    var task_result : &legion_c.legion_task_result_t =
      [&legion_c.legion_task_result_t](
        std.malloc(sizeof(legion_c.legion_task_result_t)))
    task_result.value_size = sizeof(return_type)
    task_result.value = std.malloc(sizeof(return_type))
    @[&return_type](task_result.value) = result_from_task
    return task_result
  end

  return alloc_task_result(result_from_task)
end

function lua_void_task_wrapper_in_lua(script_name, task_name,
                                 task, regions, ctx, runtime)
  legion:set_top_level_script(script_name)
  task = Task:from_cobj(task)
  local regions_lua = Array:new {}
  for i = 1, #regions do
    regions_lua:insert(PhysicalRegion:from_cobj(regions[i]))
  end
  ctx.is_context = true
  runtime.is_runtime = true
  setmetatable(runtime, legion)
  _G[task_name](task, regions_lua, ctx, runtime)
end

-- Lua equivalent of LegionRuntime::Arrays::GenericPointInRectIterator
function GenericPointInRectIterator:new(r)
  local pir = { r = r, p = r.lo:clone(),
                any_left = (r.lo <= r.hi), dim = r.lo.dim }
  setmetatable(pir, self)
  return pir
end

function GenericPointInRectIterator:has_next()
  return self.any_left
end

function GenericPointInRectIterator:next()
  for i = 0, self.dim - 1 do
    self.p.x[i] = self.p.x[i] + 1
    if self.p.x[i] <= self.r.hi.x[i] then
      return
    end
    self.p.x[i] = self.r.lo.x[i]
  end
  self.any_left = false
end

-- Lua wrapper of Legion::IndexIterator
function IndexIterator:new(arg)
  local is
  if arg.is_logical_region then
    is = arg:get_index_space()
  else
    if arg.is_index_space then
      is = arg
    else
      assert(false,
        "Error: IndexIterator takes either a logical region or " ..
        "an index space to create")
    end
  end

  local itr = { cobj = ffi.gc(legion_c.legion_index_iterator_create(is),
                              legion_c.legion_index_iterator_destroy),
                is_index_iterator = true }
  setmetatable(itr, self)
  return itr
end

function IndexIterator:has_next()
  return legion_c.legion_index_iterator_has_next(self.cobj)
end

function IndexIterator:next()
  return Ptr:from_cobj(
    legion_c.legion_index_iterator_next(self.cobj))
end

-- Lua wrapper for LegionRuntime::Accessor::RegionAccessor
function RegionAccessor:new(pr, fid)
  assert(pr.is_physical_region)

  local accessor = {}
  accessor.is_region_accessor = true
  local cobj
  if fid then
    cobj =
      ffi.gc(
        legion_c.legion_physical_region_get_field_accessor_generic(pr, fid),
        legion_c.legion_accessor_generic_destroy)
  else
    cobj =
      ffi.gc(
        legion_c.legion_physical_region_get_accessor_generic(pr),
        legion_c.legion_accessor_generic_destroy)
  end
  accessor.cobj = cobj
  accessor.fid = fid
  setmetatable(accessor, self)
  return accessor
end

function RegionAccessor:typeify(elem_type)
  self.elem_type = elem_type
  assert(elem_type, "Error: a correct type should be given")
  local terra read(accessor : legion_c.legion_accessor_generic_t,
                   ptr : legion_c.legion_ptr_t) : elem_type
    var v : elem_type
    var size = sizeof(elem_type)
    legion_c.legion_accessor_generic_read(accessor, ptr, &v, size)
    return v
  end
  local terra write(accessor : legion_c.legion_accessor_generic_t,
                    ptr : legion_c.legion_ptr_t,
                    v : elem_type)
    var size = sizeof(elem_type)
    legion_c.legion_accessor_generic_write(accessor, ptr, &v, size)
  end
  local terra read_dp(accessor : legion_c.legion_accessor_generic_t,
                      dp : legion_c.legion_domain_point_t) : elem_type
    var v : elem_type
    var size = sizeof(elem_type)
    legion_c.legion_accessor_generic_read_domain_point(accessor, dp, &v, size)
    return v
  end
  local terra write_dp(accessor : legion_c.legion_accessor_generic_t,
                    dp : legion_c.legion_domain_point_t,
                    v : elem_type)
    var size = sizeof(elem_type)
    legion_c.legion_accessor_generic_write_domain_point(accessor, dp, &v, size)
  end
  self.read = function(self, p)
    assert(self.is_region_accessor,
      "Error: the self object is not a RegionAccessor object. " ..
      "Did you forget to write ':read' to call this read method?")
    if p.is_domain_point then
      return read_dp(self.cobj, p:to_cobj())
    else if p.is_ptr then
           local v = read(self.cobj, p)
           return v
         end
         assert(false)
    end
  end
  if elem_type == legion_c.legion_ptr_t then
    self.read_cobj = self.read
    self.read = function(self, p)
      return Ptr:from_cobj(self:read_cobj(p))
    end
  end
  self.write = function(self, p, v)
    assert(self.is_region_accessor,
      "Error: the self object is not a RegionAccessor object. " ..
      "Did you forget to write ':write' to call this write method?")
    if p.is_domain_point then
      write_dp(self.cobj, p:to_cobj(), v)
    else if p.is_ptr then
           return write(self.cobj, p, v)
         end
         assert(false)
    end
  end
  self.reduce = function(self, op_type, p, v)
    assert(self.is_region_accessor,
      "Error: the self object is not a RegionAccessor object. " ..
      "Did you forget to write ':reduce' to call this reduce method?")
    assert(p.is_ptr)
    if p.is_ptr then
      local reduce = ReductionOp:get_reduce_func(op_type, self.elem_type)
      reduce(self.cobj, p, v)
    else
      assert(false)
    end
  end
  return self
end

function RegionAccessor:convert(op_type)
  local safe_reduce =
    ReductionOp:get_safe_reduce_func(op_type, self.elem_type)
  self.reduce_cobj = safe_reduce
  self.reduce = function(self, p, v)
    assert(p.is_ptr)
    self.reduce_cobj(self.cobj, p, v)
  end
  return self
end

function TaskConfigOptions:new(leaf, inner, idempotent)
  local opt = { leaf = leaf or false,
                inner = inner or false,
                idempotent = idempotent or false }
  setmetatable(opt, self)
  return opt
end

-- Lua wrapper of Legion::Machine
function Machine:from_cobj(cobj)
  local machine = { cobj = cobj }
  setmetatable(machine, self)
  return machine
end

function Machine:get_all_processors()
  local all_processors = Array:new {}
  local all_processors_size =
    legion_c.legion_machine_get_all_processors_size(self.cobj)
  local all_processors_ =
    gcmalloc(legion_c.legion_processor_t, all_processors_size)
  legion_c.legion_machine_get_all_processors(self.cobj, all_processors_,
                                             all_processors_size)
  for i = 0, all_processors_size - 1 do
    all_processors:insert(Processor:from_cobj(all_processors_[i]))
  end
  return all_processors
end

-- Lua wrapper of Legion::Processor
function Processor:from_cobj(cobj)
  local proc = { id = cobj.id }
  setmetatable(proc, self)
  return proc
end

function Processor:kind()
  return legion_c.legion_processor_kind(self)
end

function Processor:__tostring()
  return "Proc " .. self.id
end

-- Lua wrapper for Legion::MappingUtilities::MachineQueryInterface
function MachineQueryInterface:new(machine)
  local cobj =
    ffi.gc(legion_c.legion_machine_query_interface_create(machine.cobj),
           legion_c.legion_machine_query_interface_destroy)
  local interface = { cobj = cobj }
  setmetatable(interface, self)
  return interface
end

function MachineQueryInterface:find_memory_kind(proc, kind)
  return
    legion_c.legion_machine_query_interface_find_memory_kind(self.cobj,
                                                             proc, kind)
end

-- Lua wrapper for std::vector<Legion::DomainSplit>
function DomainSplitVector:from_cobj(cobj)
  local vec = { cobj = cobj }
  setmetatable(vec, self)
  return vec
end

function DomainSplitVector:insert(split)
  assert(split.is_domain_split,
    "Error: insert only takes a DomainSplit object")
  legion_terra.vector_legion_domain_split_push_back(self.cobj, split:to_cobj())
end

function DomainSplitVector:size()
  return legion_terra.vector_legion_domain_split_size(self.cobj)
end

function DomainSplitVector:get(idx)
  return
    DomainSplit:from_cobj(
      legion_terra.vector_legion_domain_split_get(self.cobj, idx))
end

function DomainSplitVector:__tostring()
  local str = ""
  local size = self:size()
  for i = 0, size - 1 do
    local split = self.get(i)
    str = str .. tostring(split) .. "\n"
  end
  return str
end
