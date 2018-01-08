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

require("legionlib")

local ffi = require("ffi")

function LegionLib:register_terra_task(id, processor, is_single, is_index)
   BindingLibInC.register_terra_task(id, processor, is_single, is_index)
end

local stdlib = terralib.includec("stdlib.h")
local str = terralib.includec("string.h")
local lg = terralib.includec("binding.h")
local lgf = terralib.includec("binding_functions.h")

function table_converter(struct_type, is_global)
   if is_global
   then
      return
         terra(terra_fn: &uint64, value: struct_type): &uint8
            var buffer_size = sizeof(struct_type) + sizeof(uint64)
            var buffer: &uint8 = [&uint8](stdlib.malloc(buffer_size))
            @([&&uint64](buffer)) = terra_fn
            str.memcpy(buffer + sizeof(uint64),
                       &value,
                       sizeof(struct_type))
            return buffer
         end
   else
      return
         terra(value: struct_type): &uint8
            var buffer_size = sizeof(struct_type)
            var buffer: &uint8 = [&uint8](stdlib.malloc(buffer_size))
            str.memcpy(buffer, &value, sizeof(struct_type))
            return buffer
         end
   end
end

function test_fun(struct_type)
   return
      terra(ptr: &uint8)
         var value: struct_type = @[&struct_type](ptr + sizeof(uint64))
         print(@([&&uint64](ptr)))
         return value
      end
end

function LegionLib:make_terra_task_argument(arg, arg_size)
   if not self.__task_args then self.__task_args = {} end
   local task_arg =
      BindingLibInC.make_terra_task_argument(tonumber(ffi.cast("intptr_t", arg)),
                                             arg_size)
   self.__task_args[#self.__task_args + 1] = task_arg
   return task_arg
end

TerraTaskLauncher = {}

function TerraTaskLauncher:new(tid, terra_fn, arg_type, arg,
                               pred, id, tag)
   local launcher = {
      tid = tid,
      terra_fn = terra_fn:getdefinitions()[1]:getpointer(),
      arg_type = arg_type,
      arg = arg,
      pred = pred or Predicate.TRUE_PRED(),
      id = id or 0,
      tag = tag or 0,
      reqs = {},
      __is_terra_task_launcher = true
   }
   setmetatable(launcher, self)
   self.__index = self
   return launcher
end

function TerraTaskLauncher:add_region_requirement(req)
   self.reqs[#self.reqs + 1] = req
   return self
end

function TerraTaskLauncher:add_region_requirements(...)
   local vargs = {...}

   for _, req in ipairs(vargs)
   do
      self:add_region_requirement(req)
   end
   return self
end

TerraIndexLauncher = {}

function TerraIndexLauncher:new(tid, domain, terra_fn,
                                global_arg_type, global_arg,
                                local_arg_type, arg_map,
                                pred, must, id, tag)
   local launcher = {
      tid = tid,
      domain = domain,
      terra_fn = terra_fn:getdefinitions()[1]:getpointer(),
      global_arg_type = global_arg_type,
      global_arg = global_arg,
      local_arg_type = local_arg_type,
      arg_map = arg_map,
      pred = pred or Predicate.TRUE_PRED(),
      must = must or false,
      id = id or 0,
      tag = tag or 0,
      reqs = {},
      __is_terra_index_launcher = true
   }
   setmetatable(launcher, self)
   self.__index = self
   return launcher
end

function TerraIndexLauncher:add_region_requirement(req)
   self.reqs[#self.reqs + 1] = req
   return self
end

function TerraIndexLauncher:add_region_requirements(...)
   local vargs = {...}

   for _, req in ipairs(vargs)
   do
      self:add_region_requirement(req)
   end
   return self
end

function LegionLib:execute_terra_task(launcher)
   if launcher.__is_terra_task_launcher
   then
      local converter = table_converter(launcher.arg_type, true)
      local terra_obj = converter(ffi.cast("uint64_t*", launcher.terra_fn),
                                  launcher.arg)
      
      -- local test_fun_instance = test_fun(launcher.arg_type)
      -- local value = test_fun_instance(terra_obj)
      -- return value
      local task_arg =
         self:make_terra_task_argument(terra_obj,
                                       terralib.sizeof(launcher.arg_type) +
                                          terralib.sizeof(uint64))
      
      local c_launcher =
         LegionRuntime.HighLevel.TaskLauncher(
            launcher.tid, task_arg,
            launcher.pred, launcher.id, launcher.tag
         )

      for _, req in ipairs(launcher.reqs)
      do
         c_launcher:add_region_requirement(req:to_c_object())
      end
      
      return self.__runtime:execute_task(self.__ctx, c_launcher)
   else if launcher.__is_terra_index_launcher
        then
           local global_converter = table_converter(launcher.global_arg_type, true)
           local global_terra_obj =
              global_converter(ffi.cast("uint64_t*", launcher.terra_fn),
                               launcher.global_arg)
           local global_arg =
              self:make_terra_task_argument(global_terra_obj,
                                            terralib.sizeof(launcher.global_arg_type) +
                                               terralib.sizeof(uint64))

           local local_converter = table_converter(launcher.local_arg_type, false)
           local local_args = self:create_argument_map()
           for point, arg in pairs(launcher.arg_map)
           do
              local terra_obj = local_converter(arg)
              local task_arg =
                 self:make_terra_task_argument(terra_obj,
                                               terralib.sizeof(launcher.local_arg_type))
              local point_in_c = DomainPoint:new(point):to_c_object()
              local_args:set_point(point_in_c, task_arg)
           end

           local c_launcher =
              LegionRuntime.HighLevel.IndexLauncher(
                 launcher.tid, launcher.domain:to_c_object(),
                 global_arg, local_args.__argmap,
                 launcher.pred, launcher.must,
                 launcher.id, launcher.tag
              )

           for _, req in ipairs(launcher.reqs)
           do
              c_launcher:add_region_requirement(req:to_c_object())
           end
           
           return self.__runtime:execute_index_space(self.__ctx, c_launcher)
        else
           assert(false, "not supported task launcher")
        end
   end
end

TIndexSpace = lg.TIndexSpace
TFieldSpace = lg.TFieldSpace
TLogicalRegion = lg.TLogicalRegion
TLogicalPartition = lg.TLogicalPartition
TPhysicalRegion = lg.TPhysicalRegion
TTask = lg.TTask

struct TerraIndexIterator
{
   rawptr: &opaque
}

terra TerraIndexIterator:init(region: TLogicalRegion)
   self.rawptr = lgf.create_index_iterator(region)
end

terra TerraIndexIterator:next(): uint32
   return lgf.next(self.rawptr)
end

terra TerraIndexIterator:has_next(): bool
   return [bool](lgf.has_next(self.rawptr))
end

terra TerraIndexIterator:close()
   lgf.destroy_index_iterator(self.rawptr)
end

function TerraAccessor(elem_type)
   local struct Accessor
   {
      rawptr: &opaque
   }

   terra Accessor:init(region: TPhysicalRegion)
      self.rawptr = lgf.create_terra_accessor(region)
   end

   terra Accessor:init_with_field(region: TPhysicalRegion,
                                  field: uint32)
      self.rawptr = lgf.create_terra_field_accessor(region, field)
   end

   terra Accessor:read(ptr: uint32): elem_type
      var value: elem_type
      lgf.read_from_accessor(self.rawptr, ptr, &value, sizeof(elem_type))
      return value
   end

   terra Accessor:write(ptr: uint32, value: elem_type): {}
      lgf.write_to_accessor(self.rawptr, ptr, &value, sizeof(elem_type))
   end

   terra Accessor:close()
      lgf.destroy_terra_accessor(self.rawptr)
   end

   return Accessor
end

function TerraReducer(structure, field_name)
   local elem_type = nil
   for _, v in pairs(structure.entries)
   do
      if v.field == field_name
      then elem_type = v.type end
   end
   if not (elem_type == int or
              elem_type == float or
              elem_type == double)
   then
      assert(false, "unsupported reducer type")
   end

   local struct Reducer
   {
      rawptr: &opaque;
      redop: uint32;
      elem_type: uint32;
      red_type: uint32;
   }

   local function init_elem_type(s)
      if elem_type == int
      then return quote s.elem_type = Primt.int end
      else if elem_type == float
           then return quote s.elem_type = PrimType.float end
           else if elem_type == double
                then return quote s.elem_type = PrimType.double end
                end
           end
      end
   end
   
   terra Reducer:init(region: TPhysicalRegion,
                      red_type: int)
      self.redop = region.redop
      [init_elem_type(self)]
      self.red_type = red_type
      self.rawptr =
         lgf.create_terra_reducer(region,
                                  [terralib.offsetof(structure, field_name)],
                                  self.redop,
                                  self.elem_type,
                                  self.red_type)
   end

   local sym_ptr, sym_value = symbol("ptr"), symbol("value")
   local function reduce(s)
      if elem_type == float
      then return quote
            lgf.reduce_terra_reducer_float(s.rawptr,
                                           s.redop,
                                           s.red_type,
                                           [sym_ptr],
                                           [sym_value])
                  end
      else if elem_type == float
           then return quote
                 lgf.reduce_terra_reducer_double(s.rawptr,
                                                 s.redop,
                                                 s.red_type,
                                                 [sym_ptr],
                                                 [sym_value])
                       end
           else if elem_type == int
                then return quote
                      lgf.reduce_terra_reducer_int(s.rawptr,
                                                   s.redop,
                                                   s.red_type,
                                                   [sym_ptr],
                                                   [sym_value])
                            end
                end
           end
      end
   end
   
   terra Reducer:reduce([sym_ptr]: uint32, [sym_value]: elem_type): {}
      [reduce(self)]
   end

   terra Reducer:close()
      lgf.destroy_terra_reducer(self.rawptr,
                                self.redop,
                                self.elem_type,
                                self.red_type)
   end

   return Reducer
end

terra TTask:get_index(): int
   return lgf.get_index(self.rawptr)
end
