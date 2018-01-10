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

require('Set')

local not_found = true

for k, _ in pairs(_G)
do
   if k == 'LegionRuntime' then
      not_found = false
      break
   end
end

if not_found then
   package.loadlib('libbinding.so', '*')
   package.loadlib('libbinding.so', 'init')()
end

local SEPARATE_FIELD_ALLOCATION = false

-- utility functions

local function cpp_vector_to_list(vec)
   local list = {}
   for i = 0, vec:size() - 1
   do
      list[i] = vec:at(i)
   end
   return list
end

-- field allocation

-- primitive types
PrimType = {
   float = 0,
   double = 1,
   int = 2
}

-- primitive type sizes
PrimTypeSize = {}

PrimTypeSize[PrimType.float] = 4
PrimTypeSize[PrimType.double] = 8
PrimTypeSize[PrimType.int] = 4

-- array type
ArrayType = {}

function ArrayType:new(ty, size)
   local set = { __elem_type = ty,
                 __array_size = size }
   return set
end

-- reduction op type

ReductionType = {
   PLUS = 100,
   MINUS = 200,
   TIMES = 300
}

-- field accessor
local function typeify(accessor, ty)
   if ty == PrimType.float then
      return accessor:typeify_float()
   end
   if ty == PrimType.double then
      return accessor:typeify_double()
   end
   if ty == PrimType.int then
      return accessor:typeify_int()
   end

   error("unsupported primitive type for accessor")
end

PrimAccessor = {}

if SEPARATE_FIELD_ALLOCATION
then

function PrimAccessor:new(untyped_accessor, ty)
   local accessor = { __type = ty,
                      __accessor = typeify(untyped_accessor, ty) }
   setmetatable(accessor, self)
   self.__index = self
   return accessor
end

else

function PrimAccessor:new(untyped_accessor, ty, redop)
   local accessor = { __type = ty,
                      __accessor = typeify(untyped_accessor, ty),
                      __redop = redop }
   setmetatable(accessor, self)
   self.__index = self
   return accessor
end

end -- if SEPARATE_FIELD_ALLOCATION

function PrimAccessor:read(ptr)
   return self.__accessor:read(ptr_t(ptr))
end

function PrimAccessor:write(ptr, value)
   self.__accessor:write(ptr_t(ptr), value)
end

function PrimAccessor:read_at_point(point)
   local point_in_c = DomainPoint:new(point):to_c_object()
   return self.__accessor:read_at_point(point_in_c)
end

function PrimAccessor:write_at_point(point, value)
   local point_in_c = DomainPoint:new(point):to_c_object()
   self.__accessor:write_at_point(point_in_c, value)
end

if SEPARATE_FIELD_ALLOCATION
then

function PrimAccessor:reduce(op, ptr, value)
   if op == ReductionType.PLUS then
      self.__accessor:reduce_plus(ptr_t(ptr), value)
   end
   if op == ReductionType.MINUS then
      self.__accessor:reduce_minus(ptr_t(ptr), value)
   end
   if op == ReductionType.TIMES then
      self.__accessor:reduce_times(ptr_t(ptr), value)
   end
end

else

function PrimAccessor:reduce(op, ptr, value)
   if self.__redop == 0
   then
      if op == ReductionType.PLUS then
         self.__accessor:reduce_plus(ptr_t(ptr), value)
      end
      if op == ReductionType.MINUS then
         self.__accessor:reduce_minus(ptr_t(ptr), value)
      end
      if op == ReductionType.TIMES then
         self.__accessor:reduce_times(ptr_t(ptr), value)
      end
   else -- self._redop != 0
      if op == ReductionType.PLUS then
         self.__accessor:convert_plus():reduce(ptr_t(ptr), value)
      end
      if op == ReductionType.MINUS then
         self.__accessor:convert_minus():reduce(ptr_t(ptr), value)
      end
      if op == ReductionType.TIMES then
         self.__accessor:convert_times():reduce(ptr_t(ptr), value)
      end
   end
end
   
end -- if SEPARATE_FIELD_ALLOCATION

ArrayAccessor = {}

function ArrayAccessor:new(untyped_accessor, ty)
   assert(type(ty) == "table")

   local accessor = { __type = ty.__elem_type,
                      __size = ty.__array_size,
                      __accessor = untyped_accessor }
   setmetatable(accessor, self)
   self.__index = self
   return accessor
end

function ArrayAccessor:read(ptr)
   local array = {}

   for i = 0, self.__size - 1
   do
      local offset = i * PrimTypeSize[self.__type]
      local elem_accessor =
         self.__accessor:get_untyped_field_accessor(offset, 0)
      array[i] = typeify(elem_accessor, self.__type):read(ptr_t(ptr))
   end

   return array
end

function ArrayAccessor:write(ptr, array)
   for i = 0, self.__size - 1
   do
      local offset = i * PrimTypeSize[self.__type]
      local elem_accessor =
         self.__accessor:get_untyped_field_accessor(offset, 0)
      typeify(elem_accessor, self.__type):write(ptr_t(ptr), array[i])
   end
end

function ArrayAccessor:read_at_point(point)
   local array = {}
   local point_in_c = DomainPoint:new(point):to_c_object()
   for i = 0, self.__size - 1
   do
      local offset = i * PrimTypeSize[self.__type]
      local elem_accessor =
         self.__accessor:get_untyped_field_accessor(offset, 0)
      array[i] = typeify(elem_accessor, self.__type):read_at_point(point_in_c)
   end

   return array
end

function ArrayAccessor:write_at_point(point, array)
   local point_in_c = DomainPoint:new(point):to_c_object()
   for i = 0, self.__size - 1
   do
      local offset = i * PrimTypeSize[self.__type]
      local elem_accessor =
         self.__accessor:get_untyped_field_accessor(offset, 0)
      typeify(elem_accessor, self.__type):write_at_point(point_in_c, array[i])
   end
end

FieldAccessor = {}

if SEPARATE_FIELD_ALLOCATION
then
   
function FieldAccessor:new(physical_region, field_id)
   local field_accessor = { __accessors = {} }

   for k, v in pairs(field_id)
   do
      local untyped_accessor
         = physical_region:get_field_accessor(v.__fid)
      local ty = v.__type

      local accessor = nil
      if type(ty) == "number" then
         accessor = PrimAccessor:new(untyped_accessor, ty)
      else
         assert(type(ty) == "table")
         accessor = ArrayAccessor:new(untyped_accessor, ty)
      end

      field_accessor.__accessors[k] = accessor
   end
   
   setmetatable(field_accessor, self)
   self.__index = function (child, idx)
      return
         rawget(self, idx) or
         rawget(rawget(child, "__accessors"), idx) or
         error("field accessor does not have field " .. idx)
   end
   return field_accessor
end

else

function FieldAccessor:new(physical_region, field_id)
   local redop = physical_region.__requirement.redop
   local parent_accessor = nil
      if redop == 0
      then
         parent_accessor =
            physical_region:get_field_accessor(field_id.__fid)
      else
         parent_accessor = physical_region:get_accessor()
      end
   local field_accessor = { __accessors = {} }

   for k, offset in pairs(field_id.__offset)
   do
      local untyped_accessor
         = parent_accessor:get_untyped_field_accessor(offset, 0)
      local ty = field_id.__type[k]

      local accessor = nil
      if type(ty) == "number" then
         accessor = PrimAccessor:new(untyped_accessor, ty, redop)
      else
         assert(type(ty) == "table")
         accessor = ArrayAccessor:new(untyped_accessor, ty)
      end

      field_accessor.__accessors[k] = accessor
   end
   
   setmetatable(field_accessor, self)
   self.__index = function (child, idx)
      return
         rawget(self, idx) or
         rawget(rawget(child, "__accessors"), idx) or
         error("field accessor does not have field " .. idx)
   end
   return field_accessor
end
   
end -- if SEPARATE_FIELD_ALLOCATION


function FieldAccessor:read(ptr)
   local field = {}
   for k, accessor in pairs(self.__accessors)
   do
      field[k] = accessor:read(ptr)
   end
   return field
end

function FieldAccessor:write(ptr, value)
   for k, _ in pairs(self.__accessors)
   do
      local v = value[k]
      if v
      then self.__accessors[k]:write(ptr, v)
      end
   end
end

function FieldAccessor:read_at_point(point)
   local field = {}
   for k, accessor in pairs(self.__accessors)
   do
      field[k] = accessor:read_at_point(point)
   end
   return field
end

function FieldAccessor:write_at_point(point, value)
   for k, _ in pairs(self.__accessors)
   do
      local v = value[k]
      if v
      then self.__accessors[k]:write_at_point(point, v)
      end
   end
end


function LegionRuntime.HighLevel.PhysicalRegion:get_lua_accessor(field_id, field_name)
   if not field_name then
      return FieldAccessor:new(self, field_id)
   else
      return FieldAccessor:new(self, { field_name = field_id[field_name] })
   end
end

-- Predicate
Predicate = LegionRuntime.HighLevel.Predicate

-- Domain
-- Domain = LegionRuntime.LowLevel.Domain
-- DomainPoint = LegionRuntime.LowLevel.DomainPoint

-- Point
-- Point = LegionRuntime.HighLevel.Point
-- Rect = LegionRuntime.HighLevel.Rect
-- Blockify = LegionRuntime.HighLevel.Blockify

-- processor type
Processor = LegionRuntime.LowLevel.Processor

-- some enumerations
PrivilegeMode = LegionRuntime.HighLevel.PrivilegeMode
CoherenceProperty = LegionRuntime.HighLevel.CoherenceProperty

-- legion interface
LegionLib = {}

function LegionLib:init_binding(task_file_name)
   local state = { __task_file_name = task_file_name }
   setmetatable(state, self)
   self.__index = self
   return state
end

function LegionLib:set_top_level_task_id(id)
   LegionRuntime.HighLevel.HighLevelRuntime.set_top_level_task_id(id)
end

function LegionLib:register_single_task(id, name, processor, is_leaf)
   BindingLibInC.register_single_task(id, name, processor, is_leaf)
end

function LegionLib:register_index_task(id, name, processor, is_leaf)
   BindingLibInC.register_index_task(id, name, processor, is_leaf)
end

function LegionLib:start(cmd_arg)
   cmd_arg.__task_file_name = self.__task_file_name
   BindingLibInC.start(cmd_arg)
end

function LegionLib:get_region_requirements()
   return cpp_vector_to_list(self.__reqs)
end

function LegionLib:get_context()
   return self.__ctx
end

function LegionLib:get_runtime()
   return self.__runtime
end

function LegionLib:hide_details(reqs, ctx, runtime)
   self.__reqs = reqs
   self.__ctx = ctx
   self.__runtime = runtime
end

local function make_index_space(ispace_id)
   local ispace = { __is_index_space = true }
   ispace.id = ispace_id
   return ispace
end

local function make_field_space(fspace_id)
   local fspace = { __is_field_space = true }
   fspace.id = fspace_id
   return fspace
end

local function make_logical_region(...) --tree_id, ispace_id, fspace_id)
   local region = { __is_logical_region = true }
   local vargs = {...}

   if #vargs == 1
   then
      local logical_region_in_c = vargs[1]
      region.tree_id = logical_region_in_c:get_tree_id()
      region.index_space =
         make_index_space(logical_region_in_c:get_index_space().id)
      region.field_space =
         make_field_space(logical_region_in_c:get_field_space():get_id())
   else
      assert(#vargs == 3)
      -- tree_id = vargs[1]
      -- ispace_id = vargs[2]
      -- fspace_id = vargs[3]
      region.tree_id = vargs[1]
      region.index_space = make_index_space(vargs[2])
      region.field_space = make_field_space(vargs[3])
   end

   return region
end

local function make_logical_partition(...)
   local partition = { __is_logical_partition = true }
   local vargs = {...}

   if #vargs == 1
   then
      local logical_partition_in_c = vargs[1]
      partition.tree_id = logical_partition_in_c:get_tree_id()
      partition.index_partition = logical_partition_in_c:get_index_partition()
      partition.field_space =
         make_field_space(logical_partition_in_c:get_field_space():get_id())
   else
      assert(#vargs == 3)
      -- tree_id = vargs[1]
      -- ipartition = vargs[2]
      -- fspace_id = vargs[3]
      partition.tree_id = vargs[1]
      partition.index_partition = vargs[2]
      partition.field_space = make_field_space(vargs[3])
   end

   return partition
end

local function make_c_index_space(ispace)
   assert(ispace.__is_index_space and ispace.id)
   return BindingLibInC.make_index_space(ispace.id)
end

local function make_c_field_space(fspace)
   assert(fspace.__is_field_space and fspace.id)
   return BindingLibInC.make_field_space(fspace.id)
end

local function make_c_logical_region(region)
   assert(region.__is_logical_region
             and region.tree_id
             and region.index_space
             and region.field_space
             and region.index_space.__is_index_space
             and region.index_space.id
             and region.field_space.__is_field_space
             and region.field_space.id)
   return BindingLibInC.make_logical_region(region.tree_id,
                                            region.index_space.id,
                                            region.field_space.id)
end

local function make_c_logical_partition(partition)
   assert(partition.__is_logical_partition
             and partition.tree_id
             and partition.index_partition
             and partition.field_space
             and partition.field_space.__is_field_space
             and partition.field_space.id)
   return BindingLibInC.make_logical_partition(partition.tree_id,
                                               partition.index_partition,
                                               partition.field_space.id)
end

function LegionLib:create_index_space(arg)
   if type(arg) == 'number'
   then
      local ispace_in_c = self.__runtime:create_index_space(self.__ctx, arg)
      return  make_index_space(ispace_in_c.id)
   else if type(arg) == 'table' and arg.__is_domain
        then
           local domain_in_c = arg:to_c_object()
           local ispace_in_c = self.__runtime:create_index_space(self.__ctx,
                                                                 domain_in_c)
           return make_index_space(ispace_in_c.id)
        else
           assert(false, "not supported")
        end
   end
end

function LegionLib:create_field_space()
   local fspace_in_c = self.__runtime:create_field_space(self.__ctx)
   return make_field_space(fspace_in_c:get_id())
end

function LegionLib:destroy_index_space(ispace)
   self.__runtime:destroy_index_space(self.__ctx,
                                      make_c_index_space(ispace))
end

function LegionLib:destroy_field_space(fspace)
   self.__runtime:destroy_field_space(self.__ctx,
                                      make_c_field_space(fspace))
end

if SEPARATE_FIELD_ALLOCATION  
then
   
function LegionLib:allocate_field(fspace, structure)
   local field_id = {}
   local allocator =
      self.__runtime:create_field_allocator(self.__ctx, fspace)

   for k, ty in pairs(structure)
   do
      if type(ty) == 'number' then
         field_id[k] = {
            __type = ty,
            __fid = allocator:allocate_field(PrimTypeSize[ty], -1)
         }
      else
         if type(ty) == 'table' then
            local array_size =
               PrimTypeSize[ty.__elem_type] * ty.__array_size
            field_id[k] = {
               __type = ty,
               __fid = allocator:allocate_field(array_size, -1)
            }
         else
            error('invalid type of value in structure descriptor')
         end
      end
   end

   return field_id
end

else

local function convert_terra_struct_to_lua_struct(terra_struct)
   local lua_struct = {}
   local function convert_terra_type(terra_type)
      local ty = nil
      if string.match(terra_type.name, "int")
      then ty = PrimType.int
      else
         if string.match(terra_type.name, "double")
         then ty = PrimType.double
         else
            if string.match(terra_type.name, "float")
            then ty = PrimType.float
            else assert(false, "unsupported type")
            end
         end
      end
      return ty
   end

   for _, v in pairs(terra_struct.entries)
   do
      local ty = nil
      if rawget(v.type, 'N')
      then
         ty = ArrayType:new(convert_terra_type(v.type.type),
                            v.type.N)
      else ty = convert_terra_type(v.type)
      end
      lua_struct[v.field] = ty
   end
   return lua_struct
end

function LegionLib:allocate_field(fspace, structure)
   local lua_struct = structure
   if structure.entries
   then -- if structure is terra struct
      lua_struct = convert_terra_struct_to_lua_struct(structure)
   end
   
   local field_id = { __offset = {}, __type = {} }
   local offset = 0
   
   if structure.entries
   then -- if structure is terra struct
      for _, v in pairs(structure.entries)
      do
         local field_name = v.field
         field_id.__offset[field_name] =
            terralib.offsetof(structure, field_name)
         field_id.__type[field_name] = lua_struct[field_name]
      end
      offset = terralib.sizeof(structure)
   else
      for k, ty in pairs(lua_struct)
      do
         field_id.__offset[k] = offset
         field_id.__type[k] = ty
         if type(ty) == 'number' then
            offset = offset + PrimTypeSize[ty]
         else
            if type(ty) == 'table' then
               local array_size =
                  PrimTypeSize[ty.__elem_type] * ty.__array_size
               offset = offset + array_size
            else
               error('invalid type of value in structure descriptor')
            end
         end
      end
   end

   local allocator =
      self.__runtime:create_field_allocator(self.__ctx,
                                            make_c_field_space(fspace))
   field_id.__fid = allocator:allocate_field(offset, -1)

   return field_id
end

end -- if SEPARATE_FIELD_ALLOCATION

function LegionLib:create_logical_region(ispace, fspace)
   local logical_region_in_c =
      self.__runtime:create_logical_region(self.__ctx,
                                           make_c_index_space(ispace),
                                           make_c_field_space(fspace))
   return make_logical_region(logical_region_in_c)
end

function LegionLib:destroy_logical_region(region)
   self.__runtime:destroy_logical_region(self.__ctx,
                                         make_c_logical_region(region))
end

function LegionLib:make_task_argument(obj)
   if not self.__task_args then self.__task_args = {} end

   local task_arg = BindingLibInC.make_task_argument(obj)
   self.__task_args[#self.__task_args + 1] = task_arg
   return task_arg
end

function LegionLib:destroy_task_arguments()
   if self.__task_args then
      for _, v in ipairs(self.__task_args)
      do
         BindingLibInC.delete_task_argument(v)
      end
   end
   self.__task_args = nil
end

-- assume that map is a table from integers to sets of values of type ptr_t.
function to_cpp_coloring(coloring)
   local cpp_coloring = LegionRuntime.HighLevel.Coloring()

   for color, set in pairs(coloring)
   do
      if not (type(color) == "number") then
         error("coloring map should be indexed by numbers")
      end
      local elems = set:to_list()
      local tmp = cpp_coloring:at(color) -- just intialize the element for the index
      for _, v in pairs(elems)
      do
         cpp_coloring:at(color).points:insert(ptr_t(v))
      end
   end

   return cpp_coloring
end

function LegionLib:create_index_partition(...)
   local vargs = {...}

   assert (vargs[1].__is_index_space)

   local ispace = vargs[1]
   local ispace_in_c = make_c_index_space(ispace)

   if #vargs == 2 then
      if type(vargs[2]) == "table" and vargs[2].__is_blockify then
         -- when blockify is given as mapper
         local blockify = vargs[2]:to_c_object()
         return
            self.__runtime:create_index_partition(self.__ctx, ispace_in_c,
                                                  blockify, -1)
      end
   end
   if #vargs == 3 then
      if type(vargs[2]) == "table" and not vargs[2].__is_blockify then
         -- when lua coloring is given
         local coloring = to_cpp_coloring(vargs[2])
         local disjoint = vargs[3]
         return
            self.__runtime:create_index_partition(self.__ctx, ispace_in_c,
                                                  coloring, disjoint, -1)
      else
         if type(vargs[3]) == "number"
         then
            local blockify = vargs[2]:to_c_object()
            local part_color = vargs[3]
            return
               self.__runtime:create_index_partition(self.__ctx, ispace_in_c,
                                                     blockify, part_color)
         end
      end
   end

   error("not supported yet")
end

function LegionLib:get_index_subspace(partition, color)
   local ispace_in_c =
      self.__runtime:get_index_subspace(self.__ctx, partition, color)
   return make_index_space(ispace_in_c.id)
end

function LegionLib:get_logical_partition(parent, index_partition)
   local logical_partition_in_c =
      self.__runtime:get_logical_partition(self.__ctx,
                                           make_c_logical_region(parent),
                                           index_partition)
   return make_logical_partition(logical_partition_in_c)
end

function LegionLib:get_logical_partition_by_tree(index_partition,
                                                 fspace, tree_id)
   local fspace_in_c = make_c_field_space(fspace)
   local logical_partition_in_c =
      self.__runtime:get_logical_partition_by_tree(self.__ctx,
                                                   index_partition,
                                                   fspace_in_c, tree_id)
   return make_logical_partition(logical_partition_in_c)
end

function LegionLib:get_logical_subregion_by_color(logical_partition, color)
   local logical_partition_in_c = make_c_logical_partition(logical_partition)
   local logical_region_in_c =
      self.__runtime:get_logical_subregion_by_color(self.__ctx,
                                                    logical_partition_in_c,
                                                    color)
   return make_logical_region(logical_region_in_c)
end

function LegionLib:get_index_partition_color_space(index_partition)
   local domain_in_c =
      self.__runtime:get_index_partition_color_space(self.__ctx,
                                                     index_partition)
   return Domain:wrap(domain_in_c)
end

function LegionLib:map_region(requirement, id, tag)
   local id_ = id or 0
   local tag_ = tag or 0
   local req = requirement:to_c_object()
   local region =
      self.__runtime:map_region(self.__ctx,
                                req, id_, tag_)
   region.__requirement = req
   return region
end 

function LegionLib:unmap_region(physical_region)
   self.__runtime:unmap_region(self.__ctx, physical_region)
end

function LegionLib:map_all_regions()
   self.__runtime:map_all_regions(self.__ctx)
end

function LegionLib:unmap_all_regions()
   self.__runtime:unmap_all_regions(self.__ctx)
end

function LegionLib:allocate_in_indexspace(ispace, num)
   local allocator =
      self.__runtime:create_index_allocator(self.__ctx,
                                            make_c_index_space(ispace))
   allocator:alloc(num)
end

function LegionLib:create_argument_map()
   return ArgumentMap:new(self.__runtime:create_argument_map(self.__ctx))
end

function LegionLib:execute_task(launcher)
   launcher.arg.__task_file_name = self.__task_file_name
   local task_arg = self:make_task_argument(launcher.arg)
   
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
end

function LegionLib:execute_index_space(launcher)
   launcher.global_arg.__task_file_name = self.__task_file_name
   local global_arg = self:make_task_argument(launcher.global_arg)
   local local_args = self:create_argument_map()
   for point, arg in pairs(launcher.arg_map)
   do
      local task_arg = self:make_task_argument(arg)
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
end

function LegionLib:register_reduction_op(op_id, op_type, ty)
   if op_type == ReductionType.PLUS then
      if ty == PrimType.float then
         BindingLibInC.register_plus_reduction_for_float(op_id)
         return
      end
      if ty == PrimType.double then
         BindingLibInC.register_plus_reduction_for_double(op_id)
         return
      end
      if ty == PrimType.int then
         BindingLibInC.register_plus_reduction_for_int(op_id)
         return
      end
   end   
   if op_type == ReductionType.MINUS then
      if ty == PrimType.float then
         BindingLibInC.register_minus_reduction_for_float(op_id)
         return
      end
      if ty == PrimType.double then
         BindingLibInC.register_minus_reduction_for_double(op_id)
         return
      end
      if ty == PrimType.int then
         BindingLibInC.register_minus_reduction_for_int(op_id)
         return
      end
   end
   if op_type == ReductionType.TIMES then
      if ty == PrimType.float then
         BindingLibInC.register_times_reduction_for_float(op_id)
         return
      end
      if ty == PrimType.double then
         BindingLibInC.register_times_reduction_for_double(op_id)
         return
      end
      if ty == PrimType.int then
         BindingLibInC.register_times_reduction_for_int(op_id)
         return
      end
   end
   error("unsupported reduction operation")
end

ArgumentMap = {}

function ArgumentMap:new(argmap)
   local argument_map = { __argmap = argmap }
   setmetatable(argument_map, self)
   self.__index = self
   return argument_map
end

function ArgumentMap:set_point(point, task_arg, replace)
   local replace_ = replace or true
   self.__argmap:set_point(point, task_arg, replace_)
end

IndexIterator = {}

function IndexIterator:new(arg)
   local iterator = {}
   if arg.__is_index_space
   then
      iterator.__iterator =
         LegionRuntime.HighLevel.IndexIterator(make_c_index_space(arg))
   else if arg.__is_logical_region
        then
           iterator.__iterator =
              LegionRuntime.HighLevel.IndexIterator(make_c_logical_region(arg))
        else
           assert(false, "not supported!")
        end
   end
   setmetatable(iterator, self)
   self.__index = self
   return iterator
end

function IndexIterator:has_next()
   return self.__iterator:has_next()
end

function IndexIterator:next()
   local ptr = self.__iterator:next()
   return ptr.value
end

RegionRequirement = {}

function RegionRequirement:new(conf)
   local req = {
      priv = conf.priv or PrivilegeMode.READ_WRITE,
      prop = conf.prop or CoherenceProperty.EXCLUSIVE,
      region = conf.region,
      parent = conf.parent or conf.region,
      part = conf.part,
      proj_id = conf.proj_id or 0,
      reduce_op = conf.reduce_op,
      tag = conf.tag or 0,
      verified = conf.verified or false,
      fields = {}
   }
   setmetatable(req, self)
   self.__index = self
   return req
end

function RegionRequirement:add_field(field, instance)
   local instance_ = instance or true

   self.fields[#self.fields + 1] =
      { field = field, instance = instance_ }
   return self
end

function RegionRequirement:add_field_raw(field, instance, key)
   local instance_ = instance or true

   self.fields[#self.fields + 1] =
      { field = {__fid = field[key]}, instance = instance_ }
   return self
end

if SEPARATE_FIELD_ALLOCATION
then

function RegionRequirement:to_c_object()
   local req = nil
   local factory = LegionRuntime.HighLevel.RegionRequirement

   local function add_field(field_id, instance) 
      assert(type(field_id) == 'table')
      for k, v in pairs(field_id)
      do
         req:add_field(v.__fid, instance)
      end
   end

   if not self.part
   then -- make requirement for the whole region
      assert(self.region)
      if not self.reduce_op
      then
         req = factory.make(self.region,
                            self.priv, self.prop,
                            self.parent, self.tag,
                            self.verified)
      else
         req = factory.make_with_reduction_op(self.region,
                                              self.reduce_op, self.prop,
                                              self.parent, self.tag,
                                              self.verified)
      end
   else -- make requirement for the partition
      assert(self.parent)
      if not self.reduce_op
      then
         req = factory.make(self.part, self.proj_id,
                            self.priv, self.prop,
                            self.parent, self.tag,
                            self.verified)
      else
         req = factory.make_with_reduction_op(self.part, self.proj_id,
                                              self.reduce_op, self.prop,
                                              self.parent, self.tag,
                                              self.verified)
      end
   end

   for _, field in ipairs(self.fields)
   do
      add_field(field.field, field.instance)
   end

   assert(req)
   return req
end

else

function RegionRequirement:to_c_object()
   local req = nil
   local factory = LegionRuntime.HighLevel.RegionRequirement

   local function add_field(field_id, instance) 
      assert(type(field_id) == 'table')
      req:add_field(field_id.__fid, instance)
   end

   if not self.part
   then -- make requirement for the whole region
      assert(self.region)
      if not self.reduce_op
      then
         req = factory.make(make_c_logical_region(self.region),
                            self.priv, self.prop,
                            make_c_logical_region(self.parent),
                            self.tag,
                            self.verified)
      else
         req = factory.make_with_reduction_op(make_c_logical_region(self.region),
                                              self.reduce_op, self.prop,
                                              make_c_logical_region(self.parent),
                                              self.tag,
                                              self.verified)
      end
   else -- make requirement for the partition
      assert(self.parent)
      if not self.reduce_op
      then
         req = factory.make(make_c_logical_partition(self.part),
                            self.proj_id,
                            self.priv, self.prop,
                            make_c_logical_region(self.parent),
                            self.tag,
                            self.verified)
      else
         req = factory.make_with_reduction_op(make_c_logical_partition(self.part),
                                              self.proj_id,
                                              self.reduce_op, self.prop,
                                              make_c_logical_region(self.parent),
                                              self.tag,
                                              self.verified)
      end
   end

   for _, field in ipairs(self.fields)
   do
      add_field(field.field, field.instance)
   end

   assert(req)
   return req
end

end -- if SEPARATE_FIELD_ALLOCATION
   
IndexLauncher = {}

function IndexLauncher:new(tid, domain, global_arg, arg_map,
                           pred, must, id, tag)
   local launcher = {
      tid = tid,
      domain = domain,
      global_arg = global_arg,
      arg_map = arg_map or {},
      pred = pred or Predicate.TRUE_PRED(),
      must = must or false,
      id = id or 0,
      tag = tag or 0,
      reqs = {}
   }
   setmetatable(launcher, self)
   self.__index = self
   return launcher
end

function IndexLauncher:add_region_requirement(req)
   self.reqs[#self.reqs + 1] = req
   return self
end

function IndexLauncher:add_region_requirements(...)
   local vargs = {...}

   for _, req in ipairs(vargs)
   do
      self:add_region_requirement(req)
   end
   return self
end

TaskLauncher = {}

function TaskLauncher:new(tid, arg,
                          pred, must, id, tag)
   local launcher = {
      tid = tid,
      arg = arg or {},
      pred = pred or Predicate.TRUE_PRED(),
      id = id or 0,
      tag = tag or 0,
      reqs = {}
   }
   setmetatable(launcher, self)
   self.__index = self
   return launcher
end

function TaskLauncher:add_region_requirement(req)
   self.reqs[#self.reqs + 1] = req
   return self
end

function TaskLauncher:add_region_requirements(...)
   local vargs = {...}

   for _, req in ipairs(vargs)
   do
      self:add_region_requirement(req)
   end
   return self
end

-- coloring
ColoredPoints = Set


-- task wrappers in lua side

function task_wrapper_in_lua(task_name, task_file_name, args, reqs,
                             region_vec, ctx, runtime)
   local binding = LegionLib:init_binding(task_file_name)
   binding:hide_details(reqs, ctx, runtime)
   local regions = cpp_vector_to_list(region_vec)
   local args_recovered = recover_metatables(args)
   for k, v in pairs(regions)
   do
      v.__requirement = reqs:at(k)
   end
   _G[task_name](binding, regions, args_recovered)
   binding:destroy_task_arguments()
end

function index_task_wrapper_in_lua(task_name, task_file_name,
                                   global_args, local_args,
                                   point, reqs, region_vec, ctx, runtime)
   local binding = LegionLib:init_binding(task_file_name)
   binding:hide_details(reqs, ctx, runtime)
   local regions = cpp_vector_to_list(region_vec)
   local global_args_recovered = recover_metatables(global_args)
   for k, v in pairs(regions)
   do
      v.__requirement = reqs:at(k)
   end
   local point_in_lua = {}
   if point.dim == 0
   then point_in_lua = DomainPoint:new(point:get_index())
   else point_in_lua = DomainPoint:new(Point:new(point:get_coords()))
   end
   _G[task_name](binding, regions, global_args_recovered, local_args, point_in_lua)
   binding:destroy_task_arguments()
end

-- Equivalent to LegionRuntime::Arrays::Point
Point = {}

function Point:new(coords)
   assert(#coords > 0,
          "points should be at least 1 dimensional")
   local point = { __coords = {} }
   for k, v in pairs(coords)
   do
      point.__coords[k] = v
   end
   point.__dim = #coords
   point.__is_point = true
   setmetatable(point, self)
   self.__index = function (child, idx)
      return
         rawget(self, idx) or
         rawget(child.__coords, idx)
   end
   return point
end

function Point:clone(other)
   if self.__coords
   then return Point:new(self.__coords)
   else if other.__coords
        then return Point:new(other.__coords)
        end
   end
end

function Point:__eq(other)
   assert(self.__dim == other.__dim,
          "dimensions should be the same")
   local this = self.__coords
   local that = other.__coords
   for k, v in pairs(this)
   do
      if v ~= that[k] then return false end
   end
   return true
end

function Point:__tostring()
   return
      "(" .. table.concat(self.__coords, ", ") .. ")"
end

function Point:__le(other)
   assert(self.__dim == other.__dim,
          "dimensions should be the same")
   local this = self.__coords
   local that = other.__coords
   for k, v in pairs(this)
   do
      if v > that[k] then return false end
   end
   return true
end
   
function Point:__lt(other)
   assert(self.__dim == other.__dim,
          "dimensions should be the same")
   local this = self.__coords
   local that = other.__coords
   for k, v in pairs(this)
   do
      if v >= that[k] then return false end
   end
   return true
end
   
function Point:zeros(dim)
   local zeros = {}
   for i = 1, dim do zeros[i] = 0 end
   return Point:new(zeros)
end

function Point:ones(dim)
   local ones = {}
   for i = 1, dim do ones[i] = 1 end
   return Point:new(ones)
end

function Point:__add(other)
   assert(self.__dim == other.__dim,
          "dimensions should be the same")
   local result = self:clone()
   local this = result.__coords
   local that = other.__coords
   for k, v in pairs(this)
   do
      this[k] = v + that[k]
   end
   return result
end

function Point:__sub(other)
   assert(self.__dim == other.__dim,
          "dimensions should be the same")
   local result = self:clone()
   local this = result.__coords
   local that = other.__coords
   for k, v in pairs(this)
   do
      this[k] = v - that[k]
   end
   return result
end

function Point:__mul(other)
   assert(self.__dim == other.__dim,
          "dimensions should be the same")
   local result = self:clone()
   local this = result.__coords
   local that = other.__coords
   for k, v in pairs(this)
   do
      this[k] = v * that[k]
   end
   return result
end

function Point:__div(other)
   assert(self.__dim == other.__dim,
          "dimensions should be the same")
   local result = self:clone()
   local this = result.__coords
   local that = other.__coords
   for k, v in pairs(this)
   do
      this[k] = v / that[k]
   end
   return result
end

function Point:__unm()
   local result = self:clone()
   local this = result.__coords
   for k, v in pairs(this)
   do
      this[k] = -v
   end
   return result
end

function Point:dot(other)
   assert(other.__is_point)
   assert(self.__dim == other.__dim,
          "dimensions should be the same")
   local result = 0
   local this = self.__coords
   local that = other.__coords
   for k, v in pairs(this)
   do
      result = result + v * that[k]
   end
   return result
end

function Point:max(x, y)
   assert(x.__is_point and y.__is_point)
   assert(x.__dim == y.__dim,
          "dimensions should be the same")
   local coords = {}
   for k, v in pairs(x.__coords)
   do
      coords[k] = math.max(v, y.__coords[k])
   end
   return Point:new(coords)
end

function Point:min(x, y)
   assert(x.__is_point and y.__is_point)
   assert(x.__dim == y.__dim,
          "dimensions should be the same")
   local coords = {}
   for k, v in pairs(x.__coords)
   do
      coords[k] = math.min(v, y.__coords[k])
   end
   return Point:new(coords)
end

function Point:to_c_object()
   return LegionRuntime.HighLevel.Point.make(unpack(self.__coords))
end

function Point:recover_metatable(tbl)
   assert(tbl.__is_point)
   return Point:new(tbl.__coords)
end

-- Equivalent to LegionRuntime::Arrays::Rect
Rect = {}

function Rect:new(...)
   local vargs = {...}
   local rect = {}
   local low = {}
   local high = {}

   if #vargs == 1
   then
      local coords = vargs[1]
      assert(type(coords) == 'table')
      assert(#coords % 2 == 0)
      local low_arr = {}
      local high_arr = {}
      local dim = #coords / 2
      for i = 1, dim
      do
         low_arr[i] = coords[i]
         high_arr[i] = coords[i + dim]
      end
      low = Point:new(low_arr)
      high = Point:new(high_arr)
   else if #vargs == 2
        then
           low = vargs[1]:clone()
           high = vargs[2]:clone()
        end
   end

   assert(low.__is_point and high.__is_point,
          "constructor requires two points as arguments")
   assert(low.__dim == high.__dim,
          "two points should have the same dimension")
   rect.__low = low
   rect.__high = high
   rect.__dim = rect.__low.__dim
   rect.__is_rect = true
   setmetatable(rect, self)
   self.__index = self
   return rect
end

function Rect:clone(other)
   if self.__low and self.__high
   then return Rect:new(self.__low, self.__high)
   else if other.__low and other.__high
        then return Rect:new(other.__low, other.__high)
        end
   end
end

function Rect:__tostring()
   return
      "[" ..
      tostring(self.__low) .. ", " ..
      tostring(self.__high) .. "]"
end

function Rect:__eq(other)
   assert(self.__dim == other.__dim,
          "dimensions should be the same")
   return
      self.__low == other.__low and
      self.__high == other.__high
end

function Rect:overlaps(other)
   assert(other.__is_rect)
   assert(self.__dim == other.__dim,
          "dimensions should be the same")
   for i = 1, self.__dim
   do
      if self.__high[i] < other.__low[i] or
         self.__low[i] > other.__high[i]
      then return false end
   end
   return true
end

function Rect:contains(other)
   assert(other.__is_point or other.__is_rect)
   assert(self.__dim == other.__dim,
          "dimensions should be the same")
   if other.__is_rect
   then
      for i = 1, self.__dim
      do
         if self.__low[i] > other.__low[i] or
            self.__high[i] < other.__high[i]
         then return false end
      end
      return true
   else -- other.__is_point
      for i = 1, self.__dim
      do
         if other[i] < self.__low[i] or
            other[i] > self.__high[i]
         then return false end
      end
      return true      
   end
end

function Rect:volume()
   local v = 1
   for i = 1, self.__dim
   do
      v = v * (self.__high[i] - self.__low[i] + 1)
   end
   return v
end

function Rect:dim_size(dim)
   assert(dim >= 1 and dim <= self.__dim)
   return self.__high[dim] - self.__low[dim] + 1
end

function Rect:__mul(other)
   assert(self.__dim == other.__dim,
          "dimensions should be the same")
   return
      Rect:new(Point:max(self.__low, other.__low),
               Point:min(self.__high, other.__high))
end

function Rect:to_c_object()
   return
      LegionRuntime.HighLevel.Rect.make(self.__low:to_c_object(),
                                        self.__high:to_c_object())
end

function Rect:recover_metatable(tbl)
   assert(tbl.__is_rect)
   return Rect:new(Point:new(tbl.__low.__coords),
                   Point:new(tbl.__high.__coords))
end

Blockify = {}

function Blockify:new(block_size)
   local blockify = {}

   if block_size.__is_point
   then blockify.__block_size = block_size
   else blockify.__block_size = Point:new(block_size)
   end
   blockify.__is_blockify = true
   setmetatable(blockify, self)
   self.__index = self
   return blockify
end

function Blockify:image(point)
   assert(point.__is_point)
   local result = {}
   for i = 1, point.__dim
   do
      result[i] = point[i] / self.__block_size[i]
   end
   return Point:new(result)
end

function Blockify:image_convex(rect)
   assert(rect.__is_rect)
   return Rect:new(
      self:image(rect.__low),
      self:image(rect.__high))
end

function Blockify:preimage(point)
   assert(point.__is_point)
   local low = {}
   local high = {}
   for i = 1, point.__dim
   do
      low[i] = point[i] * self.__block_size[i]
      high[i] = low[i] + self.__block_size[i] - 1
   end
   return Rect:new(Point:new(low), Point:new(high))
end

function Blockify:__tostring()
   return "<<" .. tostring(self.__block_size) .. ">>"
end

function Blockify:to_c_object()
   return
      LegionRuntime.HighLevel.Blockify.make(unpack(self.__block_size.__coords))
end

function Blockify:recover_metatable(tbl)
   assert(tbl.__is_blockify)
   return Blockify:new(Point:recover_metatable(tbl.__block_size))
end

PointInRectIterator = {}

function PointInRectIterator:new(rect)
   local iterator = {}
   iterator.__rect = rect
   iterator.__point = rect.__low
   iterator.__dim = rect.__dim
   iterator.__has_next = rect.__low <= rect.__high
   setmetatable(iterator, self)
   self.__index = self
   return iterator
end

function PointInRectIterator:has_next()
   return self.__has_next
end

function PointInRectIterator:next()
   local point = self.__point

   self.__point = point:clone()
   local coords = self.__point.__coords
   for i = 1, self.__dim
   do
      coords[i] = coords[i] + 1
      if coords[i] <= self.__rect.__high[i]
      then
         return point
      end
      coords[i] = self.__rect.__low[i]
   end
   self.__has_next = false
   return point
end

Domain = {}

function Domain:new(arg)
   local domain = {}
   if type(arg) == 'table' and arg.__is_rect
   then -- if domain is being created with rect
      domain.__rect = arg
   else if type(arg) == 'table' and arg.__is_index_space
        then
           domain.__ispace = arg
   -- else if type(arg) == 'userdata'
   --      then -- if domain is being created with indexspace
   --         domain.__ispace = arg
        else
           assert(false, "not supported")
        end
   end
   domain.__is_domain = true
   setmetatable(domain, self)
   self.__index = self
   return domain
end

function Domain:wrap(domain_from_c)
   local domain = {}
   domain.__is_domain = true
   domain.__domain_from_c = domain_from_c
   setmetatable(domain, self)
   self.__index = self
   return domain
end

function Domain:to_c_object()
   if self.__rect
   then
      local rect_in_c = self.__rect:to_c_object()
      return LegionRuntime.LowLevel.Domain.from_rect(rect_in_c)
   else if self.__ispace
        then
           local ispace_in_c = make_c_index_space(self.__ispace)
           return LegionRuntime.LowLevel.Domain(ispace_in_c)
        else if self.__domain_from_c
             then return self.__domain_from_c
             else assert(false, "not supported")
             end
        end
   end
end

DomainPoint = {}

function DomainPoint:new(arg)
   local point = {}
   if type(arg) == 'number'
   then
      point.__idx = arg
      point.__dim = 0
   else if type(arg) == 'table' and arg.__is_point
        then
           point.__point = arg
           point.__dim = arg.__dim
        else
           assert(false, "not supported")
        end
   end
   point.__is_domainpoint = true
   setmetatable(point, self)
   self.__index = self
   return point
end

function DomainPoint:get_point()
   assert(self.__point,
          "DomainPoint was not initialized with Point")
   return self.__point
end

function DomainPoint:get_index()
   assert(self.__idx,
          "DomainPoint was not initialized with 0-dimensional index")
   return self.__idx
end

function DomainPoint:to_c_object()
   if self.__idx
   then return LegionRuntime.LowLevel.DomainPoint(self.__idx)
   else
      local point_in_c = self.__point:to_c_object()
      return
         LegionRuntime.LowLevel.DomainPoint.from_point(point_in_c)
   end
end

function recover_metatables(args)
   if type(args) == 'table'
   then
      for k, v in pairs(args)
      do
         if type(v) == 'table'
         then
            if v.__is_blockify
            then
               args[k] = Blockify:recover_metatable(v)
            else if v.__is_rect
                 then
                    args[k] = Rect:recover_metatable(v)
                 else if v.__is_point
                      then
                         args[k] = Point:recover_metatable(v)
                      end
                 end
            end
         end
      end
   end
   return args
end

