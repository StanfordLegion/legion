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
   package.loadlib('libbinding.so', 'init')()
end

hl = LegionRuntime.HighLevel
ll = LegionRuntime.LowLevel
ColoredPoints = Set

-- field allocation

PrimType = {
   float = 0,
   double = 1,
   int = 2
}

PrimTypeSize = {}

PrimTypeSize[PrimType.float] = 4
PrimTypeSize[PrimType.double] = 8
PrimTypeSize[PrimType.int] = 4

ArrayType = {}

function ArrayType.make(ty, size)
   return { elem_type = ty, size = size }
end

function allocate_field(struct, ctx, runtime, fspace)
   local field_id = {}
   local allocator = runtime:create_field_allocator(ctx, fspace)

   for k, v in pairs(struct)
   do
      if type(v) == 'number' then
         fid = allocator:allocate_field(PrimTypeSize[v], -1)
         field_id[k] = { ty = v, fid = fid }
      else
         if type(v) == 'table' then
            local elem_type = v.elem_type
            local size = v.size
            assert(type(elem_type) == 'number')
            fid = allocator:allocate_field(PrimTypeSize[elem_type] * size, -1)
            field_id[k] = { ty = v, fid = fid }
         else
            error('invalid type of value in struct descriptor')
         end
      end
   end

   return field_id
end

function add_field(req, field_id, is_instance)
   if type(field_id) == 'table' then
      for k, v in pairs(field_id)
      do
         -- print(k .. " : " .. v.fid)
         assert(type(v) == 'table')
         req:add_field(v.fid, is_instance)
      end
   else
      error('invalid field id')
   end
end

function get_accessor(region, field_id)
   local accessors = {}

   for k, v in pairs(field_id)
   do
      -- print(k .. " : " .. v.fid)
      local generic_accessor = region:get_field_accessor(v.fid)
      local get_prim_accessor = function(accessor, ty)
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

      local accessor = nil
      if type(v.ty) == 'number' then
         accessor = get_prim_accessor(generic_accessor, v.ty)
      else
         if type(v.ty) == 'table' then
            local array_accessor = {}
            array_accessor.elem_type = v.ty.elem_type
            array_accessor.size = v.ty.size
            array_accessor.parent = generic_accessor

            array_accessor.read = function (self, ptr)
               local array = {}

               for i = 0, self.size - 1
               do
                  local offset = i * PrimTypeSize[self.elem_type]
                  local elem_accessor =
                     self.parent:get_untyped_field_accessor(offset, 0)
                  array[i] = get_prim_accessor(elem_accessor, self.elem_type):read(ptr)
               end

               return array
            end

            array_accessor.write = function (self, ptr, array)
               for i = 0, self.size - 1
               do
                  local offset = i * PrimTypeSize[self.elem_type]
                  local elem_accessor =
                     -- self.parent
                     self.parent:get_untyped_field_accessor(offset, 0)
                  local typed_elem_accessor =
                     get_prim_accessor(elem_accessor, self.elem_type)
                  typed_elem_accessor:write(ptr, array[i])
               end
            end

            local mt = {}
            mt.__index = function (self, idx)
               if self[idx] == nil
               then
                  assert(type(idx) == "number")
                  assert(0 <= idx and idx < self.size)
                  
                  local offset = idx * PrimTypeSize[self.elem_type]
                  local elem_accessor =
                     self.parent:get_untyped_field_accesor(offset, 0)
                  self[idx] = get_prim_accessor(elem_accessor, self.elem_type)
                  return elem_accessor
               else
                  return self[idx]
               end
            end
            mt.__newindex = function (self, idx, val)
               error("array_accessor is read-only")
            end

            setmetatable(array_accessor, mt)

            accessor = array_accessor
         end
      end

      accessors[k] = { ty = v.ty,
                       fid = v.fid,
                       accessor = accessor }
   end

   accessors.read = function (self, ptr)
      local field = {}
      for k, v in pairs(self)
      do
         if not (k == "read" or k == "write") then
            field[k] = self[k].accessor:read(ptr)
         end
      end
      return field
   end
   accessors.write = function (self, ptr, field)
      for k, v in pairs(field)
      do
         self[k].accessor:write(ptr, v)
      end
   end
   
   return accessors
end


-- assume that map is a table from integers to sets of values of type ptr_t.
function to_coloring(map)
   local coloring = hl.Coloring()

   for k, set in pairs(map)
   do
      if not (type(k) == "number") then
         error("coloring map should be indexed by numbers")
      end
      local elems = ColoredPoints.to_list(set)
      local tmp =  coloring:at(k) -- just intialize the element for the index
      for _, v in pairs(elems)
      do
         coloring:at(k).points:insert(ptr_t(v))
      end
   end

   return coloring
end
