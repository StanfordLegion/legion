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

TOP_LEVEL_TASK_ID = 100
TASKID_MAIN = 200
TASKID_INIT_VECTORS = 300
TASKID_ADD_VECTORS = 400

DEFAULT_NUM_BLOCKS = 64
BLOCK_SIZE = 256

FIELDID_V = 0

Entry = {
   v = PrimType.double
}

function top_level_task(binding, regions, args)
   print("top_level_task")

   local main_args = {}
   main_args.num_blocks = DEFAULT_NUM_BLOCKS;
   for k, v in pairs(args)
   do
      if v == '-blocks' then
         main_args.num_blocks = tonumber(args[k + 1])
      end
   end
   print("saxpy: num elems = " .. main_args.num_blocks * BLOCK_SIZE)
   main_args.num_elems = main_args.num_blocks * BLOCK_SIZE

   local points = Rect:new { 0, main_args.num_elems - 1 }
   main_args.ispace = binding:create_index_space(Domain:new(points))
   main_args.blkify = Blockify:new { BLOCK_SIZE }
   main_args.ipart = binding:create_index_partition(main_args.ispace, main_args.blkify)
   main_args.fspace = binding:create_field_space()
   local fid = binding:allocate_field(main_args.fspace, Entry)

   main_args.r_x = binding:create_logical_region(main_args.ispace, main_args.fspace)
   main_args.r_y = binding:create_logical_region(main_args.ispace, main_args.fspace)
   main_args.r_z = binding:create_logical_region(main_args.ispace, main_args.fspace)
   main_args.fid = fid

   local main_launcher = TaskLauncher:new(TASKID_MAIN, main_args)
   main_launcher:add_region_requirements(
      RegionRequirement:new { region = main_args.r_x }:add_field(fid),
      RegionRequirement:new { region = main_args.r_y }:add_field(fid),
      RegionRequirement:new { region = main_args.r_z }:add_field(fid))

   local f = binding:execute_task(main_launcher)
   f:get_void_result()

   binding:destroy_logical_region(main_args.r_x)
   binding:destroy_logical_region(main_args.r_y)
   binding:destroy_logical_region(main_args.r_z)
   binding:destroy_index_space(main_args.ispace)
   binding:destroy_field_space(main_args.fspace)
end

function main_task(binding, regions, args)
   math.randomseed(os.time())
   args.alpha = math.random()
   print("alpha: " .. args.alpha)

   binding:unmap_all_regions()

   -- Prepare arguments
   local global_arg = args
   local local_args = {}
   local domain = binding:get_index_partition_color_space(args.ipart)

   local init_launcher = IndexLauncher:new(TASKID_INIT_VECTORS, domain,
                                           global_arg, local_args)
   local add_launcher = IndexLauncher:new(TASKID_ADD_VECTORS, domain,
                                           global_arg, local_args)
   
   local p_x = binding:get_logical_partition(args.r_x, args.ipart)
   local p_y = binding:get_logical_partition(args.r_y, args.ipart)
   local p_z = binding:get_logical_partition(args.r_z, args.ipart)

   init_launcher:add_region_requirements(
      RegionRequirement:new { part = p_x,
                              parent = args.r_x,
                              priv = PrivilegeMode.WRITE_ONLY }
         :add_field(args.fid),
      RegionRequirement:new { part = p_y,
                              parent = args.r_y,
                              priv = PrivilegeMode.WRITE_ONLY }
         :add_field(args.fid)

   )

   add_launcher:add_region_requirements(
      RegionRequirement:new { part = p_x,
                              parent = args.r_x,
                              priv = PrivilegeMode.READ_ONLY }
         :add_field(args.fid),
      RegionRequirement:new { part = p_y,
                              parent = args.r_y,
                              priv = PrivilegeMode.READ_ONLY }
         :add_field(args.fid),
      RegionRequirement:new { part = p_z,
                              parent = args.r_z,
                              priv = PrivilegeMode.WRITE_ONLY }
         :add_field(args.fid)
   )

   print("STARTING MAIN SIMULATION LOOP")

   -- Launch tasks
   local init_f = binding:execute_index_space(init_launcher)
   init_f:wait_all_results()
   
   local add_f = binding:execute_index_space(add_launcher)
   add_f:wait_all_results()

   print("VERIFYING RESULTS")
   
   local r_x = binding:map_region(RegionRequirement:new { region = args.r_x }:add_field(args.fid))
   local r_y = binding:map_region(RegionRequirement:new { region = args.r_y }:add_field(args.fid))
   local r_z = binding:map_region(RegionRequirement:new { region = args.r_z }:add_field(args.fid))

   r_x:wait_until_valid()
   r_y:wait_until_valid()
   r_z:wait_until_valid()

   local a_x = r_x:get_lua_accessor(args.fid)
   local a_y = r_y:get_lua_accessor(args.fid)
   local a_z = r_z:get_lua_accessor(args.fid)
   

   for i = 0, args.num_blocks - 1, 1
   do
      local point = Point:new {i}
      local x = a_x:read_at_point(point)
      local y = a_y:read_at_point(point)
      local z = a_z:read_at_point(point)

      if not (args.alpha * x.v + y.v == z.v) then
         print("Verification failed : " .. args.alpha ..
                  " * " .. x.v .. " + " .. y.v .. " != " .. z.v)
      end
   end

   print("Done")

   binding:unmap_region(r_x)
   binding:unmap_region(r_y)
   binding:unmap_region(r_z)
end

function init_vectors_task(binding, regions,
                           global_arg, local_args, point)
   math.randomseed(os.time())

   -- print("init_vectors")
   
   local r_x = regions[0]
   local r_y = regions[1]
   local preimage = global_arg.blkify:preimage(point:get_point())

   local a_x = r_x:get_lua_accessor(global_arg.fid)
   local a_y = r_y:get_lua_accessor(global_arg.fid)

   local itr = PointInRectIterator:new(preimage)

   while(itr:has_next())
   do
      local point = itr:next()
      local v = math.random()
      a_x:write_at_point(point, { v = v })
      v = math.random()
      a_y:write_at_point(point, { v = v })
   end
end


function add_vectors_task(binding, regions,
                          global_arg, local_args, point)
   -- print("add_vectors")

   local r_x = regions[0]
   local r_y = regions[1]
   local r_z = regions[2]
   local preimage = global_arg.blkify:preimage(point:get_point())

   local a_x = r_x:get_lua_accessor(global_arg.fid)
   local a_y = r_y:get_lua_accessor(global_arg.fid)
   local a_z = r_z:get_lua_accessor(global_arg.fid)
   
   local itr = PointInRectIterator:new(preimage)

   while(itr:has_next())
   do
      local point = itr:next()
      local x = a_x:read_at_point(point)
      local y = a_y:read_at_point(point)
      local saxpy = global_arg.alpha * x.v + y.v
      a_z:write_at_point(point, { v = saxpy })
   end

end
