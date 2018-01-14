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
local std = terralib.includec("stdlib.h")
rawset(_G, "drand48", std.drand48)

-- In this section we use a sequential
-- implementation of daxpy to show how
-- to create physical instances of logical
-- reigons.  In later sections we will
-- show how to extend this daxpy example
-- so that it will run with sub-tasks
-- and also run in parallel.

TOP_LEVEL_TASK_ID = 0

FID_X = 0
FID_Y = 1
FID_Z = 2

function top_level_task(task, regions, ctx, runtime)
  local num_elements = 1024
  -- See if we have any command line arguments to parse
  local command_args = legion:get_input_args()
  for i = 0, #command_args do
    if command_args[i] == '-n' then
      num_elements = tonumber(command_args[i+1])
    end
  end
  printf("Running daxpy for %d elements...", num_elements)

  -- We'll create two logical regions with a common index space
  -- for storing our inputs and outputs.  The input region will
  -- have two fields for storing the 'x' and 'y' fields of the
  -- daxpy computation, and the output region will have a single
  -- field 'z' for storing the result.
  local elem_rect = Rect:new(Point:new{0},Point:new{num_elements-1})
  local is = runtime:create_index_space(ctx, Domain:from_rect(elem_rect))
  local input_fs = runtime:create_field_space(ctx)
  do
    local allocator =
      runtime:create_field_allocator(ctx, input_fs)
    allocator:allocate_field(sizeof(double),FID_X)
    allocator:allocate_field(sizeof(double),FID_Y)
  end
  local output_fs = runtime:create_field_space(ctx)
  do
    local allocator =
      runtime:create_field_allocator(ctx, output_fs)
    allocator:allocate_field(sizeof(double),FID_Z)
  end
  local input_lr = runtime:create_logical_region(ctx, is, input_fs)
  local output_lr = runtime:create_logical_region(ctx, is, output_fs)

  -- Now that we have our logical regions we want to instantiate physical
  -- instances of these regions which we can use for storing data.  One way
  -- of creating a physical instance of a logical region is via an inline
  -- mapping operation.  (We'll discuss the other way of creating physical
  -- instances in the next example.)  Inline mappings map a physical instance
  -- of logical region inside of this task's context.  This will give the
  -- task an up-to-date copy of the data stored in these logical regions.
  -- In this particular daxpy example, the data has yet to be initialized so
  -- really this just creates un-initialized physical regions for the
  -- application to use.
  --
  -- To perform an inline mapping we use an InlineLauncher object which is
  -- similar to other launcher objects shown earlier.  The argument passed
  -- to this launcher is a 'RegionRequirement' which is used to specify which
  -- logical region we should be mapping as well as with what privileges
  -- and coherence.  In this example we are mapping the input_lr logical
  -- region with READ-WRITE privilege and EXCLUSIVE coherence.  We'll see
  -- examples of other privileges in later examples.  If you're interested
  -- in learning about relaxed coherence modes we refer you to our OOPSLA paper.
  -- The last argument in the RegionRequirement is the logical region for
  -- which the enclosing task has privileges which in this case is the
  -- same input_lr logical region.  We'll discuss restrictions on privileges
  -- more in the next example.
  local req = RegionRequirement:new(input_lr, legion.READ_WRITE,
                                    legion.EXCLUSIVE, input_lr)
  -- We also need to specify which fields we plan to access in our
  -- RegionRequirement.  To do this we invoke the 'add_field' method
  -- on the RegionRequirement.
  req:add_field(FID_X)
  req:add_field(FID_Y)
  local input_launcher = InlineLauncher:new(req)

  -- Once we have set up our launcher, we as the runtime to map a physical
  -- region instance of our requested logical region with the given
  -- privileges and coherence.  This returns a PhysicalRegion object
  -- which is handle to the physical instance which contains data
  -- for the logical region.  In keeping with Legion's deferred execution
  -- model the 'map_region' call is asynchronous.  This allows the
  -- application to issue many of these operations in flight and to
  -- perform other useful work while waiting for the region to be ready.
  --
  -- One common criticism about Legion applications is that there exists
  -- a dichotomy between logical and physical regions.  Programmers
  -- are explicitly required to keep track of both kinds of regions and
  -- know when and how to use them.  If you feel this way as well, we
  -- encourage you to try out our Legion compiler in which this
  -- dichotomy does not exist.  There are simply regions and the compiler
  -- automatically manages the logical and physical nature of them
  -- in a way that is analogous to how compilers manage the mapping
  -- between variables and architectural registers.  This runtime API
  -- is designed to be expressive for all Legion programs and is not
  -- necessarily designed for programmer productivity.
  local input_region = runtime:map_region(ctx, input_launcher)
  -- The application can either poll a physical region to see when it
  -- contains valid data for the logical region using the 'is_valid'
  -- method or it can explicitly wait using the 'wait_until_valid'
  -- method.  Just like waiting on a future, if the region is not ready
  -- this task is pre-empted and other tasks may run while waiting
  -- for the region to be ready.  Note that an application does not
  -- need to explicitly wait for the physical region to be ready before
  -- using it, but any call to get an accessor (described next) on
  -- phsyical region that does not yet have valid data will implicitly
  -- call 'wait_until_valid' to guarantee correct execution by ensuring
  -- the application only can access the physical instance once the
  -- data is valid.
  input_region:wait_until_valid()

  -- To actually access data within a physical region, an application
  -- must create a RegionAccessor.  RegionAccessors provide a level
  -- of indirection between a physical instance and the application
  -- which is necessary for supporting general task code that is
  -- independent of data layout.  Note that each accessor must specify
  -- which field it is accessing and then pass the types to cast the
  -- field that is being accessed.
  local acc_x = input_region:get_field_accessor(FID_X):typeify(double)
  local acc_y = input_region:get_field_accessor(FID_Y):typeify(double)

  -- We initialize our regions with some random data.  To iterate
  -- over all the points in each of the regions we use an iterator
  -- which can be used to enumerate all the points within an array.
  -- The points in the array (the 'p' field in the iterator) are
  -- used to access different locations in each of the physical
  -- instances.
  local pir = GenericPointInRectIterator:new(elem_rect);
  while pir:has_next() do
    acc_x:write(DomainPoint:from_point(pir.p), drand48())
    acc_y:write(DomainPoint:from_point(pir.p), drand48())
    pir:next()
  end

  -- Now we map our output region so we can do the actual computation.
  -- We use another inline launcher with a different RegionRequirement
  -- that specifies our privilege to be WRITE-DISCARD.  WRITE-DISCARD
  -- says that we can discard any data presently residing the region
  -- because we are going to overwrite it.
  local output_launcher =
    InlineLauncher:new(
      RegionRequirement:new(output_lr, legion.WRITE_DISCARD,
                            legion.EXCLUSIVE, output_lr))
  output_launcher.requirement:add_field(FID_Z)

  -- Map the region
  local output_region = runtime:map_region(ctx, output_launcher)

  -- Note that this accessor invokes the implicit 'wait_until_valid'
  -- call described earlier.
  output_region:wait_until_valid()

  local acc_z = output_region:get_field_accessor(FID_Z):typeify(double)

  local alpha = drand48()
  printf("Running daxpy computation with alpha %.8g...", alpha);
  -- Iterate over our points and perform the daxpy computation.  Note
  -- we can use the same iterator because both the input and output
  -- regions were created using the same index space.
  pir = GenericPointInRectIterator:new(elem_rect);
  while pir:has_next() do
    local value =
      alpha * acc_x:read(DomainPoint:from_point(pir.p)) +
              acc_y:read(DomainPoint:from_point(pir.p))
    acc_z:write(DomainPoint:from_point(pir.p), value)
    pir:next()
  end
  print("Done!")

  -- In some cases it may be necessary to unmap regions and then
  -- remap them.  We'll give a compelling example of this in the
  -- next example.   In this case we'll remap the output region
  -- with READ-ONLY privileges to check the output result.
  -- We really could have done this directly since WRITE-DISCARD
  -- privileges are equivalent to READ-WRITE privileges in terms
  -- of allowing reads and writes, but we'll explicitly unmap
  -- and then reamp.  Unmapping is done with the unamp call.
  -- After this call the physical region no longer contains valid
  -- data and all accessors from the physical region are invalidated.
  runtime:unmap_region(ctx, output_region)

  -- We can then remap the region.  Note if we wanted to remap
  -- with the same privileges we could have used the 'remap_region'
  -- call.  However, we want different privileges so we update
  -- the launcher and then remap the region.  The 'remap_region'
  -- call also guarantees that we would get the same physical
  -- instance.  By calling 'map_region' again, we have no such
  -- guarantee.  We may get the same physical instance or a new
  -- one.  The orthogonality of correctness from mapping decisions
  -- ensures that we will access the same data regardless.
  output_launcher.requirement.privilege = legion.READ_ONLY
  output_region = runtime:map_region(ctx, output_launcher)

  -- Since we may have received a new physical instance we need
  -- to update our accessor as well.  Again this implicitly calls
  -- 'wait_until_valid' to ensure we have valid data.
  acc_z = output_region:get_field_accessor(FID_Z):typeify(double)

  print("Checking results...")
  local all_passed = true

  -- Check our results are the same
  pir = GenericPointInRectIterator:new(elem_rect);
  while pir:has_next() do
    local expected =
      alpha * acc_x:read(DomainPoint:from_point(pir.p)) +
              acc_y:read(DomainPoint:from_point(pir.p))
    local received =
      acc_z:read(DomainPoint:from_point(pir.p))
    -- Probably shouldn't check for floating point equivalence but
    -- the order of operations are the same should they should
    -- be bitwise equal.
    if expected ~= received then
      all_passed = false
    end
    pir:next()
  end

  if all_passed then
    print("SUCCESS!")
  else
    print("FAILURE!")
  end

  runtime:unmap_region(ctx, input_region)
  runtime:unmap_region(ctx, output_region)

  runtime:destroy_logical_region(ctx, input_lr)
  runtime:destroy_logical_region(ctx, output_lr)
  runtime:destroy_field_space(ctx, input_fs)
  runtime:destroy_field_space(ctx, output_fs)
  runtime:destroy_index_space(ctx, is)
end

function legion_main(arg)
  legion:set_top_level_task_id(TOP_LEVEL_TASK_ID)
  legion:register_lua_task_void(
    "top_level_task",
    TOP_LEVEL_TASK_ID, legion.LOC_PROC,
    true,  -- single
    false -- index
  )
  legion:start(arg)
end

if rawget(_G, "arg") then
  legion_main(arg)
end
