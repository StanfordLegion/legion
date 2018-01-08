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

--
-- To illustrate task launches and futures in Legion
-- we implement a program to compute the first N
-- Fibonacci numbers.  While we note that this is not
-- the fastest way to compute Fibonacci numbers, it
-- is designed to showcase the functional nature of
-- Legion tasks and futures.
--

TOP_LEVEL_TASK_ID = 0
FIBONACCI_TASK_ID = 1
SUM_TASK_ID = 2

function top_level_task(task, regions, ctx, runtime)
  local num_fibonacci = 7
  -- The command line arguments to a Legion application are
  -- available through the runtime 'get_input_args' call.  We'll
  -- use this to get the number of Fibonacci numbers to compute.
  local command_args = legion:get_input_args()
  if #command_args >= 1 then
    local i = 1
    while i <= #command_args do
      local arg = command_args[i]
      if string.sub(arg, 1, 1) == "-" then
        i = i + 2
      else
        num_fibonacci = tonumber(arg)
        break
      end
    end
    assert(num_fibonacci >= 0)
  end
  print("Computing the first " .. num_fibonacci .. " Fibonacci numbers...")

  -- This is a table which we'll use to store the future
  -- results of all the tasks that we launch.  The goal here
  -- is to launch all of our tasks up front to get them in
  -- flight before waiting on a future value.  This exposes
  -- as many tasks as possible to the Legion runtime to
  -- maximize performance.
  local fib_results = {}
  for i = 0, num_fibonacci - 1 do
    -- All Legion tasks are spawned from a launcher object.  A
    -- 'TaskLauncher' is a struct used for specifying the arguments
    -- necessary for launching a task.  Launchers contain many
    -- fields which we will explore throughout the examples.  Here
    -- we look at the first two arguments: the ID of the kind of
    -- task to launch and a 'TaskArgument'.  The ID of the task
    -- must correspond to one of the IDs registered with the Legion
    -- runtime before the application began.  A 'TaskArgument' internally
    -- allocates a buffer of the specified type and stores the value
    -- in the buffer.  It is important to note that this buffer is
    -- not actually copied until 'execute_task' is called.  The buffer
    -- would remain live until the launcher goes out of scope.
    local arg = TaskArgument:new(i, int)
    local launcher = TaskLauncher:new(FIBONACCI_TASK_ID, arg)
    -- To launch a task, a TaskLauncher object is passed to the runtime
    -- along with the context.  Legion tasks are asynchronous which means
    -- that this call returns immediately and returns a future value which
    -- we store in our table of future results.  Note that launchers can
    -- be reused to launch as many tasks as desired, and can be modified
    -- immediately after the 'execute_task' call returns.
    fib_results[i] = runtime:execute_task(ctx, launcher)
  end

  -- Print out our results
  for i = 0, num_fibonacci - 1 do
    -- One way to use a future is to explicitly ask for its value using
    -- the 'get_result' method.  This is a blocking call which will cause
    -- this task (the top-level task) to pause until the sub-task which
    -- is generating the future returns.  Note that waiting on a future
    -- that is not ready blocks this task, but does not block the processor
    -- on which the task is running.  If additional tasks have been mapped
    -- onto this processor and they are ready to execute, then they will
    -- begin running as soon as the call to 'get_result' is made.
    --
    -- The 'get_result' method takes the type of the return
    -- value which tells the Legion runtime how to interpret the bits
    -- being returned.
    local result = fib_results[i]:get_result(int)
    print("Fibonacci(" .. i .. ") = " .. result)
  end
end

function fibonacci_task(task, regions, ctx, runtime)
  -- The 'Task' object provides the 'get_args' method to retrieve
  -- the 'TaskArgument' value. As to retrieve the result from a
  -- 'Future' object, we should pass the type of the argument.
  local fib_num = task:get_args(int)

  -- Fibonacci base cases
  -- Note that tasks return values the same as Lua functions.
  -- If a task is running remotely from its parent task then
  -- Legion automatically packages up the result and returns
  -- it to the origin location.
  if fib_num == 0 then
    return 0
  end
  if fib_num == 1 then
    return 1
  end

  -- Launch fib-1
  local fib1 = fib_num - 1
  local arg1 = TaskArgument:new(fib1, int)
  local t1 = TaskLauncher:new(FIBONACCI_TASK_ID, arg1)
  local f1 = runtime:execute_task(ctx, t1)

  -- Launch fib-2
  local fib2 = fib_num - 2
  local arg2 = TaskArgument:new(fib2, int)
  local t2 = TaskLauncher:new(FIBONACCI_TASK_ID, arg2)
  local f2 = runtime:execute_task(ctx, t2)

  -- Here will illustrate a non-blocking way of using a future.
  -- Rather than waiting for the values and passing the results
  -- directly to the summation task, we instead pass the futures
  -- through the TaskLauncher object.  Legion then will
  -- ensure that the sum task does not begin until both futures
  -- are ready and that the future values are available wherever
  -- the sum task is run (even if it is run remotely).  Futures
  -- should NEVER be passed through a TaskArgument.
  local sum = TaskLauncher:new(SUM_TASK_ID, nil)
  sum:add_future(f1)
  sum:add_future(f2)
  local result = runtime:execute_task(ctx, sum)

  -- Our API does not permit returning Futures as the result of
  -- a task.  Any attempt to do so will result in a failed static
  -- assertion at compile-time.  In general, waiting for one or
  -- more futures at the end of a task is inexpensive since we
  -- have already exposed the available sub-tasks for execution
  -- to the Legion runtime so we can extract as much task-level
  -- parallelism as possible from the application.
  local value = result:get_result(int)
  return value
end

function sum_task(task, regions, ctx, runtime)
  -- Note that even though it looks like we are performing
  -- blocking calls to get these future results, the
  -- Legion runtime is smart enough to not run this task
  -- until all the future values passed through the
  -- task launcher have completed.
  local f1 = task.futures[0]
  local r1 = f1:get_result(int)
  local f2 = task.futures[1]
  local r2 = f2:get_result(int)

  return (r1 + r2)
end

function legion_main(arg)
  legion:set_top_level_task_id(TOP_LEVEL_TASK_ID)
  legion:register_lua_task_void("top_level_task",
    TOP_LEVEL_TASK_ID, legion.LOC_PROC, true,  false)
  -- Note that tasks which return values must pass the type of
  -- the return argument as the first argument.
  legion:register_lua_task(int, "fibonacci_task",
    FIBONACCI_TASK_ID, legion.LOC_PROC, true,  false)
  -- The sum-task has a very special property which is that it is
  -- guaranteed never to make any runtime calls.  We call these
  -- kinds of tasks "leaf" tasks and tell the runtime system
  -- about them using a 'TaskConfigOptions' object.  Being
  -- a leaf task allows the runtime to perform significant
  -- optimizations that minimize the overhead of leaf task
  -- execution.  Note that we also tell the runtime to
  -- automatically generate the variant ID for this task
  -- with the 'AUTO_GENERATE_ID' argument.
  legion:register_lua_task(int, "sum_task",
    SUM_TASK_ID, legion.LOC_PROC, true,  false,
    legion.AUTO_GENERATE_ID, TaskConfigOptions:new(true))
  legion:start(arg)
end

if rawget(_G, "arg") then
  legion_main(arg)
end
