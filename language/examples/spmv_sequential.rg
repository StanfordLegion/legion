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

-- SpMV for CSR matrices
--
-- This code is designed to demonstrate the auto-parallelzier's capability
-- of handling sparse matrix code and and is not for any performance comparisons.

import "regent"

local c = regentlib.c
local VAL_TYPE = double

fspace csr
{
  val : VAL_TYPE;
  ind : int1d;
}

struct Config
{
  steps       : int;
  prune       : int;
  parallelism : int;
  size        : int64;
  nnz_per_row : int32;
  variance    : bool;
}

terra parse_config()
  var config = Config { 1, 0, 2, 10, 3, false }
  var args = c.legion_runtime_get_input_args()
  var i = 0
  while i < args.argc do
    if c.strcmp(args.argv[i], "-steps") == 0 then
      i = i + 1
      config.steps = c.atoi(args.argv[i])
    elseif c.strcmp(args.argv[i], "-prune") == 0 then
      i = i + 1
      config.prune = c.atoi(args.argv[i])
    elseif c.strcmp(args.argv[i], "-p") == 0 then
      i = i + 1
      config.parallelism = c.atoi(args.argv[i])
    elseif c.strcmp(args.argv[i], "-size") == 0 then
      i = i + 1
      config.size = c.atol(args.argv[i])
    elseif c.strcmp(args.argv[i], "-nnz") == 0 then
      i = i + 1
      config.nnz_per_row = c.atoi(args.argv[i])
    elseif c.strcmp(args.argv[i], "-var") == 0 then
      config.variance = true
    end
    i = i + 1
  end
  return config
end

parse_config.replicable = true

__demand(__parallel)
task init(x : region(ispace(int1d), VAL_TYPE),
          y : region(ispace(int1d), VAL_TYPE),
          mat : region(ispace(int1d), csr),
          ranges : region(ispace(int1d), rect1d),
          config : Config)
where
  reads writes(x, y, mat, ranges)
do
  var nnz_per_row = config.nnz_per_row
  for e in x do @e = [VAL_TYPE](e) end
  for e in y do @e = 0 end
  for e in ranges do
    @e = rect1d { int64(e) * nnz_per_row, (int64(e) + 1) * nnz_per_row - 1 }
  end

  for e in mat do
    e.val = [VAL_TYPE](e)
    e.ind = int64(e) / nnz_per_row
  end
end

__demand(__parallel, __cuda)
task spmv(x : region(ispace(int1d), VAL_TYPE),
          y : region(ispace(int1d), VAL_TYPE),
          mat : region(ispace(int1d), csr),
          ranges : region(ispace(int1d), rect1d))
where
  reads writes(y),
  reads(x, mat, ranges)
do
  for i in ranges do
    var range = ranges[i]
    for k in range do
      y[i] += mat[k].val * x[mat[k].ind]
    end
  end
end

task print_summary(time : double, config : Config)
  c.printf("ELAPSED TIME = %7.3f s\n", time)
end

__demand(__replicable, __inner)
task main()
  var config = parse_config()

  var x = region(ispace(int1d, config.size), VAL_TYPE)
  var y = region(ispace(int1d, config.size), VAL_TYPE)
  var ranges = region(ispace(int1d, config.size), rect1d)

  var mat_size : int64 = config.size * config.nnz_per_row
  var mat = region(ispace(int1d, mat_size), csr)

  var cs = ispace(int1d, config.parallelism)

  var ts_start = c.legion_get_current_time_in_micros()
  var ts_end = ts_start
  var prune = config.prune
  var num_loops = config.steps + 2 * config.prune

  __parallelize_with cs do
    init(x, y, mat, ranges, config)
  end

  __parallelize_with cs do
    for i = 0, num_loops do
      if i == prune then
        __fence(__execution, __block)
        ts_start = c.legion_get_current_time_in_micros()
      end
      spmv(x, y, mat, ranges)
      if i == num_loops - prune - 1 then
        __fence(__execution, __block)
        ts_end = c.legion_get_current_time_in_micros()
      end
    end
  end
  var time = 1e-6 * (ts_end - ts_start)
  print_summary(time, config)
end

regentlib.start(main)
