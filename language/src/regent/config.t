-- Copyright 2022 Stanford University
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

-- Regent Configuration and Command Line Parsing

local common_config = require("common/config")
local data = require("common/data")

local config = {}

local expect_vars = terralib.newlist({"TERRA_PATH", "INCLUDE_PATH", "LG_RT_DIR", "USE_CMAKE", "USE_RDIR"})
if os.getenv("USE_CMAKE") == "1" then
  expect_vars:insert("CMAKE_BUILD_DIR")
end
if os.execute("bash -c \"[ `uname` == 'Darwin' ]\"") == 0 then
  expect_vars:insert("DYLD_LIBRARY_PATH")
else
  expect_vars:insert("LD_LIBRARY_PATH")
end
for _, expect_var in ipairs(expect_vars) do
  if os.getenv(expect_var) == nil then
    print("ERROR: Regent expects " .. expect_var .. " to be set, but it appears to be missing.")
    print("Did you configure LAUNCHER to pass through the right environment variables?")
    print()
    print("The following variables must be configured to pass through the LAUNCHER command.")
    for _, v in ipairs(expect_vars) do
      print("    " .. v)
    end
    os.exit(1)
  end
end

config.UNSPECIFIED = -1

local default_options = {
  -- Main user-facing correctness flags:
  ["bounds-checks"] = false,
  ["bounds-checks-targets"] = ".*",

  -- Main user-facing optimization flags:
  ["index-launch"] = true,
  ["inline"] = true,
  ["future"] = true,
  ["predicate"] = true,
  ["predicate-unroll"] = 2, -- Iterations to unroll predicated loops.
  ["leaf"] = true,
  ["inner"] = true,
  ["idempotent"] = true,
  ["replicable"] = true,
  ["mapping"] = true,
  ["openmp"] = config.UNSPECIFIED,
  ["openmp-offline"] = not data.is_luajit(),
  ["openmp-strict"] = false,
  ["skip-empty-tasks"] = false,
  ["vectorize"] = true,
  ["offline"] = not data.is_luajit(),
  ["separate"] = false,

  -- Code generation flags:
  ["fast-math"] = config.UNSPECIFIED,

  -- GPU optimization flags:
  ["gpu"] = "unspecified",
  ["gpu-offline"] = config.UNSPECIFIED,
  ["gpu-arch"] = os.getenv("GPU_ARCH") or "unspecified",

  -- Deprecated GPU flags:
  ["cuda"] = config.UNSPECIFIED,
  ["cuda-offline"] = config.UNSPECIFIED,
  ["cuda-arch"] = "unspecified",

  -- Legion runtime optimization flags:
  ["legion-leaf"] = true,
  ["legion-inner"] = true,
  ["legion-idempotent"] = true,
  ["legion-replicable"] = true,

  -- Dataflow optimization flags:
  ["flow"] = os.getenv('USE_RDIR') == '1' or false,
  ["flow-spmd"] = false,
  ["flow-spmd-shardsize"] = 1,
  ["flow-old-iteration-order"] = 0,

  -- Experimental auto-parallelization flags:
  ["parallelize"] = true,
  ["parallelize-dop"] = "4",
  ["parallelize-global"] = true,
  ["parallelize-debug"] = false,

  -- Experimental CUDA code generation flags:
  ["cuda-2d-launch"] = true,
  ["cuda-licm"] = true,
  ["cuda-generate-cubin"] = false,
  ["cuda-pretty-kernels"] = false,
  ["cuda-dump-ptx"] = false,

  -- Internal HIP code generation flags:
  ["hip-pretty-kernels"] = false,

  -- Miscellaneous, internal or special-purpose flags:
  ["aligned-instances"] = false,
  ["debug"] = false,
  ["warn-as-error"] = false,
  ["no-dynamic-branches"] = true,
  ["no-dynamic-branches-assert"] = false,
  ["override-demand-index-launch"] = false,
  ["index-launch-dynamic"] = true,
  ["override-demand-openmp"] = false,
  ["override-demand-cuda"] = false,
  ["pretty"] = false,
  ["pretty-verbose"] = false,
  ["debuginfo"] = false,
  ["layout-constraints"] = true,
  ["trace"] = true,
  ["validate"] = true,
  ["emergency-gc"] = false,
  ["jobs"] = "1",
  ["incr-comp"] = os.getenv('REGENT_INCREMENTAL') == '1' or false, -- incremental compilation
  ["opt-compile-time"] = true, -- compile time optimization

  -- Need this here to make the logger happy.
  ["log"] = "",
}

local function make_default_options(prefix, options)
  local result = terralib.newlist()
  for k, v in pairs(options) do
    result:insert(
      common_config.make_default_option(prefix .. k, k, type(v), v))
  end
  return result
end

local function check_consistency(options, args)
  if options["cuda"] ~= config.UNSPECIFIED then
    if options["gpu"] ~= "unspecified" then
      print("conflicting command line arguments: must specify at most one of -fcuda and -fgpu")
      print("note: -fcuda is deprecated, please switch to -fgpu")
      assert(false)
    end
    options["gpu"] = options["cuda"] >= 1 and "cuda" or "none"
  end

  if options["cuda-offline"] ~= config.UNSPECIFIED then
    if options["gpu-offline"] ~= config.UNSPECIFIED then
      print("conflicting command line arguments: must specify at most one of -fcuda-offline and -fgpu-offline")
      print("note: -fcuda-offline is deprecated, please switch to -fgpu-offline")
      assert(false)
    end
    options["gpu-offline"] = options["cuda-offline"]
  end
  if options["gpu-offline"] == config.UNSPECIFIED then
    options["gpu-offline"] = not data.is_luajit()
  end

  if options["cuda-arch"] ~= "unspecified" then
    if options["gpu-arch"] ~= "unspecified" then
      print("conflicting command line arguments: must specify at most one of -fcuda-arch and -fgpu-arch")
      print("note: -fcuda-arch is deprecated, please switch to -fgpu-arch")
      assert(false)
    end
    options["gpu-arch"] = options["cuda-arch"]
  end

  if options["gpu-offline"] == 1 and options["gpu-arch"] == "unspecified" then
    print("conflicting command line arguments: requested -fgpu-offline 1 but -fgpu-arch is unspecified")
    assert(false)
  end

  return options, args
end

function config.args()
  return check_consistency(
    common_config.args(
      make_default_options("-f", default_options),
      "-f"))
end

return config
