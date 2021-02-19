-- Copyright 2021 Stanford University
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

local unprefixed_options = {
  -- Output flags:
  ["o"] = "",
}

local prefixed_options = {
  -- Main user-facing correctness flags:
  ["bounds-checks"] = false,
  ["bounds-checks-targets"] = ".*",

  -- Main user-facing optimization flags:
  ["cuda"] = config.UNSPECIFIED,
  ["cuda-offline"] = not data.is_luajit(),
  ["cuda-arch"] = os.getenv("GPU_ARCH") or "fermi",
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

  -- Legion runtime optimization flags:
  ["legion-leaf"] = true,
  ["legion-inner"] = true,
  ["legion-idempotent"] = true,
  ["legion-replicable"] = true,

  -- Dataflow optimization flags:
  ["flow"] = os.getenv('USE_RDIR') == '1' or false,
  ["flow-spmd"] = false,
  ["flow-spmd-shardsize"] = 1,

  -- Experimental auto-parallelization flags:
  ["parallelize"] = true,
  ["parallelize-dop"] = "4",
  ["parallelize-global"] = true,
  ["parallelize-debug"] = false,

  -- Experimental CUDA code generation flags:
  ["cuda-2d-launch"] = true,
  ["cuda-licm"] = true,

  -- Miscellaneous, internal or special-purpose flags:
  ["aligned-instances"] = false,
  ["debug"] = false,
  ["warn-as-error"] = false,
  ["no-dynamic-branches"] = true,
  ["no-dynamic-branches-assert"] = false,
  ["override-demand-index-launch"] = false,
  ["override-demand-openmp"] = false,
  ["override-demand-cuda"] = false,
  ["allow-loop-demand-parallel"] = false,
  ["pretty"] = false,
  ["pretty-verbose"] = false,
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

function config.args()
  local options = terralib.newlist()
  common_config.add_default_options("-", unprefixed_options, options)
  common_config.add_default_options("-f", prefixed_options, options)
  return common_config.args(options, "-f")
end

return config
