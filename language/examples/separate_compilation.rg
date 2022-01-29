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

-- runs-with:
-- [["-fseparate", "1", "-fjobs", "2", "-fincr-comp", "1"]]

-- FIXME: Separate compilation in JIT mode requires either incremental
-- or parallel compilation, due to a bad interaction when using the
-- LLVM JIT linker mode.

import "regent"

assert(regentlib.config["separate"], "test requires separate compilation")

-- Make sure this all happens in a temporary directory in case we're
-- running concurrently.
local tmp_dir
do
  -- use os.tmpname to get a hopefully-unique directory to work in
  local tmpfile = os.tmpname()
  tmp_dir = tmpfile .. ".d/"
  assert(os.execute("mkdir " .. tmp_dir) == 0)
  os.remove(tmpfile)  -- remove this now that we have our directory
end

-- Compile separate tasks.
local root_dir = arg[0]:match(".*/") or "./"
local loaders = terralib.newlist()
for part = 1, 2 do
  local regent_exe = os.getenv('REGENT') or 'regent'
  local tasks_rg = "separate_compilation_tasks_part" .. part .. ".rg"
  assert(os.execute("cp " .. root_dir .. tasks_rg .. " " .. tmp_dir .. tasks_rg) == 0)
  local tasks_h = "separate_compilation_tasks_part" .. part .. ".h"
  local tasks_so = tmp_dir .. "libseparate_compilation_tasks_part" .. part .. ".so"
  if os.execute(regent_exe .. " " .. tmp_dir .. tasks_rg .. " -fseparate 1") ~= 0 then
    print("Error: failed to compile " .. tmp_dir .. tasks_rg)
    assert(false)
  end
  local tasks_c = terralib.includec(tasks_h, {"-I", tmp_dir})
  loaders:insert(tasks_c["separate_compilation_tasks_part" .. part .. "_h_register"])
  terralib.linklibrary(tasks_so)
end
terra loader()
  [loaders:map(function(thunk) return `thunk() end)]
end

struct fs {
  x : int
  y : int
  z : int
}

extern task my_regent_task(r : region(ispace(int1d), fs), x : int, y : double, z : bool)
where reads writes(r.{x, y}), reads(r.z) end

extern task other_regent_task(r : region(ispace(int1d), fs), s : region(ispace(int1d), fs))
where reads writes(r.{x, y}, s.z), reads(r.z, s.x), reduces+(s.y) end

task main()
  var r = region(ispace(int1d, 5), fs)
  var s = region(ispace(int1d, 10), fs)
  var pr = partition(equal, r, ispace(int1d, 4))
  var ps = partition(equal, s, ispace(int1d, 4))
  fill(r.{x, y, z}, 0)
  fill(s.{x, y, z}, 0)
  for i = 0, 4 do
    my_regent_task(pr[i], 1, 2, true)
  end
  for i = 0, 4 do
    other_regent_task(pr[i], ps[i])
  end
end
regentlib.start(main, loader)

-- os.execute("rm -r " .. tmp_dir)
