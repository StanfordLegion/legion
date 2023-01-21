-- Copyright 2023 Stanford University
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
-- [["-ll:py", "1", "-ll:pyimport", "python_main"]]

import "regent"

task hello(x : int64, y : double) : int64
  regentlib.c.printf("hello from Regent (got values from Python: %lld %f)\n", x, y)
  return 5678
end

fspace fs {
  x : double,
  y : double,
}

task saxpy(r : region(ispace(int1d), fs), a : double)
where reads writes(r) do
  for i in r do
    i.x = a*i.x + i.y
  end
end

extern task py_main()
py_main:set_task_id(2)

regentlib.start(py_main)
