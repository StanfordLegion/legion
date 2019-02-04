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

-- fails-with:
-- invalid_import_ispace4.rg:28: cannot import a handle that is already imported

import "regent"

terra create_index_space(runtime : regentlib.c.legion_runtime_t,
                         context : regentlib.c.legion_context_t)
  return regentlib.c.legion_index_space_create(runtime, context, 1)
end

task main()
  var raw_is = create_index_space(__runtime(), __context())
  var is = __import_ispace(int1d, raw_is)
  var is_again = __import_ispace(int1d, raw_is)
end

regentlib.start(main)
