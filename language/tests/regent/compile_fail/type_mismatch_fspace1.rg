-- Copyright 2024 Stanford University
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
-- type_mismatch_fspace1.rg:32: type mismatch in argument 3: expected ptr(list($rd), $rd) but got ptr(list($re), $re)
--   walk(rd, re, le)
--      ^

import "regent"

fspace list(ra : region(list(ra))) {
  data : int,
  next : ptr(list(ra), ra),
}

task walk(rb : region(list(rb)), rc : region(list(rc)), lb : ptr(list(rb), rb))
end
walk:compile()

task top(rd : region(list(rd)), re : region(list(re)), le : ptr(list(re), re))
  walk(rd, re, le)
end
top:compile()
