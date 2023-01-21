-- Copyright 2023 Stanford University, NVIDIA Corporation
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
-- region_name_invalid3.rg:25: unnamed task element cannot have a named region element
-- task region#r {
--    ^

import "regent"
import "bishop"

mapper

task region#r {
  target : memories[kind=sysmem];
}

end

task toplevel()
end

regentlib.start(toplevel, bishoplib.make_entry())
