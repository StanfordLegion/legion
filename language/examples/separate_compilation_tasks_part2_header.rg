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

-- This file is not meant to be run directly.

-- runs-with:
-- []

import "regent"

require("separate_compilation_common")

extern task other_regent_task(r : region(ispace(int1d), fs), s : region(ispace(int1d), fs))
where reads writes(r.{x, y}, s.z), reads(r.z, s.x), reduces+(s.y) end
