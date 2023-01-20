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
-- [
--   ["-ll:gpu", "1", "-ll:csize", "1500", "-ll:fsize", "1500", "-foverride-demand-cuda", "1",
--    "-nx", "1000", "-ny", "1000", "-ntx", "4", "-nty", "4", "-tsteps", "50", "-tprune", "30"]
-- ]

package.terrapath = package.terrapath .. ";" .. (arg[0]:match(".*/") or "./") .. "../../../examples/?.rg"

require("stencil_fast")
