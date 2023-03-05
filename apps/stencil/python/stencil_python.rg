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
--   ["-ll:cpu", "4", "-ll:py", "1", "-ll:pyimport", "stencil", "-foverride-demand-cuda", "1"]
-- ]

package.terrapath = package.terrapath .. ";" .. (arg[0]:match(".*/") or "./") .. "../../../language/examples/?.rg"

stencil_use_python_main = true
require("stencil_fast")
