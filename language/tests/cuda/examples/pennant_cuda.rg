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
--   ["pennant.tests/sedovsmall/sedovsmall.pnt",
--    "-npieces", "1", "-seq_init", "1", "-par_init", "1", "-interior", "0",
--    "-ll:gpu", "1"],
--   ["pennant.tests/sedov/sedov.pnt",
--    "-npieces", "3", "-seq_init", "1", "-par_init", "1", "-interior", "0",
--    "-absolute", "2e-6", "-relative", "1e-8", "-relative_absolute", "1e-10",
--    "-ll:gpu", "1"],
--   ["pennant.tests/leblanc/leblanc.pnt",
--    "-npieces", "2", "-seq_init", "1", "-par_init", "1", "-interior", "0",
--    "-ll:gpu", "1"]
-- ]

package.terrapath = package.terrapath .. ";" .. (arg[0]:match(".*/") or "./") .. "../../../examples/?.rg"

require("pennant")
