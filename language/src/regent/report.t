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

-- Error Reporting in Regent

local common_report = require("common/report")
local std = require("regent/std")

local report = {}

report.info = common_report.warn

report.warn = function(node, ...)
  if std.config["warn-as-error"] then
    common_report.error(node, ...)
  else
    common_report.warn(node, ...)
  end
end

report.error = common_report.error

return report
