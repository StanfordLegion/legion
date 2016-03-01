-- Copyright 2016 Stanford University
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

import "regent"

local c = regentlib.c

task hello()
  c.printf("hello world\n")
end

task main()
  hello()
end
regentlib.saveobj(main, "saveobj", "executable")
-- If this were using regentlib.start, there's no way you'd ever call
-- main() three times. (Legion is not re-entrant.)
assert(os.execute("./saveobj") == 0)
assert(os.execute("./saveobj") == 0)
assert(os.execute("./saveobj") == 0)
