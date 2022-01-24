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

-- This is a useful utility for debugging memory usage in Regent
-- programs. The print_maxrss function reports the current maxrss
-- (along with a message).

import "regent"

local c = terralib.includecstring([[
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
long getrusage_maxrss()
{
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0) abort();
  return usage.ru_maxrss;
}
]])

terra print_maxrss(message : rawstring)
  var usage = c.getrusage_maxrss()
  c.printf("%s: %.1f MB\n", message, usage / 1024.)
  return usage
end
print_maxrss("top of program")

task main()
  print_maxrss("top of main")
end
print_maxrss("calling start")
regentlib.start(main)
