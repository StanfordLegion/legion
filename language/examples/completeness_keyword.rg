-- This task is a simple use of the `complete` keyword. It should fail if the
-- keyword is not defined

import "regent"

task name(r: region(int), p : partition(disjoint, complete, r))
end
