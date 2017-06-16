# MustEpochLauncher

A MustEpochLauncher is actually a meta-launcher that simply contains other launcher objects.
The idea is that instead of launching a bunch of tasks to the Legion runtime separately and hoping they run in parallel,
applications can gather up a bunch of tasks (either a collection of individual tasks or one or more index tasks)
inside of a MustEpochLauncher and then issue them as a single launch to the Legion runtime.
The Legion runtime is then aware that all the tasks must be capable of executing in parallel and synchronizing
with each other.
Legion will first check that all of the region requirements for this set of tasks are non-interfering and
therefore capable of running in parallel.
Legion will also check any mapping decisions which might impact the ability of the tasks to run in parallel.
For example, if two tasks in a must-parallelism epoch are mapped onto the same processor they will not be able to
run in parallel and potentially synchronize with each other.
To help avoid this case, Legion provides an explicit mapping call for mapping must-parallelism epochs
map_must_epoch.
If there are any mapping decisions which would prevent the must-parallelism epoch,
Legion issues a runtime error (as opposed to silently hanging).
