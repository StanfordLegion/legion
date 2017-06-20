# Index Space Task Launch

Launch a large number of non-interfering tasks in Legion using a single index space task launch. 
IndexSpace tasks are launched in a similar manner to individual tasks using a launcher object which has
the type IndexLauncher.
IndexLauncher objects take some of the same arguments as TaskLauncher objects such as the ID of the task to launch
and a TaskArgument which is passed by value as a global argument to all the points in the index space launch.
The IndexLauncher objects also take the additional arguments of an ArgumentMap and a Domain which describes
the set of tasks to create (one for each point).
 Just like individual tasks, index space task launches are performed asynchronously.

When launching a large set of tasks in a single call, we may want to pass different arguments to each task.
ArgumentMap types allow the user to pass different TaskArgument values to tasks associated with different points.

Legion also provides FutureMap types as a mechanism for managing the many return values that are returned from an
index space task launch.
