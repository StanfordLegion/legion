# TaskLauncher

To spawn a single task we use a TaskLauncher object.
A TaskLauncher is a struct used for specifying the arguments necessary for launching a task.
Here we look at the first two arguments of TaskLauncher:

## ID - the registered ID of the task to be launched
## argument - pass-by-value input to the task

The second field has type TaskArgument which points to a buffer and specifies the size in bytes to copy by value
from the buffer.
This copy does not actually take place until the launcher object is passed to the execute_task call.
If there is more than one argument it is the responsibility of the application to pack the values into a single buffer.

Launching a task simply requires passing a TaskLauncher object and a context to the Legion runtime via the execute_task call.
The context object is an opaque handle that is passed to the enclosing parent task.
Legion task launches (like most Legion API calls) are asynchronous which means that the call returns immediately.
As a place holder for the return value of the task, the Legion runtime returns a Future.
Note that launcher objects can be re-used to launch as many tasks as desired and can be modified for
the next task launch immediately once the preceding execute_task call returns.
