Individual task launch

```cpp
// pass a single integer as argument
int i = ...;
Context ctx;
HighLevelRuntime *runtime;
TaskLauncher launcher(FIBONACCI_TASK_ID, TaskArgument(&i,sizeof(i)));
Future future = runtime->execute_task(ctx, launcher);
```
