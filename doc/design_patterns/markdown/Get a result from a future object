Get a result from a Future object

```cpp
TaskLauncher launcher(FIBONACCI_TASK_ID, TaskArgument(&i,sizeof(i)));
Future future = runtime->execute_task(ctx, launcher);
...
int result = future.get_result<int>();
```
