Preregister tasks and start Legion runtime

```cpp
enum TaskID {
  HELLO_WORLD_ID,
};

 HighLevelRuntime::set_top_level_task_id(HELLO_WORLD_ID);
  HighLevelRuntime::register_legion_task<hello_world_task>(HELLO_WORLD_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);

  return HighLevelRuntime::start(argc, argv);
```
