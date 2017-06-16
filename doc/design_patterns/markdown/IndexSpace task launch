IndexSpace task launch

```cpp
int num_points = ...;
Rect<1> launch_bounds(Point<1>(0),Point<1>(num_points-1));
  Domain launch_domain = Domain::from_rect<1>(launch_bounds);

  ArgumentMap arg_map;
  for (int i = 0; i < num_points; i++) {
    int input = i + 10;
    arg_map.set_point(DomainPoint::from_point<1>(Point<1>(i)),
        TaskArgument(&input,sizeof(input)));
  }

  IndexLauncher index_launcher(HELLO_WORLD_INDEX_ID,
                               launch_domain,
                               TaskArgument(NULL, 0),
                               arg_map);

  FutureMap fm = runtime->execute_index_space(ctx, index_launcher);
  fm.wait_all_results();
  
  bool all_passed = true;
  for (int i = 0; i < num_points; i++) {
    int expected = 2*(i+10);
    int received = fm.get_result<int>(DomainPoint::from_point<1>(Point<1>(i)));
    if (expected != received) {
      printf("Check failed for point %d: %d != %d\n", i, expected, received);
      all_passed = false;
    }
  }
  if (all_passed)
    printf("All checks passed!\n");
```
