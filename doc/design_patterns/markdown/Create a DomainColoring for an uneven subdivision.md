Create a DomainColoring for an uneven subdivision

This example show how to create a DomainColoring and IndexPartition
when the number of elements does not evenly
subdivide the domain.

```cpp
  if ((num_elements % num_subregions) != 0) {
    // Not evenly divisible
    const int lower_bound = num_elements/num_subregions;
    const int upper_bound = lower_bound+1;
    const int number_small = num_subregions - (num_elements % num_subregions);
    DomainColoring coloring;
    int index = 0;
    for (int color = 0; color < num_subregions; color++) {
      int num_elmts = color < number_small ? lower_bound : upper_bound;
      assert((index+num_elmts) <= num_elements);
      Rect<1> subrect(Point<1>(index),Point<1>(index+num_elmts-1));
      coloring[color] = Domain::from_rect<1>(subrect);
      index += num_elmts;
    }
    ip = runtime->create_index_partition(ctx, is, color_domain, 
                                      coloring, true/*disjoint*/);
  }
  LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, ip);
```  