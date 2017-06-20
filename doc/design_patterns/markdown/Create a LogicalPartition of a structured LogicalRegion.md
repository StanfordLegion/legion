Create a LogicalPartition of a structured LogicalRegion

```cpp
      LogicalRegion image = ...;
      Point<3> regionSize;
      // HDTV imgae resolution
      regionSize.x[0] = 1920;
      regionSize.x[1] = 1080;
      regionSize.x[2] = numSimulationNodes;
      Point<3> fragmentSize;
      // Size of individual subregion 16x1x1
      fragmentSize.x[0] = 16;
      fragmentSize.x[1] = 1;
      fragmentSize.x[2] = 1;
      Blockify<3> coloring(fragmentSize);
      IndexPartition imageFragmentIndexPartition =
          runtime->create_index_partition(context, image.get_index_space(), coloring);
      runtime->attach_name(imageFragmentIndexPartition, "image fragment index");
      LogicalPartition imageFragmentPartition =
          mRuntime->get_logical_partition(context, image, imageFragmentIndexPartition);
      runtime->attach_name(imageFragmentPartition, "image fragment partition");
      Point<3> numFragments;
      numFragments.x[0] = regionSize.x[0] / fragmentSize.x[0];
      numFragments.x[1] = regionSize.x[1] / fragmentSize.x[1];
      numFragments.x[2] = regionSize.x[2] / fragmentSize.x[2];
      Rect<3> fragmentBounds(Point<3>::ZEROES(), numFragments - Point<3>::ONES());
      Domain fragmentDomain = Domain::from_rect<3>(fragmentBounds);
```