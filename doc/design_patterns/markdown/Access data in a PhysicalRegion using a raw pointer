Access data in a PhysicalRegion using a raw pointer

```cpp
void createImageFieldPointer(RegionAccessor<AccessorType::Generic, float> &acc,
                                                 int fieldID,
                                                 float *&field,
                                                 Rect<3> imageBounds,
                                                 PhysicalRegion region,
                                                 ByteOffset offset[3]) {
      acc = region.get_field_accessor(fieldID).typeify<float>();
      Rect<3> tempBounds;
      field = acc.raw_rect_ptr<3>(imageBounds, tempBounds, offset);
      assert(imageBounds == tempBounds);
    }
```