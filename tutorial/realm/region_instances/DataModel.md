# Realm Data Model
The core storage for application data is RegionInstance. Each
RegionInstance is associated with a Memory where application data is
persistenly stored and cannot be moved. Any data migration in Realm
involves a copy operation.

RegionInstance can represent a logical array of structures as
multi-dimensional rectangle for dense data or a bitmask for sparse
unstructured data.

```
void top_level_task() {
  RegionInstance instance;
  ...
}
```
