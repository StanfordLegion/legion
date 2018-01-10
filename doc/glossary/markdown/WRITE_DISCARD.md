# WRITE_DISCARD

WRITE_DISCARD is a special form of READ_WRITE privilege that permits the task to perform any kind of operation,
but informs the runtime that the task intends to overwrite all previous data stored in the LogicalRegion without reading it.
This enables the runtime to perform several performance optimizations associated with removing unnecessary data movement
operations.
