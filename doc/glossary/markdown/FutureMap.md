# FutureMap

Legion provides FutureMap types as a mechanism for managing the many return values that are returned
from an index space task launch.
FutureMap objects store a future value for every point in the index space task launch.
Applications can either wait on the FutureMap for all tasks in the index space launch to complete,
or can extract individual Future objects associated with specific points in the index space task launch.
