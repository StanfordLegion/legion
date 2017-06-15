# Inline Mapping

To perform an inline mapping, applications create an InlineLauncher object similar to other launcher objects
for launching tasks.
The argument passed to the InlineLauncher constructor is a RegionRequirement which is used to describe the
LogicalRegion requested.
Once we have have set up the launcher, we invoke the map_region runtime method and pass the launcher.
This call returns a PhysicalRegion handle which represents the physical instance of the data.
In keeping with Legion’s deferred execution model, the map_region call is asynchronous,
allowing the application to issue many operations in flight and perform other useful work
while waiting for the region to be ready.
