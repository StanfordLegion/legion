# Virtually Mapped

In some cases tasks need only pass privileges for accessing a region without needing an explicit physical instance.
In these cases, the mapper which maps the task may request that one or more LogicalRegion requirements be virtually mapped.
In these cases no physical instance is created, but the task is still granted privileges for the requested
LogicalRegion and fields.
Tasks can test whether a PhysicalRegion has been virtually mapped by invoking the is_mapped method which
will return false if virtually mapped.
The Legion default mapper will never virtually map a region, but other mappers may choose to do so
and tasks should be implemented to handle such scenarios.
