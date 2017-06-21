# TaskArgument

A class for describing an untyped task argument.
Note that task arguments do not make copies of the data they point to.
Copies are only made upon calls to the runtime to avoid double copying.
It is up to the user to make sure that the the data described by a task argument
is valid throughout the duration of its lifetime.
