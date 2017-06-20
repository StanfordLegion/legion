# FieldAllocator

Fields are allocated by invoking the allocate_field method on a FieldAllocator.
When a field is allocated the application must specify the required data size for a field entry in bytes.
The allocate_field method will return a FieldID which is used to name the field.
Users may optionally specify the ID to be associated with the field being allocated using the
second parameter to allocate_field.
If this is done, then it is the responsibility of the user to ensure that each FieldID is used only
once for a each field space.
Legion supports parallel field allocation in the same field space by different tasks,
but undefined behavior will result if the same FieldID is allocated in the same field space by two different tasks.
