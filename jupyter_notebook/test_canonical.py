import legion_cffi
from legion_top import legion_canonical_python_main, legion_canonical_python_cleanup, top_level, is_control_replicated

print("Start canonical python test")
assert legion_cffi.is_legion_python == False
legion_canonical_python_main()
legion_canonical_python_main()
print(legion_cffi.ffi, legion_cffi.lib)
print(top_level.runtime, top_level.context, top_level.task)
print("Control replicated: ", is_control_replicated())
_max_dim = None
for dim in range(1, 9):
    try:
        getattr(legion_cffi.lib, 'legion_domain_get_rect_{}d'.format(dim))
    except AttributeError:
        break
    _max_dim = dim
assert _max_dim is not None, 'Unable to detect LEGION_MAX_DIM'
print("Max DIM:", _max_dim)
legion_canonical_python_cleanup()
legion_canonical_python_cleanup()