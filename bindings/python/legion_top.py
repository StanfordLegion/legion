#!/usr/bin/env python

# Copyright 2020 Stanford University, NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import absolute_import, division, print_function, unicode_literals

import gc
import os
import sys
import code
import types
import struct
import threading
import importlib

from legion_cffi import ffi, lib as c

try:
    unicode # Python 2
except NameError:
    unicode = str # Python 3

# This has to match the unique name in main.cc
_unique_name = 'legion_python'

# Storage for variables that apply to the top-level task.
# IMPORTANT: They are valid ONLY in the top-level task.
# or in global import tasks.
top_level = threading.local()
# Fields:
#     top_level.runtime
#     top_level.context
#     top_level.task
#     top_level.cleanup_items


def input_args(filter_runtime_options=False):
    raw_args = c.legion_runtime_get_input_args()

    args = []
    for i in range(raw_args.argc):
        args.append(ffi.string(raw_args.argv[i]).decode('utf-8'))

    if filter_runtime_options:
        i = 1 # Skip program name

        prefixes = ['-lg:', '-hl:', '-realm:', '-ll:', '-cuda:', '-numa:',
                    '-dm:', '-bishop:']
        while i < len(args):
            match = False
            for prefix in prefixes:
                if args[i].startswith(prefix):
                    match = True
                    break
            if args[i] == '-level':
                match = True
            if args[i] == '-logfile':
                match = True
            if match:
                args.pop(i)
                # Assume that every option has an argument, as long as
                # the subsequent value does **NOT** start with a dash.
                if i < len(args) and not args[i].startswith('-'):
                    args.pop(i)
                continue
            i += 1
    return args


def run_repl():
    try:
        shell = code.InteractiveConsole()
        shell.interact(banner='Welcome to Legion Python interactive console')
    except SystemExit:
        pass


def run_cmd(cmd, run_name=None):
    import imp
    module = imp.new_module(run_name)
    setattr(module, '__name__', run_name)
    setattr(module, '__package__', None)

    # Hide the current module if it exists.
    sys.modules[run_name] = module
    code = compile(cmd, '<string>', 'eval')
    exec(code, module.__dict__)


# We can't use runpy for this since runpy is aggressive about
# cleaning up after itself and removes the module before execution
# has completed.
def run_path(filename, run_name=None):
    import imp
    module = imp.new_module(run_name)
    setattr(module, '__name__', run_name)
    setattr(module, '__file__', filename)
    setattr(module, '__loader__', None)
    setattr(module, '__package__', run_name.rpartition('.')[0])

    # Hide the current module if it exists.
    old_module = sys.modules[run_name] if run_name in sys.modules else None
    sys.modules[run_name] = module

    sys.path.append(os.path.dirname(filename))

    with open(filename) as f:
        code = compile(f.read(), filename, 'exec')
        exec(code, module.__dict__)

    # FIXME: Can't restore the old module because tasks may be
    # continuing to execute asynchronously. We could fix this with
    # an execution fence but it doesn't seem worth it given that
    # we'll be cleaning up the process right after this.

    # sys.modules[run_name] = old_module


def import_global(module, check_depth=True):
    try:
        # We should only be doing something for this if we're the top-level task
        if c.legion_task_get_depth(top_level.task[0]) > 0 and check_depth:
            return
    except AttributeError:
        raise RuntimeError('"import_global" must be called in a legion_python task')
    if isinstance(module,str):
        name = module
    elif isinstance(module,unicode):
        name = module
    elif isinstance(module,types.ModuleType):
        name = module.__name__
    else:
        raise TypeError('"module" arg to "import_global" must be a ModuleType or str type')
    mapper = c.legion_runtime_generate_library_mapper_ids(
            top_level.runtime[0], _unique_name.encode('utf-8'), 1)
    future = c.legion_runtime_select_tunable_value(
            top_level.runtime[0], top_level.context[0], 0, mapper, 0)
    num_python_procs = struct.unpack_from('i',
            ffi.buffer(c.legion_future_get_untyped_pointer(future),4))[0]
    c.legion_future_destroy(future)
    assert num_python_procs > 0
    # Launch an index space task across all the python 
    # processors to import the module in every interpreter
    task_id = c.legion_runtime_generate_library_task_ids(
            top_level.runtime[0], _unique_name.encode('utf-8'), 2) + 1
    rect = ffi.new('legion_rect_1d_t *')
    rect[0].lo.x[0] = 0
    rect[0].hi.x[0] = num_python_procs - 1
    domain = c.legion_domain_from_rect_1d(rect[0])
    packed = name.encode('utf-8')
    arglen = len(packed)
    array = ffi.new('char[]', arglen)
    ffi.buffer(array, arglen)[:] = packed
    args = ffi.new('legion_task_argument_t *')
    args[0].args = array
    args[0].arglen = arglen
    argmap = c.legion_argument_map_create()
    launcher = c.legion_index_launcher_create(task_id, domain, 
            args[0], argmap, c.legion_predicate_true(), False, mapper, 0)
    future = c.legion_index_launcher_execute_reduction(top_level.runtime[0], 
            top_level.context[0], launcher, c.LEGION_REDOP_SUM_INT32)
    c.legion_index_launcher_destroy(launcher)
    c.legion_argument_map_destroy(argmap)
    result = struct.unpack_from('i',
            ffi.buffer(c.legion_future_get_untyped_pointer(future),4))[0]
    c.legion_future_destroy(future)
    if result > 0:
        raise ImportError('failed to globally import '+name+' on '+str(result)+' nodes')


# In general we discourage the use of this function, but some libraries are
# not safe to use with control replication so this will give them a way
# to check whether they are running in a safe context or not
def is_control_replicated():
    try:
        # We should only be doing something for this if we're the top-level task
        return c.legion_context_get_num_shards(top_level.runtime[0],
                top_level.context[0], True) > 1
    except AttributeError:
        raise RuntimeError('"is_control_replicated" must be called in a legion_python task')


# Helper class for deduplicating output streams with control replication
class LegionOutputStream(object):
    def __init__(self, shard_id, stream):
        self.shard_id = shard_id
        # This is the original stream
        self.stream = stream

    def close(self):
        self.stream.close()

    def flush(self):
        self.stream.flush()

    def write(self, string):
        # Only do the write if we are shard 0
        if self.shard_id == 0:
            self.stream.write(string)

    def writelines(self, sequence):
        # Only do the write if we are shard 0
        if self.shard_id == 0:
            self.stream.writelines(sequence)

    def isatty(self):
        return self.stream.isatty()


def legion_python_main(raw_args, user_data, proc):
    raw_arg_ptr = ffi.new('char[]', bytes(raw_args))
    raw_arg_size = len(raw_args)

    # Execute preamble to obtain Legion API context.
    task = ffi.new('legion_task_t *')
    raw_regions = ffi.new('legion_physical_region_t **')
    num_regions = ffi.new('unsigned *')
    context = ffi.new('legion_context_t *')
    runtime = ffi.new('legion_runtime_t *')
    c.legion_task_preamble(
        raw_arg_ptr, raw_arg_size, proc,
        task, raw_regions, num_regions, context, runtime)

    top_level.runtime, top_level.context, top_level.task = runtime, context, task
    top_level.cleanup_items = []

    # If we're control replicated, deduplicate stdout
    if is_control_replicated():
        shard_id = c.legion_context_get_shard_id(runtime[0], context[0], True)
        sys.stdout = LegionOutputStream(shard_id, sys.stdout)

    # Run user's script.
    args = input_args(True)
    start = 1
    if len(args) > 1 and args[1] == '--nocr':
        start += 1
    if len(args) < (start+1) or args[start] == '-':
        run_repl()
    elif args[start] == '-c':
        assert len(args) >= 3
        sys.argv = list(args)
        run_cmd(args[start+1], run_name='__main__')
    else:
        assert len(args) >= (start+1) 
        sys.argv = list(args)
        run_path(args[start], run_name='__main__')

    # # Hack: Keep this thread alive because otherwise Python will reuse
    # # it for task execution and Pygion's thread-local state (_my.ctx)
    # # will get messed up.
    # c.legion_future_get_void_result(
    #     c.legion_runtime_issue_execution_fence(runtime[0], context[0]))

    for cleanup in top_level.cleanup_items:
        cleanup()

    del top_level.runtime
    del top_level.context
    del top_level.task
    del top_level.cleanup_items

    # Force a garbage collection so that we know that all objects whic can 
    # be collected are actually collected before we exit the top-level task
    gc.collect()

    # Execute postamble.
    c.legion_task_postamble(runtime[0], context[0], ffi.NULL, 0)


# This is our helper task for ensuring that python modules are imported
# globally on all python processors across the system
def legion_python_import_global(raw_args, user_data, proc):
    raw_arg_ptr = ffi.new('char[]', bytes(raw_args))
    raw_arg_size = len(raw_args)

    # Execute preamble to obtain Legion API context.
    task = ffi.new('legion_task_t *')
    raw_regions = ffi.new('legion_physical_region_t **')
    num_regions = ffi.new('unsigned *')
    context = ffi.new('legion_context_t *')
    runtime = ffi.new('legion_runtime_t *')
    c.legion_task_preamble(
        raw_arg_ptr, raw_arg_size, proc,
        task, raw_regions, num_regions, context, runtime)

    top_level.runtime, top_level.context, top_level.task = runtime, context, task

    # Get the name of the task 
    module_name = ffi.unpack(ffi.cast('char*', c.legion_task_get_args(task[0])), 
            c.legion_task_get_arglen(task[0])).decode('utf-8')
    try:
        globals()[module_name] = importlib.import_module(module_name)
        failures = 0
    except ImportError:
        failures = 1

    del top_level.runtime
    del top_level.context
    del top_level.task

    result = struct.pack('i',failures)

    c.legion_task_postamble(runtime[0], context[0], ffi.from_buffer(result), 4)

